from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Document
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.retrievers import AutoMergingRetriever
from typing import Dict, List, Optional, AsyncIterator, Callable
from .utils import get_supported_file_types, format_supported_types
from .file_handlers import ExcelReader, DocxReader
import os
import re
import asyncio
import logging
from contextlib import contextmanager
import shutil
import datetime
import json
from lib.utils.debug import debug_box
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.events.retrieval import RetrievalStartEvent, RetrievalEndEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler

dispatcher = instrument.get_dispatcher(__name__)
class RetrievalEventHandler(BaseEventHandler):
    def handle(self, event):
        print(f"Retrieval Event: (event)")

event_handler = RetrievalEventHandler()
dispatcher.add_event_handler(event_handler)


def apply_minimum_content_threshold(text, score, min_length=20):
    """Penalize documents that are too short to be meaningful."""
    if len(text) < min_length:
        penalty = 0.5 * (len(text) / min_length)
        return score * penalty
    return score

def normalize_by_length(score, text_length, max_length=5000):
    """Normalize score based on document length to prevent bias toward longer docs."""
    # Penalize very short documents (might be noise) and very long documents
    length_factor = min(text_length, max_length) / max_length
    # Apply a sigmoid-like normalization that favors mid-length documents
    length_adjustment = (4 * length_factor * (1 - length_factor)) ** 0.5
    return score * length_adjustment

def boost_for_exact_matches(query_text, node_text, base_score):
    """Boost scores for documents with exact matches of important terms."""
    # Extract important terms (non-stopwords)
    # Simple stopwords list for demonstration
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                  'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 
                  'by', 'about', 'against', 'between', 'into', 'through', 'during', 
                  'before', 'after', 'above', 'below', 'from', 'up', 'down', 'of', 
                  'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
                  'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
                  'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
                  'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 
                  'just', 'should', 'now'}
    
    # Extract important terms from query
    important_terms = [term.lower() for term in re.findall(r'\b\w+\b', query_text) 
                      if term.lower() not in stop_words and len(term) > 3]
    
    # Count exact matches in document
    match_count = sum(1 for term in important_terms if re.search(r'\b' + re.escape(term) + r'\b', 
                                                               node_text.lower()))
    
    # Apply boost based on match ratio
    if important_terms:
        match_ratio = match_count / len(important_terms)
        debug_box(f"Important terms: {important_terms}")
        debug_box(f"Match count {match_count} Match ratio {match_ratio}")
        print(f"Match ratio: {match_ratio}")
        boost_factor = 1.0 + (match_ratio * 0.5)  # Up to 60% boost
        return min(base_score * boost_factor, 1.0)
    return base_score

def adjust_score_by_metadata(metadata, query_text, base_score):
    """Adjust score based on document metadata."""
    file_name = metadata.get('file_name', '').lower()
    file_type = metadata.get('file_type', '').lower()
    
    # Extract key terms from query
    query_terms = set(query_text.lower().split())
    
    # Check if filename contains query terms
    filename_match = any(term in file_name for term in query_terms if len(term) > 3)
    
    # Boost for relevant file types (e.g., PDF for forms)
    file_type_boost = 1.0
    if 'form' in query_text.lower() and 'pdf' in file_type:
        file_type_boost = 1.15
    
    # Apply boosts
    adjusted_score = base_score
    if filename_match:
        adjusted_score *= 1.2  # 20% boost for filename match
    
    adjusted_score *= file_type_boost
    
    return min(adjusted_score, 1.0)  # Cap at 1.0

logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass

def get_file_handlers(supported_types: Dict[str, bool]) -> Dict[str, callable]:
    """Get dictionary of file handlers based on available support."""
    handlers = {}
    if supported_types[".xlsx"]:
        handlers.update({".xlsx": ExcelReader(), ".xls": ExcelReader()})
    if supported_types[".docx"]:
        handlers[".docx"] = DocxReader()
    return handlers

@contextmanager
def atomic_index_update(kb_instance: 'HierarchicalKnowledgeBase'):
    """Context manager for atomic index updates with rollback capability."""
    # Create backup of persist_dir if it exists
    backup_dir = None
    if os.path.exists(kb_instance.persist_dir):
        backup_dir = f"{kb_instance.persist_dir}_backup"
        shutil.copytree(kb_instance.persist_dir, backup_dir)
    
    try:
        yield
        # If we get here, the operation succeeded
        if backup_dir and os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
    except Exception as e:
        # Restore from backup if operation failed
        if backup_dir and os.path.exists(backup_dir):
            if os.path.exists(kb_instance.persist_dir):
                shutil.rmtree(kb_instance.persist_dir)
            shutil.move(backup_dir, kb_instance.persist_dir)
            # Reload index from backup
            kb_instance.storage_context = StorageContext.from_defaults(
                persist_dir=kb_instance.persist_dir)
            kb_instance.index = load_index_from_storage(kb_instance.storage_context)
        raise DocumentProcessingError(f"Index update failed: {str(e)}") from e

class HierarchicalKnowledgeBase:
    def __init__(self, persist_dir: str, 
                 chunk_sizes: List[int] = [2048, 512, 128],
                 embedding_model: Optional[str] = None,
                 batch_size: int = 10):
        # Cache for retrievers
        self._retriever = None
        self.persist_dir = persist_dir
        self.chunk_sizes = chunk_sizes
        self.supported_types = get_supported_file_types()
        self.file_handlers = get_file_handlers(self.supported_types)
        self.supported_types = get_supported_file_types()
        self.batch_size = batch_size
        
        # Configure embedding model        # Configure embedding model
        if embedding_model is None or embedding_model.lower() == 'openai':
            self.embed_model = None  # LlamaIndex will use OpenAI default
        else:
            # Assume it's a HuggingFace model name
            from llama_index.embeddings import HuggingFaceEmbedding
            self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes
        )
        
        self.index = None
        self._clear_retriever_cache()
        
        # Load existing index if storage exists
        if os.path.exists(persist_dir):
            os.makedirs(persist_dir, exist_ok=True)

    def _index_files_exist(self) -> bool:
        """Check if all required index files exist."""
        required_files = [
            'docstore.json',
            'default__vector_store.json',
            'index_store.json'
        ]
        return all(
            os.path.exists(os.path.join(self.persist_dir, fname))
            for fname in required_files
        )

    def load_if_exists(self) -> bool:
        """Load index from storage if it exists. Returns True if loaded."""
        if self._index_files_exist():
            self.storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(self.storage_context)
            return True
        return False
    
    async def process_documents(self, documents, progress_callback: Optional[Callable] = None) -> AsyncIterator[List]:
        """Process documents in batches, yielding nodes."""
        total_docs = len(documents)
        processed = 0
        current_batch = []
        
        for doc in documents:
            current_batch.append(doc)
            if len(current_batch) >= self.batch_size:
                nodes = self.node_parser.get_nodes_from_documents(current_batch)
                processed += len(current_batch)
                if progress_callback:
                    progress_callback(processed / total_docs)
                current_batch = []
                yield nodes
        
        if current_batch:  # Process remaining documents
            nodes = self.node_parser.get_nodes_from_documents(current_batch)
            if progress_callback:
                progress_callback(1.0)
            yield nodes
    
    async def create_index(self, data_dir: str, progress_callback: Optional[Callable] = None, skip_if_exists: bool = True):
        """Create a new index from a directory of documents."""
        try:
            # Configure custom file handlers based on available support
            # PDF is handled automatically when PyPDF2/pypdf is installed
            # .txt and .md are handled by default
            file_extractor = {}
            if self.supported_types[".xlsx"]:
                file_extractor.update({
                    ".xlsx": ExcelReader(),
                    ".xls": ExcelReader()
                })
            if self.supported_types[".docx"]:
                file_extractor[".docx"] = DocxReader()
            
            # Skip if index exists and flag is set
           
            # Load documents with custom handlers
            documents = SimpleDirectoryReader(data_dir, file_extractor=file_extractor).load_data()
            if not documents:
                raise ValueError(f"No documents found in {data_dir}")
                
            all_nodes = []
            async for nodes in self.process_documents(documents, progress_callback):
                print("Processing node..")
                all_nodes.extend(nodes)

            if skip_if_exists and self.load_if_exists():
                print("Using existing index")
                self.index.refresh_ref_docs(documents)
                self.index.storage_context.persist(persist_dir=self.persist_dir)
                self._clear_retriever_cache()

                return self.index
 

            with atomic_index_update(self):
                self.index = VectorStoreIndex(
                    all_nodes,
                    embed_model=self.embed_model
                )
                # Persist the index
                self.index.storage_context.persist(persist_dir=self.persist_dir)
                self._clear_retriever_cache()
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise DocumentProcessingError(f"Index creation failed: {str(e)}") from e
            
        return self.index
    
    async def add_document(self, file_path: str, 
                          progress_callback: Optional[Callable] = None,
                          refresh_mode: bool = True):
        """Add a single new document to the index.
        
        Args:
            file_path: Path to the document to add
            progress_callback: Optional callback for progress updates
            refresh_mode: If True, use refresh_ref_docs instead of insert_nodes
        """
        if not self.index:
            raise ValueError("Index not initialized. Call create_index first.")

        try:
            # Load and process new document
            file_extractor = self._get_file_extractors()
            documents = SimpleDirectoryReader(input_files=[file_path], 
                                              filename_as_id=True,
                                              file_extractor=file_extractor).load_data()
            
            if not documents:
                raise ValueError(f"No content found in {file_path}")
            
            if refresh_mode:
                # Use refresh_ref_docs for automatic updates
                self.index.refresh_ref_docs(documents)
                self.index.storage_context.persist(persist_dir=self.persist_dir)
                self._clear_retriever_cache()
            else:
                # Use traditional insert method
                await self._add_document_insert(file_path, progress_callback)

        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise DocumentProcessingError(f"Document addition failed: {str(e)}") from e

    async def _add_document_insert(self, file_path: str, 
                          progress_callback: Optional[Callable] = None):
        """Add a single new document to the index."""
        if not self.index:
            raise ValueError("Index not initialized. Call create_index first.")
            
        try:
            # Load and process new document
            # Configure custom file handlers based on available support
            file_extractor = self._get_file_extractors()

            # Load document with custom handlers
            documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
            if not documents:
                raise ValueError(f"No content found in {file_path}")
            
            all_nodes = []
            async for nodes in self.process_documents(documents, progress_callback):
                all_nodes.extend(nodes)
            
            # Update index atomically
            with atomic_index_update(self):
                for node in all_nodes:
                    self.index.insert_nodes([node])
                self.index.storage_context.persist(persist_dir=self.persist_dir)
                self._clear_retriever_cache()
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise DocumentProcessingError(f"Document addition failed: {str(e)}") from e
    
    async def remove_document(self, doc_id: str):
        """Remove a document and all its hierarchical nodes from the index."""
        if not self.index:
            raise ValueError("Index not initialized.")
            
        try:
            # Get all nodes associated with this document
            nodes_to_remove = set()
            for node_id, node in self.index.docstore.docs.items():
                if hasattr(node, 'relationships') and hasattr(node.relationships, 'source'):
                    if node.relationships.source.node_id == doc_id:
                        nodes_to_remove.add(node_id)
                        # Also add any child nodes
                        if hasattr(node.relationships, 'children'):
                            for child in node.relationships.children:
                                nodes_to_remove.add(child.node_id)
        
            with atomic_index_update(self):
                # Remove nodes from docstore and vector store
                for node_id in nodes_to_remove:
                    # Remove from docstore
                    if node_id in self.index.docstore.docs:
                        del self.index.docstore.docs[node_id]
                    # Remove from vector store
                    self.index.vector_store.delete(node_id)
                
                # Persist updates
                self.index.storage_context.persist(persist_dir=self.persist_dir)
                self._clear_retriever_cache()
        except Exception as e:
            logger.error(f"Failed to remove document: {str(e)}")
            raise DocumentProcessingError(f"Document removal failed: {str(e)}") from e
    
    def get_document_info(self) -> List[Dict]:
        """Get information about all documents and their hierarchical structure."""
        if not self.index:
            return []
            
        docs_info = {}
        seen_doc_ids = set()

        for id, doc in self.index.docstore.docs.items():
            print(id, doc.metadata)
            if doc.metadata['file_path'] not in docs_info:
                docs_info[doc.metadata['file_path']] = doc.metadata
            print("--------------------------------------------------------------------")
        # convert from dict to list
        as_list = []
        for k, v in docs_info.items():
            as_list.append(v)
        return as_list

        for node_id, node in self.index.docstore.docs.items(): #.docstore.docs.items():
            print(node_id, node)
            x="""
            if True or hasattr(node, 'relationships') and hasattr(node.relationships, 'source'):
                try:
                    doc_id = node.relationships.source.node_id
                    if doc_id not in seen_doc_ids:
                        doc = self.index.docstore.docs[doc_id]
                        # Count nodes at each level
                        level_counts = {size: 0 for size in self.chunk_sizes}
                        for n in self.index.docstore.docs.values():
                            if (hasattr(n, 'relationships') and 
                                hasattr(n.relationships, 'source') and
                                n.relationships.source.node_id == doc_id):
                                # Approximate which level this node belongs to
                                node_size = len(n.text)
                                for size in self.chunk_sizes:
                                    if node_size <= size:
                                        level_counts[size] += 1
                                        break
                        
                        docs_info.append({
                            'doc_id': doc_id,
                            'filename': doc.metadata.get('file_name', 'Unknown'),
                            'file_path': doc.metadata.get('file_path', 'Unknown'),
                            'creation_date': doc.metadata.get('creation_date', 'Unknown'),
                            'level_counts': level_counts,
                            'total_nodes': sum(level_counts.values())
                        })
                        seen_doc_ids.add(doc_id)
                except Exception as e:
                    pass
                    #logger.error(f"Failed to get document info: {str(e)}")
                    #    raise DocumentProcessingError(f"Document info retrieval failed: {str(e)}") from e
                """

        return docs_info
    
    async def query(self, query_text: str, similarity_top_k: int = 12):
        """Query the index."""
        if not self.index:
            raise ValueError("Index not initialized.")
            
        try:
            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k)
            return await query_engine.aquery(query_text)
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise DocumentProcessingError(f"Query failed: {str(e)}") from e
    
    def _get_file_extractors(self) -> Dict:
        """Get file extractors dictionary based on supported types."""
        file_extractor = {}
        if self.supported_types[".xlsx"]:
            file_extractor.update({".xlsx": ExcelReader(), ".xls": ExcelReader()})
        if self.supported_types[".docx"]:
            file_extractor[".docx"] = DocxReader()
        return file_extractor

    def _clear_retriever_cache(self):
        """Clear the cached retrievers when index changes."""
        self._retriever = None
        print("Cleared retriever cache")

    @dispatcher.span
    async def retrieve_relevant_nodes(self, query_text: str, similarity_top_k: int = 15):
        """Get raw retrieval results without LLM synthesis.
        
        Returns:
            List of tuples (node text, metadata, score, chunk_size)
            chunk_size helps identify which level of the hierarchy the node is from
        """
        if not self.index:
            raise ValueError("Index not initialized.")
        dispatcher.event(RetrievalStartEvent(query=query_text))
        
        retriever_start = datetime.datetime.now()

        # Use cached retriever or create new one
        if self._retriever is None:
            #self._retriever = AutoMergingRetriever(
            #    self.index.as_retriever(similarity_top_k=similarity_top_k),
            #    storage_context=self.index.storage_context
            #)
            self._retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
            print(f"Created new retriever: {datetime.datetime.now() - retriever_start}")
        else:
            print("Using cached retriever")
        
        retriever_end = datetime.datetime.now()
        print(f"Total retriever setup time: {retriever_end - retriever_start}")
        
        nodes = await self._retriever.aretrieve(query_text)
        
        # Get raw results
        raw_results = [(node.node.text, 
                 node.node.metadata,
                 node.score,
                 len(node.node.text)) for node in nodes]
        
        # Apply score adjustments
        enhanced_results = []
        for node_text, metadata, score, chunk_size in raw_results:
            # Penalize very short documents
            adjusted_score = apply_minimum_content_threshold(node_text, score)
            
            # Boost for exact keyword matches
            adjusted_score = boost_for_exact_matches(query_text, node_text, adjusted_score)
            
            # Normalize by length to prevent bias toward very long documents
            adjusted_score = normalize_by_length(adjusted_score, len(node_text))
            
            # Apply metadata-based adjustments
            adjusted_score = adjust_score_by_metadata(metadata, query_text, adjusted_score)
            
            enhanced_results.append((node_text, metadata, adjusted_score, chunk_size))
        
        # Re-sort by adjusted scores
        enhanced_results.sort(key=lambda x: x[2], reverse=True)

        dispatcher.event(RetrievalEndEvent(query=query_text))

        return enhanced_results
    
    async def get_relevant_context(self, query_text: str, 
                            similarity_top_k: int = 15,
                            format_type: str = "detailed",
                            min_score: float = 0.65) -> str:
        """Get formatted context from relevant nodes.
        
        Args:
            query_text: The query to match against
            similarity_top_k: Number of matches to retrieve
            format_type: How to format output ('markdown', 'plain', or 'detailed')
            min_score: Minimum similarity score to include (0.0 to 1.0)
        
        Returns:
            Formatted string with relevant context
        """
        query_stats = {}
        # start timing
        start_time = datetime.datetime.now()

        results = await self.retrieve_relevant_nodes(query_text, similarity_top_k)
        
        # Filter by minimum score
        results = [r for r in results if r[2] >= min_score]
        
        if not results:
            return ""
            
        if format_type == "markdown":
            context = "### [Retrieved from knowledge base]\n\n"
            for text, _, score, _ in results:
                context += f"> {text}\n\n"
        elif format_type == "plain":
            context = "[Retrieved from knowledge base]\n\n"
            context += "\n\n".join(text for text, _, _, _ in results)
        else:  # detailed
            context = "### [Retrieved Knowledge Base Results]\n"
            context += "Note: Results are ranked by relevance score (0-1). Higher scores indicate stronger matches.\n"
            context += "Some results with lower scores may be less relevant or unrelated to user query.\n\n"
            context += "=" * 80 + "\n\n"  # Distinctive separator at start

            for text, metadata, score, chunk_size in results:
                # Extract metadata fields
                file_name = metadata.get('file_name', 'Unknown')
                file_type = metadata.get('file_type', 'Unknown')
                creation_date = metadata.get('creation_date', 'Unknown')
                # Format metadata header
                context += f"[File: {file_name} | Type: {file_type} | "
                context += f"Created: {creation_date} | "
                context += f"Score: {score:.3f} | Size: {chunk_size}]\n"
                context += f"{text}\n"
                context += "-" * 80 + "\n\n"  # Separator between chunks
            
            context += "=" * 80 + "\n"  # Distinctive separator at end
            context += "[End of Retrieved Results]\n"

        # end timing
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        query_stats['total_time'] = total_time
        query_stats['retriever_creation'] = bool(self._retriever is None)
        
        print(f"Time taken to get relevant context: {total_time}")
        print(f"Retrievers {'created' if query_stats['retriever_creation'] else 'reused'}")

        return context.strip(), query_stats



