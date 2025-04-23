from llama_index.core.indices import VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.storage import StorageContext
from llama_index.core import Document, load_index_from_storage
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient
from typing import Dict, List, Optional, AsyncIterator, Callable
from .utils import get_supported_file_types, format_supported_types
from .file_handlers import ExcelReader, DocxReader
import os
from .keyword_matching.enhanced_matching import enhance_search_results
import re
import asyncio
import logging
from contextlib import contextmanager
import csv
import shutil
import datetime
import json
from lib.utils.debug import debug_box
import re
import hashlib
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.events.retrieval import RetrievalStartEvent, RetrievalEndEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
import traceback
import jsonpickle
import hashlib

dispatcher = instrument.get_dispatcher(__name__)
class RetrievalEventHandler(BaseEventHandler):
    def handle(self, event):
        print(f"Retrieval Event: (event)")

event_handler = RetrievalEventHandler()
dispatcher.add_event_handler(event_handler)

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
        self.using_old_style = False  # Flag to track if we're using old-style indices
        # Cache for retrievers
        # Cache for retrievers
        self._retriever = None
        self.persist_dir = persist_dir
        self.chunk_sizes = chunk_sizes
        self.supported_types = get_supported_file_types()
        self.file_handlers = get_file_handlers(self.supported_types)
        self.supported_types = get_supported_file_types()
        self.batch_size = batch_size
        
        # ChromaDB setup
        self.chroma_dir = os.path.join(persist_dir, "chroma_db")
        os.makedirs(self.chroma_dir, exist_ok=True)
        self.chroma_client = PersistentClient(path=self.chroma_dir)
        
        # Create collections for text and metadata
        self.text_collection_name = "text_index"
        self.metadata_collection_name = "metadata_index"
        
        # Add verbatim document tracking
        self.verbatim_docs_dir = os.path.join(persist_dir, "verbatim_docs")
        self.verbatim_docs_index_path = os.path.join(persist_dir, "verbatim_docs_index.json")
        
        # Add URL document tracking
        self.url_docs_dir = os.path.join(persist_dir, "url_docs")
        self.url_docs_index_path = os.path.join(persist_dir, "url_docs_index.json")
        self.verbatim_docs = {}
        
        # Add CSV document tracking
        self.csv_docs_dir = os.path.join(persist_dir, "csv_docs")
        self.csv_docs_index_path = os.path.join(persist_dir, "csv_docs_index.json")
        self.url_docs = {}
        
        # Create verbatim docs directory if it doesn't exist
        os.makedirs(self.verbatim_docs_dir, exist_ok=True)
        
        # Load verbatim docs index if it exists
        if os.path.exists(self.verbatim_docs_index_path):
            try:
                with open(self.verbatim_docs_index_path, 'r') as f:
                    self.verbatim_docs = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load verbatim docs index: {str(e)}")
        
        # Create CSV docs directory if it doesn't exist
        os.makedirs(self.csv_docs_dir, exist_ok=True)
        
        # Load CSV docs index if it exists
        self.csv_docs = {}
        if os.path.exists(self.csv_docs_index_path):
            try:
                with open(self.csv_docs_index_path, 'r') as f:
                    self.csv_docs = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load CSV docs index: {str(e)}")
        
        # Create URL docs directory if it doesn't exist
        os.makedirs(self.url_docs_dir, exist_ok=True)
        
        # Load URL docs index if it exists
        if os.path.exists(self.url_docs_index_path):
            try:
                with open(self.url_docs_index_path, 'r') as f:
                    self.url_docs = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load URL docs index: {str(e)}")
        
        # Configure embedding model        # Configure embedding model
        if embedding_model is None or embedding_model.lower() in ['openai', 'default']:
            self.embed_model = None  # LlamaIndex will use OpenAI default
        else:
            # Assume it's a HuggingFace model name
            from llama_index.embeddings import HuggingFaceEmbedding
            self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes
        )
        
        self.index = None
        self.text_index = None
        self.metadata_index = None
        
        # Load existing index if storage exists
        if os.path.exists(persist_dir):
            os.makedirs(persist_dir, exist_ok=True)
            if self.load_if_exists():
                from .csv_handler import CSVDocumentHandler
                self.csv_handler = CSVDocumentHandler(self)
                print("Loaded existing index and initialized CSV handler")

    def _index_files_exist(self) -> bool:
        """Check if all required index files exist."""
        # Check for ChromaDB files first (preferred)
        chroma_exists = os.path.exists(self.chroma_dir) and os.path.isdir(self.chroma_dir)
        if chroma_exists:
            # Check if collections exist
            try:
                self.chroma_client.get_collection(self.text_collection_name)
                return True
            except Exception:
                chroma_exists = False  # Collection doesn't exist or can't be accessed
                pass
        
        # If we get here, ChromaDB files don't exist or collection couldn't be accessed
        self.using_old_style = False
        
        # Check for old-style index files as fallback
        old_style_files = [
            'docstore.json',
            'default__vector_store.json',
            'index_store.json'
        ]
        old_style_exists = all(
            os.path.exists(os.path.join(self.persist_dir, fname))
            for fname in old_style_files
        )

        if old_style_exists:
            self.using_old_style = True
        
        return chroma_exists or old_style_exists

    def _setup_vector_stores(self):
        """Set up ChromaDB vector stores for text and metadata"""
        # Set up text vector store
        try:
            self.text_collection = self.chroma_client.get_or_create_collection(self.text_collection_name)
            self.text_vector_store = ChromaVectorStore(chroma_collection=self.text_collection)
        except Exception as e:
            logger.error(f"Failed to set up text vector store: {str(e)}")
            raise
        
        # Set up metadata vector store
        try:
            self.metadata_collection = self.chroma_client.get_or_create_collection(self.metadata_collection_name)
            self.metadata_vector_store = ChromaVectorStore(chroma_collection=self.metadata_collection)
        except Exception as e:
            logger.error(f"Failed to set up metadata vector store: {str(e)}")
            raise

    def load_if_exists(self) -> bool:
        """Load index from storage if it exists. Returns True if loaded."""
        # Check if any index files exist and determine which type to load
        index_exists = self._index_files_exist()
        
        if index_exists and self.using_old_style:
            self.storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(self.storage_context)
            self.text_index = self.index  # For backward compatibility
            return True
        elif index_exists:
            # Set up vector stores
            self._setup_vector_stores()
            # Load ChromaDB indices
            self.text_index = VectorStoreIndex.from_vector_store(self.text_vector_store, embed_model=self.embed_model)
            self.index = self.text_index  # For backward compatibility
            return True
        
        # If no index files exist at all, return False
        return False
    
    def _migrate_from_old_index(self):
        """Migration completely disabled. Always returns empty list."""
        logger.info("Migration from old index format is disabled")
        return []  # Migration disabled - unreachable code below
        
        # Try to load ChromaDB indices
        try:
            # Set up vector stores
            self._setup_vector_stores()
            logger.info(f"ChromaDB vector stores set up successfully in {self.chroma_dir}")
            
            # Create storage contexts
            text_storage_context = StorageContext.from_defaults(vector_store=self.text_vector_store)
            metadata_storage_context = StorageContext.from_defaults(vector_store=self.metadata_vector_store)
            
            # Create indices
            self.text_index = VectorStoreIndex.from_vector_store(
                self.text_vector_store,
                storage_context=text_storage_context,
                embed_model=self.embed_model
            )
            
            self.metadata_index = VectorStoreIndex.from_vector_store(
                self.metadata_vector_store,
                storage_context=metadata_storage_context,
                embed_model=self.embed_model
            )
            
            # For backward compatibility, set index to text_index
            self.index = self.text_index
            
            # Initialize CSV handler
            from .csv_handler import CSVDocumentHandler
            self.csv_handler = CSVDocumentHandler(self)
            logger.info("ChromaDB indices loaded successfully")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load ChromaDB indices: {str(e)}")
            return False
    
    def _encode_metadata_for_indexing(self, metadata):
        """Convert metadata to a string for vector indexing"""
        # Combine relevant metadata fields into a searchable string
        metadata_text = ""
        for key, value in metadata.items():
            # Skip internal fields and empty values
            if key in ['is_deleted', 'is_csv_row'] or not value:
                continue
                
            # Add key-value pair to metadata text
            metadata_text += f"{key}: {value} "
            
        return metadata_text
    
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
            
            documents = SimpleDirectoryReader(data_dir, file_extractor=file_extractor).load_data(num_workers=5)

            if not documents:
                raise ValueError(f"No documents found in {data_dir}")
                
            # Migration disabled - always empty list
            old_documents = []
            all_nodes = []
            async for nodes in self.process_documents(documents, progress_callback):
                print("Processing node..")
                all_nodes.extend(nodes)

            if skip_if_exists and self.load_if_exists():
                # If we loaded an existing index, just refresh it with the new documents
                print("Using existing index")
                self.text_index.refresh_ref_docs(documents)
                self.text_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "text_index"))
                self._clear_retriever_cache()

                return self.index
 

            # Create a new index with ChromaDB
            with atomic_index_update(self):
                # Set up vector stores if not already done
                if not hasattr(self, 'text_vector_store') or not self.text_vector_store:
                    self._setup_vector_stores()
                
                # Create storage contexts
                text_storage_context = StorageContext.from_defaults(vector_store=self.text_vector_store)
                metadata_storage_context = StorageContext.from_defaults(vector_store=self.metadata_vector_store)
                
                # Create text index
                self.text_index = VectorStoreIndex(
                    all_nodes,
                    storage_context=text_storage_context,
                    embed_model=self.embed_model
                )
                
                # Create metadata documents and index
                metadata_documents = []
                for node in all_nodes:
                    metadata_text = self._encode_metadata_for_indexing(node.metadata)
                    if metadata_text.strip():
                        metadata_doc = Document(
                            text=metadata_text,
                            metadata={
                                "source_node_id": node.id_,
                                **node.metadata
                            }
                        )
                        metadata_documents.append(metadata_doc)
                
                if metadata_documents:
                    metadata_nodes = self.node_parser.get_nodes_from_documents(metadata_documents)
                    self.metadata_index = VectorStoreIndex(
                        metadata_nodes,
                        storage_context=metadata_storage_context,
                        embed_model=self.embed_model
                    )
                
                # For backward compatibility, set index to text_index
                self.index = self.text_index
                
                # Add old documents if migrating
                if old_documents:
                    logger.info(f"Adding {len(old_documents)} documents from old index")
                    self.text_index.refresh_ref_docs(old_documents)
                
                # Persist indices
                self.text_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "text_index"))
                if hasattr(self, 'metadata_index') and self.metadata_index:
                    self.metadata_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "metadata_index"))
                
                # Initialize CSV handler
                from .csv_handler import CSVDocumentHandler
                self.csv_handler = CSVDocumentHandler(self)
                
                self._clear_retriever_cache()
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise DocumentProcessingError(f"Index creation failed: {str(e)}\n{traceback.format_exc()}") from e
            
        return self.index
    
    async def add_document(self, file_path: str, 
                          progress_callback: Optional[Callable[[float], None]] = None,
                          refresh_mode: bool = True,
                          always_include_verbatim: bool = False,
                          force_verbatim: bool = False):
        """Add a single new document to the index.
        
        Args:
            file_path: Path to the document to add
            progress_callback: Optional callback for progress updates
            refresh_mode: If True, use refresh_ref_docs instead of insert_nodes
            always_include_verbatim: If True, always include this document's full text
                                    in retrieval results regardless of query relevance
            force_verbatim: If True, include document as verbatim even if it's large
        """
        if not hasattr(self, 'text_index') or not self.text_index:
            raise ValueError("Index not initialized. Call create_index first.")

        try:            
            # Check if file exists
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            # Handle verbatim document if requested
            if always_include_verbatim:
                await self._add_verbatim_document(file_path, progress_callback, force_verbatim)
                
            # Check if we should use the CSV handler for CSV files
            if file_path.lower().endswith('.csv'):
                # For CSV files, we'll use a special handler in a separate method
                # This is just a placeholder - the actual implementation would need to detect CSV format
                # and prompt the user for configuration
                logger.info(f"CSV file detected: {file_path}. Use csv_handler.add_csv_document instead.")
                return
            
            # Load and process new document
            file_extractor = self._get_file_extractors()
            documents = SimpleDirectoryReader(input_files=[file_path], 
                                              filename_as_id=True,
                                              file_extractor=file_extractor).load_data()
 
            if not documents:
                raise ValueError(f"No content found in {file_path}")
            
            # Create metadata documents
            metadata_documents = []
            for doc in documents:
                metadata_text = self._encode_metadata_for_indexing(doc.metadata)
                if metadata_text.strip():
                    metadata_doc = Document(
                        text=metadata_text,
                        metadata={
                            "source_doc_id": doc.doc_id if hasattr(doc, 'doc_id') else None,
                            **doc.metadata
                        }
                    )
                    metadata_documents.append(metadata_doc)
            
            # Call progress callback with initial progress
            if progress_callback:
                progress_callback(0.1)  # 10% progress for loading document
                
            if refresh_mode:
                # Use refresh_ref_docs for automatic updates
                self.text_index.refresh_ref_docs(documents)
                if metadata_documents and hasattr(self, 'metadata_index') and self.metadata_index:
                    self.metadata_index.refresh_ref_docs(metadata_documents)
                
                # Persist indices
                self.text_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "text_index"))
                if hasattr(self, 'metadata_index') and self.metadata_index:
                    self.metadata_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "metadata_index"))
                
                self._clear_retriever_cache()
            else:
                # Use traditional insert method
                await self._add_document_insert(file_path, progress_callback)
            
            # Call progress callback with completion
            if progress_callback:
                progress_callback(1.0)  # 100% progress for completion

        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise DocumentProcessingError(f"Document addition failed: {str(e)}\n{traceback.format_exc()}") from e

    async def _add_document_insert(self, file_path: str, 
                          progress_callback: Optional[Callable] = None):
        """Add a single new document to the index using the insert_nodes method."""
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
                
            # Create metadata documents
            metadata_documents = []
            for doc in documents:
                metadata_text = self._encode_metadata_for_indexing(doc.metadata)
                if metadata_text.strip():
                    metadata_doc = Document(
                        text=metadata_text,
                        metadata={
                            "source_doc_id": doc.doc_id if hasattr(doc, 'doc_id') else None,
                            **doc.metadata
                        }
                    )
                    metadata_documents.append(metadata_doc)
            
            all_nodes = []
            async for nodes in self.process_documents(documents, progress_callback):
                all_nodes.extend(nodes)
                
                # Update progress callback for processing phase (10%-90%)
                if progress_callback:
                    progress_callback(0.1 + (0.8 * len(all_nodes) / (len(documents) * 3)))  # Estimate progress
            
            # Update index atomically
            metadata_nodes = []
            if metadata_documents:
                metadata_nodes = self.node_parser.get_nodes_from_documents(metadata_documents)
                
            with atomic_index_update(self):
                for node in all_nodes:
                    self.text_index.insert_nodes([node])
                    
                # Add metadata nodes if available
                if hasattr(self, 'metadata_index') and self.metadata_index and metadata_nodes:
                    for node in metadata_nodes:
                        self.metadata_index.insert_nodes([node])
                
                # Persist indices
                self.text_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "text_index"))
                if hasattr(self, 'metadata_index') and self.metadata_index:
                    self.metadata_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "metadata_index"))
                
                self._clear_retriever_cache()
                
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")


    async def _add_verbatim_document(self, file_path: str, progress_callback: Optional[Callable] = None, force_verbatim: bool = False):
        """Extract and store the full text of a document for verbatim inclusion in results.
        
        Args:
            file_path: Path to the document to add as verbatim
            progress_callback: Optional callback for progress updates
            force_verbatim: If True, include document as verbatim even if it's large
        """
        try:
            # Use the existing text_extractor module
            from .text_extractor import extract_text_from_file, get_file_metadata
            
            # Extract full text
            full_text = await extract_text_from_file(file_path)
            
            # Check document size - limit verbatim inclusion for very large documents
            MAX_VERBATIM_SIZE = 125000  # ~50KB, adjust as needed
            if len(full_text) > MAX_VERBATIM_SIZE and not force_verbatim:
                logger.warning(f"Document {os.path.basename(file_path)} exceeds verbatim size limit "
                              f"({len(full_text)} > {MAX_VERBATIM_SIZE}). "
                              f"Not adding as verbatim. Use force_verbatim=True to override.")
                return
            
            # Get file metadata
            metadata = await get_file_metadata(file_path)
            
            # Update progress callback
            if progress_callback:
                progress_callback(0.2)  # 20% progress for extraction
            
            # Generate a unique ID for the document
            doc_id = os.path.basename(file_path)
            safe_doc_id = re.sub(r'[^\w\-\.]', '_', doc_id)  # Make filename safe
            
            # Store metadata about the document
            verbatim_path = os.path.join(self.verbatim_docs_dir, f"{safe_doc_id}.txt")
            self.verbatim_docs[safe_doc_id] = {
                "file_path": file_path,
                "file_name": metadata["file_name"],
                "file_type": metadata["file_type"],
                "added_at": datetime.datetime.now().isoformat(),
                "size": len(full_text),
                "verbatim_path": verbatim_path
            }
            
            # Save the full text to a file
            with open(verbatim_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            # Save the updated verbatim docs index
            with open(self.verbatim_docs_index_path, 'w') as f:
                json.dump(self.verbatim_docs, f, indent=2)
            
            logger.info(f"Added verbatim document: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to add verbatim document: {str(e)}")
            raise DocumentProcessingError(f"Verbatim document addition failed: {str(e)}") from e

    async def add_url_document(self, url: str, progress_callback=None, always_include_verbatim=True):
        """Add content from a URL to the knowledge base.
        
        Args:
            url: The URL to fetch content from
            progress_callback: Optional callback for progress updates
            always_include_verbatim: If True, include as verbatim document
            
        Returns:
            Dict with URL document information
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized. Call create_index first.")
                
            from .url_fetcher import fetch_content_from_url
            
            # Update progress callback
            if progress_callback:
                progress_callback(0.1)  # 10% progress for starting
            
            # Create a hash of the URL for unique identification
            url_hash = hashlib.md5(url.encode()).hexdigest()
            
            # Fetch content from URL
            content = await fetch_content_from_url(url)
            
            if progress_callback:
                progress_callback(0.3)  # 30% progress after fetching
            
            if not content:
                raise ValueError(f"No content could be extracted from URL: {url}")
                
            # Create a temporary file with the content
            url_filename = f"url_{url_hash}.txt"
            url_path = os.path.join(self.url_docs_dir, url_filename)
            
            with open(url_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            if progress_callback:
                progress_callback(0.5)  # 50% progress after saving
                
            # Save metadata about the URL document
            self.url_docs[url_hash] = {
                "url": url,
                "added_at": datetime.datetime.now().isoformat(),
                "last_refreshed": datetime.datetime.now().isoformat(),
                "file_path": url_path,
                "size": len(content)
            }
            
            # Save the updated URL docs index
            with open(self.url_docs_index_path, 'w') as f:
                json.dump(self.url_docs, f, indent=2)
                
            # Add the document to the index
            await self.add_document(url_path, progress_callback=progress_callback, 
                                  always_include_verbatim=always_include_verbatim)
            
            if progress_callback:
                progress_callback(1.0)  # 100% progress for completion
                
            logger.info(f"Added URL document: {url}")
            return self.url_docs[url_hash]
            
        except Exception as e:
            logger.error(f"Failed to add URL document: {str(e)}")
            raise DocumentProcessingError(f"URL document addition failed: {str(e)}") from e
            
    async def refresh_url_document(self, url_hash, progress_callback=None):
        """Refresh content for a URL document.
        
        Args:
            url_hash: The hash of the URL to refresh
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with updated URL document information
        """
        try:
            if url_hash not in self.url_docs:
                raise ValueError(f"URL with hash {url_hash} not found in URL documents")
                
            url_info = self.url_docs[url_hash]
            url = url_info["url"]
            file_path = url_info["file_path"]
            
            # Update progress callback
            if progress_callback:
                progress_callback(0.1)  # 10% progress for starting
                
            from .url_fetcher import fetch_content_from_url
            
            # Fetch fresh content
            content = await fetch_content_from_url(url)
            
            if progress_callback:
                progress_callback(0.3)  # 30% progress after fetching
                
            if not content:
                raise ValueError(f"No content could be extracted from URL: {url}")
                
            # First remove the document from the index
            await self.remove_document(file_path)
            
            if progress_callback:
                progress_callback(0.5)  # 50% progress after removing
                
            # Update the file with new content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # Update metadata
            self.url_docs[url_hash]["last_refreshed"] = datetime.datetime.now().isoformat()
            self.url_docs[url_hash]["size"] = len(content)
            
            # Save the updated URL docs index
            with open(self.url_docs_index_path, 'w') as f:
                json.dump(self.url_docs, f, indent=2)
                
            # Add the document back to the index
            await self.add_document(file_path, progress_callback=progress_callback, 
                                  always_include_verbatim=True)
            
            if progress_callback:
                progress_callback(1.0)  # 100% progress for completion
                
            logger.info(f"Refreshed URL document: {url}")
            return self.url_docs[url_hash]
            
        except Exception as e:
            logger.error(f"Failed to refresh URL document: {str(e)}")
            raise DocumentProcessingError(f"URL document refresh failed: {str(e)}") from e

    async def remove_url_document(self, url_or_hash: str):
        """Remove a URL document.
        
        Args:
            url_or_hash: The URL or hash of the URL to remove
        """
        try:
            # Determine if input is a URL or hash
            if url_or_hash.startswith(('http://', 'https://')):
                # Create hash from URL
                url_hash = hashlib.md5(url_or_hash.encode()).hexdigest()
            else:
                # Assume input is already a hash
                url_hash = url_or_hash
                
            # Check if URL exists in our index
            if url_hash not in self.url_docs:
                logger.warning(f"URL document not found with hash: {url_hash}")
                return
                
            # Get file path from URL info
            file_path = self.url_docs[url_hash]["file_path"]
            
            # If this document is also in verbatim documents, remove it from there first
            # This ensures all references to the document are removed
            await self.remove_verbatim_document(file_path)
            
            # Remove document from the index
            await self.remove_document(file_path)
            
            # Remove the URL document file
            if os.path.exists(file_path):
                os.remove(file_path)
                
            # Remove from the URL docs index
            del self.url_docs[url_hash]
            
            # Save the updated URL docs index
            with open(self.url_docs_index_path, 'w') as f:
                json.dump(self.url_docs, f, indent=2)
                
            logger.info(f"Removed URL document with hash: {url_hash}")
            
        except Exception as e:
            logger.error(f"Failed to remove URL document: {str(e)}")
            raise DocumentProcessingError(f"URL document removal failed: {str(e)}") from e

    async def remove_document(self, file_path: str, remove_from_metadata_index: bool = True, remove_from_chroma: bool = True):
        """Remove a document and all its hierarchical nodes from the index."""
        if not self.index:
            raise ValueError("Index not initialized.")
            
        try:
            debug_box("_________________________________________")
            print("Ref doc info: ", self.index.ref_doc_info)

            nodes_to_remove = set()
            #for node_id, node in self.index.docstore.docs.items():
            for doc_id, doc in self.index.ref_doc_info.items():
                doc_file_path = doc.metadata.get('file_path', 'Unknown')
                print(f"Checking doc: {doc_file_path} to match file path: {file_path}")

                if doc_file_path == file_path:
                    nodes_to_remove.add(doc_id)
                    print("doc =", doc)
                    print("doc metadata = ", doc.metadata)
                    print(f"Removing doc: {doc_id} with matching file path {file_path}")
                else:
                    print(f"Did not match. doc metadata was {doc.metadata}")
        
            if len(nodes_to_remove) == 0:
                raise ValueError(f"No docs found for document: {file_path}")

            with atomic_index_update(self):
                for doc_id in nodes_to_remove:
                    self.text_index.delete_ref_doc(doc_id, delete_from_docstore=True, delete_from_vectorstore=remove_from_chroma)
                    print("Deleted doc: ", doc_id)
                    
                    # Also remove from metadata index if it exists
                    if remove_from_metadata_index and hasattr(self, 'metadata_index') and self.metadata_index:
                        # Find and delete corresponding metadata nodes
                        meta_nodes_to_remove = []
                        for meta_node_id, meta_node in self.metadata_index.docstore.docs.items():
                            if meta_node.metadata.get("file_path") == file_path:
                                meta_nodes_to_remove.append(meta_node_id)
                        
                        for meta_node_id in meta_nodes_to_remove:
                            self.metadata_index.delete_ref_doc(meta_node_id, delete_from_docstore=True, delete_from_vectorstore=remove_from_chroma)
                            print(f"Deleted metadata node: {meta_node_id}")
                
                # Persist updates
                self.text_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "text_index"))
                self.text_index.docstore.persist()
                
                # Persist metadata index if it exists
                if hasattr(self, 'metadata_index') and self.metadata_index:
                    self.metadata_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "metadata_index"))
                    self.metadata_index.docstore.persist()

                print("Saved docstore")
                self._clear_retriever_cache()
        except Exception as e:
            logger.error(f"Failed to remove document: {str(e)}")
            raise DocumentProcessingError(f"Document removal failed: {str(e)}. {traceback.format_exc()}") from e 
    async def remove_verbatim_document(self, file_path: str):
        """Remove a verbatim document.
        
        Args:
            file_path: Path to the document to remove
        """
        try:
            # Find the document in the verbatim docs index
            doc_id = None
            for id, info in self.verbatim_docs.items():
                if info["file_path"] == file_path:
                    doc_id = id
                    break
            
            if doc_id is None:
                logger.warning(f"Verbatim document not found: {file_path}")
                return

            # Check if verbatim path exists before trying to remove
            verbatim_path = self.verbatim_docs[doc_id]["verbatim_path"]
            if os.path.exists(verbatim_path):
                try:
                    os.remove(verbatim_path)
                except Exception as e:
                    logger.warning(f"Could not remove verbatim file {verbatim_path}: {str(e)}")
            
            # Remove from the index
            del self.verbatim_docs[doc_id]
            
            # Save the updated verbatim docs index
            with open(self.verbatim_docs_index_path, 'w') as f:
                json.dump(self.verbatim_docs, f, indent=2)
            
            logger.info(f"Removed verbatim document: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to remove verbatim document: {str(e)}")
            raise DocumentProcessingError(f"Verbatim document removal failed: {str(e)}") from e

    def get_document_info(self) -> List[Dict]:
        """Get information about all documents and their hierarchical structure."""
        print("Get document info")
        if not self.index:
            return []
        print('1')
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
            # Check if this is a URL document
            url_hash = None
            is_url = False
            url_info = {}
            
            # Look through url_docs to find if this file is from a URL
            for hash_id, info in self.url_docs.items():
                if info.get("file_path") == k:
                    url_hash = hash_id
                    is_url = True
                    url_info = info
                    break
                    
            # Add URL information if this document is from a URL
            if is_url:
                v['is_url'] = True
                v['url'] = url_info.get('url', '')
                v['url_hash'] = url_hash
                v['last_refreshed'] = url_info.get('last_refreshed', '')
            else:
                v['is_url'] = False
                
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

    def find_root_node(self, node):
        if node.parent_node is None:
            return node
        else:
            return self.find_root_node(node.parent_node)
 

    def _get_verbatim_documents(self):
        """Get all verbatim documents.
        
        Returns:
            List of tuples (text, metadata, score, chunk_size) for all verbatim documents
        """
        verbatim_results = []
        
        for doc_id, info in self.verbatim_docs.items():
            try:
                # Read the verbatim text file
                verbatim_path = info["verbatim_path"]
                if os.path.exists(verbatim_path):
                    with open(verbatim_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Create metadata for the document
                    metadata = {
                        "file_name": info["file_name"],
                        "file_path": info["file_path"],
                        "file_type": info.get("file_type", "Unknown"),
                        "is_verbatim": True,
                        "added_at": info["added_at"]
                    }
                    
                    # Add to results with a high score to ensure inclusion
                    # Use 2.0 to ensure it ranks higher than any similarity match
                    verbatim_results.append((text, metadata, 2.0, len(text)))
            except Exception as e:
                logger.error(f"Failed to read verbatim document {doc_id}: {str(e)}")
        
        return verbatim_results

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
    async def retrieve_relevant_nodes(self, query_text: str, similarity_top_k: int = 15, final_top_k: int = 6, min_score: float = 0.69, include_verbatim: bool = True, search_metadata: bool = True, metadata_only: bool = False):
        """Get raw retrieval results without LLM synthesis.
        
        Returns:
            List of tuples (node text, metadata, score, chunk_size)
            chunk_size helps identify which level of the hierarchy the node is from
        """
        debug_box("Starting retrieval")
        
        # Add verbatim documents if requested
        verbatim_results = []
        if include_verbatim:
            verbatim_results = self._get_verbatim_documents()
            logger.info(f"Including {len(verbatim_results)} verbatim documents in retrieval results")
        
        text_results = [] if metadata_only else []
        metadata_results = []
        
        try:
            if not self.index:
                raise ValueError("Index not initialized.")
            
            retriever_start = datetime.datetime.now()

            # Get results from text index if not metadata_only
            if not metadata_only:
                text_retriever = self.text_index.as_retriever(similarity_top_k=similarity_top_k)
                text_nodes = await text_retriever.aretrieve(query_text)
                
                # Get raw results from text index
                text_results = [(node.node.text, 
                        node.node.metadata,
                        node.score,
                        len(node.node.text)) for node in text_nodes]
                text_results = [r for r in text_results if r[2] >= min_score]
            
            # Get results from metadata index if requested
            if search_metadata and hasattr(self, 'metadata_index') and self.metadata_index:
                metadata_retriever = self.metadata_index.as_retriever(similarity_top_k=similarity_top_k)
                metadata_nodes = await metadata_retriever.aretrieve(query_text)
                
                # Process metadata results
                for node in metadata_nodes:
                    if node.score >= min_score:
                        # Find the corresponding text node
                        source_node_id = node.node.metadata.get("source_node_id")
                        if source_node_id and source_node_id in self.text_index.docstore.docs:
                            text_node = self.text_index.docstore.docs[source_node_id]
                            metadata_results.append((text_node.text, 
                                                  text_node.metadata,
                                                  node.score,  # Use metadata match score
                                                  len(text_node.text)))
                        elif "file_path" in node.node.metadata:
                            # If we can't find the source node, use the metadata node itself
                            # This can happen if the metadata node was created separately
                            metadata_results.append((node.node.text, 
                                                  node.node.metadata,
                                                  node.score,
                                                  len(node.node.text)))
            
            retriever_end = datetime.datetime.now()
            print(f"Total retriever setup time: {retriever_end - retriever_start}")
            
            # Combine results, removing duplicates
            combined_results = text_results.copy()
            seen_texts = set(r[0] for r in text_results)
            
            for result in metadata_results:
                if result[0] not in seen_texts and result[0].strip():
                    combined_results.append(result)
                    seen_texts.add(result[0])
             
            # Apply enhanced keyword matching and filtering
            enhanced_results = enhance_search_results(query_text, combined_results, 
                                                    initial_top_k=similarity_top_k,
                                                    final_top_k=final_top_k)
            
            # Add verbatim documents if available
            if verbatim_results:
                # Combine results
                combined_results = verbatim_results + enhanced_results
                # Sort by score (descending)
                combined_results.sort(key=lambda x: x[2], reverse=True)
                # Limit to final_top_k + number of verbatim docs
                # This ensures all verbatim docs are included plus up to final_top_k regular results
                return combined_results[:len(verbatim_results) + final_top_k]
           
            return enhanced_results
            
        except Exception as e:
            trace = traceback.format_exc()
            print(f"Retrieval failed: {str(e)} \n {trace}")
            raise DocumentProcessingError(f"Retrieval failed: {str(e)} \n {trace}") from e


    async def get_relevant_context(self, query_text: str, 
                            similarity_top_k: int = 15,
                            final_top_k: int = 6,
                            min_score: float = 0.65,
                            include_verbatim: bool = False, search_metadata: bool = True, metadata_only: bool = False) -> str:
        """Get formatted context from relevant nodes.
        
        Args:
            query_text: The query to match against
            similarity_top_k: Number of matches to retrieve
            final_top_k: Number of results to return after score enhancement
            min_score: Minimum similarity score to include (0.0 to 1.0)
            search_metadata: Whether to search metadata index in addition to text index
            metadata_only: Whether to search only the metadata index
            include_verbatim: Whether to include verbatim documents
        
        Returns:
            Formatted string with relevant context
        """
        query_stats = {}
        
        # Start timing
        start_time = datetime.datetime.now()

        results = await self.retrieve_relevant_nodes(query_text, similarity_top_k, final_top_k, 
                                                   min_score, include_verbatim, search_metadata,
                                                   metadata_only)
        if not results:
            return ""
            
        # Separate verbatim and regular results
        verbatim_results = []
        regular_results = []
        
        for result in results:
            text, metadata, score, chunk_size = result
            if metadata.get("is_verbatim", False):
                verbatim_results.append(result)
            else:
                regular_results.append(result)
        
        # If we're not including verbatim documents, clear the verbatim_results list
        if not include_verbatim:
            verbatim_results = []
            print("Excluding verbatim documents as requested")
        
        # Format for detailed view
        context = "### [Retrieved Knowledge Base Results]\n"
            
        # Add verbatim documents first with special formatting
        if verbatim_results:
            context += "## ESSENTIAL DOCUMENTS\n\n"
            context += "=" * 80 + "\n\n"  # Distinctive separator
            
            for text, metadata, _, chunk_size in verbatim_results:
                # Format metadata header
                context += f"[ESSENTIAL: {metadata.get('file_name', 'Document')} | "
                context += f"Path: {metadata.get('file_path', 'Unknown')}]\n"
                context += f"{text}\n"
                context += "=" * 80 + "\n\n"  # Distinctive separator
        
        # Add regular results
        if regular_results:
            context += "Note: Results are ranked by relevance score. Higher scores indicate stronger matches.\n"
            context += "Some results with lower scores may be less relevant or unrelated to user query.\n\n"
            
            for text, metadata, score, chunk_size in regular_results:
                # Extract metadata fields
                file_name = metadata.get('file_name', 'Unknown')
                file_type = metadata.get('file_type', 'Unknown')
                creation_date = metadata.get('creation_date', 'Unknown')
                # Format metadata header
                context += "| File | Score | Type | Creation Date | Size |\n"
                context += "|------|------|------|-------------|---------|\n"
                context += f"|{file_name} | {score: .3f} | {file_type} | {creation_date} | {chunk_size} |\n"
                context += "\n"
                context += f"{text}\n\n"
                context += "___\n\n"
            
            context += "## End of Retrieved Results"

        # end timing
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        query_stats['total_time'] = total_time
        query_stats['retriever_creation'] = bool(self._retriever is None)
        query_stats['verbatim_docs'] = len(verbatim_results)
        query_stats['metadata_search'] = search_metadata
        query_stats['metadata_only'] = metadata_only
        
        print(f"Time taken to get relevant context: {total_time}")
        print(f"Retrievers {'created' if query_stats['retriever_creation'] else 'reused'}")
        print(f"Verbatim documents included: {len(verbatim_results)}")

        return context.strip(), query_stats

    def _clear_retriever_cache(self):
        """Clear the cached retrievers when index changes."""
        self._retriever = None
        print("Cleared retriever cache")
        
        # Also clear any cached retrievers for text and metadata indices
        if hasattr(self, 'text_index') and self.text_index:
            self.text_index._retriever = None

    async def add_csv_document(self, file_path: str, config: dict, 
                             progress_callback: Optional[Callable[[float], None]] = None):
        """Add a CSV file as a collection of row-based documents.
        
        Args:
            file_path: Path to the CSV file
            config: Dictionary with configuration:
                - text_column: Index of column containing main text
                - id_column: Index of column to use as document ID
                - key_metadata_columns: List of column indices to display in UI
                - metadata_columns: List of column indices to store as metadata
            progress_callback: Optional callback for progress updates
        """
        if not hasattr(self, 'text_index') or not self.text_index:
            raise ValueError("Index not initialized. Call create_index first.")

        try:            
            # Check if file exists
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            # Generate a unique ID for the CSV source
            source_id = os.path.basename(file_path)
            safe_source_id = re.sub(r'[^\w\-\.]', '_', source_id)  # Make filename safe
            
            # Create a directory for this CSV source
            source_dir = os.path.join(self.csv_docs_dir, safe_source_id)
            os.makedirs(source_dir, exist_ok=True)
            
            # Save the configuration
            config_path = os.path.join(source_dir, "config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            # Call progress callback with initial progress
            if progress_callback:
                progress_callback(0.1)  # 10% progress for setup

            
            # Parse the CSV file
            documents = []
            text_col = config.get("text_column")
            id_col = config.get("id_column")
            key_metadata_cols = config.get("key_metadata_columns", [])
            metadata_cols = config.get("metadata_columns", [])
            
            # Combine all metadata columns
            all_metadata_cols = list(set(key_metadata_cols + metadata_cols))
            
            # Use preprocessed rows if available in config, otherwise parse the file
            rows = []
            if "preprocessed_rows" in config and config["preprocessed_rows"]:
                rows = config["preprocessed_rows"]
                print(f"Using {len(rows)} preprocessed rows from config")
            else:
                print("No preprocessed rows found in config, parsing CSV file")
                # Read the CSV file
                with open(file_path, 'r', encoding='utf-8', newline='') as f:
                    # Try to detect the dialect
                    try:
                        dialect = csv.Sniffer().sniff(f.read(1024))
                        f.seek(0)
                    except:
                        dialect = 'excel'  # Default to Excel dialect if detection fails
                    
                    # Read the CSV with the detected dialect
                    reader = csv.reader(f, dialect=dialect)
                    rows = list(reader)

            # Get column headers if available (first row) based on has_header config
            has_header = config.get("has_header", True)
            headers = rows[0] if rows and has_header else []
            
            # Process each row
            total_rows = len(rows)
            print("Found {} rows in CSV file".format(total_rows))

            for i, row in enumerate(rows):
                # Skip header row if it exists and is configured to be skipped
                if i == 0 and has_header:
                    continue
                    
                try:
                    # Check if row has enough columns
                    max_required_col = max([text_col, id_col] + all_metadata_cols) if all_metadata_cols else max(text_col, id_col)
                    if len(row) <= max_required_col:
                        logger.warning(f"Row {i} has insufficient columns ({len(row)} <= {max_required_col}), skipping")
                        continue
                    
                    # Extract text and document ID
                    text = row[text_col].strip()
                    doc_id = row[id_col].strip()
                    
                    # Skip empty rows
                    if not text or not doc_id:
                        logger.warning(f"Row {i} has empty text or ID, skipping")
                        continue
                    
                    # Create metadata
                    metadata = {
                        "file_path": file_path,
                        "file_name": os.path.basename(file_path),
                        "file_type": "csv",
                        "row_index": i,
                        "csv_source_id": safe_source_id,
                        "doc_id": doc_id,
                        "is_csv_row": True
                    }
                    
                    # Add column headers as metadata keys if available
                    for col_idx in all_metadata_cols:
                        if col_idx < len(row):
                            col_name = f"col_{col_idx}"
                            if headers and col_idx < len(headers):
                                col_name = headers[col_idx]
                            metadata[col_name] = row[col_idx]
                    
                    # Create document
                    doc = Document(text=text, metadata=metadata)
                    documents.append(doc)
                    print(f"Processed row {i}: {text[:50]}...")  # Print first 50 chars of text for debugging
                    # Update progress periodically
                    if progress_callback and i % 10 == 0:
                        progress_callback(0.1 + (0.8 * i / total_rows))
                        
                except Exception as e:
                    logger.warning(f"Error processing row {i}: {str(e)}")
            
            # Save metadata about the CSV source
            self.csv_docs[safe_source_id] = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "added_at": datetime.datetime.now().isoformat(),
                "row_count": len(documents),
                "config": config,
                "config_path": config_path
            }
            
            # Save the updated CSV docs index
            with open(self.csv_docs_index_path, 'w') as f:
                json.dump(self.csv_docs, f, indent=2)

            if documents:
                self.index.refresh_ref_docs(documents)
                self.index.storage_context.persist(persist_dir=self.persist_dir)
                self._clear_retriever_cache()
            
            # Call progress callback with completion
            if progress_callback:
                progress_callback(1.0)  # 100% progress for completion

            logger.info(f"Added CSV document with {len(documents)} rows: {file_path}")
            return self.csv_docs[safe_source_id]
            
        except Exception as e:
            logger.error(f"Failed to add CSV document: {str(e)}")
            raise DocumentProcessingError(f"CSV document addition failed: {str(e)}") from e

    # Initialize CSV handler in __init__ method
    async def sync_csv_document(self, file_path: str, progress_callback: Optional[Callable] = None):
        """Sync a CSV document with the index, updating/adding/removing rows as needed.
        
        Args:
            file_path: Path to the CSV file
            progress_callback: Optional callback for progress updates
        """
        if not hasattr(self, 'text_index') or not self.text_index:
            raise ValueError("Index not initialized.")
            
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            # Find the CSV source in our CSV docs index
            source_id = os.path.basename(file_path)
            safe_source_id = re.sub(r'[^\w\-\.]', '_', source_id)  # Make filename safe
            
            if safe_source_id not in self.csv_docs:
                raise ValueError(f"CSV source not found: {file_path}")
                
            # Get the configuration
            config = self.csv_docs[safe_source_id]["config"]
            
            # Call progress callback with initial progress
            if progress_callback:
                progress_callback(0.1)  # 10% progress for setup
            
            # Find all existing documents from this CSV source
            existing_docs = {}
            for node_id, node in self.text_index.docstore.docs.items():
                if node.metadata.get("csv_source_id") == safe_source_id:
                    doc_id = node.metadata.get("doc_id")
                    if doc_id:
                        existing_docs[doc_id] = node_id
            
            # Parse the CSV file to get new documents
            text_col = config.get("text_column")
            id_col = config.get("id_column")
            key_metadata_cols = config.get("key_metadata_columns", [])
            metadata_cols = config.get("metadata_columns", [])
            all_metadata_cols = list(set(key_metadata_cols + metadata_cols))
            
            # Read the CSV file
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                # Try to detect the dialect
                try:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    f.seek(0)
                except:
                    dialect = 'excel'  # Default to Excel dialect if detection fails
                
                # Read the CSV with the detected dialect
                reader = csv.reader(f, dialect=dialect)
                rows = list(reader)
            
            # Get column headers if available (first row)
            headers = rows[0] if rows else []
            
            # Process each row to build new documents
            new_docs = {}
            for i, row in enumerate(rows):
                # Skip header row if it exists and is configured to be skipped
                if i == 0 and config.get("has_header", True):
                    continue
                    
                try:
                    # Check if row has enough columns
                    max_required_col = max([text_col, id_col] + all_metadata_cols) if all_metadata_cols else max(text_col, id_col)
                    if len(row) <= max_required_col:
                        logger.warning(f"Row {i} has insufficient columns ({len(row)} <= {max_required_col}), skipping")
                        continue
                    
                    # Extract text and document ID
                    text = row[text_col].strip()
                    doc_id = row[id_col].strip()
                    
                    # Skip empty rows
                    if not text or not doc_id:
                        continue
                    
                    # Create metadata
                    metadata = {
                        "file_path": file_path,
                        "file_name": os.path.basename(file_path),
                        "file_type": "csv",
                        "row_index": i,
                        "csv_source_id": safe_source_id,
                        "doc_id": doc_id,
                        "is_csv_row": True
                    }
                    
                    # Add column headers as metadata keys if available
                    for col_idx in all_metadata_cols:
                        if col_idx < len(row):
                            col_name = f"col_{col_idx}"
                            if headers and col_idx < len(headers):
                                col_name = headers[col_idx]
                            metadata[col_name] = row[col_idx]
                    
                    # Create document
                    new_docs[doc_id] = Document(text=text, metadata=metadata)
                except Exception as e:
                    logger.warning(f"Error processing row {i} during sync: {str(e)}")
            
            # Determine which documents to add, update, or delete
            to_add = []
            to_update = []
            to_delete = set(existing_docs.keys()) - set(new_docs.keys())
            
            for doc_id, doc in new_docs.items():
                if doc_id not in existing_docs:
                    to_add.append(doc)
                else:
                    # Compare text to detect changes
                    node_id = existing_docs[doc_id]
                    existing_node = self.index.docstore.docs[node_id]
                    if existing_node.text != doc.text:
                        to_update.append((node_id, doc))
            
            # Update progress
            if progress_callback:
                progress_callback(0.3)  # 30% progress after analysis
            
            # Update index atomically
            with atomic_index_update(self):
                # Delete removed rows
                for doc_id in list(to_delete):  # Convert to list to avoid modification during iteration
                    self.index.delete_ref_doc(existing_docs[doc_id], delete_from_docstore=True)
                
                # Update changed rows
                for node_id, doc in to_update:
                    self.index.delete_ref_doc(node_id, delete_from_docstore=True)
                    nodes = self.node_parser.get_nodes_from_documents([doc])
                    for node in nodes:
                        self.text_index.insert_nodes([node])
                
                # Add new rows
                if to_add:
                    nodes = self.node_parser.get_nodes_from_documents(to_add)
                    for node in nodes:
                        self.index.insert_nodes([node])
                
                # Add metadata documents if we have a metadata index
                if hasattr(self, 'metadata_index') and self.metadata_index:
                    # Create and add metadata documents for new rows
                    # This would be similar to the code in add_csv_document
                    pass
                
                # Persist updates
                self.text_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "text_index"))
                if hasattr(self, 'metadata_index') and self.metadata_index:
                    self.metadata_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "metadata_index"))
                self._clear_retriever_cache()
            
            # Update CSV source metadata
            self.csv_docs[safe_source_id]["row_count"] = len(new_docs)
            self.csv_docs[safe_source_id]["last_synced"] = datetime.datetime.now().isoformat()
            
            # Save the updated CSV docs index
            with open(self.csv_docs_index_path, 'w') as f:
                json.dump(self.csv_docs, f, indent=2)
            
            # Call progress callback with completion
            if progress_callback:
                progress_callback(1.0)  # 100% progress for completion
            
            logger.info(f"CSV sync complete: {len(to_add)} added, {len(to_update)} updated, {len(to_delete)} deleted")
            return {
                "added": len(to_add),
                "updated": len(to_update),
                "deleted": len(to_delete),
                "total": len(new_docs)
            }
            
        except Exception as e:
            logger.error(f"Failed to sync CSV document: {str(e)}")
            raise DocumentProcessingError(f"CSV document sync failed: {str(e)}") from e

    async def update_csv_row(self, csv_source_id: str, doc_id: str, new_text: str, new_metadata: dict = None, update_metadata_index: bool = True):
        """Update a single row in a CSV document.
        
        Args:
            csv_source_id: ID of the CSV source
            doc_id: Document ID of the row to update
            new_text: New text content for the row
            new_metadata: Optional new metadata for the row 
            update_metadata_index: Whether to update the metadata index
        """
        if not hasattr(self, 'text_index') or not self.text_index:
            raise ValueError("Index not initialized.")
            
        try:
            # Check if CSV source exists
            if csv_source_id not in self.csv_docs:
                raise ValueError(f"CSV source not found: {csv_source_id}")
            
            # Find the document in the index
            node_id_to_update = None
            for node_id, node in self.text_index.docstore.docs.items():
                if (node.metadata.get("csv_source_id") == csv_source_id and 
                    node.metadata.get("doc_id") == doc_id):
                    node_id_to_update = node_id
                    break
            
            if not node_id_to_update:
                raise ValueError(f"Row with document ID {doc_id} not found in CSV source {csv_source_id}")
            
            # Get the existing node to preserve metadata
            existing_node = self.text_index.docstore.docs[node_id_to_update]
            existing_metadata = existing_node.metadata.copy()
            
            # Update metadata if provided
            if new_metadata:
                existing_metadata.update(new_metadata)
            
            # Create a new document with updated text and metadata
            updated_doc = Document(text=new_text, metadata=existing_metadata)
            
            # Update index atomically
            with atomic_index_update(self):
                # Delete the existing node
                self.index.delete_ref_doc(node_id_to_update, delete_from_docstore=True)
                
                # Create metadata document if needed
                metadata_doc = None
                if update_metadata_index and hasattr(self, 'metadata_index') and self.metadata_index:
                    metadata_text = self._encode_metadata_for_indexing(existing_metadata)
                    if metadata_text.strip():
                        metadata_doc = Document(
                            text=metadata_text,
                            metadata={
                                "csv_row_id": doc_id,
                                "csv_source_id": csv_source_id,
                                **existing_metadata
                            }
                        )
                
                # Add the updated document
                nodes = self.node_parser.get_nodes_from_documents([updated_doc])
                for node in nodes:
                    self.text_index.insert_nodes([node])
                
                # Update metadata index if needed
                if metadata_doc and hasattr(self, 'metadata_index') and self.metadata_index:
                    # Find and delete existing metadata node
                    for meta_node_id, meta_node in self.metadata_index.docstore.docs.items():
                        if (meta_node.metadata.get("csv_source_id") == csv_source_id and 
                            meta_node.metadata.get("csv_row_id") == doc_id):
                            self.metadata_index.delete_ref_doc(meta_node_id, delete_from_docstore=True)
                            break
                    
                    # Add new metadata node
                    meta_nodes = self.node_parser.get_nodes_from_documents([metadata_doc])
                    for node in meta_nodes:
                        self.metadata_index.insert_nodes([node])
                
                # Persist updates
                self.text_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "text_index"))
                if hasattr(self, 'metadata_index') and self.metadata_index:
                    self.metadata_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "metadata_index"))
                self._clear_retriever_cache()
            
            logger.info(f"Updated row with document ID {doc_id} in CSV source {csv_source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update CSV row: {str(e)}")
            raise DocumentProcessingError(f"CSV row update failed: {str(e)}") from e
    
    async def delete_csv_row(self, csv_source_id: str, doc_id: str):
        """Delete a single row from a CSV document.
        
        Args:
            csv_source_id: ID of the CSV source
            doc_id: Document ID of the row to delete
        """
        if not hasattr(self, 'text_index') or not self.text_index:
            raise ValueError("Index not initialized.")
            
        try:
            # Check if CSV source exists
            if csv_source_id not in self.csv_docs:
                raise ValueError(f"CSV source not found: {csv_source_id}")
            
            # Find the document in the index
            node_id_to_delete = None
            for node_id, node in self.text_index.docstore.docs.items():
                if (node.metadata.get("csv_source_id") == csv_source_id and 
                    node.metadata.get("doc_id") == doc_id):
                    node_id_to_delete = node_id
                    break
            
            if not node_id_to_delete:
                raise ValueError(f"Row with document ID {doc_id} not found in CSV source {csv_source_id}")
            
            # Update index atomically
            with atomic_index_update(self):
                # Delete the node
                self.text_index.delete_ref_doc(node_id_to_delete, delete_from_docstore=True)
                
                # Delete from metadata index if it exists
                if hasattr(self, 'metadata_index') and self.metadata_index:
                    for meta_node_id, meta_node in self.metadata_index.docstore.docs.items():
                        if (meta_node.metadata.get("csv_source_id") == csv_source_id and 
                            meta_node.metadata.get("csv_row_id") == doc_id):
                            self.metadata_index.delete_ref_doc(meta_node_id, delete_from_docstore=True)
                            break
                
                # Persist updates
                self.text_index.storage_context.persist(persist_dir=os.path.join(self.persist_dir, "text_index"))
                self._clear_retriever_cache()
            
            # Update CSV source metadata
            self.csv_docs[csv_source_id]["row_count"] = self.csv_docs[csv_source_id].get("row_count", 0) - 1
            
            # Save the updated CSV docs index
            with open(self.csv_docs_index_path, 'w') as f:
                json.dump(self.csv_docs, f, indent=2)
            
            logger.info(f"Deleted row with document ID {doc_id} from CSV source {csv_source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete CSV row: {str(e)}")
            raise DocumentProcessingError(f"CSV row deletion failed: {str(e)}") from e

    def get_csv_rows(self, csv_source_id: str) -> List[Dict]:
        """Get all rows from a CSV source.
        
        Args:
            csv_source_id: ID of the CSV source
            
        Returns:
            List of dictionaries with row data
        """
        # Add debug logging
        logger.info(f"Getting CSV rows for source {csv_source_id}")
        
        if not hasattr(self, 'text_index') or not self.text_index:
            logger.warning("No text_index available, returning empty list")
            return []
            
        try:
            # Check if CSV source exists
            if not hasattr(self, 'csv_docs') or csv_source_id not in self.csv_docs:
                logger.warning(f"CSV source {csv_source_id} not found in csv_docs")
                raise ValueError(f"CSV source not found: {csv_source_id}")
            
            # Find all documents from this CSV source
            rows = []
            seen_doc_ids = set()  # Track doc_ids we've already processed
            for node_id, node in self.index.docstore.docs.items():
                logger.info("looking at node")
                if node.metadata.get("csv_source_id") == csv_source_id:
                    logger.info(f"Found node with csv_source_id {csv_source_id}: {node_id}")
                    # Create a row object with text and metadata
                    row = {
                        "node_id": node_id,
                        "text": node.text,
                        "doc_id": node.metadata.get("doc_id", ""),
                        "row_index": node.metadata.get("row_index", 0)
                    }
                    
                    # Skip deleted rows
                    logger.info(f"Checking if row is deleted: {node.metadata.get('is_deleted', False)}")
                    if node.metadata.get("is_deleted", False):
                        continue
                        
                    # Skip if we've already seen this doc_id to avoid duplicates
                    doc_id = node.metadata.get("doc_id", "")
                    if doc_id in seen_doc_ids:
                        continue
                    logger.info(f"Adding row with doc_id {doc_id} to results")
                    
                    seen_doc_ids.add(doc_id)
                    
                    # Add all metadata fields that start with "col_" or are in the metadata
                    for key, value in node.metadata.items():
                        if key.startswith("col_") or key in ["doc_id", "row_index"]:
                            row[key] = value
                    
                    rows.append(row)
                else:
                    logger.info(f"Node {node.metadata.get('csv_source_id')} does not match csv_source_id {csv_source_id}, skipping")

            logger.info(f"Found {len(rows)} rows for CSV source {csv_source_id}")
            # Sort rows by row_index
            rows.sort(key=lambda x: x.get("row_index", 0))
            
            logger.info(f"Returning {len(rows)} sorted rows")
            return rows
            
        except Exception as e:
            logger.error(f"Failed to get CSV rows: {str(e)}")
            return []
