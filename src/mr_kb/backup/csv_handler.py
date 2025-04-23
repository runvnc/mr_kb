import os
import json
import logging
import datetime
import re
from typing import Dict, List, Optional, Callable

from llama_index.core import Document
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class CSVDocumentHandler:
    def __init__(self, kb_instance):
        self.kb = kb_instance
        self.csv_docs_dir = os.path.join(kb_instance.persist_dir, "csv_docs")
        self.csv_docs_index_path = os.path.join(kb_instance.persist_dir, "csv_docs_index.json")
        self.csv_docs = {}
        
        # Create CSV docs directory if it doesn't exist
        os.makedirs(self.csv_docs_dir, exist_ok=True)
        
        # Load CSV docs index if it exists
        if os.path.exists(self.csv_docs_index_path):
            try:
                with open(self.csv_docs_index_path, 'r') as f:
                    self.csv_docs = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load CSV docs index: {str(e)}")
    
    def _encode_metadata_for_indexing(self, metadata):
        """Convert metadata to a string for vector indexing"""
        # Combine relevant metadata fields into a searchable string
        metadata_text = ""
        for key, value in metadata.items():
            # Skip internal fields
            if key in ['is_deleted', 'is_csv_row']:
                continue
            metadata_text += f"{key}: {value} "
        return metadata_text

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
        if not hasattr(self.kb, 'text_index') or not self.kb.text_index:
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

            # Process each row to build documents
            documents = []
            metadata_documents = []
            
            # Parse the CSV file
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
            else:
                # Read the CSV file
                import csv
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
                        "is_csv_row": True,
                        "is_deleted": False
                    }
                    
                    # Add column headers as metadata keys if available
                    for col_idx in all_metadata_cols:
                        if col_idx < len(row):
                            col_name = f"col_{col_idx}"
                            if headers and col_idx < len(headers):
                                col_name = headers[col_idx]
                            metadata[col_name] = row[col_idx]
                    
                    # Create document for text index
                    doc = Document(text=text, metadata=metadata)
                    documents.append(doc)
                    
                    # Create document for metadata index
                    metadata_text = self._encode_metadata_for_indexing(metadata)
                    if metadata_text.strip():
                        metadata_doc = Document(
                            text=metadata_text,
                            metadata={
                                "csv_row_id": doc_id,
                                "csv_source_id": safe_source_id,
                                **metadata
                            }
                        )
                        metadata_documents.append(metadata_doc)
                    
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
            
            # Add documents to indices
            if documents:
                # Add to text index
                self.kb.text_index.refresh_ref_docs(documents)
                
                # Add to metadata index if it exists
                if hasattr(self.kb, 'metadata_index') and metadata_documents:
                    self.kb.metadata_index.refresh_ref_docs(metadata_documents)
                
                # Persist indices
                self.kb.text_index.storage_context.persist(persist_dir=os.path.join(self.kb.persist_dir, "text_index"))
                if hasattr(self.kb, 'metadata_index'):
                    self.kb.metadata_index.storage_context.persist(persist_dir=os.path.join(self.kb.persist_dir, "metadata_index"))
                
                self.kb._clear_retriever_cache()
            
            # Call progress callback with completion
            if progress_callback:
                progress_callback(1.0)  # 100% progress for completion
            
            logger.info(f"Added CSV document with {len(documents)} rows: {file_path}")
            return self.csv_docs[safe_source_id]
            
        except Exception as e:
            logger.error(f"Failed to add CSV document: {str(e)}")
            from .kb import DocumentProcessingError
            raise DocumentProcessingError(f"CSV document addition failed: {str(e)}") from e
    async def update_csv_row(self, csv_source_id: str, doc_id: str, new_text: str, new_metadata: dict = None):
        """Update a single row in a CSV document."""
        if not hasattr(self.kb, 'text_index') or not self.kb.text_index:
            raise ValueError("Index not initialized.")
            
        try:
            # Check if CSV source exists
            if csv_source_id not in self.csv_docs:
                raise ValueError(f"CSV source not found: {csv_source_id}")
            
            # Find the document in the index
            node_id_to_update = None
            for node_id, node in self.kb.text_index.docstore.docs.items():
                if (node.metadata.get("csv_source_id") == csv_source_id and 
                    node.metadata.get("doc_id") == doc_id):
                    node_id_to_update = node_id
                    break
            
            if not node_id_to_update:
                raise ValueError(f"Row with document ID {doc_id} not found in CSV source {csv_source_id}")
            
            # Get the existing node to preserve metadata
            existing_node = self.kb.text_index.docstore.docs[node_id_to_update]
            existing_metadata = existing_node.metadata.copy()
            
            # Update metadata if provided
            if new_metadata:
                existing_metadata.update(new_metadata)
            
            # Create a new document with updated text and metadata
            updated_doc = Document(text=new_text, metadata=existing_metadata)
            
            # Update index atomically
            from .kb import atomic_index_update
            with atomic_index_update(self.kb):
                # Delete the existing node
                self.kb.text_index.delete_ref_doc(node_id_to_update, delete_from_docstore=True)
                
                # Add the updated document
                nodes = self.kb.node_parser.get_nodes_from_documents([updated_doc])
                for node in nodes:
                    self.kb.text_index.insert_nodes([node])
                
                # Update metadata index if it exists
                if hasattr(self.kb, 'metadata_index'):
                    # Find and delete the corresponding metadata node
                    meta_node_id_to_update = None
                    for meta_node_id, meta_node in self.kb.metadata_index.docstore.docs.items():
                        if (meta_node.metadata.get("csv_source_id") == csv_source_id and 
                            meta_node.metadata.get("csv_row_id") == doc_id):
                            meta_node_id_to_update = meta_node_id
                            break
                    
                    if meta_node_id_to_update:
                        self.kb.metadata_index.delete_ref_doc(meta_node_id_to_update, delete_from_docstore=True)
                    
                    # Create and add new metadata document
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
                        meta_nodes = self.kb.node_parser.get_nodes_from_documents([metadata_doc])
                        for node in meta_nodes:
                            self.kb.metadata_index.insert_nodes([node])
                
                # Persist updates
                self.kb.text_index.storage_context.persist(persist_dir=os.path.join(self.kb.persist_dir, "text_index"))
                if hasattr(self.kb, 'metadata_index'):
                    self.kb.metadata_index.storage_context.persist(persist_dir=os.path.join(self.kb.persist_dir, "metadata_index"))
                
                self.kb._clear_retriever_cache()
            
            logger.info(f"Updated row with document ID {doc_id} in CSV source {csv_source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update CSV row: {str(e)}")
            from .kb import DocumentProcessingError
            raise DocumentProcessingError(f"CSV row update failed: {str(e)}") from e
    async def delete_csv_row(self, csv_source_id: str, doc_id: str):
        """Mark a row as deleted rather than removing it completely"""
        if not hasattr(self.kb, 'text_index') or not self.kb.text_index:
            raise ValueError("Index not initialized.")
            
        try:
            # Check if CSV source exists
            if csv_source_id not in self.csv_docs:
                raise ValueError(f"CSV source not found: {csv_source_id}")
            
            # Find the document in the index
            node_id_to_update = None
            for node_id, node in self.kb.text_index.docstore.docs.items():
                if (node.metadata.get("csv_source_id") == csv_source_id and 
                    node.metadata.get("doc_id") == doc_id):
                    node_id_to_update = node_id
                    break
            
            if not node_id_to_update:
                raise ValueError(f"Row with document ID {doc_id} not found in CSV source {csv_source_id}")
            
            # Get the existing node to update its metadata
            existing_node = self.kb.text_index.docstore.docs[node_id_to_update]
            existing_metadata = existing_node.metadata.copy()
            
            # Mark as deleted in metadata
            existing_metadata["is_deleted"] = True
            
            # Create a new document with empty text but preserved metadata
            # We use empty text so it won't be retrieved in searches
            updated_doc = Document(text="", metadata=existing_metadata)
            
            # Update index atomically
            from .kb import atomic_index_update
            with atomic_index_update(self.kb):
                # Delete the existing node
                self.kb.text_index.delete_ref_doc(node_id_to_update, delete_from_docstore=True)
                
                # Add the updated document with empty text but preserved metadata
                nodes = self.kb.node_parser.get_nodes_from_documents([updated_doc])
                for node in nodes:
                    self.kb.text_index.insert_nodes([node])
                
                # Also update the metadata index
                if hasattr(self.kb, 'metadata_index'):
                    # Find and delete the corresponding metadata node
                    meta_node_id_to_update = None
                    for meta_node_id, meta_node in self.kb.metadata_index.docstore.docs.items():
                        if (meta_node.metadata.get("csv_source_id") == csv_source_id and 
                            meta_node.metadata.get("csv_row_id") == doc_id):
                            meta_node_id_to_update = meta_node_id
                            break
                    
                    if meta_node_id_to_update:
                        self.kb.metadata_index.delete_ref_doc(meta_node_id_to_update, delete_from_docstore=True)
                
                # Persist updates
                self.kb.text_index.storage_context.persist(persist_dir=os.path.join(self.kb.persist_dir, "text_index"))
                if hasattr(self.kb, 'metadata_index'):
                    self.kb.metadata_index.storage_context.persist(persist_dir=os.path.join(self.kb.persist_dir, "metadata_index"))
                
                self.kb._clear_retriever_cache()
            
            # Update CSV source metadata
            self.csv_docs[csv_source_id]["row_count"] = self.csv_docs[csv_source_id].get("row_count", 0) - 1
            
            # Save the updated CSV docs index
            with open(self.csv_docs_index_path, 'w') as f:
                json.dump(self.csv_docs, f, indent=2)
            
            logger.info(f"Marked row with document ID {doc_id} as deleted in CSV source {csv_source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete CSV row: {str(e)}")
            from .kb import DocumentProcessingError
            raise DocumentProcessingError(f"CSV row deletion failed: {str(e)}") from e
    def get_csv_rows(self, csv_source_id: str) -> List[Dict]:
        """Get all rows from a CSV source.
        
        Args:
            csv_source_id: ID of the CSV source
            
        Returns:
            List of dictionaries with row data
        """
        if not hasattr(self.kb, 'text_index') or not self.kb.text_index:
            return []
            
        try:
            # Check if CSV source exists
            if csv_source_id not in self.csv_docs:
                raise ValueError(f"CSV source not found: {csv_source_id}")
            
            # Find all documents from this CSV source
            rows = []
            seen_doc_ids = set()  # Track doc_ids we've already processed
            for node_id, node in self.kb.text_index.docstore.docs.items():
                if node.metadata.get("csv_source_id") == csv_source_id:
                    # Skip deleted rows
                    if node.metadata.get("is_deleted", False):
                        continue
                        
                    # Create a row object with text and metadata
                    row = {
                        "node_id": node_id,
                        "text": node.text,
                        "doc_id": node.metadata.get("doc_id", ""),
                        "row_index": node.metadata.get("row_index", 0)
                    }
                    
                    # Skip if we've already seen this doc_id to avoid duplicates
                    doc_id = node.metadata.get("doc_id", "")
                    if doc_id in seen_doc_ids:
                        continue
                    seen_doc_ids.add(doc_id)
                    
                    # Add all metadata fields that start with "col_" or are in the metadata
                    for key, value in node.metadata.items():
                        if key.startswith("col_") or key in ["doc_id", "row_index"]:
                            row[key] = value
                    
                    rows.append(row)
            
            # Sort rows by row_index
            rows.sort(key=lambda x: x.get("row_index", 0))
            
            return rows
            
        except Exception as e:
            logger.error(f"Failed to get CSV rows: {str(e)}")
            return []

    async def sync_csv_document(self, file_path: str, progress_callback: Optional[Callable] = None):
        """Sync a CSV document with the index, updating/adding/removing rows as needed."""
        if not hasattr(self.kb, 'text_index') or not self.kb.text_index:
            raise ValueError("Index not initialized.")
            
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            # Find the CSV source in our index
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
            for node_id, node in self.kb.text_index.docstore.docs.items():
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
            import csv
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
            has_header = config.get("has_header", True)
            headers = rows[0] if rows and has_header else []
            
            # Process each row to build new documents
            new_docs = {}
            metadata_docs = {}
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
                        continue
                    
                    # Create metadata
                    metadata = {
                        "file_path": file_path,
                        "file_name": os.path.basename(file_path),
                        "file_type": "csv",
                        "row_index": i,
                        "csv_source_id": safe_source_id,
                        "doc_id": doc_id,
                        "is_csv_row": True,
                        "is_deleted": False
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
                    
                    # Create metadata document
                    metadata_text = self._encode_metadata_for_indexing(metadata)
                    if metadata_text.strip():
                        metadata_docs[doc_id] = Document(
                            text=metadata_text,
                            metadata={
                                "csv_row_id": doc_id,
                                "csv_source_id": safe_source_id,
                                **metadata
                            }
                        )
                except Exception as e:
                    logger.warning(f"Error processing row {i} during sync: {str(e)}")
            
            # Determine which documents to add, update, or delete
            to_add = []
            to_add_metadata = []
            to_update = []
            to_update_metadata = []
            to_delete = set(existing_docs.keys()) - set(new_docs.keys())
            
            for doc_id, doc in new_docs.items():
                if doc_id not in existing_docs:
                    to_add.append(doc)
                    if doc_id in metadata_docs:
                        to_add_metadata.append(metadata_docs[doc_id])
                else:
                    # Compare text to detect changes
                    node_id = existing_docs[doc_id]
                    existing_node = self.kb.text_index.docstore.docs[node_id]
                    if existing_node.text != doc.text:
                        to_update.append((node_id, doc))
                        if doc_id in metadata_docs:
                            to_update_metadata.append((doc_id, metadata_docs[doc_id]))
            
            # Update progress
            if progress_callback:
                progress_callback(0.3)  # 30% progress after analysis
            
            # Update index atomically
            from .kb import atomic_index_update
            with atomic_index_update(self.kb):
                # Delete removed rows
                for doc_id in to_delete:
                    node_id = existing_docs[doc_id]
                    self.kb.text_index.delete_ref_doc(node_id, delete_from_docstore=True)
                    
                    # Also delete from metadata index
                    if hasattr(self.kb, 'metadata_index'):
                        for meta_node_id, meta_node in self.kb.metadata_index.docstore.docs.items():
                            if (meta_node.metadata.get("csv_source_id") == safe_source_id and 
                                meta_node.metadata.get("csv_row_id") == doc_id):
                                self.kb.metadata_index.delete_ref_doc(meta_node_id, delete_from_docstore=True)
                                break
                
                # Update changed rows
                for node_id, doc in to_update:
                    self.kb.text_index.delete_ref_doc(node_id, delete_from_docstore=True)
                    nodes = self.kb.node_parser.get_nodes_from_documents([doc])
                    for node in nodes:
                        self.kb.text_index.insert_nodes([node])
                
                # Update metadata for changed rows
                if hasattr(self.kb, 'metadata_index'):
                    for doc_id, meta_doc in to_update_metadata:
                        # Find and delete existing metadata node
                        for meta_node_id, meta_node in self.kb.metadata_index.docstore.docs.items():
                            if (meta_node.metadata.get("csv_source_id") == safe_source_id and 
                                meta_node.metadata.get("csv_row_id") == doc_id):
                                self.kb.metadata_index.delete_ref_doc(meta_node_id, delete_from_docstore=True)
                                break
                        
                        # Add updated metadata node
                        meta_nodes = self.kb.node_parser.get_nodes_from_documents([meta_doc])
                        for node in meta_nodes:
                            self.kb.metadata_index.insert_nodes([node])
                
                # Add new rows
                if to_add:
                    nodes = self.kb.node_parser.get_nodes_from_documents(to_add)
                    for node in nodes:
                        self.kb.text_index.insert_nodes([node])
                
                # Add new metadata
                if hasattr(self.kb, 'metadata_index') and to_add_metadata:
                    meta_nodes = self.kb.node_parser.get_nodes_from_documents(to_add_metadata)
                    for node in meta_nodes:
                        self.kb.metadata_index.insert_nodes([node])
                
                # Persist updates
                self.kb.text_index.storage_context.persist(persist_dir=os.path.join(self.kb.persist_dir, "text_index"))
                if hasattr(self.kb, 'metadata_index'):
                    self.kb.metadata_index.storage_context.persist(persist_dir=os.path.join(self.kb.persist_dir, "metadata_index"))
                
                self.kb._clear_retriever_cache()
            
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
            from .kb import DocumentProcessingError
            raise DocumentProcessingError(f"CSV document sync failed: {str(e)}") from e