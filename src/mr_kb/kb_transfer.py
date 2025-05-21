#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MindRoot Knowledge Base Export/Import Utilities

This module provides functions to export a complete knowledge base to a portable format
and import it into another MindRoot instance.
"""

import os
import shutil
import logging
import json
import time
import zipfile
from typing import Dict, List, Optional, Union, Any, Tuple

from .portable import export_collection_to_parquet, import_from_parquet_to_chromadb

logger = logging.getLogger(__name__)

async def export_kb(kb_instance, output_path: str, progress_callback=None):
    """
    Export a complete knowledge base to a portable format.
    
    Args:
        kb_instance: HierarchicalKnowledgeBase instance
        output_path: Path to save the exported KB (zip file)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to the created export file
    """
    start_time = time.time()
    logger.info(f"Starting export of knowledge base to {output_path}")
    
    # Create a temporary directory for the export
    temp_dir = f"{output_path}_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 1. Export ChromaDB collections to Parquet
        if progress_callback:
            progress_callback(0.1, "Exporting text collection")
            
        text_parquet_path = os.path.join(temp_dir, "text_collection.parquet")
        export_collection_to_parquet(
            collection=kb_instance.text_collection,
            output_path=text_parquet_path,
            chunk_size=1000
        )
        
        if progress_callback:
            progress_callback(0.3, "Exporting metadata collection")
            
        metadata_parquet_path = os.path.join(temp_dir, "metadata_collection.parquet")
        export_collection_to_parquet(
            collection=kb_instance.metadata_collection,
            output_path=metadata_parquet_path,
            chunk_size=1000
        )
        
        # 2. Copy document directories
        if progress_callback:
            progress_callback(0.4, "Copying verbatim documents")
            
        # Verbatim documents
        verbatim_docs_dir = os.path.join(temp_dir, "verbatim_docs")
        os.makedirs(verbatim_docs_dir, exist_ok=True)
        if os.path.exists(kb_instance.verbatim_docs_dir):
            for file in os.listdir(kb_instance.verbatim_docs_dir):
                src_path = os.path.join(kb_instance.verbatim_docs_dir, file)
                dst_path = os.path.join(verbatim_docs_dir, file)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
        
        # URL documents
        if progress_callback:
            progress_callback(0.5, "Copying URL documents")
            
        url_docs_dir = os.path.join(temp_dir, "url_docs")
        os.makedirs(url_docs_dir, exist_ok=True)
        if os.path.exists(kb_instance.url_docs_dir):
            for file in os.listdir(kb_instance.url_docs_dir):
                src_path = os.path.join(kb_instance.url_docs_dir, file)
                dst_path = os.path.join(url_docs_dir, file)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
        
        # CSV documents
        if progress_callback:
            progress_callback(0.6, "Copying CSV documents")
            
        csv_docs_dir = os.path.join(temp_dir, "csv_docs")
        os.makedirs(csv_docs_dir, exist_ok=True)
        if os.path.exists(kb_instance.csv_docs_dir):
            for item in os.listdir(kb_instance.csv_docs_dir):
                src_path = os.path.join(kb_instance.csv_docs_dir, item)
                dst_path = os.path.join(csv_docs_dir, item)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                elif os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
        
        # 3. Copy index files
        if progress_callback:
            progress_callback(0.7, "Copying index files")
            
        # Verbatim docs index
        if os.path.exists(kb_instance.verbatim_docs_index_path):
            shutil.copy2(kb_instance.verbatim_docs_index_path, 
                        os.path.join(temp_dir, "verbatim_docs_index.json"))
        
        # URL docs index
        if os.path.exists(kb_instance.url_docs_index_path):
            shutil.copy2(kb_instance.url_docs_index_path, 
                        os.path.join(temp_dir, "url_docs_index.json"))
        
        # CSV docs index
        if os.path.exists(kb_instance.csv_docs_index_path):
            shutil.copy2(kb_instance.csv_docs_index_path, 
                        os.path.join(temp_dir, "csv_docs_index.json"))
        
        # 4. Create metadata file with KB information
        if progress_callback:
            progress_callback(0.8, "Creating metadata")
            
        kb_metadata = {
            "export_version": "1.0.0",
            "export_timestamp": time.time(),
            "export_date": time.ctime(),
            "chunk_sizes": kb_instance.chunk_sizes,
            "components": {
                "text_collection": os.path.exists(text_parquet_path),
                "metadata_collection": os.path.exists(metadata_parquet_path),
                "verbatim_docs": os.path.exists(kb_instance.verbatim_docs_dir),
                "url_docs": os.path.exists(kb_instance.url_docs_dir),
                "csv_docs": os.path.exists(kb_instance.csv_docs_dir)
            }
        }
        
        with open(os.path.join(temp_dir, "kb_metadata.json"), 'w') as f:
            json.dump(kb_metadata, f, indent=2)
        
        # 5. Create zip archive
        if progress_callback:
            progress_callback(0.9, "Creating archive")
            
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Export completed in {duration:.2f} seconds")
        
        if progress_callback:
            progress_callback(1.0, "Export complete")
            
        return output_path
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

async def import_kb(kb_instance, import_path: str, progress_callback=None):
    """
    Import a knowledge base from a portable format.
    
    Args:
        kb_instance: HierarchicalKnowledgeBase instance
        import_path: Path to the KB export file (zip)
        progress_callback: Optional callback for progress updates
        
    Returns:
        True if successful
    """
    start_time = time.time()
    logger.info(f"Starting import of knowledge base from {import_path}")
    
    # Create a temporary directory for the import
    temp_dir = f"{import_path}_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 1. Extract zip archive
        if progress_callback:
            progress_callback(0.1, "Extracting archive")
            
        with zipfile.ZipFile(import_path, 'r') as zipf:
            zipf.extractall(temp_dir)
        
        # 2. Read metadata file
        if progress_callback:
            progress_callback(0.2, "Reading metadata")
            
        metadata_path = os.path.join(temp_dir, "kb_metadata.json")
        if not os.path.exists(metadata_path):
            raise ValueError("Invalid KB export: metadata file not found")
            
        with open(metadata_path, 'r') as f:
            kb_metadata = json.load(f)
        
        # 3. Import ChromaDB collections from Parquet
        if progress_callback:
            progress_callback(0.3, "Importing text collection")
            
        text_parquet_path = os.path.join(temp_dir, "text_collection.parquet")
        if os.path.exists(text_parquet_path):
            import_from_parquet_to_chromadb(
                chroma_client=kb_instance.chroma_client,
                parquet_path=text_parquet_path,
                collection_name=kb_instance.text_collection_name,
                embedding_function=kb_instance.embed_model,
                chunk_size=500,
                reset_collection=True
            )
        
        if progress_callback:
            progress_callback(0.5, "Importing metadata collection")
            
        metadata_parquet_path = os.path.join(temp_dir, "metadata_collection.parquet")
        if os.path.exists(metadata_parquet_path):
            import_from_parquet_to_chromadb(
                chroma_client=kb_instance.chroma_client,
                parquet_path=metadata_parquet_path,
                collection_name=kb_instance.metadata_collection_name,
                embedding_function=kb_instance.embed_model,
                chunk_size=500,
                reset_collection=True
            )
        
        # 4. Copy document directories
        if progress_callback:
            progress_callback(0.6, "Copying document directories")
            
        # Verbatim documents
        verbatim_docs_dir = os.path.join(temp_dir, "verbatim_docs")
        if os.path.exists(verbatim_docs_dir):
            if os.path.exists(kb_instance.verbatim_docs_dir):
                shutil.rmtree(kb_instance.verbatim_docs_dir)
            shutil.copytree(verbatim_docs_dir, kb_instance.verbatim_docs_dir)
        
        # URL documents
        url_docs_dir = os.path.join(temp_dir, "url_docs")
        if os.path.exists(url_docs_dir):
            if os.path.exists(kb_instance.url_docs_dir):
                shutil.rmtree(kb_instance.url_docs_dir)
            shutil.copytree(url_docs_dir, kb_instance.url_docs_dir)
        
        # CSV documents
        csv_docs_dir = os.path.join(temp_dir, "csv_docs")
        if os.path.exists(csv_docs_dir):
            if os.path.exists(kb_instance.csv_docs_dir):
                shutil.rmtree(kb_instance.csv_docs_dir)
            shutil.copytree(csv_docs_dir, kb_instance.csv_docs_dir)
        
        # 5. Copy index files
        if progress_callback:
            progress_callback(0.8, "Copying index files")
            
        # Verbatim docs index
        verbatim_index_path = os.path.join(temp_dir, "verbatim_docs_index.json")
        if os.path.exists(verbatim_index_path):
            shutil.copy2(verbatim_index_path, kb_instance.verbatim_docs_index_path)
            # Load the index into memory
            with open(kb_instance.verbatim_docs_index_path, 'r') as f:
                kb_instance.verbatim_docs = json.load(f)
        
        # URL docs index
        url_index_path = os.path.join(temp_dir, "url_docs_index.json")
        if os.path.exists(url_index_path):
            shutil.copy2(url_index_path, kb_instance.url_docs_index_path)
            # Load the index into memory
            with open(kb_instance.url_docs_index_path, 'r') as f:
                kb_instance.url_docs = json.load(f)
        
        # CSV docs index
        csv_index_path = os.path.join(temp_dir, "csv_docs_index.json")
        if os.path.exists(csv_index_path):
            shutil.copy2(csv_index_path, kb_instance.csv_docs_index_path)
            # Load the index into memory
            with open(kb_instance.csv_docs_index_path, 'r') as f:
                kb_instance.csv_docs = json.load(f)
        
        # 6. Reload indices
        if progress_callback:
            progress_callback(0.9, "Reloading indices")
            
        # Set up vector stores
        kb_instance._setup_vector_stores()
        
        # Load ChromaDB indices
        from llama_index.core.indices import VectorStoreIndex
        kb_instance.text_index = VectorStoreIndex.from_vector_store(
            kb_instance.text_vector_store, 
            embed_model=kb_instance.embed_model
        )
        kb_instance.metadata_index = VectorStoreIndex.from_vector_store(
            kb_instance.metadata_vector_store, 
            embed_model=kb_instance.embed_model
        )
        
        # For backward compatibility
        kb_instance.index = kb_instance.text_index
        
        # Clear retriever cache
        kb_instance._clear_retriever_cache()
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Import completed in {duration:.2f} seconds")
        
        if progress_callback:
            progress_callback(1.0, "Import complete")
            
        return True
        
    except Exception as e:
        logger.error(f"Import failed: {str(e)}")
        raise
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def create_command_handlers():
    """
    Create command handlers for the KB transfer module.
    
    This function creates command handlers that can be registered with the MindRoot
    command system to provide KB export/import functionality to agents.
    
    Returns:
        Dictionary of command handlers
    """
    from lib.providers.commands import command
    
    @command()
    async def export_knowledge_base(output_path, context=None):
        """
        Export a knowledge base to a portable format.
        
        Args:
            output_path: Path to save the exported KB (zip file)
            
        Returns:
            Path to the created export file
        """
        if not context or not hasattr(context, 'kb'):
            return {"error": "No knowledge base available in context"}
            
        kb_instance = context.kb
        
        try:
            result = await export_kb(kb_instance, output_path)
            return {"success": True, "path": result}
        except Exception as e:
            return {"error": str(e)}
    
    @command()
    async def import_knowledge_base(import_path, context=None):
        """
        Import a knowledge base from a portable format.
        
        Args:
            import_path: Path to the KB export file (zip)
            
        Returns:
            Success status
        """
        if not context or not hasattr(context, 'kb'):
            return {"error": "No knowledge base available in context"}
            
        kb_instance = context.kb
        
        try:
            result = await import_kb(kb_instance, import_path)
            return {"success": result}
        except Exception as e:
            return {"error": str(e)}
    
    return {
        "export_knowledge_base": export_knowledge_base,
        "import_knowledge_base": import_knowledge_base
    }

# Command-line interface for standalone usage
def main():
    """
    Command-line interface for KB transfer utilities.
    
    Usage:
        python -m mr_kb.kb_transfer export --kb-dir=/path/to/kb --output=/path/to/export.zip
        python -m mr_kb.kb_transfer import --kb-dir=/path/to/kb --input=/path/to/export.zip
    """
    import argparse
    import asyncio
    from .kb import HierarchicalKnowledgeBase
    
    parser = argparse.ArgumentParser(description="MindRoot Knowledge Base Transfer Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a knowledge base")
    export_parser.add_argument("--kb-dir", required=True, help="Path to the knowledge base directory")
    export_parser.add_argument("--output", required=True, help="Path to save the exported KB (zip file)")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import a knowledge base")
    import_parser.add_argument("--kb-dir", required=True, help="Path to the knowledge base directory")
    import_parser.add_argument("--input", required=True, help="Path to the KB export file (zip)")
    
    args = parser.parse_args()
    
    if args.command == "export":
        # Initialize KB
        kb = HierarchicalKnowledgeBase(persist_dir=args.kb_dir)
        
        # Define progress callback
        def progress_callback(progress, message=None):
            if message:
                print(f"[{progress:.1%}] {message}")
            else:
                print(f"[{progress:.1%}]")
        
        # Run export
        try:
            result = asyncio.run(export_kb(kb, args.output, progress_callback))
            print(f"Export completed successfully: {result}")
            return 0
        except Exception as e:
            print(f"Export failed: {str(e)}")
            return 1
    
    elif args.command == "import":
        # Initialize KB
        kb = HierarchicalKnowledgeBase(persist_dir=args.kb_dir)
        
        # Define progress callback
        def progress_callback(progress, message=None):
            if message:
                print(f"[{progress:.1%}] {message}")
            else:
                print(f"[{progress:.1%}]")
        
        # Run import
        try:
            result = asyncio.run(import_kb(kb, args.input, progress_callback))
            print(f"Import completed successfully")
            return 0
        except Exception as e:
            print(f"Import failed: {str(e)}")
            return 1
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
