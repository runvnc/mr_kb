#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portable ChromaDB Import/Export Utilities

This module provides functions to export ChromaDB collections to portable formats
and import them back, with special handling for large datasets through chunking.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('chromadb_portable')

def export_collection_to_parquet(
    collection,
    output_path: str,
    chunk_size: int = 1000,
    include_embeddings: bool = True,
    include_documents: bool = True,
    include_metadatas: bool = True,
    compression: str = 'snappy'
) -> str:
    """
    Export a ChromaDB collection to a Parquet file with chunking for large datasets.
    
    Args:
        collection: ChromaDB collection object
        output_path: Path to save the Parquet file
        chunk_size: Number of records to process at once
        include_embeddings: Whether to include embeddings in the export
        include_documents: Whether to include documents in the export
        include_metadatas: Whether to include metadata in the export
        compression: Compression algorithm for Parquet (snappy, gzip, brotli, etc.)
        
    Returns:
        Path to the created Parquet file
    """
    start_time = time.time()
    logger.info(f"Starting export of collection to {output_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Get collection count
    try:
        count = collection.count()
        logger.info(f"Collection contains {count} records")
    except Exception as e:
        # Some versions of ChromaDB might not have count() method
        logger.warning(f"Could not get collection count: {e}")
        count = None
    
    # Prepare what to include
    include = ["ids"]
    if include_embeddings:
        include.append("embeddings")
    if include_documents:
        include.append("documents")
    if include_metadatas:
        include.append("metadatas")
    
    # Create schema based on first chunk
    logger.info("Fetching first chunk to determine schema")
    first_chunk = collection.get(limit=1, include=include)
    
    # Discover metadata schema from first record
    metadata_fields = set()
    if include_metadatas and first_chunk["metadatas"] and first_chunk["metadatas"][0]:
        metadata_fields.update(first_chunk["metadatas"][0].keys())
    
    # Create PyArrow schema
    fields = [
        pa.field('id', pa.string())
    ]
    
    if include_embeddings:
        # Store embeddings as list of floats
        fields.append(pa.field('embedding', pa.list_(pa.float32())))
        # Also store the embedding dimension for reference
        if first_chunk["embeddings"] and len(first_chunk["embeddings"]) > 0:
            fields.append(pa.field('embedding_dimension', pa.int32()))
    
    if include_documents:
        fields.append(pa.field('document', pa.string()))
    
    # Add metadata fields
    for field in metadata_fields:
        # Use string type as default for metadata fields
        # This could be improved with type inference
        fields.append(pa.field(f"metadata_{field}", pa.string()))
    
    # Add export metadata
    fields.extend([
        pa.field('export_timestamp', pa.timestamp('s')),
        pa.field('export_version', pa.string())
    ])
    
    schema = pa.schema(fields)
    logger.info(f"Created schema with {len(fields)} fields")
    
    # Create Parquet writer
    writer = pq.ParquetWriter(output_path, schema, compression=compression)
    
    # Process in chunks
    offset = 0
    total_records = 0
    
    while True:
        logger.info(f"Processing chunk starting at offset {offset}")
        
        # Get chunk of data
        try:
            results = collection.get(
                limit=chunk_size,
                offset=offset,
                include=include
            )
        except Exception as e:
            # Some versions might not support offset
            if offset == 0:
                logger.warning(f"Error with offset parameter: {e}. Trying without pagination.")
                results = collection.get(include=include)
            else:
                logger.info("Finished processing all available records")
                break
        
        # Check if we got any results
        if not results["ids"] or len(results["ids"]) == 0:
            logger.info("No more records to process")
            break
        
        chunk_size_actual = len(results["ids"])
        logger.info(f"Processing {chunk_size_actual} records")
        
        # Prepare data for this chunk
        chunk_data = {
            "id": results["ids"],
            "export_timestamp": [int(time.time())] * chunk_size_actual,
            "export_version": ["1.0.0"] * chunk_size_actual
        }
        
        # Add embeddings if included
        if include_embeddings and "embeddings" in results and results["embeddings"]:
            chunk_data["embedding"] = results["embeddings"]
            # Add embedding dimension
            if results["embeddings"][0]:
                dim = len(results["embeddings"][0])
                chunk_data["embedding_dimension"] = [dim] * chunk_size_actual
        
        # Add documents if included
        if include_documents and "documents" in results and results["documents"]:
            chunk_data["document"] = results["documents"]
        
        # Process metadata
        if include_metadatas and "metadatas" in results and results["metadatas"]:
            # Discover any new metadata fields
            for meta in results["metadatas"]:
                if meta:  # Check if metadata exists
                    metadata_fields.update(meta.keys())
            
            # Add each metadata field as a separate column
            for field in metadata_fields:
                chunk_data[f"metadata_{field}"] = [
                    str(meta.get(field)) if meta and field in meta else None 
                    for meta in results["metadatas"]
                ]
        
        # Convert to PyArrow table
        try:
            # Create a pandas DataFrame first
            df = pd.DataFrame(chunk_data)
            
            # Convert DataFrame to PyArrow Table
            table = pa.Table.from_pandas(df, schema=schema)
            
            # Write to Parquet file
            writer.write_table(table)
            
            total_records += chunk_size_actual
            logger.info(f"Wrote {chunk_size_actual} records to Parquet file")
            
            # Update offset for next chunk
            offset += chunk_size_actual
            
        except Exception as e:
            logger.error(f"Error writing chunk to Parquet: {e}")
            raise
    
    # Close the writer
    writer.close()
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Export completed: {total_records} records written to {output_path} in {duration:.2f} seconds")
    
    return output_path

def import_from_parquet_to_chromadb(
    chroma_client,
    parquet_path: str,
    collection_name: str,
    embedding_function=None,
    chunk_size: int = 500,
    create_if_not_exists: bool = True,
    reset_collection: bool = False
) -> Tuple[Any, int]:
    """
    Import data from a Parquet file into a ChromaDB collection with chunking for large datasets.
    
    Args:
        chroma_client: ChromaDB client instance
        parquet_path: Path to the Parquet file
        collection_name: Name of the collection to import into
        embedding_function: Optional embedding function to use for the collection
        chunk_size: Number of records to process at once
        create_if_not_exists: Whether to create the collection if it doesn't exist
        reset_collection: Whether to reset the collection before importing
        
    Returns:
        Tuple of (collection object, number of records imported)
    """
    start_time = time.time()
    logger.info(f"Starting import from {parquet_path} to collection {collection_name}")
    
    # Check if file exists
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    # Get or create collection
    try:
        if reset_collection:
            logger.info(f"Resetting collection {collection_name}")
            try:
                chroma_client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Error deleting collection: {e}")
            
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Created new collection: {collection_name}")
        else:
            # Try to get the collection if it exists
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Using existing collection: {collection_name}")
    except Exception as e:
        if create_if_not_exists:
            logger.info(f"Collection does not exist, creating: {collection_name}")
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Created new collection: {collection_name}")
        else:
            logger.error(f"Collection does not exist and create_if_not_exists is False: {e}")
            raise
    
    # Open the Parquet file for reading
    parquet_file = pq.ParquetFile(parquet_path)
    
    # Get total number of row groups for progress reporting
    total_row_groups = parquet_file.num_row_groups
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"Parquet file contains {total_rows} rows in {total_row_groups} row groups")
    
    # Process in chunks
    total_imported = 0
    batch_num = 0
    
    # Read and process the Parquet file in chunks
    for batch_num, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
        # Convert PyArrow batch to pandas DataFrame
        df = batch.to_pandas()
        logger.info(f"Processing batch {batch_num+1} with {len(df)} records")
        
        # Extract IDs
        ids = df["id"].tolist()
        
        # Extract embeddings if present
        embeddings = None
        if "embedding" in df.columns:
            # Convert embeddings from lists to numpy arrays then back to lists
            # This ensures consistent format
            embeddings = [np.array(emb, dtype=np.float32).tolist() 
                         if isinstance(emb, (list, np.ndarray)) else emb 
                         for emb in df["embedding"]]
        
        # Extract documents if present
        documents = None
        if "document" in df.columns:
            documents = df["document"].tolist()
        
        # Reconstruct metadata dictionaries
        metadata_columns = [col for col in df.columns if col.startswith("metadata_")]
        metadatas = None
        
        if metadata_columns:
            metadatas = []
            for _, row in df.iterrows():
                metadata = {}
                for col in metadata_columns:
                    # Extract the original key name by removing "metadata_" prefix
                    key = col[9:]  # len("metadata_") = 9
                    if pd.notna(row[col]):
                        metadata[key] = row[col]
                metadatas.append(metadata)
        
        # Add data to collection
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            total_imported += len(ids)
            logger.info(f"Added batch {batch_num+1}: {len(ids)} records")
        except Exception as e:
            logger.error(f"Error adding batch {batch_num+1} to collection: {e}")
            raise
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Import completed: {total_imported} records imported to collection '{collection_name}' in {duration:.2f} seconds")
    
    return collection, total_imported


def get_parquet_info(parquet_path: str) -> Dict[str, Any]:
    """
    Get information about a Parquet file exported from ChromaDB.
    
    Args:
        parquet_path: Path to the Parquet file
        
    Returns:
        Dictionary with information about the Parquet file
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    # Open the Parquet file
    parquet_file = pq.ParquetFile(parquet_path)
    
    # Get basic file info
    info = {
        "file_path": parquet_path,
        "file_size_bytes": os.path.getsize(parquet_path),
        "file_size_mb": os.path.getsize(parquet_path) / (1024 * 1024),
        "num_rows": parquet_file.metadata.num_rows,
        "num_row_groups": parquet_file.num_row_groups,
        "created_at": time.ctime(os.path.getctime(parquet_path)),
        "modified_at": time.ctime(os.path.getmtime(parquet_path)),
    }
    
    # Get schema information
    schema = parquet_file.schema
    fields = []
    
    for field in schema.names:
        field_type = schema.field(field).type
        fields.append({
            "name": field,
            "type": str(field_type),
            "is_metadata": field.startswith("metadata_")
        })
    
    info["schema"] = {
        "fields": fields,
        "num_fields": len(fields)
    }
    
    # Check for embedding information
    if "embedding" in schema.names:
        # Try to get embedding dimension from the first row
        try:
            first_row = next(parquet_file.iter_batches(batch_size=1)).to_pandas()
            if "embedding_dimension" in first_row.columns:
                info["embedding_dimension"] = int(first_row["embedding_dimension"].iloc[0])
            elif "embedding" in first_row.columns and len(first_row["embedding"]) > 0:
                # Try to infer from the first embedding
                first_embedding = first_row["embedding"].iloc[0]
                if isinstance(first_embedding, (list, np.ndarray)):
                    info["embedding_dimension"] = len(first_embedding)
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
    
    # Get metadata field information
    metadata_fields = [f for f in fields if f["is_metadata"]]
    if metadata_fields:
        info["metadata_fields"] = [f["name"][9:] for f in metadata_fields]  # Remove "metadata_" prefix
        info["num_metadata_fields"] = len(metadata_fields)
    
    return info


def command_line_interface():
    """
    Command line interface for the portable ChromaDB utilities.
    
    Usage:
        python -m mr_kb.portable export --collection=my_collection --output=my_export.parquet
        python -m mr_kb.portable import --parquet=my_export.parquet --collection=new_collection
        python -m mr_kb.portable info --parquet=my_export.parquet
    """
    import argparse
    import json
    import chromadb
    
    parser = argparse.ArgumentParser(description="ChromaDB Portable Import/Export Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a ChromaDB collection to Parquet")
    export_parser.add_argument("--collection", required=True, help="Name of the collection to export")
    export_parser.add_argument("--output", required=True, help="Path to save the Parquet file")
    export_parser.add_argument("--chunk-size", type=int, default=1000, help="Number of records to process at once")
    export_parser.add_argument("--no-embeddings", action="store_true", help="Don't include embeddings in the export")
    export_parser.add_argument("--no-documents", action="store_true", help="Don't include documents in the export")
    export_parser.add_argument("--no-metadatas", action="store_true", help="Don't include metadata in the export")
    export_parser.add_argument("--compression", default="snappy", help="Compression algorithm for Parquet")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import a Parquet file into ChromaDB")
    import_parser.add_argument("--parquet", required=True, help="Path to the Parquet file")
    import_parser.add_argument("--collection", required=True, help="Name of the collection to import into")
    import_parser.add_argument("--chunk-size", type=int, default=500, help="Number of records to process at once")
    import_parser.add_argument("--reset", action="store_true", help="Reset the collection before importing")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about a Parquet file")
    info_parser.add_argument("--parquet", required=True, help="Path to the Parquet file")
    
    args = parser.parse_args()
    
    if args.command == "export":
        # Initialize ChromaDB client
        client = chromadb.Client()
        
        # Get collection
        try:
            collection = client.get_collection(args.collection)
        except Exception as e:
            logger.error(f"Error getting collection: {e}")
            return 1
        
        # Export collection
        try:
            output_path = export_collection_to_parquet(
                collection=collection,
                output_path=args.output,
                chunk_size=args.chunk_size,
                include_embeddings=not args.no_embeddings,
                include_documents=not args.no_documents,
                include_metadatas=not args.no_metadatas,
                compression=args.compression
            )
            logger.info(f"Successfully exported collection to {output_path}")
            return 0
        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            return 1
    
    elif args.command == "import":
        # Initialize ChromaDB client
        client = chromadb.Client()
        
        # Import collection
        try:
            collection, count = import_from_parquet_to_chromadb(
                chroma_client=client,
                parquet_path=args.parquet,
                collection_name=args.collection,
                chunk_size=args.chunk_size,
                reset_collection=args.reset
            )
            logger.info(f"Successfully imported {count} records to collection {args.collection}")
            return 0
        except Exception as e:
            logger.error(f"Error importing collection: {e}")
            return 1
    
    elif args.command == "info":
        # Get info about Parquet file
        try:
            info = get_parquet_info(args.parquet)
            print(json.dumps(info, indent=2, default=str))
            return 0
        except Exception as e:
            logger.error(f"Error getting Parquet info: {e}")
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(command_line_interface())
