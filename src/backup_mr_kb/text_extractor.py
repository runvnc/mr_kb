# text_extractor.py
from typing import Dict, Optional
from llama_index.core.readers import SimpleDirectoryReader
from .utils import get_supported_file_types
from .file_handlers import ExcelReader, DocxReader
import os
import logging

logger = logging.getLogger(__name__)

class DocumentExtractionError(Exception):
    """Raised when document extraction fails."""
    pass

def get_file_extractors() -> Dict[str, callable]:
    """Get dictionary of file handlers based on available support."""
    supported_types = get_supported_file_types()
    handlers = {}
    
    if supported_types[".xlsx"]:
        handlers.update({".xlsx": ExcelReader(), ".xls": ExcelReader()})
    if supported_types[".docx"]:
        handlers[".docx"] = DocxReader()
    
    return handlers

async def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file using the same process as document indexing.
    
    Args:
        file_path: Path to the file to extract text from
        
    Returns:
        str: The extracted text content from the file
        
    Raises:
        DocumentExtractionError: If text extraction fails
        ValueError: If no content is found in the file
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
            
        # Get file extractors based on supported types
        file_extractor = get_file_extractors()
        
        # Load document with custom handlers
        documents = SimpleDirectoryReader(
            input_files=[file_path], 
            file_extractor=file_extractor
        ).load_data()
        
        if not documents:
            raise ValueError(f"No content found in {file_path}")
        
        # Return the text from the document(s)
        # For most files, SimpleDirectoryReader returns one document per file
        if len(documents) == 1:
            return documents[0].text
        else:
            # If multiple documents were extracted, combine them with separators
            combined_text = ""
            for i, doc in enumerate(documents):
                combined_text += doc.text
                if i < len(documents) - 1:  # Add separator between documents
                    combined_text += "\n\n---\n\n"
            return combined_text
        
    except Exception as e:
        logger.error(f"Failed to extract text: {str(e)}")
        raise DocumentExtractionError(f"Text extraction failed: {str(e)}") from e

# Utility function to get file metadata
async def get_file_metadata(file_path: str) -> Dict:
    """Extract metadata from a file without processing its content.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict: File metadata including name, path, type, and size
    """
    try:
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
            
        # Get basic file info
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        _, file_ext = os.path.splitext(file_path)
        file_ext = file_ext.lower()
        
        # Determine file type
        supported_types = get_supported_file_types()
        file_type = file_ext[1:] if file_ext.startswith('.') else file_ext
        is_supported = file_ext in supported_types and supported_types[file_ext]
        
        return {
            "file_name": file_name,
            "file_path": file_path,
            "file_type": file_type,
            "file_size": file_size,
            "is_supported": is_supported
        }
        
    except Exception as e:
        logger.error(f"Failed to get file metadata: {str(e)}")
        raise DocumentExtractionError(f"Metadata extraction failed: {str(e)}") from e
