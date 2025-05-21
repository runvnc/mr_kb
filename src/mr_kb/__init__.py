# MindRoot Knowledge Base Plugin

# Import main classes and functions for easier access
from .kb import HierarchicalKnowledgeBase, DocumentProcessingError
from .csv_handler import CSVDocumentHandler
from .portable import export_collection_to_parquet, import_from_parquet_to_chromadb, get_parquet_info
from .kb_transfer import export_kb, import_kb, create_command_handlers

# Version information
__version__ = '1.0.0'
# Import Hugging Face integration if available
try:
    from .hf import upload_kb_to_hf, download_kb_from_hf, list_kb_repos
except ImportError:
    # huggingface_hub not installed, HF functions won't be available
    pass
