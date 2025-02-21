from typing import Dict

def get_supported_file_types() -> Dict[str, bool]:
    """Get dictionary of supported file types and their availability.
    
    Returns:
        Dict mapping file extensions to boolean indicating if support is available
    """
    support_status = {
        ".txt": True,  # Always supported
        ".md": True,  # Always supported
        ".pdf": False,
        ".xlsx": False,
        ".xls": False,
        ".docx": False
    }
    
    # Check PDF support
    try:
        import PyPDF2
        support_status[".pdf"] = True
    except ImportError:
        try:
            import pypdf
            support_status[".pdf"] = True
        except ImportError:
            pass
    
    # Check Excel support
    try:
        import pandas as pd
        support_status[".xlsx"] = True
        support_status[".xls"] = True
    except ImportError:
        pass
        
    # Check Word support
    try:
        import docx
        support_status[".docx"] = True
    except ImportError:
        pass
    
    return support_status

def format_supported_types(supported_types: Dict[str, bool]) -> str:
    """Format supported file types into a readable string."""
    available = [ext for ext, supported in supported_types.items() if supported]
    unavailable = [ext for ext, supported in supported_types.items() if not supported]
    
    result = "Supported file types:\n"
    if available:
        result += "✓ Available:\n  " + ", ".join(available) + "\n"
    if unavailable:
        result += "✗ Requires additional packages:\n  " + ", ".join(unavailable)
    
    return result
