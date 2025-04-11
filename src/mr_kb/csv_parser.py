import csv
import os
import re

# Set a high field size limit for large text fields
csv.field_size_limit(1000000)

def detect_csv_format(file_content):
    """
    Detect the format of a CSV file based on its content.
    
    Args:
        file_content (str): The content of the CSV file
        
    Returns:
        tuple: (dialect, has_header, has_multiple_quotes)
    """
    # Check if this is a file with special quoting
    has_multiple_quotes = '"""' in file_content
    
    # Create a custom dialect for files with multiple quotes or special formatting
    if has_multiple_quotes:
        class CustomDialect(csv.excel):
            quotechar = '\"'
            doublequote = True
            quoting = csv.QUOTE_ALL
            skipinitialspace = False
        dialect = CustomDialect()
        has_header = False  # For files with multiple quotes, assume no headers by default
    # Check for other special formats that might need custom handling
    elif '""' in file_content or re.search(r'[^\r\n]""[^\r\n]', file_content):
        # Files with double quotes inside fields
        class CustomDialect(csv.excel):
            quotechar = '\"'
            doublequote = True
            quoting = csv.QUOTE_ALL
            skipinitialspace = False
        dialect = CustomDialect()
        has_header = False
    else:
        # Try to detect the dialect for standard CSV files
        try:
            dialect = csv.Sniffer().sniff(file_content[:1024])
            has_header = csv.Sniffer().has_header(file_content[:1024])
        except:
            dialect = csv.excel
            has_header = False  # Default to no headers if detection fails
    
    return dialect, has_header, has_multiple_quotes

def parse_csv(file_path=None, file_content=None, dialect=None, max_rows=None):
    """
    Parse a CSV file with the appropriate settings.
    
    Args:
        file_path (str, optional): Path to the CSV file
        file_content (str, optional): Content of the CSV file
        dialect (csv.Dialect, optional): Dialect to use for parsing
        max_rows (int, optional): Maximum number of rows to parse
        
    Returns:
        list: List of rows from the CSV file
        Note: This will return ALL rows, not just a preview, unless max_rows is specified
    """
    if file_path and not file_content:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            file_content = f.read()
    
    if not dialect:
        dialect, _, _ = detect_csv_format(file_content)
    
    # Print debug info
    print(f"Parsing CSV with dialect: {dialect.__class__.__name__}, quotechar: {dialect.quotechar}, doublequote: {dialect.doublequote}")
    
    # Parse the CSV with appropriate settings
    rows = []
    reader = csv.reader(file_content.splitlines(), dialect=dialect)
    for i, row in enumerate(reader):
        rows.append(row)
        if max_rows is not None and i >= max_rows - 1:
            break
    
    # Print row count for debugging
    print(f"Parsed {len(rows)} rows from CSV")
    
    return rows

def generate_column_names(rows):
    """
    Generate default column names for a CSV file.
    
    Args:
        rows (list): List of rows from the CSV file
        
    Returns:
        list: List of column names
    """
    if not rows:
        return []
    
    # Get the maximum number of columns
    max_cols = max(len(row) for row in rows)
    
    # Generate column names like 'A', 'B', 'C' or 'Column 1', 'Column 2', etc.
    column_names = []
    for i in range(max_cols):
        # Use alphabetical names for first 26 columns, then switch to numbers
        column_names.append(chr(65 + i) if i < 26 else f"Column {i+1}")
    
    return column_names

def create_column_map(rows):
    """
    Create a column map for a CSV file.
    
    Args:
        rows (list): List of rows from the CSV file
        
    Returns:
        dict: Dictionary mapping column indices to column names
    """
    column_names = generate_column_names(rows)
    return {i: name for i, name in enumerate(column_names)}

def get_csv_preview(file_path, max_rows=10):
    """
    Get a preview of a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        max_rows (int, optional): Maximum number of rows to include in the preview
        
    Returns:
        dict: Preview data including rows, dialect, column names, etc.
    """
    print(f"Getting CSV preview for file: {file_path}")
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            file_content = f.read()
        
        # Detect the CSV format
        dialect, has_header, has_multiple_quotes = detect_csv_format(file_content)
        
        # Print file info for debugging
        print(f"File content length: {len(file_content)} bytes")
        print(f"Detected format: has_header={has_header}, has_multiple_quotes={has_multiple_quotes}")
        
        # Parse the CSV
        rows = parse_csv(file_content=file_content, dialect=dialect, max_rows=max_rows)
        
        # Count total rows for comparison later
        # Generate default column names
        default_column_names = generate_column_names(rows)
        
        # Get column count
        max_cols = len(default_column_names)
        
        # Create preview data
        # Get a more accurate row count by parsing the entire file
        print("Parsing entire file to get accurate row count...")
        all_rows = parse_csv(file_content=file_content, dialect=dialect)
        total_rows = len(all_rows)
        print(f"Total rows in file (actual): {total_rows}")
        
        # For debugging, also show the line count
        line_count = len(file_content.splitlines())
        print(f"Total lines in file: {line_count} (not the same as rows for multiline fields)")
        
        # Use the actual parsed row count as the expected count
        preview_data = {
            "rows": rows,
            "dialect": {
                "delimiter": dialect.delimiter,
                "doublequote": dialect.doublequote,
                "escapechar": dialect.escapechar,
                "lineterminator": repr(dialect.lineterminator),
                "quotechar": dialect.quotechar,
                "quoting": dialect.quoting,
                "skipinitialspace": dialect.skipinitialspace
            } if hasattr(dialect, 'delimiter') else str(dialect),
            "has_header": has_header,
            "has_multiple_quotes": has_multiple_quotes,
            "default_column_names": default_column_names,
            "column_count": max_cols,
            "preview_row_count": len(rows),
            "row_count": total_rows,  # This is the total number of rows in the file
            "temp_path": file_path
        }
        
        return preview_data
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}")
