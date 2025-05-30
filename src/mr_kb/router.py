from fastapi import APIRouter, Request, UploadFile, File, Form, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from lib.templates import render
from .kb import HierarchicalKnowledgeBase
from .mod import get_kb_instance, create_kb, list_kbs, delete_kb, add_to_kb, add_url_to_kb, refresh_url_in_kb
import os
import tempfile
import uuid
import shutil
import asyncio
import json
import datetime
import re
import csv
from lib.utils.debug import debug_box
from .csv_parser import detect_csv_format, parse_csv, generate_column_names, create_column_map, get_csv_preview
import logging
import traceback

logger = logging.getLogger(__name__)


# Dictionary to store processing tasks and their status
processing_tasks = {}

router = APIRouter()

@router.get("/admin/kb")
async def admin_page(request: Request):
    """Render KB admin page"""
    html = await render('kb_admin', {
        "user": request.state.user.username
    })
    return HTMLResponse(html)

@router.get("/api/kb/list")
async def list_knowledge_bases(request: Request):
    """List all knowledge bases"""
    try:
        kbs = await list_kbs()
        return JSONResponse({"success": True, "data": kbs})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.post("/api/kb/{name}/csv/{source_id}/row")
async def add_csv_row(name: str, source_id: str, request: Request):
    """Add a new row to a CSV source"""
    try:
        data = await request.json()
        new_text = data.get("text")
        new_metadata = data.get("metadata")

        if not new_text:
            return JSONResponse({"success": False, "message": "Text is required"}, status_code=400)
        
        kb = await get_kb_instance(name)
        
        # Check if kb has CSV document support
        if not hasattr(kb, 'csv_docs'):
            return JSONResponse({"success": False, "message": "This knowledge base was created with an older version and doesn't support CSV documents"}, status_code=400)
        
        # Get the CSV source configuration to determine which column is the ID column
        csv_config = kb.csv_docs[source_id].get("config", {})
        id_column = csv_config.get("id_column")
        
        # If we have an ID column defined in the config, try to use the corresponding value from metadata
        custom_doc_id = None
        if id_column is not None and new_metadata:
            # The column in metadata will be named col_X where X is the column index
            id_col_name = f"col_{id_column}"
            
            if id_col_name in new_metadata and new_metadata[id_col_name]:
                custom_doc_id = new_metadata[id_col_name]
                print(f"Using {id_col_name} value as doc_id: {custom_doc_id}")
                
                # Validate that the ID is unique
                existing_rows = kb.get_csv_rows(source_id)
                existing_ids = [row.get("doc_id") for row in existing_rows]
                if custom_doc_id in existing_ids:
                    return JSONResponse({
                        "success": False, 
                        "message": f"A row with ID '{custom_doc_id}' already exists. Please use a unique ID."
                    }, status_code=400)
        
        # Generate a unique document ID
        import uuid
        import datetime
        
        # Check if CSV source exists
        if source_id not in kb.csv_docs:
            return JSONResponse({"success": False, "message": f"CSV source not found: {source_id}"}, status_code=404)
        
        
        # Create a timestamp-based ID with a UUID suffix for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = custom_doc_id or f"{timestamp}-{str(uuid.uuid4())[:8]}"
        
        # Get the row count to determine the next row index
        rows = kb.get_csv_rows(source_id)
        next_row_index = len(rows) + 1
        
        # Add the new row
        await kb.add_csv_row(source_id, unique_id, new_text, next_row_index, new_metadata)
        
        return JSONResponse({
            "success": True,
            "doc_id": unique_id
        })
    except Exception as e:
        import traceback
        print(f"Error adding CSV row: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.post("/api/kb/create")
async def create_knowledge_base(request: Request):
    """Create a new knowledge base"""
    try:
        data = await request.json()
        name = data.get('name')
        description = data.get('description', '')
        
        if not name:
            return JSONResponse({"success": False, "message": "Name is required"}, status_code=400)
            
        kb_data = await create_kb(name, description)
        return JSONResponse({"success": True, "data": kb_data})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)
@router.delete("/api/kb/{name}")
async def delete_knowledge_base(name: str, request: Request):
    """Delete a knowledge base"""
    try:
        await delete_kb(name)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

# Agent KB settings endpoints

@router.get("/api/kb/agent/{agent_name}/settings")
async def get_agent_kb_settings(agent_name: str, request: Request):
    """Get KB settings for an agent"""
    try:
        # Load agent KB settings from a dedicated file
        settings_dir = "data/kb/agent_settings"
        os.makedirs(settings_dir, exist_ok=True)
        settings_file = f"{settings_dir}/{agent_name}.json"
        
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            return JSONResponse(settings)
        else:
            return JSONResponse({"kb_access": []})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.post("/api/kb/agent/{agent_name}/settings")
async def update_agent_kb_settings(agent_name: str, request: Request):
    """Update KB settings for an agent"""
    try:
        data = await request.json()
        kb_access = data.get('kb_access', [])
        
        # Validate that all KBs exist
        kbs = await list_kbs()
        valid_kbs = set(kbs.keys())
        
        # Filter out any non-existent KBs
        kb_access = [kb for kb in kb_access if kb in valid_kbs]
        
        # Save agent KB settings to a dedicated file
        settings_dir = "data/kb/agent_settings"
        os.makedirs(settings_dir, exist_ok=True)
        settings_file = f"{settings_dir}/{agent_name}.json"
        
        with open(settings_file, 'w') as f:
            json.dump({"kb_access": kb_access}, f)
            
        return JSONResponse({"success": True, "kb_access": kb_access})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def cleanup_task(task_id, delay_seconds):
    """Clean up task after delay"""
    await asyncio.sleep(delay_seconds)
    if task_id in processing_tasks:
        del processing_tasks[task_id]

async def process_document_with_progress(name, file_path, task_id):
    """Process document with progress tracking"""
    try:
        # Debug information
        print(f"Starting document processing for KB: {name}, file: {file_path}, task_id: {task_id}")
        
        # Get KB instance
        kb = await get_kb_instance(name)
        
        # Update task status to indicate processing has started
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["progress"] = 0
        
        # Define progress callback
        def progress_callback(progress):
            processing_tasks[task_id]["progress"] = int(progress * 100)
        
        # Add document to KB with progress tracking
        # Check if file exists
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
            
        print(f"File exists, adding document to KB: {file_path}")
        await kb.add_document(file_path, progress_callback=progress_callback)
        
        # Update task status to indicate completion
        processing_tasks[task_id]["status"] = "complete"
        processing_tasks[task_id]["progress"] = 100
        processing_tasks[task_id]["permanent_path"] = file_path  # Store the permanent path
        
        # Schedule task cleanup after some time
        asyncio.create_task(cleanup_task(task_id, 300))  # Clean up after 5 minutes
        
    except Exception as e:
        # Print detailed error information
        import traceback
        print(f"Error in process_document_with_progress: {str(e)}")
        print(traceback.format_exc())
        
        # Update task status to indicate error
        processing_tasks[task_id]["status"] = "error"
        processing_tasks[task_id]["message"] = str(e)
        
        # Clean up temp file
        if os.path.exists(file_path):
            os.unlink(file_path)

async def process_csv_document(name, file_path, config, task_id):
    """Process CSV document with progress tracking"""
    try:
        # Get KB instance
        print(f"Processing CSV document: {file_path}")
        kb = await get_kb_instance(name)
        
        # Update task status to indicate processing has started
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["progress"] = 0
        
        # Define progress callback
        def progress_callback(progress):
            processing_tasks[task_id]["progress"] = int(progress * 100)
        
        # Get the expected row count from the config if available
        expected_row_count = config.get("expected_row_count")
        
        # Parse the CSV file using our consolidated parser
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            file_content = f.read()
        
        # Log the file content for debugging
        print(f"CSV file content length: {len(file_content)} bytes")
        print(f"First 200 chars: {file_content[:200]}")
        
        dialect, has_header, has_multiple_quotes = detect_csv_format(file_content)
        rows = parse_csv(file_content=file_content, dialect=dialect)
        
        # Check if the row count matches what we expected
        actual_row_count = len(rows)
        print(f"Parsed {actual_row_count} rows from CSV file")
        print(f"Expected row count: {expected_row_count}")
        if expected_row_count is not None and actual_row_count != expected_row_count:
            error_msg = f"Row count mismatch: expected {expected_row_count} rows but parsed {actual_row_count} rows. CSV parsing may be incorrect."
            print(error_msg)
            processing_tasks[task_id]["status"] = "error"
            processing_tasks[task_id]["message"] = error_msg
            return
        
        # Add the parsed rows to the config
        config["preprocessed_rows"] = rows
        config["has_header"] = has_header
        config["has_multiple_quotes"] = has_multiple_quotes
       
        result = await kb.add_csv_document(file_path, config, progress_callback=progress_callback)
        
        # Update task status to indicate completion
        processing_tasks[task_id]["status"] = "complete"
        processing_tasks[task_id]["progress"] = 100
        processing_tasks[task_id]["result"] = result
        
        # Schedule task cleanup after some time
        asyncio.create_task(cleanup_task(task_id, 300))  # Clean up after 5 minutes
        
    except Exception as e:
        # Print detailed error information
        import traceback
        print(f"Error in process_csv_document: {str(e)}")
        print(traceback.format_exc())
        
        # Update task status to indicate error
        processing_tasks[task_id]["status"] = "error"
        processing_tasks[task_id]["message"] = str(e)
        
        # Clean up temp file
        if os.path.exists(file_path):
            os.unlink(file_path)

async def process_csv_sync(name, file_path, task_id):
    """Process CSV sync with progress tracking"""
    try:
        # Get KB instance
        kb = await get_kb_instance(name)
        
        # Update task status to indicate processing has started
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["progress"] = 0
        
        # Define progress callback
        def progress_callback(progress):
            processing_tasks[task_id]["progress"] = int(progress * 100)
        
        # Sync CSV document with KB
        result = await kb.sync_csv_document(file_path, progress_callback=progress_callback)
        
        # Update task status to indicate completion
        processing_tasks[task_id]["status"] = "complete"
        processing_tasks[task_id]["progress"] = 100
        processing_tasks[task_id]["result"] = result
        
        # Clean up temp file
        if os.path.exists(file_path):
            os.unlink(file_path)
        
        # Schedule task cleanup after some time
        asyncio.create_task(cleanup_task(task_id, 300))  # Clean up after 5 minutes
        
    except Exception as e:
        # Print detailed error information
        import traceback
        print(f"Error in process_csv_sync: {str(e)}")
        print(traceback.format_exc())
        
        # Update task status to indicate error
        processing_tasks[task_id]["status"] = "error"
        processing_tasks[task_id]["message"] = str(e)

async def process_url_with_progress(name, url, task_id, always_include_verbatim=True):
    """Process URL with progress tracking"""
    try:
        # Update task status to indicate processing has started
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["progress"] = 0
        
        # Define progress callback
        def progress_callback(progress):
            processing_tasks[task_id]["progress"] = int(progress * 100)
        
        # Add URL to KB with progress tracking
        result = await add_url_to_kb(name, url, always_include_verbatim)
        
        # Update task status to indicate completion
        processing_tasks[task_id]["status"] = "complete"
        processing_tasks[task_id]["progress"] = 100
        
        # Store the URL hash for future reference
        if result and isinstance(result, dict):
            url_hash = None
            for k, v in result.items():
                if k == "url_hash" or (k == "url" and url == v):
                    url_hash = k
                    break
            if url_hash:
                processing_tasks[task_id]["url_hash"] = url_hash
        
        # Schedule task cleanup after some time
        asyncio.create_task(cleanup_task(task_id, 300))  # Clean up after 5 minutes
        
    except Exception as e:
        # Print detailed error information
        import traceback
        print(f"Error in process_url_with_progress: {str(e)}")
        print(traceback.format_exc())
        
        # Update task status to indicate error
        processing_tasks[task_id]["status"] = "error"
        processing_tasks[task_id]["message"] = str(e)

async def refresh_url_with_progress(name, url_or_hash, task_id):
    """Refresh URL with progress tracking"""
    try:
        # Update task status to indicate processing has started
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["progress"] = 0
        
        # Define progress callback
        def progress_callback(progress):
            processing_tasks[task_id]["progress"] = int(progress * 100)
        
        # Refresh URL in KB with progress tracking
        result = await refresh_url_in_kb(name, url_or_hash)
        
        # Update task status to indicate completion
        processing_tasks[task_id]["status"] = "complete"
        processing_tasks[task_id]["progress"] = 100
        
        # Schedule task cleanup after some time
        asyncio.create_task(cleanup_task(task_id, 300))  # Clean up after 5 minutes
        
    except Exception as e:
        # Print detailed error information
        import traceback
        print(f"Error in refresh_url_with_progress: {str(e)}")
        print(traceback.format_exc())
        
        # Update task status to indicate error
        processing_tasks[task_id]["status"] = "error"
        processing_tasks[task_id]["message"] = str(e)

@router.post("/api/kb/{name}/csv/preview")
async def preview_csv_file(name: str, file: UploadFile = File(...), request: Request = None):
    try:
        # Create a temporary file to store the uploaded CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        # Read the CSV file to get a preview
        preview_data = {}
        try:
            # Get CSV preview using the consolidated parser
            preview_data = get_csv_preview(temp_path, max_rows=10)
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
        
        return JSONResponse({
            "success": True, 
            "preview": preview_data
        })
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.post("/api/kb/{name}/csv/upload")
async def upload_csv_document(name: str, 
                            file: UploadFile = File(None),
                            temp_path: str = Form(None),
                            config: str = Form(...),
                            request: Request = None):
    """Upload and process a CSV file with configuration"""
    try:
        # Parse the configuration
        config_dict = json.loads(config)
        
        # Add default column names if not provided in config
        # Get KB instance
        kb = await get_kb_instance(name)
        
        # Handle file upload or use temp path
        if file:
            # Create a temporary file to store the uploaded CSV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                content = await file.read()
                tmp.write(content)
                temp_path = tmp.name
        elif not temp_path or not os.path.exists(temp_path):
            return JSONResponse({
                "success": False, 
                "message": "No file provided and no valid temporary path"
            }, status_code=400)
        
        # Create a permanent path for the CSV file
        uploads_dir = os.path.join(kb.persist_dir, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate a safe filename
        original_filename = file.filename if file else os.path.basename(temp_path)
        suffix = os.path.splitext(original_filename)[1]
        safe_filename = re.sub(r'[^\w\-\.]', '_', original_filename)
        
        # Create a unique filename to avoid overwriting
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{safe_filename}"
        permanent_path = os.path.join(uploads_dir, unique_filename)
        
        # Copy to permanent location
        shutil.copy2(temp_path, permanent_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        # Create a task ID for tracking progress
        task_id = str(uuid.uuid4())
        
        # Store task info
        processing_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "file_name": original_filename,
            "permanent_path": permanent_path,
            "config": config_dict
        }
        
        # Use our consolidated CSV parser to prepare the file for processing
        try:
            # Get CSV preview to extract format information
            preview_data = get_csv_preview(permanent_path)
            
            # Store the expected row count for validation during processing
            config_dict["expected_row_count"] = preview_data["row_count"]
            
            # Add the detected format information to the config
            config_dict["has_header"] = preview_data["has_header"]
            config_dict["has_multiple_quotes"] = preview_data["has_multiple_quotes"]
            
            # If no column_map is provided, use the default column names
            if "column_map" not in config_dict or not config_dict["column_map"]:
                column_names = preview_data["default_column_names"]
                config_dict["column_map"] = {i: name for i, name in enumerate(column_names)}
        
        except Exception as e:
            print(f"Error preparing CSV file: {str(e)}")
        
        # Start background task for processing
        asyncio.create_task(process_csv_document(name, permanent_path, config_dict, task_id))
        
        return JSONResponse({
            "success": True, 
            "task_id": task_id
        })
    except Exception as e:
        # Print detailed error information
        import traceback
        print(f"Error in upload_csv_document: {str(e)}")
        print(traceback.format_exc())
        
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return JSONResponse({"success": False, "message": f"Upload failed: {str(e)}"}, status_code=500)

@router.post("/api/kb/{name}/csv/sync")
async def sync_csv_document(name: str, file: UploadFile = File(...), request: Request = None):
    """Sync a CSV document with the knowledge base"""
    try:
        # Create a temporary file to store the uploaded CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        # Create a task ID for tracking progress
        task_id = str(uuid.uuid4())
        
        # Store task info
        processing_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "file_name": file.filename,
            "temp_path": temp_path
        }
        
        # Start background task for processing
        asyncio.create_task(process_csv_sync(name, temp_path, task_id))
        
        return JSONResponse({
            "success": True, 
            "task_id": task_id
        })
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.get("/api/kb/{name}/csv/sources")
async def get_csv_sources(name: str, request: Request):
    """Get all CSV sources in a knowledge base"""
    try:
        kb = await get_kb_instance(name)
        
        # Check if kb has CSV document support
        if not hasattr(kb, 'csv_docs'):
            return JSONResponse({"success": False, 
                              "message": "This knowledge base was created with an older version and doesn't support CSV documents"}, 
                              status_code=400)
        
        return JSONResponse({
            "success": True, 
            "data": kb.csv_docs
        })
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.get("/api/kb/{name}/csv/{source_id}/rows")
async def get_csv_rows(name: str, source_id: str, request: Request):
    """Get all rows from a CSV source"""
    try:
        print("Getting rows!!")
        kb = await get_kb_instance(name)
        
        # Check if kb has CSV document support
        if not hasattr(kb, 'csv_docs'):
            return JSONResponse({"success": False, 
                              "message": "This knowledge base was created with an older version and doesn't support CSV documents"}, 
                              status_code=400)
        print("loading rows")
        rows = kb.get_csv_rows(source_id)
        return JSONResponse({
            "success": True, 
            "data": rows
        })
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.get("/api/kb/{name}/csv/{source_id}/search")
async def search_csv_rows(name: str, source_id: str, query: str = "", limit: int = 3, request: Request = None):
    """Search for CSV rows that match a query"""
    try:
        kb = await get_kb_instance(name)
        
        # Check if kb has CSV document support
        if not hasattr(kb, 'csv_docs'):
            return JSONResponse({"success": False, 
                              "message": "This knowledge base was created with an older version and doesn't support CSV documents"}, 
                              status_code=400)
        
        # If query is empty, return a limited number of rows
        if not query.strip():
            rows = kb.get_csv_rows(source_id)[:limit]
        else:
            rows = await kb.csv_handler.search_csv_rows(source_id, query, limit)
            
        return JSONResponse({
            "success": True, 
            "data": rows
        })
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)
@router.post("/api/kb/{name}/csv/{source_id}/row/{doc_id}")
async def update_csv_row(name: str, source_id: str, doc_id: str, request: Request):
    """Update a single row in a CSV source"""
    try:
        data = await request.json()
        new_text = data.get("text")
        new_metadata = data.get("metadata")
        
        if not new_text:
            return JSONResponse({"success": False, "message": "Text is required"}, status_code=400)
        
        kb = await get_kb_instance(name)
        
        # Check if kb has CSV document support
        if not hasattr(kb, 'csv_docs'):
            return JSONResponse({"success": False, 
                              "message": "This knowledge base was created with an older version and doesn't support CSV documents"}, 
                              status_code=400)
        
        # Check if CSV source exists
        if source_id not in kb.csv_docs:
            return JSONResponse({"success": False, "message": f"CSV source not found: {source_id}"}, status_code=404)
        
                
        # Check if kb has CSV document support
        if not hasattr(kb, 'csv_docs'):
            return JSONResponse({"success": False, 
                              "message": "This knowledge base was created with an older version and doesn't support CSV documents"}, 
                              status_code=400)
        
        await kb.update_csv_row(source_id, doc_id, new_text, new_metadata)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.delete("/api/kb/{name}/csv/{source_id}/row/{doc_id}")
async def delete_csv_row(name: str, source_id: str, doc_id: str, request: Request):
    """Delete a single row from a CSV source"""
    try:
        kb = await get_kb_instance(name)
        
        # Check if kb has CSV document support
        if not hasattr(kb, 'csv_docs'):
            return JSONResponse({"success": False, 
                              "message": "This knowledge base was created with an older version and doesn't support CSV documents"}, 
                              status_code=400)
        
        await kb.delete_csv_row(source_id, doc_id)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.post("/api/kb/{name}/url")
async def add_url_to_knowledge_base(name: str, request: Request):
    """Add a URL to a knowledge base"""
    try:
        data = await request.json()
        url = data.get('url')
        always_include_verbatim = data.get('verbatim', True)
        
        if not url:
            return JSONResponse({
                "success": False, 
                "message": "URL is required"
            }, status_code=400)
            
        # Create a task ID for tracking progress
        task_id = str(uuid.uuid4())
        
        # Store task info
        processing_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "url": url
        }
        
        # Start background task for processing
        asyncio.create_task(process_url_with_progress(
            name, 
            url, 
            task_id, 
            always_include_verbatim
        ))
        
        return JSONResponse({
            "success": True, 
            "task_id": task_id
        })
    except Exception as e:
        return JSONResponse({
            "success": False, 
            "message": f"Error adding URL: {str(e)}"
        }, status_code=500)

@router.post("/api/kb/{name}/url/refresh")
async def refresh_url_in_knowledge_base(name: str, request: Request):
    """Refresh content for a URL in a knowledge base"""
    try:
        data = await request.json()
        url_or_hash = data.get('url_or_hash')
        
        if not url_or_hash:
            return JSONResponse({
                "success": False, 
                "message": "URL or hash is required"
            }, status_code=400)
            
        # Create a task ID for tracking progress
        task_id = str(uuid.uuid4())
        
        # Store task info
        processing_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "url_or_hash": url_or_hash
        }
        
        # Start background task for refreshing
        asyncio.create_task(refresh_url_with_progress(name, url_or_hash, task_id))
        
        return JSONResponse({
            "success": True, 
            "task_id": task_id
        })
    except Exception as e:
        return JSONResponse({
            "success": False, 
            "message": f"Error refreshing URL: {str(e)}"
        }, status_code=500)

@router.post("/api/kb/{name}/upload")
async def upload_document(name: str, file: UploadFile = File(...), request: Request = None):
    """Handle document upload to specific KB"""
    try:
        # Debug information
        print(f"Starting upload for KB: {name}, file: {file.filename}")
        
        # Get KB instance to get storage directory
        kb = await get_kb_instance(name)
        
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(kb.persist_dir, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        # Verify the directory was created
        if not os.path.exists(uploads_dir):
            raise ValueError(f"Failed to create uploads directory: {uploads_dir}")
        print(f"Uploads directory: {uploads_dir}")
        
        # Generate a safe filename
        original_filename = file.filename
        print(f"Original filename: {original_filename}")
        suffix = os.path.splitext(file.filename)[1]
        safe_filename = re.sub(r'[^\w\-\.]', '_', original_filename)
        
        # Create a unique filename to avoid overwriting
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        print(f"Timestamp: {timestamp}")
        unique_filename = f"{timestamp}_{safe_filename}"
        permanent_path = os.path.join(uploads_dir, unique_filename)
        
        # First save to temp file
        with tempfile.NamedTemporaryFile(delete=False, prefix=file.filename, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        print(f"Temp path: {temp_path}, Permanent path: {permanent_path}")
        
        # Copy to permanent location
        shutil.copy2(temp_path, permanent_path)
        
        # Verify the file was copied successfully
        if not os.path.exists(permanent_path):
            raise ValueError(f"Failed to copy file to permanent location: {permanent_path}")
        print(f"File copied successfully to: {permanent_path}")
        
        # Clean up temp file after copying
        if os.path.exists(temp_path):
            os.unlink(temp_path)

        # Create a task ID for tracking progress
        task_id = str(uuid.uuid4())
        
        # Store task info
        processing_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "file_name": file.filename,
            "permanent_path": permanent_path
        }
        
        # Start background task for processing
        # Use permanent path for processing
        asyncio.create_task(process_document_with_progress(name, permanent_path, task_id))
        
        return JSONResponse({
            "success": True, 
            "task_id": task_id
        })

    except Exception as e:
        # Print detailed error information
        import traceback
        print(f"Error in upload_document: {str(e)}")
        print(traceback.format_exc())
        
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path) and 'permanent_path' not in locals():
            os.unlink(temp_path)
        return JSONResponse({"success": False, "message": f"Upload failed: {str(e)}"}, status_code=500)

@router.get("/api/kb/{name}/task/{task_id}")
async def get_task_status(name: str, task_id: str, request: Request = None):
    """Get status of document processing task"""
    if task_id not in processing_tasks:
        return JSONResponse({"success": False, "message": "Task not found"}, status_code=404)
    
    return JSONResponse({
        "success": True,
        "status": processing_tasks[task_id]["status"],
        "progress": processing_tasks[task_id]["progress"],
        "file_name": processing_tasks[task_id].get("file_name", ""),
        "url": processing_tasks[task_id].get("url", ""),
        "url_hash": processing_tasks[task_id].get("url_hash", ""),
        "permanent_path": processing_tasks[task_id].get("permanent_path", ""),
        "message": processing_tasks[task_id].get("message", "")
    })

@router.get("/api/kb/{name}/documents")
async def get_documents(name: str, request: Request):
    """Get list of documents in specific KB"""
    try:
        kb = await get_kb_instance(name)
        logger.info(f"Getting documents for KB: {name}")
        docs = kb.get_document_info()
        logger.info(f"Documents found: {len(docs)}")
        # Check if kb has verbatim_docs attribute (for backward compatibility)
        has_verbatim_support = hasattr(kb, 'verbatim_docs')
        
        # Add verbatim status to each document
        for doc in docs:
            file_path = doc.get('file_path')
            is_verbatim = False
            
            # Check if document is in verbatim docs (if supported)
            if has_verbatim_support:
                for doc_id, info in kb.verbatim_docs.items():
                    if info.get('file_path') == file_path:
                        is_verbatim = True
                        break
                    
            doc['is_verbatim'] = is_verbatim
            
        return JSONResponse({"success": True, "data": docs})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.post("/api/kb/{name}/documents/toggle_verbatim")
async def toggle_document_verbatim(name: str, request: Request):
    """Toggle verbatim status for a document"""
    try:
        data = await request.json()
        file_path = data.get("file_path")
        verbatim = data.get("verbatim", False)
        force_verbatim = data.get("force_verbatim", False)
        
        debug_box("----------------------------------------------")
        print({"verbatim": verbatim})
        print({"force_verbatim": force_verbatim})

        if not file_path:
            return JSONResponse({"success": False, "message": "File path is required"}, status_code=400)
            
        kb = await get_kb_instance(name)
        
        # Check if kb has verbatim support
        if not hasattr(kb, 'verbatim_docs'):
            return JSONResponse({"success": False, 
                               "message": "This knowledge base was created with an older version and doesn't support verbatim documents"}, 
                               status_code=400)
        
        # Check if file exists at the given path
        if not os.path.exists(file_path):
            # Try to find the file in the uploads directory
            kb_instance = kb
            if isinstance(kb_instance, HierarchicalKnowledgeBase):
                uploads_dir = os.path.join(kb_instance.persist_dir, "uploads")
                filename = os.path.basename(file_path)
                # Look for files with the same name in the uploads directory
                for f in os.listdir(uploads_dir):
                    if f.endswith(filename):
                        file_path = os.path.join(uploads_dir, f)
                        break
        
        # If turning off verbatim, remove from verbatim docs
        if not verbatim:
            print(f"Removing verbatim document: {file_path}")
            await kb.remove_verbatim_document(file_path)
            return JSONResponse({"success": True, "verbatim": False})
        
        # If turning on verbatim, add as verbatim document
        print(f"Adding verbatim document: {file_path}")
        await kb.add_document(
            file_path=file_path,
            always_include_verbatim=True,
            force_verbatim=force_verbatim
        )
        print("Done")
        return JSONResponse({"success": True, "verbatim": True})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.delete("/api/kb/{name}/documents")
async def delete_document(name: str, request: Request):
    """Delete document from specific KB"""
    try:
        # Get file_path from query parameters
        file_path = request.query_params.get("path")
        if not file_path:
            return JSONResponse({"success": False, "message": "File path is required"}, status_code=400)
            
        kb = await get_kb_instance(name)
        
        # Check if file exists at the given path
        if not os.path.exists(file_path):
            # Try to find the file in the uploads directory
            kb_instance = kb
            if isinstance(kb_instance, HierarchicalKnowledgeBase):
                uploads_dir = os.path.join(kb_instance.persist_dir, "uploads")
                filename = os.path.basename(file_path)
                # Look for files with the same name in the uploads directory
                for f in os.listdir(uploads_dir):
                    if f.endswith(filename):
                        permanent_path = os.path.join(uploads_dir, f)
                        # Delete the permanent file
                        try:
                            if os.path.exists(permanent_path):
                                os.remove(permanent_path)
                                print(f"Deleted permanent file: {permanent_path}")
                        except Exception as e:
                            print(f"Warning: Error deleting permanent file: {str(e)}")
                        break
        
        # First, check if document is in verbatim docs and remove it if it is (if supported)
        if hasattr(kb, 'verbatim_docs'):
            try:
                await kb.remove_verbatim_document(file_path)
            except Exception as e:
                print(f"Warning: Error removing verbatim document: {str(e)}")
        else:
            print(f"Knowledge base {name} doesn't support verbatim documents")
            
        await kb.remove_document(file_path)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.delete("/api/kb/{name}/url")
async def delete_url_document(name: str, request: Request):
    """Delete URL document from specific KB"""
    try:
        data = await request.json()
        url_or_hash = data.get('url_or_hash')
        
        if not url_or_hash:
            return JSONResponse({"success": False, "message": "URL or hash is required"}, status_code=400)
            
        kb = await get_kb_instance(name)
        
        # Check if kb has URL document support
        if not hasattr(kb, 'url_docs'):
            return JSONResponse({"success": False, 
                              "message": "This knowledge base was created with an older version and doesn't support URL documents"}, 
                              status_code=400)
            
        # Remove URL document
        await kb.remove_url_document(url_or_hash)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


csv_doc_cache = {}

@router.get("/api/kb/{name}/csv/documents/match")
async def match_csv_metadata(name: str, field: str, val: str, limit: int = 10, request: Request = None):
    """Match CSV metadata field to a value and return full document"""
    try:
        kb = await get_kb_instance(name)
        
        if not hasattr(kb, 'csv_docs'):
            return JSONResponse({"success": False, 
                              "message": "This knowledge base was created with an older version and doesn't support CSV documents"}, 
                              status_code=400)
       
        matches = []
        print("searcing in router")
        if field in csv_doc_cache:
            print(f"found field {field} in doc cache")
            if val in csv_doc_cache[field]:
                print(f"found val {val} in doc cache")
                matches = csv_doc_cache[field][val]
            else:
                print(f"did not find value in doc cache, searching kb")
                matches = await kb.match_csv_metadata(field, val, limit)
                print(f"match results for val {matches}")
                if matches is not None and len(matches) > 0:
                    csv_doc_cache[field][val] = matches
        else:
            print(f"did not find field {field} in csv_doc_cache, call kb method")
            matches = await kb.match_csv_metadata(field, val, limit)
            print(f"matches {matches}")
            if not field in csv_doc_cache:
                csv_doc_cache[field] = {}
            if matches is not None and len(matches) > 0:
                csv_doc_cache[field][val] = matches
         
        return JSONResponse({
            "success": True, 
            "matches": matches[:limit]
        })
    except Exception as e:
        traceback.print_exc()
        print(e)

        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

