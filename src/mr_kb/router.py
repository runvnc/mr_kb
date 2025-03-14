from fastapi import APIRouter, Request, UploadFile, File
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
from lib.utils.debug import debug_box

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

async def cleanup_task(task_id, delay_seconds):
    """Clean up task after delay"""
    await asyncio.sleep(delay_seconds)
    if task_id in processing_tasks:
        del processing_tasks[task_id]

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
        docs = kb.get_document_info()
        
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
