from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from lib.templates import render
from .mod import get_kb_instance, create_kb, list_kbs, delete_kb, add_to_kb
import os
import tempfile
import uuid
import asyncio
import json

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
        # Get KB instance
        kb = await get_kb_instance(name)
        
        # Update task status to indicate processing has started
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["progress"] = 0
        
        # Define progress callback
        def progress_callback(progress):
            processing_tasks[task_id]["progress"] = int(progress * 100)
        
        # Add document to KB with progress tracking
        await kb.add_document(file_path, progress_callback=progress_callback)
        
        # Update task status to indicate completion
        processing_tasks[task_id]["status"] = "complete"
        processing_tasks[task_id]["progress"] = 100
        
        # Clean up temp file
        if os.path.exists(file_path):
            os.unlink(file_path)
            
        # Schedule task cleanup after some time
        asyncio.create_task(cleanup_task(task_id, 300))  # Clean up after 5 minutes
        
    except Exception as e:
        # Update task status to indicate error
        processing_tasks[task_id]["status"] = "error"
        processing_tasks[task_id]["message"] = str(e)
        
        # Clean up temp file
        if os.path.exists(file_path):
            os.unlink(file_path)

async def cleanup_task(task_id, delay_seconds):
    """Clean up task after delay"""
    await asyncio.sleep(delay_seconds)
    if task_id in processing_tasks:
        del processing_tasks[task_id]

@router.post("/api/kb/{name}/upload")
async def upload_document(name: str, file: UploadFile = File(...), request: Request = None):
    """Handle document upload to specific KB"""
    try:
        # Create temp file with original extension
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, prefix=file.filename, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        # Create a task ID for tracking progress
        task_id = str(uuid.uuid4())
        
        # Store task info
        processing_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "file_name": file.filename
        }
        
        # Start background task for processing
        asyncio.create_task(process_document_with_progress(name, temp_path, task_id))
        
        return JSONResponse({
            "success": True, 
            "task_id": task_id
        })

    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.get("/api/kb/{name}/task/{task_id}")
async def get_task_status(name: str, task_id: str, request: Request = None):
    """Get status of document processing task"""
    if task_id not in processing_tasks:
        return JSONResponse({"success": False, "message": "Task not found"}, status_code=404)
    
    return JSONResponse({
        "success": True,
        "status": processing_tasks[task_id]["status"],
        "progress": processing_tasks[task_id]["progress"],
        "file_name": processing_tasks[task_id]["file_name"],
        "message": processing_tasks[task_id].get("message", "")
    })

@router.get("/api/kb/{name}/documents")
async def list_documents(name: str, request: Request):
    """Get list of documents in specific KB"""
    try:
        kb = await get_kb_instance(name)
        docs = kb.get_document_info()
        return JSONResponse({"success": True, "data": docs})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.delete("/api/kb/{name}/documents/{file_path}")
async def delete_document(name: str, file_path: str, request: Request):
    """Delete document from specific KB"""
    try:
        kb = await get_kb_instance(name)
        await kb.remove_document(file_path)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)
