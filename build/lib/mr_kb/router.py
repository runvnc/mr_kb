from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from lib.templates import render
from .mod import get_kb_instance, create_kb, list_kbs, delete_kb, add_to_kb
import os
import tempfile

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

        try:
            # Add to knowledge base
            await add_to_kb(name, temp_path)
            return JSONResponse({"success": True})
        finally:
            # Clean up temp file
            os.unlink(temp_path)

    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.get("/api/kb/{name}/documents")
async def list_documents(name: str, request: Request):
    """Get list of documents in specific KB"""
    try:
        kb = await get_kb_instance(name)
        docs = kb.get_document_info()
        return JSONResponse({"success": True, "data": docs})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)

@router.delete("/api/kb/{name}/documents/{doc_id}")
async def delete_document(name: str, doc_id: str, request: Request):
    """Delete document from specific KB"""
    try:
        kb = await get_kb_instance(name)
        await kb.remove_document(doc_id)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)
