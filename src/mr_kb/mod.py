from lib.providers.services import service
from lib.providers.commands import command
from lib.pipelines.pipe import pipe
from .vector_only_kb import HierarchicalKnowledgeBase
import os
import json
import datetime

# Global KB instances cache
_kb_instances = {}

# KB metadata file
KB_METADATA_FILE = "data/kb/metadata.json"

def load_kb_metadata():
    """Load metadata about all KBs"""
    if os.path.exists(KB_METADATA_FILE):
        with open(KB_METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_kb_metadata(metadata):
    """Save KB metadata"""
    os.makedirs(os.path.dirname(KB_METADATA_FILE), exist_ok=True)
    with open(KB_METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

@service()
async def create_kb(name: str, description: str = ""):
    """Create a new knowledge base"""
    metadata = load_kb_metadata()
    if name in metadata:
        raise ValueError(f"Knowledge base '{name}' already exists")

    storage_dir = f"data/kb/bases/{name}"
    os.makedirs(storage_dir, exist_ok=True)

    kb = HierarchicalKnowledgeBase(storage_dir)
    await kb.create_index(storage_dir)
    _kb_instances[name] = kb

    metadata[name] = {
        "name": name,
        "description": description,
        "created_at": str(datetime.datetime.now()),
        "storage_dir": storage_dir
    }
    save_kb_metadata(metadata)
    return metadata[name]

@service()
async def get_kb_instance(name: str):
    """Get a specific KB instance"""
    if name in _kb_instances:
        return _kb_instances[name]

    metadata = load_kb_metadata()
    if name not in metadata:
        raise ValueError(f"Knowledge base '{name}' not found")

    storage_dir = metadata[name]['storage_dir']
    kb = HierarchicalKnowledgeBase(storage_dir)
    if not kb.load_if_exists():
        await kb.create_index(storage_dir)

    _kb_instances[name] = kb
    return kb

@service()
async def list_kbs():
    """List all available knowledge bases"""
    return load_kb_metadata()

@service()
async def delete_kb(name: str):
    """Delete a knowledge base"""
    metadata = load_kb_metadata()
    if name not in metadata:
        raise ValueError(f"Knowledge base '{name}' not found")

    storage_dir = metadata[name]['storage_dir']
    if os.path.exists(storage_dir):
        import shutil
        shutil.rmtree(storage_dir)

    if name in _kb_instances:
        del _kb_instances[name]

    del metadata[name]
    save_kb_metadata(metadata)

@service()
async def add_to_kb(name: str, file_path: str):
    """Add a document to specified KB"""
    kb = await get_kb_instance(name)
    await kb.add_document(file_path)

@command()
async def query_kb(name: str, query: str, context=None):
    """Query a specific knowledge base
    
    Example:
    { "query_kb": { "name": "general", "query": "What is the capital of France?" } }
    """
    kb = await get_kb_instance(name)
    results = await kb.get_relevant_context(
        query,
        similarity_top_k=5,
        format_type="detailed"
    )
    return results

@command()
async def add_to_kb_cmd(name: str, file_path: str, context=None):
    """Add a document to a specific knowledge base
    
    Example:
    { "add_to_kb": { "name": "general", "file_path": "/path/to/document.pdf" } }
    """
    kb = await get_kb_instance(name)
    await kb.add_document(file_path)
    return f"Added {file_path} to knowledge base '{name}'"

@pipe(name='pre_process_msg', priority=10)
async def enrich_with_kb(data: dict, context=None) -> dict:
    """Add relevant knowledge base context to messages"""
    if not data.get('message'):
        return data
        
    # Get message text
    message = data['message']
    if isinstance(message, list):
        # Handle multipart messages
        text_parts = [p['text'] for p in message if p['type'] == 'text']
        query_text = ' '.join(text_parts)
    else:
        query_text = message

    try:
        # For now, query all KBs and combine results
        metadata = load_kb_metadata()
        all_results = []
        
        for kb_name in metadata.keys():
            try:
                kb = await get_kb_instance(kb_name)
                context_data = await kb.get_relevant_context(
                    query_text,
                    similarity_top_k=3,  # Fewer results per KB since we're querying multiple
                    format_type="markdown"
                )
                if context_data:
                    all_results.append(f"From KB '{kb_name}':\n{context_data}")
            except Exception as e:
                print(f"Error querying KB '{kb_name}': {e}")
                continue

        if all_results:
            combined_results = "\n\n".join(all_results)
            if isinstance(message, list):
                data['message'].append({
                    "type": "text",
                    "text": combined_results
                })
            else:
                data['message'] = f"{message}\n\n{combined_results}"
    except Exception as e:
        print(f"Error enriching message with KB data: {e}")
        # Continue without KB enrichment on error
        pass
            
    return data
