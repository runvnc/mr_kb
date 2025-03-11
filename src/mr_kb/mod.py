from lib.providers.services import service
from lib.providers.commands import command
from lib.pipelines.pipe import pipe
from .kb import HierarchicalKnowledgeBase
import os
import json
import traceback
import datetime
from .chat_utils import KB_START_DELIMITER, KB_END_DELIMITER, clean_chat_messages
from lib.utils.debug import debug_box

import hashlib
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
    debug_box("create_kb")
    if name in metadata:
        raise ValueError(f"Knowledge base '{name}' already exists")
    
    storage_dir = f"data/kb/bases/{name}"
    os.makedirs(storage_dir, exist_ok=True)

    with open(f"{storage_dir}/init.txt", "w") as f:
        f.write("!!")

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

@service()
async def add_url_to_kb(name: str, url: str, always_include_verbatim: bool = True):
    """Add content from a URL to a knowledge base
    
    Args:
        name: Name of the knowledge base
        url: URL to fetch and add
        always_include_verbatim: Whether to include the URL content verbatim
        
    Returns:
        Dict with URL document information
    """
    kb = await get_kb_instance(name)
    return await kb.add_url_document(url, always_include_verbatim=always_include_verbatim)

@service()
async def refresh_url_in_kb(name: str, url_or_hash: str):
    """Refresh content for a URL in a knowledge base
    
    Args:
        name: Name of the knowledge base
        url_or_hash: URL or hash of the URL to refresh
        
    Returns:
        Dict with updated URL document information
    """
    kb = await get_kb_instance(name)
    
    # Determine if input is a URL or hash
    if url_or_hash.startswith(('http://', 'https://')):
        # Create hash from URL
        url_hash = hashlib.md5(url_or_hash.encode()).hexdigest()
    else:
        # Assume input is already a hash
        url_hash = url_or_hash
        
    return await kb.refresh_url_document(url_hash)

@command()
async def query_kb(kb_name: str, match_text: str, context=None):
    """Query a specific knowledge base
    
    Params:

        kb - String.            The name of the knowledgebase to search

        match_text - String.    A snippet of text to try to match in the KB.
                                This will be converted to an embedding and 
                                the vector index will be searched for similar meaning
                                blocks of text.

    Note: The match_text should not necessarily just be a simple phrase or question.
          If possible, try to construct a snippet that might have a similar semantic embedding
          to the real information from the KB you are looking for.

    Example:

    { "query_kb": { "kb_name": "general", "match_text": "The capital of France is ..." } }
    """
    kb = await get_kb_instance(kb_name)
    results = await kb.get_relevant_context(
        match_text,
        similarity_top_k=15,
        final_top_k=15
    )
    str_results = f"From KB:\n\n{results}"
    return str_results


@command()
async def add_to_kb_cmd(name: str, file_path: str, context=None):
    """Add a document to a specific knowledge base
    
    Example:
    { "add_to_kb": { "name": "general", "file_path": "/path/to/document.pdf" } }
    """
    kb = await get_kb_instance(name)
    await kb.add_document(file_path)
    return f"Added {file_path} to knowledge base '{name}'"

@command()
async def add_url_to_kb_cmd(name: str, url: str, context=None):
    """Add content from a URL to a knowledge base
    
    Args:
        name: The name of the knowledge base to add to
        url: The URL to fetch and extract content from
        
    Returns:
        str: Confirmation message with URL added
        
    Example:
        { "add_url_to_kb_cmd": { "name": "general", "url": "https://www.example.com/article" } }
    """
    try:
        result = await add_url_to_kb(name, url)
        return f"Added content from {url} to knowledge base '{name}'"
    except Exception as e:
        return f"Error adding URL: {str(e)}"

@pipe(name='pre_process_msg', priority=10)
async def enrich_with_kb(data: dict, context=None) -> dict:
    """Add relevant knowledge base context to messages"""
    debug_box("Top of enrich_with_kb")
    # get the name of the agent from the context
    # find the kbs that the agent is set to use
    # only query those kbs
    try:
        agent_name = context.agent_name
        if not agent_name:
            return data
    except Exception as e:
        print(f"Error accessing agent_name from context: {e}")
        return data
    
    # Load agent KB settings
    settings_dir = "data/kb/agent_settings"
    settings_file = f"{settings_dir}/{agent_name}.json"
    allowed_kbs = []
    
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                allowed_kbs = settings.get('kb_access', [])
        except Exception as e:
            print(f"Error loading KB settings for agent {agent_name}: {e}")
    
    # If no KBs are allowed, return data unchanged
    if not allowed_kbs:
        return data
    debug_box(f"Allowed KBs: {allowed_kbs}")

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
        
        # Only query KBs that the agent is allowed to access
        kb_names_to_query = [kb_name for kb_name in allowed_kbs if kb_name in metadata]
        
        if not kb_names_to_query:
            return data
        
        for kb_name in kb_names_to_query:
            try:
                kb = await get_kb_instance(kb_name)
                print(f"Querying KB '{kb_name}' for agent '{agent_name}'")
                context_data = await kb.get_relevant_context(
                    query_text, 
                    similarity_top_k=19,
                    final_top_k=15,
                    include_verbatim=False  # Don't include verbatim docs in regular results
                )
                # get_relevant_context returns a tuple or just a string - handle both cases
                if isinstance(context_data, tuple):
                    context_data = context_data[0]  # Extract the context string from the tuple
                elif not isinstance(context_data, str):
                    context_data = str(context_data)
                if context_data:
                    all_results.append(f"From KB '{kb_name}':\n{context_data}")
            except Exception as e:
                print(f"Error querying KB '{kb_name}': {e}")
                continue

        # Handle regular search results - keep in user message as before
        if all_results:
            combined_results = "\n\n".join(all_results)
            # Add KB delimiters
            delimited_results = f"{KB_START_DELIMITER}\n{combined_results}\n{KB_END_DELIMITER}"
            
            # Add to user message (original behavior)
            if isinstance(message, list):
                data['message'].append({
                    "type": "text",
                    "text": delimited_results
                })
            else:
                data['message'] = f"{message}\n\n{delimited_results}"
    except Exception as e:
        print(f"Error enriching message with KB data: {e}")
        # Continue without KB enrichment on error
        pass
            
    return data

@pipe(name='filter_messages', priority=10)
async def filter_kb_messages(data: dict, context=None) -> dict:
    """Filter KB content from messages, preserving only the two most recent non-assistant messages with KB content."""
    try:
        debug_box("Top of filter_kb_messages")
        
        # Clean up KB content from previous messages if they exist
        if 'messages' in data and isinstance(data['messages'], list):
            # First, add verbatim documents to the system message (first message)
            try:
                # Get the agent name from the context
                agent_name = context.agent_name
                if not agent_name:
                    return data
                
                # Load agent KB settings
                settings_dir = "data/kb/agent_settings"
                settings_file = f"{settings_dir}/{agent_name}.json"
                allowed_kbs = []
                
                if os.path.exists(settings_file):
                    with open(settings_file, 'r') as f:
                        settings = json.load(f)
                        allowed_kbs = settings.get('kb_access', [])
                
                # Only proceed if there are allowed KBs
                if allowed_kbs:
                    # Get message text for the last user message
                    user_message = None
                    for msg in reversed(data['messages']):
                        if msg['role'] == 'user':
                            user_message = msg
                            break
                    
                    if user_message:
                        # Extract text from the user message
                        if isinstance(user_message.get('content'), str):
                            query_text = user_message['content']
                        elif isinstance(user_message.get('content'), list):
                            text_parts = [p['text'] for p in user_message['content'] if p.get('type') == 'text']
                            query_text = ' '.join(text_parts)
                        else:
                            query_text = ""
                        
                        if query_text:
                            # Query each KB for verbatim documents
                            metadata = load_kb_metadata()
                            verbatim_results = []
                            
                            kb_names_to_query = [kb_name for kb_name in allowed_kbs if kb_name in metadata]
                            
                            for kb_name in kb_names_to_query:
                                try:
                                    kb = await get_kb_instance(kb_name)
                                    # Get verbatim documents separately
                                    verbatim_docs = kb._get_verbatim_documents()
                                    if verbatim_docs:
                                        # Format verbatim documents
                                        verbatim_text = "### [Essential Knowledge Base Documents]\n\n"
                                        verbatim_text += "=" * 80 + "\n\n"  # Distinctive separator
                                        
                                        for text, metadata, _, chunk_size in verbatim_docs:
                                            # Format metadata header
                                            verbatim_text += f"[ESSENTIAL: {metadata.get('file_name', 'Document')} | "
                                            verbatim_text += f"Path: {metadata.get('file_path', 'Unknown')}]\n"
                                            verbatim_text += f"{text}\n"
                                            verbatim_text += "=" * 80 + "\n\n"  # Distinctive separator
                                        
                                        verbatim_results.append(f"From KB '{kb_name}' (Verbatim):\n{verbatim_text}")
                                except Exception as e:
                                    print(f"Error getting verbatim docs from KB '{kb_name}': {e}")
                                    continue
                            
                            # If we have verbatim results, add them to the system message (first message)
                            if verbatim_results and len(data['messages']) > 0:
                                combined_verbatim = "\n\n".join(verbatim_results)
                                # Add KB delimiters
                                # Use different delimiters to ensure this content is never removed
                                # We don't want the verbatim docs to be affected by the clean_chat_messages function
                                delimited_verbatim = f"<!-- VERBATIM_DOCS_START -->\n{combined_verbatim}\n<!-- VERBATIM_DOCS_END -->"
                                
                                # Add to the system message (first message)
                                if data['messages'][0]['role'] == 'system':
                                    if isinstance(data['messages'][0]['content'], str):
                                        data['messages'][0]['content'] += f"\n\n{delimited_verbatim}"
                                    # Could handle other content types if needed here
            except Exception as e:
                trace = traceback.format_exc()
                print(f"Error adding verbatim docs to system message: {str(e)}\n {trace}")
            
            # Clean messages to retain KB content only in the two most recent non-assistant messages
            data['messages'] = clean_chat_messages(data['messages'])
        else:
            debug_box('No messages in data')
    except Exception as e:
        trace = traceback.format_exc()
        print(f"Error filtering KB content from messages: {str(e)}\n {trace}")
        # Continue without filtering on error
        pass
            
    return data
