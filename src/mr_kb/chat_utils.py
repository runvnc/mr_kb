from typing import List, Dict, Union, Any
import re
import uuid
from lib.utils.debug import debug_box


# Generate unique delimiters that won't appear in normal text
_MARKER = uuid.uuid4().hex[:8]
KB_START_DELIMITER = f"<!--KB_START_{_MARKER}-->"
KB_END_DELIMITER = f"<!--KB_END_{_MARKER}-->"
KB_REPLACEMENT = "[Knowledge base content omitted for brevity]"

def add_kb_delimiters(text: str) -> str:
    """Add end delimiter to knowledge base content if not present.
    Use this when formatting KB content for chat messages."""
    if not text.strip():
        return text
        
    if text.startswith(KB_START_DELIMITER) and not text.endswith(KB_END_DELIMITER):
        return text + KB_END_DELIMITER
    return text

def remove_kb_content(text: str) -> str:
    """Remove knowledge base content from a string, replacing with a brief note."""
    if not text or KB_START_DELIMITER not in text:
        return text
        
    pattern = f"{KB_START_DELIMITER}.*?{KB_END_DELIMITER}"
    return re.sub(pattern, KB_REPLACEMENT, text, flags=re.DOTALL).strip()

def clean_chat_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean a list of chat messages, removing KB content from all but the two most recent non-assistant messages.
    
    Handles both string content and lists of content objects.
    Only processes content objects of type 'text'.
    
    Args:
        messages: List of message dictionaries in OpenAI chat format
                 Each message should have 'role' and 'content' keys
                 Content can be string or list of objects with 'type' and 'text'
    
    
    Returns:
        Cleaned copy of messages list with KB content preserved in the two most recent non-assistant messages
    """
    debug_box("top of clean chat messages ------------------------------------------------------------------")
    cleaned_messages = []
    
    # Find the indices of the two most recent non-assistant messages (if any)
    preserve_kb_indices = []
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]['role'] != 'assistant':
            preserve_kb_indices.append(i)
            if len(preserve_kb_indices) >= 2:
                break
    
    print(f"len of messages: {len(messages)}, preserve_kb_indices: {preserve_kb_indices}, len(preserve_kb_indices): {len(preserve_kb_indices)}")

    # Convert to a set for faster lookups
    preserve_kb_indices = set(preserve_kb_indices)
    
    # Process each message with its index
    for i, message in enumerate(messages):
        cleaned_message = message.copy()
        
        # Only clean non-assistant messages that aren't the most recent one
        if message['role'] != 'assistant' and (
            i not in preserve_kb_indices
        ):
            content = message.get('content')
            # Handle string content
            if isinstance(content, str):
                cleaned_message['content'] = remove_kb_content(content)
                
            # Handle list of content objects
            elif isinstance(content, list):
                cleaned_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            # Clean text content
                            cleaned_item = item.copy()
                            cleaned_item['text'] = remove_kb_content(item['text'])
                            cleaned_content.append(cleaned_item)
                        else:
                            # Keep non-text content as-is
                            cleaned_content.append(item)
                cleaned_message['content'] = cleaned_content
                
        cleaned_messages.append(cleaned_message)
    print(f"cleaned_messages completed.")
    return cleaned_messages


if __name__ == "__main__":
    # Example usage
    
    # Test string content
    test_text = f"""Here's a question about X.

{KB_START_DELIMITER}
Relevant chunk 1...
Relevant chunk 2...
{KB_END_DELIMITER}

Continuing with the question..."""
    
    print("Original text:")
    print(test_text)
    print("\nCleaned text:")
    print(remove_kb_content(test_text))
    
    # Test chat messages
    test_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": test_text},
                {"type": "image", "image_url": "http://example.com/image.jpg"}
            ]
        },
        {
            "role": "assistant",
            "content": f"Response including {KB_START_DELIMITER}\nsome KB content\n{KB_END_DELIMITER}"
        },
        {
            "role": "user",
            "content": test_text
        }
    ]
    
    print("\nCleaned messages:")
    import json
    print(json.dumps(clean_chat_messages(test_messages), indent=2))
