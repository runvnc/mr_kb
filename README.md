# Hierarchical Knowledge Base

A flexible knowledge base that supports hierarchical chunking and multiple file types. Built on LlamaIndex with async support.

## Features

- Hierarchical document chunking
- Multiple file type support
- Async operations
- Progress tracking
- Atomic updates with rollback
- Flexible embedding model selection

## Installation

Basic installation:
```bash
pip install .
```

With all file type support:
```bash
pip install .[all]
```

With specific file type support:
```bash
pip install .[excel,pdf]  # For Excel and PDF support
pip install .[word]      # For Word document support
```

## Supported File Types

- Text files (.txt) - Always supported
- Markdown (.md) - Always supported
- PDF files (.pdf) - Requires PyPDF2
- Excel files (.xlsx, .xls) - Requires pandas and openpyxl
- Word documents (.docx) - Requires python-docx

## Usage

```python
import asyncio
from kb import HierarchicalKnowledgeBase

async def main():
# Initialize KB
kb = HierarchicalKnowledgeBase("./storage")

# Check supported file types
print(kb.supported_types)

# Create index from directory
await kb.create_index("./data")

# Add individual document
await kb.add_document("new_document.pdf")

# Get relevant context without LLM synthesis
context = await kb.get_relevant_context(
"your question here",
similarity_top_k=15,
format_type="markdown"  # or "plain" or "detailed"
)

print(context)

if __name__ == "__main__":
asyncio.run(main())
```

## Embedding Models

By default, uses OpenAI's embeddings. Can be configured to use other models:

```python
# Use OpenAI (default)
kb = HierarchicalKnowledgeBase("./storage")

# Use specific HuggingFace model
kb = HierarchicalKnowledgeBase(
"./storage",
embedding_model="BAAI/bge-small-en-v1.5"
)
```

## Document Processing

Documents are processed in a hierarchical manner:
1. Split into large chunks (2048 chars)
2. Then medium chunks (512 chars)
3. Then small chunks (128 chars)

This allows for matching at different granularity levels.

## Progress Tracking

All operations support progress callbacks:

```python
def progress_callback(progress: float):
print(f"Processing: {progress * 100:.2f}%")

await kb.create_index("./data", progress_callback=progress_callback)
```

## Error Handling

All operations use atomic updates with automatic rollback on failure:

```python
try:
await kb.add_document("doc.pdf")
except DocumentProcessingError as e:
print(f"Failed to add document: {e}")
# Original index state is preserved
```

## Requirements

Core requirements:
- llama-index
- python-dotenv
- openai
- nltk
- python-magic
- tiktoken

Optional requirements (for file type support):
- pandas + openpyxl (Excel files)
- PyPDF2 (PDF files)
- python-docx (Word documents)

## License

MIT