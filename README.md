# MindRoot Knowledge Base Plugin

A hierarchical knowledge base plugin for MindRoot that provides vector-based document search and context enrichment for chat interactions.

## Features

- Hierarchical document chunking for better context matching
- Support for multiple file formats (PDF, Word, Excel, Text, Markdown)
- Vector-based semantic search using LlamaIndex
- Automatic chat context enrichment
- Admin interface for document management
- Per-user knowledge bases

## Installation

```bash
pip install -e .
```

## Usage

### As a Chat Enhancement

Once installed, the plugin automatically enhances chat interactions by:
1. Processing user messages
2. Finding relevant context from stored documents
3. Enriching messages with this context

### Admin Interface

Access the knowledge base management interface at `/admin/kb` to:
- Upload new documents
- View stored documents
- Delete documents
- Monitor knowledge base status

### Agent Commands

Agents can interact with the knowledge base using these commands:

```json
// Query the knowledge base
{ "query_kb": { "query": "What is the capital of France?" } }

// Add a document
{ "add_to_kb": { "file_path": "/path/to/document.pdf" } }
```

## Supported File Types

- PDF files (.pdf)
- Word documents (.docx)
- Excel spreadsheets (.xlsx, .xls)
- Text files (.txt)
- Markdown files (.md)

## Configuration

The plugin stores knowledge bases in `/data/kb/{username}/` with each user getting their own isolated knowledge base instance.

## Development

To modify the plugin:

1. Make your changes
2. Run `pip install -e .` to install in development mode
3. Restart MindRoot to load changes

## License

MIT
