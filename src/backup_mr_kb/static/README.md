# MindRoot Knowledge Base Plugin - File Upload UI

## Overview

This directory contains the frontend components for the MindRoot Knowledge Base plugin, including an enhanced file upload UI with progress tracking.

## Components

### 1. File Uploader Component

The `file-uploader.js` component provides a modern, user-friendly interface for uploading files to knowledge bases with real-time progress tracking. It includes:

- Drag and drop file upload support
- Multiple file upload handling with queue management
- Real-time upload progress indicators
- Processing status tracking
- Error handling with retry options
- Cancellation support for ongoing uploads

### 2. Knowledge Base Manager

The `kb-manager.js` component is the main interface for managing knowledge bases. It has been updated to integrate the file uploader component.

## Implementation Details

### Frontend

1. **Component Architecture**
   - Separated file upload functionality into its own web component
   - Used event-based communication between components
   - Implemented CSS in separate files for better maintainability

2. **Upload Progress Tracking**
   - Replaced Fetch API with XMLHttpRequest for progress event support
   - Implemented queue-based upload system for multiple files
   - Added visual progress bars for individual files and overall progress

3. **Processing Status**
   - Added polling mechanism to track document processing after upload
   - Integrated with backend task tracking system

### Backend

1. **Task Tracking System**
   - Added task management for long-running document processing
   - Implemented progress callback support in document processing
   - Created API endpoints for checking task status

2. **Progress Reporting**
   - Enhanced vector_only_kb.py to report progress during document processing
   - Added proper cleanup of temporary files and task data

## Usage

The file uploader component can be used independently or as part of the KB manager:

```html
<file-uploader 
  kb-name="my-knowledge-base"
  @upload-started="handleUploadStarted"
  @upload-complete="handleUploadComplete"
  @upload-error="handleUploadError">
</file-uploader>
```

## Events

The file uploader component emits the following events:

- `upload-started`: When files are added to the upload queue
- `upload-complete`: When a file has been successfully uploaded and processed
- `upload-error`: When an error occurs during upload or processing

## Styling

The component uses CSS variables for theming and can be customized by overriding the styles in `file-uploader.css`.
