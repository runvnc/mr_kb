import { LitElement, html } from '/admin/static/js/lit-core.min.js';
import { BaseEl } from '/admin/static/js/base.js';

class FileUploader extends BaseEl {
  static properties = {
    kbName: { type: String },
    uploadQueue: { type: Array },
    overallProgress: { type: Number }
  };

  constructor() {
    super();
    this.kbName = '';
    this.uploadQueue = [];
    this.overallProgress = 0;
  }

  handleUpload(e) {
    if (!this.kbName) {
      this.dispatchEvent(new CustomEvent('upload-error', {
        detail: { message: 'Please select a knowledge base first' },
        bubbles: true
      }));
      return;
    }

    const files = e.target.files || e.dataTransfer.files;
    if (!files.length) return;

    // Add files to upload queue with initial state
    const newUploads = Array.from(files).map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file: file,
      name: file.name,
      size: file.size,
      type: file.type,
      progress: 0,
      status: 'queued', // queued, uploading, processing, complete, error
      error: null,
      xhr: null
    }));
    
    this.uploadQueue = [...this.uploadQueue, ...newUploads];
    
    // Process queue
    this.processUploadQueue();
    
    // Notify parent component
    this.dispatchEvent(new CustomEvent('upload-started', {
      detail: { files: newUploads },
      bubbles: true
    }));
  }

  processUploadQueue() {
    // Find next queued file
    const nextUpload = this.uploadQueue.find(item => item.status === 'queued');
    if (!nextUpload) return;
    
    // Update status
    this.updateUploadItem(nextUpload.id, { status: 'uploading' });
    
    // Create XHR for upload
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', nextUpload.file);
    
    // Store XHR reference for cancellation
    this.updateUploadItem(nextUpload.id, { xhr });
    
    // Track upload progress
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) {
        const progress = Math.round((event.loaded / event.total) * 100);
        this.updateUploadItem(nextUpload.id, { progress });
        this.calculateOverallProgress();
      }
    });
    
    // Handle completion
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const response = JSON.parse(xhr.responseText);
          if (response.success) {
            // Move to processing state
            this.updateUploadItem(nextUpload.id, { 
              status: 'processing', 
              progress: 100,
              taskId: response.task_id // If backend provides task ID
            });
            
            // Start polling for processing status if backend supports it
            if (response.task_id) {
              this.pollProcessingStatus(nextUpload.id, response.task_id);
            } else {
              // Otherwise mark as complete
              this.updateUploadItem(nextUpload.id, { status: 'complete' });
            }
            
            // Notify parent component
            this.dispatchEvent(new CustomEvent('upload-complete', {
              detail: { fileId: nextUpload.id, fileName: nextUpload.name },
              bubbles: true
            }));
            
            // Process next file
            this.processUploadQueue();
          } else {
            this.handleUploadError(nextUpload.id, response.message || 'Upload failed');
          }
        } catch (e) {
          this.handleUploadError(nextUpload.id, 'Invalid server response');
        }
      } else {
        this.handleUploadError(nextUpload.id, `Server error: ${xhr.status}`);
      }
    });
    
    // Handle network errors
    xhr.addEventListener('error', () => {
      this.handleUploadError(nextUpload.id, 'Network error occurred');
    });
    
    // Handle aborted uploads
    xhr.addEventListener('abort', () => {
      this.updateUploadItem(nextUpload.id, { status: 'cancelled', progress: 0 });
      this.processUploadQueue(); // Move to next file
    });
    
    // Start upload
    xhr.open('POST', `/api/kb/${this.kbName}/upload`);
    xhr.send(formData);
  }

  updateUploadItem(id, updates) {
    this.uploadQueue = this.uploadQueue.map(item => 
      item.id === id ? { ...item, ...updates } : item
    );
  }

  handleUploadError(id, errorMessage) {
    this.updateUploadItem(id, { 
      status: 'error', 
      error: errorMessage 
    });
    
    // Notify parent component
    this.dispatchEvent(new CustomEvent('upload-error', {
      detail: { fileId: id, message: errorMessage },
      bubbles: true
    }));
    
    this.processUploadQueue(); // Move to next file
  }

  calculateOverallProgress() {
    if (this.uploadQueue.length === 0) {
      this.overallProgress = 0;
      return;
    }
    
    const totalProgress = this.uploadQueue.reduce((sum, item) => {
      // Only include items that are queued, uploading, or complete
      if (['queued', 'uploading', 'processing', 'complete'].includes(item.status)) {
        return sum + item.progress;
      }
      return sum;
    }, 0);
    
    const activeItems = this.uploadQueue.filter(item => 
      ['queued', 'uploading', 'processing', 'complete'].includes(item.status)
    ).length;
    
    this.overallProgress = Math.round(totalProgress / (activeItems || 1));
  }

  // Optional: Poll for processing status if backend supports it
  async pollProcessingStatus(uploadId, taskId) {
    try {
      const response = await fetch(`/api/kb/${this.kbName}/task/${taskId}`);
      const result = await response.json();
      
      if (result.status === 'complete') {
        this.updateUploadItem(uploadId, { status: 'complete' });
        return;
      } else if (result.status === 'error') {
        this.updateUploadItem(uploadId, { 
          status: 'error', 
          error: result.message || 'Processing failed'
        });
        return;
      }
      
      // Update processing progress if available
      if (result.progress) {
        this.updateUploadItem(uploadId, { processingProgress: result.progress });
      }
      
      // Continue polling
      setTimeout(() => this.pollProcessingStatus(uploadId, taskId), 1000);
    } catch (e) {
      console.error('Error polling task status:', e);
      // Don't mark as error, just stop polling
    }
  }

  cancelUpload(id) {
    const upload = this.uploadQueue.find(item => item.id === id);
    if (upload && upload.status === 'uploading' && upload.xhr) {
      upload.xhr.abort();
    } else if (upload) {
      // For queued items, just mark as cancelled
      this.updateUploadItem(id, { status: 'cancelled' });
    }
  }

  retryUpload(id) {
    const upload = this.uploadQueue.find(item => item.id === id);
    if (upload && ['error', 'cancelled'].includes(upload.status)) {
      this.updateUploadItem(id, { 
        status: 'queued', 
        progress: 0, 
        error: null,
        xhr: null
      });
      this.processUploadQueue();
    }
  }

  clearCompleted() {
    this.uploadQueue = this.uploadQueue.filter(item => 
      !['complete', 'cancelled'].includes(item.status)
    );
  }

  formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' bytes';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
  }

  render() {
    return html`
      <link rel="stylesheet" href="/mr_kb/static/css/file-uploader.css">
      
      <!-- Upload Zone -->
      <div class="upload-zone" 
           @click=${() => this.shadowRoot.querySelector('input[type="file"]').click()}
           @dragover=${e => e.preventDefault()}
           @drop=${e => {
             e.preventDefault();
             this.handleUpload(e);
           }}>
        <input type="file" 
               style="display: none"
               @change=${this.handleUpload}
               multiple>
        <p>Drop files here or click to upload to "${this.kbName}"</p>
      </div>

      <!-- Upload Progress Section -->
      ${this.uploadQueue.length > 0 ? html`
        <div class="upload-progress-container">
          <div class="upload-header">
            <h4>Uploads</h4>
            <div class="upload-actions">
              <button @click=${this.clearCompleted}
                      ?disabled=${!this.uploadQueue.some(i => ['complete', 'cancelled'].includes(i.status))}>
                Clear Completed
              </button>
            </div>
          </div>
          
          <!-- Overall Progress -->
          <div class="overall-progress">
            <div class="progress-label">
              <span>Overall Progress: ${this.overallProgress}%</span>
              <span>${this.uploadQueue.filter(i => i.status === 'complete').length}/${this.uploadQueue.length} Complete</span>
            </div>
            <div class="progress-bar">
              <div class="progress-fill" style="width: ${this.overallProgress}%"></div>
            </div>
          </div>
          
          <!-- Individual File Progress -->
          <div class="file-progress-list">
            ${this.uploadQueue.map(item => html`
              <div class="file-item ${item.status}">
                <div class="file-info">
                  <div class="file-name">${item.name}</div>
                  <div class="file-meta">${this.formatFileSize(item.size)} | ${item.type}</div>
                </div>
                
                <div class="file-progress">
                  <div class="progress-bar">
                    <div class="progress-fill" 
                         style="width: ${item.status === 'processing' ? 
                                       (item.processingProgress || 100) : item.progress}%"></div>
                  </div>
                </div>
                
                <div class="file-status">
                  ${item.status === 'queued' ? 'Queued' : ''}
                  ${item.status === 'uploading' ? `Uploading ${item.progress}%` : ''}
                  ${item.status === 'processing' ? `Processing ${item.processingProgress || ''}` : ''}
                  ${item.status === 'complete' ? 'Complete' : ''}
                  ${item.status === 'error' ? html`Error: <span class="error-message">${item.error}</span>` : ''}
                  ${item.status === 'cancelled' ? 'Cancelled' : ''}
                </div>
                
                <div class="file-actions">
                  ${item.status === 'uploading' ? html`
                    <button class="cancel" @click=${() => this.cancelUpload(item.id)}>Cancel</button>
                  ` : ''}
                  
                  ${['error', 'cancelled'].includes(item.status) ? html`
                    <button class="retry" @click=${() => this.retryUpload(item.id)}>Retry</button>
                  ` : ''}
                </div>
              </div>
            `)}
          </div>
        </div>
      ` : ''}
    `;
  }
}

customElements.define('file-uploader', FileUploader);
