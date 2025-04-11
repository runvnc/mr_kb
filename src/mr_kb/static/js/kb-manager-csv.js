import { LitElement, html, css } from '/admin/static/js/lit-core.min.js';
import { BaseEl } from '/admin/static/js/base.js';
import './csv-config.js';
import './csv-viewer.js';

class KbManagerCsv extends BaseEl {
  static properties = {
    kbName: { type: String },
    csvSources: { type: Array },
    selectedSourceId: { type: String },
    loading: { type: Boolean },
    error: { type: String },
    showConfig: { type: Boolean },
    previewData: { type: Object },
    uploadTaskId: { type: String },
    selectedFile: { type: Object }
  };

  static styles = css`
    :host {
      display: block;
      width: 100%;
      font-family: var(--font-family, system-ui, sans-serif);
    }

    .kb-manager-csv {
      background: var(--component-bg, var(--background-color));
      color: var(--component-text, var(--text-color));
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 1rem;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    h3 {
      margin-top: 0;
      margin-bottom: 1rem;
      font-size: 1.2rem;
    }

    .sources-list {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 20px;
    }

    .source-card {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      padding: 10px;
      cursor: pointer;
      transition: all 0.3s ease;
      min-width: 200px;
    }

    .source-card:hover {
      background: rgba(255, 255, 255, 0.1);
    }

    .source-card.selected {
      border-color: #4a9eff;
      background: rgba(74, 158, 255, 0.1);
    }

    .source-card h4 {
      margin: 0;
      font-size: 1.1em;
    }

    .source-card p {
      margin: 5px 0;
      font-size: 0.9em;
      opacity: 0.8;
    }

    .upload-section {
      margin-top: 1.5rem;
      padding-top: 1.5rem;
      border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    .file-input-wrapper {
      position: relative;
      overflow: hidden;
      display: inline-block;
    }

    .file-input-wrapper input[type=file] {
      position: absolute;
      left: 0;
      top: 0;
      opacity: 0;
      width: 100%;
      height: 100%;
      cursor: pointer;
    }

    .file-name {
      margin-left: 1rem;
      font-style: italic;
    }

    button {
      background: #2a2a40;
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 0.5rem 1rem;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.2s;
    }

    button:hover {
      background: #3a3a50;
    }

    button.primary {
      background: #4a9eff;
    }

    button.primary:hover {
      background: #3a8eff;
    }

    .error {
      color: #ff4a4a;
      margin-top: 1rem;
      padding: 0.5rem;
      border: 1px solid rgba(255, 74, 74, 0.3);
      border-radius: 4px;
      background: rgba(255, 74, 74, 0.1);
    }

    .loading {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
      font-style: italic;
      opacity: 0.7;
    }

    .task-status {
      margin-top: 1rem;
      padding: 0.5rem;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.05);
    }

    .progress-bar {
      height: 8px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      margin-top: 0.5rem;
      overflow: hidden;
    }

    .progress-bar-fill {
      height: 100%;
      background: #4a9eff;
      width: 0%;
      transition: width 0.3s ease;
    }
  `;

  constructor() {
    super();
    this.kbName = '';
    this.csvSources = [];
    this.selectedSourceId = null;
    this.loading = false;
    this.error = '';
    this.showConfig = false;
    this.previewData = null;
    this.uploadTaskId = null;
  }

  connectedCallback() {
    super.connectedCallback();
    if (this.kbName) {
      this.loadSources();
    }
  }

  updated(changedProperties) {
    if (changedProperties.has('kbName') && this.kbName) {
      this.loadSources();
    }
  }

  async loadSources() {
    this.loading = true;
    this.error = '';
    
    try {
      const response = await fetch(`/api/kb/${this.kbName}/csv/sources`);
      const result = await response.json();
      
      if (result.success) {
        this.csvSources = Object.entries(result.data).map(([id, data]) => ({
          id,
          ...data
        }));
        
        // Select the first source by default if available
        if (this.csvSources.length > 0 && !this.selectedSourceId) {
          this.selectedSourceId = this.csvSources[0].id;
        }
      } else {
        this.error = result.message || 'Failed to load CSV sources';
      }
    } catch (error) {
      console.error('Error loading CSV sources:', error);
      this.error = error.message || 'An error occurred while loading CSV sources';
    } finally {
      this.loading = false;
    }
  }

  render() {
    if (this.loading && !this.uploadTaskId) {
      return html`<div class="loading">Loading CSV sources...</div>`;
    }

    if (this.showConfig && this.previewData) {
      return html`
        <csv-config 
          .previewData=${this.previewData}
          .kbName=${this.kbName}
          @cancel=${this.cancelConfig}
          @upload-started=${this.handleUploadStarted}>
        </csv-config>
      `;
    }

    return html`
      <div class="kb-manager-csv">
        <h3>CSV Knowledge Base Management</h3>
        
        ${this.error ? html`<div class="error">${this.error}</div>` : ''}
        
        ${this.uploadTaskId ? this.renderTaskStatus() : ''}
        
        ${this.csvSources.length > 0 ? html`
          <div class="sources-list">
            ${this.csvSources.map(source => html`
              <div class="source-card ${source.id === this.selectedSourceId ? 'selected' : ''}" 
                   @click=${() => this.selectSource(source.id)}>
                <h4>${source.file_name}</h4>
                <p>Rows: ${source.row_count || 0}</p>
                <p>Added: ${new Date(source.added_at).toLocaleString()}</p>
              </div>
            `)}
          </div>
          
          ${this.selectedSourceId ? html`
            <csv-viewer 
              .kbName=${this.kbName}
              .sourceId=${this.selectedSourceId}>
            </csv-viewer>
          ` : ''}
        ` : html`
          <p>No CSV sources found. Upload a CSV file to get started.</p>
        `}
        
        <div class="upload-section">
          <h4>Upload New CSV</h4>
          <p>Upload a CSV file to create a new database source.</p>
          
          <div class="file-input-wrapper">
            <button>Select CSV File</button>
            <input type="file" accept=".csv" @change=${this.handleFileSelected}>
          </div>
          
          ${this.selectedFile ? html`
            <span class="file-name">${this.selectedFile.name}</span>
            <button class="primary" @click=${this.uploadFile}>Upload</button>
          ` : ''}
        </div>
      </div>
    `;
  }

  renderTaskStatus() {
    return html`
      <div class="task-status">
        <p>Processing CSV file... ${this.taskProgress}%</p>
        <div class="progress-bar">
          <div class="progress-bar-fill" style="width: ${this.taskProgress}%"></div>
        </div>
      </div>
    `;
  }

  selectSource(sourceId) {
    this.selectedSourceId = sourceId;
  }

  handleFileSelected(e) {
    this.selectedFile = e.target.files[0];
    console.log('File selected:', this.selectedFile);
  }

  async uploadFile() {
    if (!this.selectedFile) {
      return;
    }
    
    this.error = '';
    
    try {
      // Upload for preview first
      const formData = new FormData();
      formData.append('file', this.selectedFile);
      
      const response = await fetch(`/api/kb/${this.kbName}/csv/preview`, {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (result.success) {
        this.previewData = result.preview;
        this.showConfig = true;
      } else {
        this.error = result.message || 'Failed to preview CSV';
      }
    } catch (error) {
      console.error('Error uploading CSV for preview:', error);
      this.error = error.message || 'An error occurred while uploading the CSV';
    }
  }

  cancelConfig() {
    this.showConfig = false;
    this.previewData = null;
    this.selectedFile = null;
  }

  handleUploadStarted(e) {
    this.uploadTaskId = e.detail.taskId;
    this.taskProgress = 0;
    this.showConfig = false;
    this.pollTaskStatus();
  }

  async pollTaskStatus() {
    if (!this.uploadTaskId) return;
    
    try {
      const response = await fetch(`/api/kb/${this.kbName}/task/${this.uploadTaskId}`);
      const result = await response.json();
      
      if (result.success) {
        this.taskProgress = result.progress;
        
        if (result.status === 'complete') {
          // Task completed successfully
          setTimeout(() => {
            this.uploadTaskId = null;
            this.loadSources();
          }, 1000);
        } else if (result.status === 'error') {
          // Task failed
          this.error = result.message || 'Error processing CSV';
          this.uploadTaskId = null;
        } else {
          // Still processing, poll again
          setTimeout(() => this.pollTaskStatus(), 1000);
        }
      } else {
        this.error = result.message || 'Failed to check task status';
        this.uploadTaskId = null;
      }
    } catch (error) {
      console.error('Error checking task status:', error);
      this.error = error.message || 'An error occurred while checking task status';
      this.uploadTaskId = null;
    }
  }
}

customElements.define('kb-manager-csv', KbManagerCsv);