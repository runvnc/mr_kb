console.log("hi from kb-manager.js");

import { LitElement, html, css } from '/admin/static/js/lit-core.min.js';
import { BaseEl } from '/admin/static/js/base.js';
import './file-uploader.js';

class KnowledgeBaseManager extends BaseEl {
  static properties = {
    kbs: { type: Array },
    documents: { type: Array },
    selectedKb: { type: String },
    loading: { type: Boolean },
    uploadStatus: { type: String },
    urlStatus: { type: String }
  }

  static styles = css`
    :host {
      display: block;
      width: 100%;
      height: 100%;
    }

    .kb-manager {
      display: flex;
      flex-direction: column;
      width: 100%;
      max-width: 1200px;
      margin: 0 auto;
      gap: 20px;
    }

    .section {
      background: rgb(10, 10, 25);
      border-radius: 8px;
      padding: 1rem;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .kb-list {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 20px;
    }

    .kb-card {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      padding: 10px;
      cursor: pointer;
      transition: all 0.3s ease;
    }

    .kb-card:hover {
      background: rgba(255, 255, 255, 0.1);
    }

    .kb-card.selected {
      border-color: #4a9eff;
      background: rgba(74, 158, 255, 0.1);
    }

    .kb-card h3 {
      margin: 0;
      font-size: 1.1em;
    }

    .kb-card p {
      margin: 5px 0;
      font-size: 0.9em;
      opacity: 0.8;
    }

    .help-text {
      margin-top: 10px;
      font-size: 0.9em;
      opacity: 0.8;
    }

    .doc-list {
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
      table-layout: fixed;
      overflow-x: auto;
      display: block;
    }

    .doc-list th,
    .doc-list td {
      padding: 0.75rem;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 200px;
    }

    .doc-list th:first-child, .doc-list td:first-child {
      max-width: none;
      white-space: normal;
      word-break: break-word;
    }

    .doc-list th {
      background: rgba(0, 0, 0, 0.2);
      font-weight: 500;
    }

    .doc-list input[type="checkbox"] {
      cursor: pointer;
      width: 18px;
      height: 18px;
    }

    .doc-list td:nth-child(5) {
      text-align: center;
    }

    .actions {
      display: flex;
      gap: 10px;
      align-items: center;
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

    button.delete {
      background: #402a2a;
    }

    button.delete:hover {
      background: #503a3a;
    }

    button.refresh {
      background: #2a402a;
      padding: 0.5rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    button.refresh:hover {
      background: #3a503a;
    }

    .status {
      margin-top: 10px;
      padding: 10px;
      border-radius: 4px;
    }

    .status.error {
      background: rgba(255, 0, 0, 0.1);
      border: 1px solid rgba(255, 0, 0, 0.2);
    }

    .status.success {
      background: rgba(0, 255, 0, 0.1);
      border: 1px solid rgba(0, 255, 0, 0.2);
    }

    .create-kb {
      margin-bottom: 20px;
    }

    .create-kb input, .url-form input {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: white;
      padding: 8px;
      border-radius: 4px;
      margin-right: 10px;
    }

    .url-form {
      margin: 20px 0;
      padding: 15px;
      background: rgba(0, 0, 0, 0.2);
      border-radius: 4px;
    }

    .url-form input[type="url"] {
      width: 60%;
    }

    .url-source {
      color: #4a9eff;
    }

    .url-source a {
      color: #4a9eff;
      text-decoration: underline;
    }

    .url-source small {
      display: block;
      color: rgba(255, 255, 255, 0.6);
      font-size: 0.8em;
      margin-top: 4px;
    }
  `;

  constructor() {
    super();
    this.kbs = [];
    this.documents = [];
    this.selectedKb = null;
    this.loading = false;
    this.uploadStatus = '';
    this.urlStatus = '';
    this.fetchKBs();
  }

  async fetchKBs() {
    this.loading = true;
    try {
      const response = await fetch('/api/kb/list');
      const result = await response.json();
      if (result.success) {
        this.kbs = Object.values(result.data);
        if (this.kbs.length > 0 && !this.selectedKb) {
          this.selectKB(this.kbs[0].name);
        }
      }
    } catch (error) {
      console.error('Error fetching KBs:', error);
    } finally {
      this.loading = false;
    }
  }

  async createKB(e) {
    e.preventDefault();
    const nameInput = this.shadowRoot.querySelector('#kb-name');
    const descInput = this.shadowRoot.querySelector('#kb-description');
    const name = nameInput.value.trim();
    const description = descInput.value.trim();

    if (!name) return;

    try {
      const response = await fetch('/api/kb/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description })
      });

      const result = await response.json();
      if (result.success) {
        nameInput.value = '';
        descInput.value = '';
        await this.fetchKBs();
        this.selectKB(name);
      }
    } catch (error) {
      console.error('Error creating KB:', error);
    }
  }

  async deleteKB(name) {
    if (!confirm(`Are you sure you want to delete the knowledge base "${name}"?`)) return;

    try {
      const response = await fetch(`/api/kb/${name}`, {
        method: 'DELETE'
      });

      const result = await response.json();
      if (result.success) {
        await this.fetchKBs();
        if (this.selectedKb === name) {
          this.selectedKb = null;
          this.documents = [];
        }
      }
    } catch (error) {
      console.error('Error deleting KB:', error);
    }
  }

  async selectKB(name) {
    this.selectedKb = name;
    await this.fetchDocuments();
  }

  async fetchDocuments() {
    if (!this.selectedKb) return;

    this.loading = true;
    try {
      const response = await fetch(`/api/kb/${this.selectedKb}/documents`);
      const result = await response.json();
      if (result.success) {
        this.documents = result.data;
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
    } finally {
      this.loading = false;
    }
  }

  handleUploadStarted(e) {
    this.uploadStatus = 'Upload started...';
  }

  handleUploadComplete(e) {
    this.uploadStatus = `Successfully uploaded ${e.detail.fileName}`;
    this.fetchDocuments();
  }

  handleUploadError(e) {
    this.uploadStatus = e.detail.message || 'Upload failed';
  }

  async addUrl(e) {
    e.preventDefault();
    const urlInput = this.shadowRoot.querySelector('#url-input');
    const url = urlInput.value.trim();
    const verbatimCheckbox = this.shadowRoot.querySelector('#url-verbatim');
    const verbatim = verbatimCheckbox ? verbatimCheckbox.checked : true;

    if (!url) return;

    this.urlStatus = 'Adding URL, please wait...';

    try {
      const response = await fetch(`/api/kb/${this.selectedKb}/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, verbatim })
      });

      const result = await response.json();

      if (result.success) {
        const taskId = result.task_id;
        this.pollUrlTaskStatus(taskId, urlInput);
      } else {
        this.urlStatus = `Error: ${result.message || 'Failed to add URL'}`;
      }
    } catch (error) {
      console.error('Error adding URL:', error);
      this.urlStatus = `Error: ${error.message || 'Failed to add URL'}`;
    }
  }

  async pollUrlTaskStatus(taskId, urlInput) {
    try {
      const response = await fetch(`/api/kb/${this.selectedKb}/task/${taskId}`);
      const result = await response.json();

      if (result.success) {
        const status = result.status;
        const progress = result.progress;

        if (status === 'complete') {
          this.urlStatus = `Successfully added URL: ${result.url || 'URL'}`;
          urlInput.value = '';
          this.fetchDocuments();
          return;
        } else if (status === 'error') {
          this.urlStatus = `Error: ${result.message || 'Failed to add URL'}`;
          return;
        } else {
          this.urlStatus = `Adding URL... ${progress}%`;
          // Continue polling
          setTimeout(() => this.pollUrlTaskStatus(taskId, urlInput), 1000);
        }
      } else {
        this.urlStatus = `Error: ${result.message || 'Failed to check status'}`;
      }
    } catch (error) {
      console.error('Error polling task status:', error);
      this.urlStatus = `Error: ${error.message || 'Failed to check status'}`;
    }
  }

  async refreshUrl(urlHash, url) {
    if (!confirm(`Are you sure you want to refresh the content from ${url}?`)) return;

    this.urlStatus = 'Refreshing URL content, please wait...';

    try {
      const response = await fetch(`/api/kb/${this.selectedKb}/url/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url_or_hash: urlHash || url })
      });

      const result = await response.json();

      if (result.success) {
        const taskId = result.task_id;
        this.pollRefreshTaskStatus(taskId);
      } else {
        this.urlStatus = `Error: ${result.message || 'Failed to refresh URL'}`;
      }
    } catch (error) {
      console.error('Error refreshing URL:', error);
      this.urlStatus = `Error: ${error.message || 'Failed to refresh URL'}`;
    }
  }

  async pollRefreshTaskStatus(taskId) {
    try {
      const response = await fetch(`/api/kb/${this.selectedKb}/task/${taskId}`);
      const result = await response.json();

      if (result.success) {
        const status = result.status;
        const progress = result.progress;

        if (status === 'complete') {
          this.urlStatus = `Successfully refreshed URL content`;
          this.fetchDocuments();
          return;
        } else if (status === 'error') {
          this.urlStatus = `Error: ${result.message || 'Failed to refresh URL content'}`;
          return;
        } else {
          this.urlStatus = `Refreshing URL content... ${progress}%`;
          // Continue polling
          setTimeout(() => this.pollRefreshTaskStatus(taskId), 1000);
        }
      } else {
        this.urlStatus = `Error: ${result.message || 'Failed to check status'}`;
      }
    } catch (error) {
      console.error('Error polling task status:', error);
      this.urlStatus = `Error: ${error.message || 'Failed to check status'}`;
    }
  }

  async deleteDocument(file_path) {
    if (!confirm('Are you sure you want to delete this document?')) return;
    
    try {
      const response = await fetch(`/api/kb/${this.selectedKb}/documents?path=${encodeURIComponent(file_path)}`, {
        method: 'DELETE'
      });

      const result = await response.json();
      if (result.success) {
        await this.fetchDocuments();
      } else {
        throw new Error(result.message);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
    }
  }

  async deleteUrlDocument(url, urlHash) {
    if (!confirm(`Are you sure you want to delete this URL document?`)) return;
    
    try {
      const response = await fetch(`/api/kb/${this.selectedKb}/url`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url_or_hash: urlHash || url })
      });

      const result = await response.json();
      if (result.success) {
        await this.fetchDocuments();
      } else {
        throw new Error(result.message);
      }
    } catch (error) {
      console.error('Error deleting URL document:', error);
    }
  }

  // Toggle verbatim status for a document
  async toggleVerbatim(file_path, isVerbatim, event) {
    // Prevent the event from bubbling up to parent elements
    if (event) {
      event.stopPropagation();
    }
    
    try {
      const response = await fetch(`/api/kb/${this.selectedKb}/documents/toggle_verbatim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_path: file_path,
          verbatim: isVerbatim,
          force_verbatim: false // Could add a UI for this option later
        })
      });

      const result = await response.json();
      if (result.success) {
        await this.fetchDocuments(); // Refresh document list
      } else {
        // Show error message
        this.uploadStatus = `Error: ${result.message || 'Failed to toggle verbatim status'}`;
        // Refresh documents to reset checkbox state
        await this.fetchDocuments();
        // Clear error after 5 seconds
        setTimeout(() => {
          this.uploadStatus = '';
        }, 5000);
      }
    } catch (error) {
      console.error('Error toggling verbatim status:', error);
      this.uploadStatus = `Error: ${error.message || 'Failed to toggle verbatim status'}`;
    }
  }

  formatDate(dateString) {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleString();
  }

  render() {
    return html`
      <div class="kb-manager">
        <div class="section">
          <h2>Knowledge Base Management</h2>
          
          <!-- Create KB Form -->
          <div class="create-kb">
            <form @submit=${this.createKB}>
              <input type="text" id="kb-name" placeholder="KB Name" required>
              <input type="text" id="kb-description" placeholder="Description">
              <button type="submit">Create New KB</button>
            </form>
          </div>

          <!-- KB List -->
          <div class="kb-list">
            ${this.kbs.map(kb => html`
              <div class="kb-card ${kb.name === this.selectedKb ? 'selected' : ''}"
                   @click=${() => this.selectKB(kb.name)}>
                <h3>${kb.name}</h3>
                <p>${kb.description || 'No description'}</p>
                <small>Created: ${new Date(kb.created_at).toLocaleString()}</small>
                <button class="delete"
                        @click=${(e) => {
                          e.stopPropagation();
                          this.deleteKB(kb.name);
                        }}>Delete</button>
              </div>
            `)}
          </div>

          ${this.selectedKb ? html`
            <!-- URL Input Form -->
            <div class="url-form">
              <h3>Add Content from URL</h3>
              <form @submit=${this.addUrl}>
                <input type="url" id="url-input" placeholder="https://example.com/page" required>
                <label>
                  <input type="checkbox" id="url-verbatim" checked>
                  Add as verbatim
                </label>
                <button type="submit">Add URL</button>
              </form>
              ${this.urlStatus ? html`
                <div class="status ${this.urlStatus.includes('Error') ? 'error' : 'success'}">
                  ${this.urlStatus}
                </div>
              ` : ''}
            </div>

            <!-- File Uploader Component -->
            <file-uploader 
              .kbName=${this.selectedKb}
              @upload-started=${this.handleUploadStarted}
              @upload-complete=${this.handleUploadComplete}
              @upload-error=${this.handleUploadError}>
            </file-uploader>

            ${this.uploadStatus ? html`
              <div class="status ${this.uploadStatus.includes('failed') || this.uploadStatus.includes('Failed') ? 'error' : 'success'}">
                ${this.uploadStatus}
              </div>
            ` : ''}

            <!-- Help text for verbatim feature -->
            <div class="help-text">
              <p><strong>Verbatim:</strong> When checked, the document will always be included in full in search results, regardless of query relevance.</p>
            </div>

            <!-- Documents List -->
            <table class="doc-list">
              <thead>
                <tr>
                  <th>Document</th>
                  <th>Size</th>
                  <th>Verbatim</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                ${this.documents.map(doc => html`
                  <tr>
                    <td>
                      ${doc.is_url ? html`
                        <a href="${doc.url}" target="_blank">${doc.url}</a>
                        <br><small>Last updated: ${this.formatDate(doc.last_refreshed)}</small>
                      ` : html`${doc.file_name}`}
                    </td>
                    <td>${doc.size}</td>
                    <td>
                      <input type="checkbox" 
                             ?checked=${doc.is_verbatim} 
                             @change=${(e) => this.toggleVerbatim(doc.file_path, e.target.checked, e)}
                             @click=${(e) => e.stopPropagation()}
                             title="When checked, this document will always be included in full in search results">
                    </td>
                    <td>
                      <div class="actions">
                        ${doc.is_url ? html`
                          <button class="refresh"
                                  @click=${() => this.refreshUrl(doc.url_hash, doc.url)}
                                  title="Refresh content from URL">
                            ⟳
                          </button>
                        ` : ''}
                        <button class="delete"
                                @click=${() => doc.is_url ? 
                                        this.deleteUrlDocument(doc.url, doc.url_hash) : 
                                        this.deleteDocument(doc.file_path)}>
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                `)}
              </tbody>
            </table>
          ` : html`
            <p>Select a knowledge base to manage documents</p>
          `}
        </div>
      </div>
    `;
  }
}

console.log("Registering kb-manager");

customElements.define('knowledge-base-manager', KnowledgeBaseManager);
