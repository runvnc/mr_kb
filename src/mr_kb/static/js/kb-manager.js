import { LitElement, html, css } from '/admin/static/js/lit-core.min.js';
import { BaseEl } from '/admin/static/js/base.js';
import './file-uploader.js';

class KnowledgeBaseManager extends BaseEl {
  static properties = {
    kbs: { type: Array },
    documents: { type: Array },
    selectedKb: { type: String },
    loading: { type: Boolean },
    uploadStatus: { type: String }
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

    .doc-list {
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
    }

    .doc-list th,
    .doc-list td {
      padding: 0.75rem;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }

    .doc-list th {
      background: rgba(0, 0, 0, 0.2);
      font-weight: 500;
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

    .create-kb input {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: white;
      padding: 8px;
      border-radius: 4px;
      margin-right: 10px;
    }
  `;

  constructor() {
    super();
    this.kbs = [];
    this.documents = [];
    this.selectedKb = null;
    this.loading = false;
    this.uploadStatus = '';
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

  async deleteDocument(docId) {
    if (!confirm('Are you sure you want to delete this document?')) return;

    try {
      const response = await fetch(`/api/kb/${this.selectedKb}/documents/${docId}`, {
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

            <!-- Documents List -->
            <table class="doc-list">
              <thead>
                <tr>
                  <th>Document</th>
                  <th>Type</th>
                  <th>Added</th>
                  <th>Size</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                ${this.documents.map(doc => html`
                  <tr>
                    <td>${doc.file_name}</td>
                    <td>${doc.file_type}</td>
                    <td>${new Date(doc.creation_date).toLocaleString()}</td>
                    <td>${doc.size}</td>
                    <td>
                      <button class="delete"
                              @click=${() => this.deleteDocument(doc.doc_id)}>
                        Delete
                      </button>
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

customElements.define('knowledge-base-manager', KnowledgeBaseManager);
