import { LitElement, html, css } from '/admin/static/js/lit-core.min.js';
import { BaseEl } from '/admin/static/js/base.js';

class CsvViewer extends BaseEl {
  static properties = {
    kbName: { type: String },
    sourceId: { type: String },
    rows: { type: Array },
    addingRow: { type: Boolean },
    newRow: { type: Object },
    columns: { type: Array },
    config: { type: Object },
    loading: { type: Boolean },
    error: { type: String },
    editingRow: { type: Object },
    selectedFile: { type: Object }
  };

  static styles = css`
    :host {
      display: block;
      width: 100%;
      font-family: var(--font-family, system-ui, sans-serif);
    }

    .csv-viewer {
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
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .row-table {
      width: 100%;
      overflow-x: auto;
      margin-bottom: 1.5rem;
      border-collapse: collapse;
    }

    .row-table th, .row-table td {
      padding: 0.5rem;
      text-align: left;
      border: 1px solid rgba(255, 255, 255, 0.1);
      white-space: nowrap;
      max-width: 200px;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .row-table th {
      background: rgba(0, 0, 0, 0.2);
      position: sticky;
      top: 0;
    }

    .row-table tr:nth-child(even) {
      background: rgba(255, 255, 255, 0.03);
    }

    .actions {
      display: flex;
      gap: 0.5rem;
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

    button.delete {
      background: #ff4a4a;
    }

    button.delete:hover {
      background: #ff3a3a;
    }

    button.icon {
      padding: 0.25rem 0.5rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
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

    .modal {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.7);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }

    .modal-content {
      background: var(--component-bg, var(--background-color));
      color: var(--component-text, var(--text-color));
      border-radius: 8px;
      padding: 1.5rem;
      width: 90%;
      max-width: 600px;
      max-height: 90vh;
      overflow-y: auto;
    }

    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }

    .modal-header h3 {
      margin: 0;
    }

    .modal-body {
      margin-bottom: 1.5rem;
    }

    .form-group {
      margin-bottom: 1rem;
    }

    .form-group label {
      display: block;
      margin-bottom: 0.5rem;
      font-weight: 500;
    }

    .form-group textarea, .form-group input {
      width: 100%;
      padding: 0.5rem;
      border-radius: 4px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      background: rgba(0, 0, 0, 0.2);
      color: var(--component-text, var(--text-color));
      font-family: inherit;
    }

    .form-group textarea {
      min-height: 100px;
      resize: vertical;
    }

    .modal-footer {
      display: flex;
      justify-content: flex-end;
      gap: 1rem;
    }

    .sync-section {
      margin-top: 1.5rem;
      padding-top: 1.5rem;
      border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    .sync-form {
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-top: 0.5rem;
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

    .stats {
      margin-top: 0.5rem;
      font-size: 0.9rem;
      opacity: 0.8;
    }
  `;

  constructor() {
    super();
    this.kbName = '';
    this.sourceId = '';
    this.rows = [];
    this.columns = [];
    this.config = {};
    this.loading = false;
    this.error = '';
    this.editingRow = null;
    this.selectedFile = null;
    this.addingRow = false;
  }

  connectedCallback() {
    super.connectedCallback();
    if (this.kbName && this.sourceId) {
      this.loadData();
    }
  }

  updated(changedProperties) {
    if (changedProperties.has('kbName') || changedProperties.has('sourceId')) {
      if (this.kbName && this.sourceId) {
        this.loadData();
      }
    }
  }

  async loadData() {
    this.loading = true;
    this.error = '';
    
    try {
      // Load rows
      const response = await fetch(`/api/kb/${this.kbName}/csv/${this.sourceId}/rows`);
      const result = await response.json();
      
      if (result.success) {
        this.rows = result.data;
        
        // Extract columns from the first row
        if (this.rows.length > 0) {
          this.columns = Object.keys(this.rows[0]).filter(key => 
            key.startsWith('col_') || key === 'doc_id' || key === 'text'
          );
        }
      } else {
        this.error = result.message || 'Failed to load CSV rows';
      }
    } catch (error) {
      console.error('Error loading CSV data:', error);
      this.error = error.message || 'An error occurred while loading the CSV data';
    } finally {
      this.loading = false;
    }
  }

  render() {
    if (this.loading) {
      return html`<div class="loading">Loading CSV data...</div>`;
    }

    return html`
      <div class="csv-viewer">
        <h3>
          <span>CSV Data Viewer</span>
          <div class="actions">
            <button class="primary" @click=${this.showAddRowModal}>Add Row</button>
            <button @click=${this.refresh}>Refresh</button>
          </div>
        </h3>
        
        ${this.error ? html`<div class="error">${this.error}</div>` : ''}
        
        ${this.rows.length > 0 ? html`
          <div class="stats">
            Showing ${this.rows.length} rows
          </div>
          
          <div class="table-container">
            <table class="row-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Text</th>
                  ${this.columns.filter(col => col !== 'doc_id' && col !== 'text' && col !== 'node_id').map(col => html`
                    <th>${col}</th>
                  `)}
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                ${this.rows.map(row => html`
                  <tr>
                    <td title="${row.doc_id}">${row.doc_id}</td>
                    <td title="${row.text}">${row.text}</td>
                    ${this.columns.filter(col => col !== 'doc_id' && col !== 'text' && col !== 'node_id').map(col => html`
                      <td title="${row[col] || ''}">${row[col] || ''}</td>
                    `)}
                    <td>
                      <div class="actions">
                        <button class="icon" @click=${() => this.editRow(row)} title="Edit">
                          ✏️
                        </button>
                        <button class="icon delete" @click=${() => this.deleteRow(row)} title="Delete">
                          🗑️
                        </button>
                      </div>
                    </td>
                  </tr>
                `)}
              </tbody>
            </table>
          </div>
          
          <div class="sync-section">
            <h4>Sync with Updated CSV</h4>
            <p>Upload a new version of this CSV to sync changes.</p>
            
            <div class="sync-form">
              <div class="file-input-wrapper">
                <button>Select CSV File</button>
                <input type="file" accept=".csv" @change=${this.handleFileSelected}>
              </div>
              
              ${this.selectedFile ? html`
                <span class="file-name">${this.selectedFile.name}</span>
              ` : ''}
              
              <button class="primary" @click=${this.syncCsv} ?disabled=${!this.selectedFile}>
                Sync Changes
              </button>
            </div>
          </div>
        ` : html`
          <p>No rows found in this CSV source.</p>
        `}
      </div>
      
      ${this.editingRow ? this.renderEditModal() : ''}
      ${this.addingRow ? this.renderAddRowModal() : ''}
    `;
  }

  renderEditModal() {
    return html`
      <div class="modal" @click=${this.closeModal}>
        <div class="modal-content" @click=${e => e.stopPropagation()}>
          <div class="modal-header">
            <h3>Edit Row</h3>
            <button class="icon" @click=${this.closeModal}>✕</button>
          </div>
          
          <div class="modal-body">
            <div class="form-group">
              <label for="edit-text">Text Content</label>
              <textarea id="edit-text" .value=${this.editingRow.text} @input=${this.handleTextChange}></textarea>
            </div>
            
            ${this.columns.filter(col => col !== 'doc_id' && col !== 'text' && col !== 'node_id' && col.startsWith('col_')).map(col => html`
              <div class="form-group">
                <label for="edit-${col}">${col}</label>
                <input type="text" id="edit-${col}" .value=${this.editingRow[col] || ''} 
                       @input=${e => this.handleMetadataChange(col, e.target.value)}>
              </div>
            `)}
          </div>
          
          <div class="modal-footer">
            <button @click=${this.closeModal}>Cancel</button>
            <button class="primary" @click=${this.saveRow}>Save Changes</button>
          </div>
        </div>
      </div>
    `;
  }

  renderAddRowModal() {
    // Create a template object with empty values for a new row
    const newRowTemplate = {
      text: '',
    };

    // Add empty fields for all columns that start with col_
    if (this.rows.length > 0) {
      const sampleRow = this.rows[0];
      for (const key in sampleRow) {
        if (key.startsWith('col_')) {
          newRowTemplate[key] = '';
        }
      }
    }

    return html`
      <div class="modal" @click=${this.closeAddRowModal}>
        <div class="modal-content" @click=${e => e.stopPropagation()}>
          <div class="modal-header">
            <h3>Add New Row</h3>
            <button class="icon" @click=${this.closeAddRowModal}>✕</button>
          </div>
          
          <div class="modal-body">
            <div class="form-group">
              <label for="add-text">Text Content</label>
              <textarea id="add-text" .value=${newRowTemplate.text} @input=${e => this.handleAddRowTextChange(e)}></textarea>
            </div>
            
            ${Object.keys(newRowTemplate).filter(key => key !== 'text' && key.startsWith('col_')).map(col => html`
              <div class="form-group">
                <label for="add-${col}">${col}</label>
                <input type="text" id="add-${col}" .value=${newRowTemplate[col] || ''} 
                       @input=${e => this.handleAddRowMetadataChange(col, e.target.value)}>
              </div>
            `)}
          </div>
          
          <div class="modal-footer">
            <button @click=${this.closeAddRowModal}>Cancel</button>
            <button class="primary" @click=${this.addRow}>Add Row</button>
          </div>
        </div>
      </div>
    `;
  }

  refresh() {
    this.loadData();
  }

  editRow(row) {
    this.editingRow = { ...row };
  }

  showAddRowModal() {
    this.addingRow = true;
    this.newRow = { text: '' };
    
    // Initialize empty fields for all columns
    if (this.rows.length > 0) {
      const sampleRow = this.rows[0];
      for (const key in sampleRow) {
        if (key.startsWith('col_')) {
          this.newRow[key] = '';
        }
      }
    }
  }

  closeModal() {
    this.editingRow = null;
  }

  closeAddRowModal() {
    this.addingRow = false;
    this.newRow = null;
  }

  handleTextChange(e) {
    this.editingRow = { ...this.editingRow, text: e.target.value };
  }

  handleMetadataChange(key, value) {
    this.editingRow = { ...this.editingRow, [key]: value };
  }

  handleAddRowTextChange(e) {
    this.newRow = { ...this.newRow, text: e.target.value };
  }

  handleAddRowMetadataChange(key, value) {
    this.newRow = { ...this.newRow, [key]: value };
  }

  handleFileSelected(e) {
    this.selectedFile = e.target.files[0];
  }

  async saveRow() {
    this.error = '';
    
    try {
      // Prepare metadata from editable columns
      const metadata = {};
      for (const key of Object.keys(this.editingRow)) {
        if (key.startsWith('col_')) {
          metadata[key] = this.editingRow[key];
        }
      }
      
      const response = await fetch(`/api/kb/${this.kbName}/csv/${this.sourceId}/row/${this.editingRow.doc_id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: this.editingRow.text,
          metadata: metadata
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        this.closeModal();
        this.loadData(); // Refresh the data
      } else {
        this.error = result.message || 'Failed to update row';
      }
    } catch (error) {
      console.error('Error updating row:', error);
      this.error = error.message || 'An error occurred while updating the row';
    }
  }

  async addRow() {
    if (!this.newRow || !this.newRow.text) {
      this.error = 'Text content is required';
      return;
    }
    
    this.error = '';
    
    try {
      // Prepare metadata from editable columns
      const metadata = {};
      for (const key of Object.keys(this.newRow)) {
        if (key.startsWith('col_')) {
          metadata[key] = this.newRow[key];
        }
      }
      
      const response = await fetch(`/api/kb/${this.kbName}/csv/${this.sourceId}/row`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: this.newRow.text,
          metadata: metadata
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        this.closeAddRowModal();
        this.loadData(); // Refresh the data
      } else {
        this.error = result.message || 'Failed to add row';
      }
    } catch (error) {
      console.error('Error adding row:', error);
      this.error = error.message || 'An error occurred while adding the row';
    }
  }

  async deleteRow(row) {
    if (!confirm(`Are you sure you want to delete the row with ID "${row.doc_id}"?`)) {
      return;
    }
    
    this.error = '';
    
    try {
      const response = await fetch(`/api/kb/${this.kbName}/csv/${this.sourceId}/row/${row.doc_id}`, {
        method: 'DELETE'
      });
      
      const result = await response.json();
      
      if (result.success) {
        this.loadData(); // Refresh the data
      } else {
        this.error = result.message || 'Failed to delete row';
      }
    } catch (error) {
      console.error('Error deleting row:', error);
      this.error = error.message || 'An error occurred while deleting the row';
    }
  }

  async syncCsv() {
    if (!this.selectedFile) {
      return;
    }
    
    this.error = '';
    this.loading = true;
    
    try {
      const formData = new FormData();
      formData.append('file', this.selectedFile);
      
      const response = await fetch(`/api/kb/${this.kbName}/csv/sync`, {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (result.success) {
        // Poll for task status
        await this.pollTaskStatus(result.task_id);
      } else {
        this.error = result.message || 'Failed to sync CSV';
      }
    } catch (error) {
      console.error('Error syncing CSV:', error);
      this.error = error.message || 'An error occurred while syncing the CSV';
    } finally {
      this.loading = false;
      this.selectedFile = null;
    }
  }

  async pollTaskStatus(taskId) {
    const checkStatus = async () => {
      try {
        const response = await fetch(`/api/kb/${this.kbName}/task/${taskId}`);
        const result = await response.json();
        
        if (result.success) {
          if (result.status === 'complete') {
            this.loadData(); // Refresh data when complete
            return true;
          } else if (result.status === 'error') {
            this.error = result.message || 'Error during CSV sync';
            return true;
          } else {
            // Still processing
            return false;
          }
        } else {
          this.error = result.message || 'Failed to check task status';
          return true;
        }
      } catch (error) {
        console.error('Error checking task status:', error);
        this.error = error.message || 'An error occurred while checking task status';
        return true;
      }
    };
    
    // Poll every second until complete or error
    while (!(await checkStatus())) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
}

customElements.define('csv-viewer', CsvViewer);