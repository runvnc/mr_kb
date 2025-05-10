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
    selectedFile: { type: Object },
    searchQuery: { type: String },
    searchMode: { type: Boolean },
    searchTimeout: { type: Object },
    searchInProgress: { type: Boolean }
  };

  static styles = css`
    :host {
      display: block;
      width: 100%;
      font-family: var(--font-family, system-ui, sans-serif);
    }

    .csv-viewer {
      /* background: var(--component-bg, var(--background-color)); */
      background: rgba(0, 0, 0);
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

    .search-status {
      display: inline-block;
      padding: 0.25rem 0.5rem;
      margin-bottom: 0.5rem;
      background: rgba(74, 158, 255, 0.1);
      border: 1px solid rgba(74, 158, 255, 0.3);
      border-radius: 4px;
      font-style: italic;
    }

    .search-container {
      display: flex;
      align-items: center;
      margin-bottom: 1rem;
      gap: 0.5rem;
    }

    .search-input {
      flex: 1;
      padding: 0.5rem;
      border-radius: 4px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      background: rgba(0, 0, 0, 0.2);
      color: var(--component-text, var(--text-color));
      font-family: inherit;
    }

    .search-input::placeholder {
      color: rgba(255, 255, 255, 0.5);
    }

    .toggle-container {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-left: 1rem;
    }

    .toggle-switch {
      position: relative;
      display: inline-block;
      width: 40px;
      height: 20px;
    }

    .toggle-label {
      font-size: 0.9rem;
      opacity: 0.8;
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
      /* background: var(--component-bg, var(--background-color)); */
      background: black;
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
      min-height: 300px;
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
    this.searchQuery = '';
    this.searchMode = true; // Default to search mode
    this.searchInProgress = false;
    this.searchTimeout = null;
  }

  connectedCallback() {
    super.connectedCallback();
    if (this.kbName && this.sourceId) {
      // Only load data if we have both kbName and sourceId
      this.error = ''; // Clear any previous errors
      if (this.searchMode) {
        this.searchRows(''); // Use search with empty query to get limited rows
      } else {
        this.loadData(); // Load all rows
      }
    }
  }

  updated(changedProperties) {
    if (changedProperties.has('kbName') || changedProperties.has('sourceId')) {
      // Reset error state when source changes
      this.error = '';
      
      // Clear rows when source is deselected
      if (!this.sourceId) {
        this.rows = [];
      } else if (this.kbName && this.sourceId) {
        if (this.searchMode) {
          this.searchRows(''); // Use search with empty query to get limited rows
        } else {
          this.loadData(); // Load all rows
        }
      }
    }
  }

  async searchRows(query) {
    // Don't show full loading state for search to avoid disrupting typing
    // Prevent searching if no source is selected
    if (!this.sourceId) {
      // Clear any existing search results when no source is selected
      this.rows = [];
      this.searchQuery = '';
      if (query) {
        this.error = 'Please select a CSV source first.';
      }
      this.error = 'Please select a CSV source first.';
      return;
    }
    
    const searchElement = this.shadowRoot?.querySelector('.search-input');
    const searchHasFocus = searchElement === document.activeElement;
    
    // Set search in progress flag but don't show full loading state
    this.searchInProgress = true;
    
    // Keep the current loading state if already loading
    const wasLoading = this.loading;
    
    // Don't show loading indicator during search to avoid disrupting typing
    this.error = '';
    const oldRows = [...this.rows]; // Save current rows in case of error
    
    try {
      if (query == '') return
      const url = new URL(`/api/kb/${this.kbName}/csv/${this.sourceId}/search`, window.location.origin);
      url.searchParams.append('query', query);
      url.searchParams.append('limit', '15'); // Reasonable default limit
      
      const response = await fetch(url);
      const result = await response.json();
      
      if (result.success) {
        this.rows = result.data;
        
        // Extract columns from the first row if we have results
        if (this.rows.length > 0) {
          this.columns = Object.keys(this.rows[0]).filter(key => 
            key.startsWith('col_') || key === 'doc_id' || key === 'text'
          );
        }
      } else {
        this.error = result.message || 'Failed to search CSV rows';
      }
    } catch (error) {
      console.error('Error searching CSV data:', error);
      this.error = error.message || 'An error occurred while searching the CSV data';
    } finally {
      // Reset search in progress flag
      this.searchInProgress = false;
      // Keep loading state as it was
      this.loading = wasLoading;
    }
  }

  async loadData() {
    // Prevent loading data if no source is selected
    if (!this.sourceId) {
      this.error = 'Please select a CSV source first.';
      return;
    }
    
    this.loading = true;
    this.error = '';
    
    try {
      // If in search mode with a query, use search endpoint, otherwise load all rows
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
    if (this.loading && !this.searchInProgress) {
      return html`<div class="loading">Loading CSV data...</div>`;
    }

    const handleSearchInput = (e) => {
      // Prevent search if no source is selected
      if (!this.sourceId) {
        this.error = 'Please select a CSV source first.';
        e.target.value = ''; // Clear the input
        return;
      }
      
      const query = e.target.value;
      this.searchQuery = query;
      
      // Clear any existing timeout
      this.searchInProgress = false;
      if (this.searchTimeout) {
        clearTimeout(this.searchTimeout);
      }
      
      // Set a new timeout for debouncing
      this.searchTimeout = setTimeout(() => {
        if (this.searchMode) {
          this.searchInProgress = true;
          this.searchRows(query);
        }
      }, 300); // 300ms debounce delay
    };
    
    const toggleSearchMode = () => {
      // Prevent toggling if no source is selected
      if (!this.sourceId) {
        this.error = 'Please select a CSV source first.';
        return;
      }
      
      // Toggle search mode
      this.searchInProgress = false;
      this.searchMode = !this.searchMode;
      
      // Update the UI based on the new mode
      if (this.searchMode) {
        this.searchRows(this.searchQuery || ''); // Use current query or empty string
      } else {
        this.loadData(); // Show all rows
      }
    };

    const isSearching = this.searchInProgress;

    return html`
      <div class="csv-viewer">
        <h3>
          <span>CSV Data Viewer</span>
          <pre>${this.sourceId}</pre>
          ${this.sourceId ? html`
            <div class="actions">
              <button class="primary" @click=${this.showAddRowModal}>Add Row</button>
              <button @click=${this.refresh}>Refresh</button>
            </div>
          ` : ''}
        </h3>
        
        
        ${this.error ? html`<div class="error">${this.error}</div>` : ''}
        
        ${!this.sourceId ? html`
          <div class="error">
            Please select a CSV source from the list above before performing any operations.
          </div>
        ` : html`${isSearching && this.sourceId ? html`<div class="search-status">Searching...</div>` : ''}`}
        
        ${this.sourceId ? html`
        <div class="search-container">
          <input 
            type="text" 
            class="search-input" 
            placeholder="Search by metadata or content..." 
            ?disabled=${!this.sourceId}
            .value=${this.searchQuery}
            @input=${handleSearchInput}
          />
          <div class="toggle-container">
            <span class="toggle-label">${this.searchMode ? 'Search Mode' : 'Show All Mode'}</span>
            <button @click=${toggleSearchMode} ?disabled=${!this.sourceId}>
              Switch to ${this.searchMode ? 'Show All Mode' : 'Search Mode'}
            </button>
          </div>
        </div>
        
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
          
          <div class="sync-section" style="display:none;">
            <h4>Sync with Updated CSV</h4>
            <p>Upload a new version of this CSV to sync changes.</p>
            
            <div class="sync-form">
              <div class="file-input-wrapper">
                <button>Select CSV File</button>
                <input type="file" accept=".csv" @change=${this.handleFileSelected}>
              </div>
              
              ${this.selectedFile && this.sourceId ? html`
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
      </div>` : ''}
      
      ${this.editingRow ? this.renderEditModal() : ''}
      ${this.addingRow ? this.renderAddRowModal() : ''}
    `;
  }

  renderEditModal() {
    return html`
      <div class="modal" @click=${this.closeModal}>
        <div class="modal-content" @click=${e => e.stopPropagation()}>
          <div class="modal-header">
            <h3>Edit Row${this.editingRow ? ` - ${this.editingRow.doc_id}` : ''}</h3>
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
    // Use this.newRow directly to ensure we're using the most up-to-date column structure
    const newRowTemplate = this.newRow;

    return html`
      <div class="modal" @click=${this.closeAddRowModal}>
        <div class="modal-content" @click=${e => e.stopPropagation()}>
          <div class="modal-header">
            <h3>Add New Row${this.sourceId ? ` to ${this.sourceId}` : ''}</h3>
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
            <button class="primary" @click=${this.addRow} ?disabled=${!this.sourceId}>Add Row</button>
          </div>
        </div>
      </div>
    `;
  }

  refresh() {
    if (this.sourceId) {
      this.loadData();
    }
  }

  editRow(row) {
    // Prevent editing rows if no source is selected
    if (!this.sourceId) {
      this.error = 'Please select a CSV source first.';
      return;
    }
    
    this.editingRow = { ...row };
  }

  async showAddRowModal() {
    // Prevent adding rows if no source is selected
    if (!this.sourceId) {
      this.error = 'Please select a CSV source first.';
      return;
    }
    
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
    
    // If rows are empty, try to load them first
    if (this.rows.length === 0) {
      try {
        this.error = '';
        const loadingIndicator = this.shadowRoot?.querySelector('.loading');
        if (loadingIndicator) loadingIndicator.style.display = 'block';
        
        // First try to get the CSV source configuration - this is more efficient
        // than loading all rows, especially for large CSV files
        console.log('Fetching CSV source configuration for Add Row form');
        try {
          const configResponse = await fetch(`/api/kb/${this.kbName}/csv/sources`);
          const configResult = await configResponse.json();
          
          if (configResult.success && configResult.data && configResult.data[this.sourceId]) {
            const sourceConfig = configResult.data[this.sourceId].config;
            console.log('Found source config:', sourceConfig);
            
            // If we have column configuration, create form fields based on it
            if (sourceConfig && sourceConfig.key_metadata_columns) {
              // Add fields for key metadata columns
              for (const colIdx of sourceConfig.key_metadata_columns) {
                const colName = `col_${colIdx}`;
                if (!this.newRow[colName]) {
                  this.newRow[colName] = ''; 
                }
              }
            }
            
            // Now that we have the column structure, show the modal
            this.requestUpdate();
          }
        } catch (configError) {
          console.error('Error fetching CSV source configuration:', configError);
          
          // Fall back to loading rows if we couldn't get the configuration
          console.log('Falling back to loading rows for column structure');
          try {
            const response = await fetch(`/api/kb/${this.kbName}/csv/${this.sourceId}/rows`);
            const result = await response.json();
            
            if (result.success && result.data.length > 0) {
              this.rows = result.data;
              
              // Now that we have rows, add column fields to the newRow template
              const sampleRow = this.rows[0];
              for (const key in sampleRow) {
                if (key.startsWith('col_') && !this.newRow[key]) {
                  this.newRow[key] = '';
                }
              }
            }
            
            // Now that we have the column structure, show the modal
            this.requestUpdate();
          } catch (rowError) {
            console.error('Error loading rows:', rowError);
          }
          this.requestUpdate();
        }
        
        if (loadingIndicator) loadingIndicator.style.display = 'none'; 
      } catch (error) {
        console.error('Error loading rows for add form:', error);
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
    // Prevent changes if no source is selected
    if (!this.sourceId) {
      this.error = 'Please select a CSV source first.';
      return;
    }
    this.newRow = { ...this.newRow, text: e.target.value };
  }

  handleAddRowMetadataChange(key, value) {
    // Prevent changes if no source is selected
    if (!this.sourceId) {
      this.error = 'Please select a CSV source first.';
      return;
    }
    this.newRow = { ...this.newRow, [key]: value };
  }

  handleFileSelected(e) {
    // Prevent file selection if no source is selected
    if (!this.sourceId) {
      this.error = 'Please select a CSV source first.';
      return;
    }
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
        //this.loadData(); // Refresh the data
      } else {
        this.error = result.message || 'Failed to update row';
      }
    } catch (error) {
      console.error('Error updating row:', error);
      this.error = error.message || 'An error occurred while updating the row';
    }
  }

  async addRow() {
    // Prevent adding rows if no source is selected
    if (!this.sourceId) {
      this.error = 'Please select a CSV source first.';
      return;
    }
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
        //this.loadData(); // Refresh the data
      } else {
        this.error = result.message || 'Failed to add row';
      }
    } catch (error) {
      console.error('Error adding row:', error);
      this.error = error.message || 'An error occurred while adding the row';
    }
  }

  async deleteRow(row) {
    // Prevent deleting rows if no source is selected
    if (!this.sourceId) {
      this.error = 'Please select a CSV source first.';
      return;
    }
    
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
        //this.loadData(); // Refresh the data
      } else {
        this.error = result.message || 'Failed to delete row';
      }
    } catch (error) {
      console.error('Error deleting row:', error);
      this.error = error.message || 'An error occurred while deleting the row';
    }
  }

  async syncCsv() {
    // Prevent syncing if no source is selected
    if (!this.sourceId) {
      this.error = 'Please select a CSV source first.';
      return;
    }
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
            //this.loadData(); // Refresh data when complete
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
