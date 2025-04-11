import { LitElement, html, css } from '/admin/static/js/lit-core.min.js';
import { BaseEl } from '/admin/static/js/base.js';

class CsvConfig extends BaseEl {
  static properties = {
    previewData: { type: Object },
    loading: { type: Boolean },
    kbName: { type: String },
    config: { type: Object },
    error: { type: String }
  };

  static styles = css`
    :host {
      display: block;
      width: 100%;
      font-family: var(--font-family, system-ui, sans-serif);
    }

    .csv-config {
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

    .preview-table {
      width: 100%;
      overflow-x: auto;
      margin-bottom: 1.5rem;
      border-collapse: collapse;
    }

    .preview-table th, .preview-table td {
      padding: 0.5rem;
      text-align: left;
      border: 1px solid rgba(255, 255, 255, 0.1);
      white-space: nowrap;
      max-width: 200px;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .preview-table th {
      background: rgba(0, 0, 0, 0.2);
      position: sticky;
      top: 0;
    }

    .preview-table tr:nth-child(even) {
      background: rgba(255, 255, 255, 0.03);
    }

    .config-section {
      margin-bottom: 1.5rem;
    }

    .config-section h4 {
      margin-top: 0;
      margin-bottom: 0.5rem;
    }

    .config-section p {
      margin-top: 0;
      margin-bottom: 0.5rem;
      font-size: 0.9rem;
      opacity: 0.8;
    }

    .column-selector {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .column-option {
      display: flex;
      align-items: center;
      padding: 0.5rem;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .column-option:hover {
      background: rgba(255, 255, 255, 0.05);
    }

    .column-option.selected {
      background: rgba(74, 158, 255, 0.1);
      border-color: #4a9eff;
    }

    .column-option input {
      margin-right: 0.5rem;
    }

    .actions {
      display: flex;
      justify-content: flex-end;
      gap: 1rem;
      margin-top: 1rem;
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

    .checkbox-list {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.5rem;
    }

    .checkbox-item {
      display: flex;
      align-items: center;
      padding: 0.25rem 0.5rem;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 4px;
    }

    .checkbox-item input {
      margin-right: 0.5rem;
    }
  `;

  constructor() {
    super();
    this.previewData = null;
    this.loading = false;
    this.kbName = '';
    this.config = {
      text_column: null,
      id_column: null,
      key_metadata_columns: [],
      metadata_columns: [],
      has_header: true
    };
    this.error = '';
  }

  render() {
    if (this.loading) {
      return html`<div class="loading">Loading CSV preview...</div>`;
    }

    if (!this.previewData) {
      return html`<div class="csv-config">
        <h3>CSV Configuration</h3>
        <p>No CSV data available for preview.</p>
      </div>`;
    }

    const { rows, column_count, has_header } = this.previewData;
    const headers = has_header ? rows[0] : Array.from({ length: column_count }, (_, i) => `Column ${i}`);
    const dataRows = has_header ? rows.slice(1) : rows;

    return html`
      <div class="csv-config">
        <h3>CSV Configuration</h3>
        
        <div class="config-section">
          <h4>Preview</h4>
          <p>Preview of the first ${dataRows.length} rows from your CSV file.</p>
          
          <div class="preview-table-container">
            <table class="preview-table">
              <thead>
                <tr>
                  <th>#</th>
                  ${headers.map((header, i) => html`
                    <th>
                      ${header} (${i})
                    </th>
                  `)}
                </tr>
              </thead>
              <tbody>
                ${dataRows.map((row, rowIndex) => html`
                  <tr>
                    <td>${rowIndex}</td>
                    ${row.map((cell, cellIndex) => html`
                      <td title="${cell}">${cell}</td>
                    `)}
                  </tr>
                `)}
              </tbody>
            </table>
          </div>
        </div>

        <div class="config-section">
          <h4>Text Column</h4>
          <p>Select the column that contains the main text content.</p>
          
          <div class="column-selector">
            ${headers.map((header, i) => html`
              <div class="column-option ${this.config.text_column === i ? 'selected' : ''}" 
                   @click=${() => this.selectTextColumn(i)}>
                <input type="radio" name="text-column" 
                       ?checked=${this.config.text_column === i}
                       @change=${() => this.selectTextColumn(i)}>
                <span>${header} (${i})</span>
              </div>
            `)}
          </div>
        </div>

        <div class="config-section">
          <h4>Document ID Column</h4>
          <p>Select the column that contains a unique identifier for each row.</p>
          
          <div class="column-selector">
            ${headers.map((header, i) => html`
              <div class="column-option ${this.config.id_column === i ? 'selected' : ''}" 
                   @click=${() => this.selectIdColumn(i)}>
                <input type="radio" name="id-column" 
                       ?checked=${this.config.id_column === i}
                       @change=${() => this.selectIdColumn(i)}>
                <span>${header} (${i})</span>
              </div>
            `)}
          </div>
        </div>

        <div class="config-section">
          <h4>Key Metadata Columns</h4>
          <p>Select columns to display in the main UI view (for easy identification).</p>
          
          <div class="checkbox-list">
            ${headers.map((header, i) => html`
              <div class="checkbox-item">
                <input type="checkbox" id="key-meta-${i}" 
                       ?checked=${this.config.key_metadata_columns.includes(i)}
                       @change=${(e) => this.toggleKeyMetadataColumn(i, e.target.checked)}>
                <label for="key-meta-${i}">${header} (${i})</label>
              </div>
            `)}
          </div>
        </div>

        <div class="config-section">
          <h4>Additional Metadata Columns</h4>
          <p>Select additional columns to store as metadata (searchable but not displayed by default).</p>
          
          <div class="checkbox-list">
            ${headers.map((header, i) => html`
              <div class="checkbox-item">
                <input type="checkbox" id="meta-${i}" 
                       ?checked=${this.config.metadata_columns.includes(i)}
                       @change=${(e) => this.toggleMetadataColumn(i, e.target.checked)}>
                <label for="meta-${i}">${header} (${i})</label>
              </div>
            `)}
          </div>
        </div>

        <div class="config-section">
          <h4>Header Row</h4>
          <div class="checkbox-item">
            <input type="checkbox" id="has-header" 
                   ?checked=${this.config.has_header}
                   @change=${(e) => this.setHasHeader(e.target.checked)}>
            <label for="has-header">First row contains column headers</label>
          </div>
        </div>

        ${this.error ? html`<div class="error">${this.error}</div>` : ''}

        <div class="actions">
          <button @click=${this.cancel}>Cancel</button>
          <button class="primary" @click=${this.submitConfig} ?disabled=${!this.isConfigValid()}>Process CSV</button>
        </div>
      </div>
    `;
  }

  selectTextColumn(index) {
    this.config = { ...this.config, text_column: index };
  }

  selectIdColumn(index) {
    this.config = { ...this.config, id_column: index };
  }

  toggleKeyMetadataColumn(index, checked) {
    const keyMetadataColumns = [...this.config.key_metadata_columns];
    
    if (checked && !keyMetadataColumns.includes(index)) {
      keyMetadataColumns.push(index);
    } else if (!checked && keyMetadataColumns.includes(index)) {
      const idx = keyMetadataColumns.indexOf(index);
      keyMetadataColumns.splice(idx, 1);
    }
    
    this.config = { ...this.config, key_metadata_columns: keyMetadataColumns };
  }

  toggleMetadataColumn(index, checked) {
    const metadataColumns = [...this.config.metadata_columns];
    
    if (checked && !metadataColumns.includes(index)) {
      metadataColumns.push(index);
    } else if (!checked && metadataColumns.includes(index)) {
      const idx = metadataColumns.indexOf(index);
      metadataColumns.splice(idx, 1);
    }
    
    this.config = { ...this.config, metadata_columns: metadataColumns };
  }

  setHasHeader(checked) {
    this.config = { ...this.config, has_header: checked };
  }

  isConfigValid() {
    return this.config.text_column !== null && 
           this.config.id_column !== null;
  }

  cancel() {
    this.dispatch('cancel');
  }

  async submitConfig() {
    if (!this.isConfigValid()) {
      this.error = 'Please select both a text column and an ID column.';
      return;
    }

    this.error = '';
    this.loading = true;
    
    try {
      const formData = new FormData();
      formData.append('config', JSON.stringify(this.config));
      formData.append('temp_path', this.previewData.temp_path);
      
      const response = await fetch(`/api/kb/${this.kbName}/csv/upload`, {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (result.success) {
        this.dispatch('upload-started', { taskId: result.task_id });
      } else {
        this.error = result.message || 'Failed to process CSV';
      }
    } catch (error) {
      console.error('Error submitting CSV config:', error);
      this.error = error.message || 'An error occurred while processing the CSV';
    } finally {
      this.loading = false;
    }
  }
}

customElements.define('csv-config', CsvConfig);