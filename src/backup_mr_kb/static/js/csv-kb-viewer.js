import { LitElement, html, css } from '/admin/static/js/lit-core.min.js';
import { BaseEl } from '/admin/static/js/base.js';

class CsvKbViewer extends BaseEl {
  static properties = {
    rows: { type: Array },
    columns: { type: Array },
    config: { type: Object },
    loading: { type: Boolean }
  };

  static styles = css`
    :host {
      display: block;
      padding: 1rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      padding: 0.5rem;
      border: 1px solid rgba(255, 255, 255, 0.1);
      text-align: left;
    }
    th {
      background: rgba(255, 255, 255, 0.05);
    }
    .actions button {
      margin-right: 0.5rem;
    }
  `;

  constructor() {
    super();
    this.rows = [];
    this.columns = [];
    this.config = {};
    this.loading = false;
  }

  render() {
    return html`
      <div>
        <h3>CSV Knowledge Base Viewer</h3>
        ${this.loading ? html`<p>Loading...</p>` : html`
          <table>
            <thead>
              <tr>
                ${this.columns.map(col => html`<th>${col}</th>`)}
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              ${this.rows.map(row => html`
                <tr>
                  ${this.columns.map(col => html`<td>${row[col]}</td>`)}
                  <td class="actions">
                    <button @click=${() => this.editRow(row)}>Edit</button>
                    <button @click=${() => this.deleteRow(row)}>Delete</button>
                  </td>
                </tr>
              `)}
            </tbody>
          </table>
        `}
      </div>
    `;
  }

  editRow(row) {
    console.log('Edit row:', row);
    // Placeholder for edit logic
  }

  deleteRow(row) {
    console.log('Delete row:', row);
    // Placeholder for delete logic
  }
}

customElements.define('csv-kb-viewer', CsvKbViewer);