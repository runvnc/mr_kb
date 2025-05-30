import { LitElement, html, css } from '/admin/static/js/lit-core.min.js';
import { BaseEl } from '/admin/static/js/base.js';

class KbSettings extends BaseEl {
  static properties = {
    agentName: { type: String },
    kbs: { type: Array },
    selectedKbs: { type: Array },
    loading: { type: Boolean }
  };

  static styles = css`
    :host {
      display: block;
      margin-top: 20px;
    }

    .kb-section {
      padding: 15px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.02);
      margin-bottom: 20px;
    }

    h3 {
      margin-top: 0;
      color: #f0f0f0;
      font-size: 1.1rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding-bottom: 8px;
    }

    .kb-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
      gap: 12px;
      margin-bottom: 30px;
    }

    .kb-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 6px;
    }

    .kb-info {
      flex: 1;
      margin-right: 12px;
    }

    .kb-name {
      color: #f0f0f0;
      font-weight: 500;
    }

    .kb-description {
      font-size: 0.85rem;
      color: rgba(255, 255, 255, 0.7);
    }

    .empty-state {
      text-align: center;
      padding: 20px;
      color: rgba(255, 255, 255, 0.7);
    }
  `;

  constructor() {
    super();
    console.log("Constructor for kb settings")
    this.kbs = [];
    this.selectedKbs = [];
    this.loading = true;
    this.agentName = '';
    this.boundHandleAgentSelected = this.handleAgentSelected.bind(this);
    this.eventListenersSetup = false;
  }

  connectedCallback() {
    super.connectedCallback();
    try {
       this.loadAgentName();
    } catch (error) {
      console.error('Error loading agent name:', error);
    }
    // Delay setup to ensure all components are ready
    setTimeout(() => {
      this.setupEventListeners();
    }, 100);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.removeEventListeners();
  }

  setupEventListeners() {
    if (this.eventListenersSetup) {
      console.log('Event listeners already set up for kb-settings');
      return;
    }
    
    console.log('%c Setting up event listeners for kb-settings', "color: blue; background: yellow;");
    
    // Listen for agent-selected events from the agent-form
    const agentEditor = document.querySelector('agent-editor');
    if (agentEditor && agentEditor.shadowRoot) {
      const agentForm = agentEditor.shadowRoot.querySelector('agent-form');
      if (agentForm) {
        agentForm.addEventListener('agent-selected', this.boundHandleAgentSelected);
        console.log('Added agent-selected event listener to agent-form');
        this.eventListenersSetup = true;
      } else {
        console.warn('Could not find agent-form in agent-editor shadow root');
        // Try again later
        setTimeout(() => {
          this.setupEventListeners();
        }, 500);
      }
    } else {
      console.warn('Could not find agent-editor or its shadow root');
      // Try again later
      setTimeout(() => {
        this.setupEventListeners();
      }, 500);
    }
  }

  removeEventListeners() {
    if (!this.eventListenersSetup) return;
    
    const agentEditor = document.querySelector('agent-editor');
    if (agentEditor && agentEditor.shadowRoot) {
      const agentForm = agentEditor.shadowRoot.querySelector('agent-form');
      if (agentForm) {
        agentForm.removeEventListener('agent-selected', this.boundHandleAgentSelected);
        console.log('Removed agent-selected event listener from agent-form');
        this.eventListenersSetup = false;
      }
    }
  }

  handleAgentSelected(event) {
    console.log('%c Agent selected event received:', "color: green;", event.detail);
    const agent = event.detail;
    if (agent && agent.name) {
      const newAgentName = agent.name;
      console.log(`Agent selected: ${newAgentName}`);
      if (newAgentName !== this.agentName) {
        console.log(`Agent changed from ${this.agentName} to ${newAgentName}`);
        this.agentName = newAgentName;
        this.fetchKnowledgeBases();
        this.fetchAgentKbSettings();
      } else {
        console.log(`Agent name remains the same: ${this.agentName}`);
      }
    } else {
      console.log("No agent name found in event detail");
      this.agentName = '';
      this.selectedKbs = [];
      this.requestUpdate();
    }
  }

  loadAgentName() {
    // Find the agent name from the URL or parent component
    const url = new URL(window.location.href);
    const pathParts = url.pathname.split('/');
    const agentIndex = pathParts.indexOf('agents') + 1;
    
    if (agentIndex > 0 && agentIndex < pathParts.length) {
      this.agentName = pathParts[agentIndex];
      this.fetchKnowledgeBases();
      this.fetchAgentKbSettings();
    } else {
      // Try to get agent name from the form element
      setTimeout(() => {
        const agentEditor = document.querySelector('agent-editor');
        if (agentEditor && agentEditor.shadowRoot) {
          const agentForm = agentEditor.shadowRoot.querySelector('agent-form');
          if (agentForm && agentForm.agent && agentForm.agent.name) {
            this.agentName = agentForm.agent.name;
            this.fetchKnowledgeBases();
            this.fetchAgentKbSettings();
          } else {
            this.loading = false;
          }
        }
      }, 500);
    }
  }

  async fetchKnowledgeBases() {
    try {
      this.loading = true;
      const response = await fetch('/api/kb/list');
      if (!response.ok) throw new Error('Failed to fetch knowledge bases');
      const data = await response.json();
      if (data.success && data.data) {
        this.kbs = Object.values(data.data);
      } else {
        this.kbs = [];
      }
    } catch (error) {
      console.error('Error loading knowledge bases:', error);
      this.kbs = [];
    } finally {
      this.loading = false;
    }
  }

  async fetchAgentKbSettings() {
    if (!this.agentName) return;
    
    try {
      const response = await fetch(`/api/kb/agent/${this.agentName}/settings`);
      if (response.ok) {
        const data = await response.json();
        this.selectedKbs = data.kb_access || [];
      } else {
        this.selectedKbs = [];
      }
    } catch (error) {
      console.error('Error loading agent KB settings:', error);
      this.selectedKbs = [];
    }
  }

  async handleKbToggle(kbName, checked) {
    if (!this.agentName) return;
    
    // Update local state
    if (checked) {
      this.selectedKbs = [...this.selectedKbs, kbName];
    } else {
      this.selectedKbs = this.selectedKbs.filter(name => name !== kbName);
    }
    
    // Save to server
    try {
      const response = await fetch(`/api/kb/agent/${this.agentName}/settings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          kb_access: this.selectedKbs
        })
      });
      
      if (!response.ok) {
        console.error('Failed to save KB settings:', await response.text());
      }
    } catch (error) {
      console.error('Error saving KB settings:', error);
    }
  }

  _render() {
    console.log("Rendering kb settings for agent:", this.agentName);
    if (this.loading) {
      return html`<div class="kb-section"><p>Loading knowledge bases...</p></div>`;
    }

    if (!this.agentName) {
      return html`<div class="kb-section"><p>Please select an agent to configure knowledge base access.</p></div>`;
    }

    if (this.kbs.length === 0) {
      return html`
        <div class="kb-section">
          <h3>Knowledge Bases</h3>
          <div class="empty-state">
            <p>No knowledge bases available. Create knowledge bases in the KB admin section first.</p>
          </div>
        </div>
      `;
    }

    return html`
      <div class="kb-section">
        <h3>Knowledge Bases</h3>
        <p>Select which knowledge bases this agent can access:</p>
        
        <div class="kb-grid">
          ${this.kbs.map(kb => html`
            <div class="kb-item">
              <div class="kb-info">
                <div class="kb-name">${kb.name}</div>
                <div class="kb-description">${kb.description || ''}</div>
              </div>
              <toggle-switch 
                .checked=${this.selectedKbs.includes(kb.name)}
                @toggle-change=${(e) => this.handleKbToggle(kb.name, e.detail.checked)}>
              </toggle-switch>
            </div>
          `)}
        </div>
      </div>
    `;
  }
}

customElements.define('kb-settings', KbSettings);
