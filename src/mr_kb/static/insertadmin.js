function insertKBSettings() {
  const agentEditor = document.querySelector('agent-editor');
  const agentForm = agentEditor.shadowRoot.querySelector('agent-form');
  if (agentForm && agentForm.shadowRoot) {
    const kbSettings = document.createElement('kb-settings');
    const agentInsertEnd = agentForm.shadowRoot.querySelectorAll('.agent-insert-end')[0];
    if (agentInsertEnd) {
      agentInsertEnd.insertAdjacentElement('afterend', kbSettings);
      console.log('KB settings component inserted into agent form');
    } else {
      console.warn('Could not find agent-insert-end in agent form');
    }
  } else {
    console.warn('Agent form not found or shadow root not accessible');
  }
}

const agentEditor = document.querySelector('agent-editor');
const shadowRoot = agentEditor.shadowRoot;
if (shadowRoot) {
  console.log("FOUND SHADOW ROOT for agent editor")
}

setTimeout(insertKBSettings, 1000)
