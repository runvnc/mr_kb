function insertKBSettings() {
  const agentEditor = document.querySelector('agent-editor');
  const agentForm = agentEditor.shadowRoot.querySelector('agent-form');
    if (agentForm && agentForm.shadowRoot) {
      // Load our component script if not already loaded
      
      // Create our KB settings component
      const kbSettings = document.createElement('kb-settings');
      // actually, just add it to the end of the form right before the Save button
      agentForm.shadowRoot.appendChild(kbSettings);

      console.log('KB settings component inserted into agent form');
      kbSettings.style.border = "3px solid red"
      kbSettings.style.height = "250px";
      kbSettings.style.width = "400px";
      /*
      const commandsSection = agentForm.shadowRoot.querySelector('.commands-section');
      if (commandsSection) {
        // Insert our component before the commands section
        commandsSection.parentNode.insertBefore(kbSettings, commandsSection);
        console.log('KB settings component inserted into agent form');
      } else {
        console.warn('Could not find commands section in agent form');
      }*/
    } else {
      console.warn('Agent form not found or shadow root not accessible');
    }
}

const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        console.log("kbsettings mutation observer")
        console.log(mutation)
        const editor = document.querySelector('agent-editor');
        const form = editor.shadowRoot.querySelector('agent-form');
        if (form) {
          setTimeout(insertKBSettings, 100)
        }
    })
})

console.log("hi from insert admin kb settings agent kb")
const agentEditor = document.querySelector('agent-editor');
console.log({agentEditor})
const shadowRoot = agentEditor.shadowRoot;
console.log({shadowRoot})
if (shadowRoot) {
  console.log("FOUND SHADOW ROOT for agent editor")
}
insertKBSettings();

observer.observe(agentEditor.shadowRoot, {
    childList: true,
    subtree: true
})
