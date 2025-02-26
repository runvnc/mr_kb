document.addEventListener('DOMContentLoaded', function() {
  // Wait for the agent form to be fully loaded
  setTimeout(function() {
    const agentForm = document.querySelector('agent-form');
    if (agentForm && agentForm.shadowRoot) {
      // Load our component script if not already loaded
      if (!document.querySelector('script[src="/kb/static/js/kb-settings.js"]')) {
        const script = document.createElement('script');
        script.type = 'module';
        script.src = '/kb/static/js/kb-settings.js';
        document.head.appendChild(script);
      }
      
      // Create our KB settings component
      const kbSettings = document.createElement('kb-settings');
      
      // Find the commands section to insert before
      const commandsSection = agentForm.shadowRoot.querySelector('.commands-section');
      if (commandsSection) {
        // Insert our component before the commands section
        commandsSection.parentNode.insertBefore(kbSettings, commandsSection);
        console.log('KB settings component inserted into agent form');
      } else {
        console.warn('Could not find commands section in agent form');
      }
    } else {
      console.warn('Agent form not found or shadow root not accessible');
    }
  }, 1000);
});
