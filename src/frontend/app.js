(() => {
  const chatArea  = document.getElementById('chatArea');
  const messages  = document.getElementById('messages');
  const welcome   = document.getElementById('welcome');
  const input     = document.getElementById('userInput');
  const sendBtn   = document.getElementById('sendBtn');
  const statusDot = document.getElementById('statusDot');
  const statusTxt = document.getElementById('statusText');

  let isStreaming = false;

  /* ── Status helpers ──────────────────────────── */
  function setStatus(state) {
    // states: 'ready' | 'thinking' | 'streaming' | 'error'
    statusDot.className = 'status-indicator' + (state !== 'ready' ? ' active' : '');
    statusTxt.textContent = state;
  }

  /* ── Auto-resize textarea ────────────────────── */
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  });

  /* ── Send on Enter (Shift+Enter = newline) ───── */
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming) send();
    }
  });

  sendBtn.addEventListener('click', () => { if (!isStreaming) send(); });

  /* ── Timestamp ───────────────────────────────── */
  function now() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  /* ── Append a message bubble ─────────────────── */
  function appendMessage(role, text = '') {
    if (welcome && !welcome.classList.contains('hidden')) {
      welcome.classList.add('hidden');
    }

    const wrap = document.createElement('div');
    wrap.className = `msg ${role}`;

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = role === 'user' ? `you · ${now()}` : `assistant · ${now()}`;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;

    wrap.appendChild(meta);
    wrap.appendChild(bubble);
    messages.appendChild(wrap);
    scrollToBottom();

    return bubble; // return bubble so we can stream into it
  }

  /* ── Scroll to latest ────────────────────────── */
  function scrollToBottom() {
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
  }

  /* ── Main send ───────────────────────────────── */
  async function send() {
    const text = input.value.trim();
    if (!text) return;

    // Reset input
    input.value = '';
    input.style.height = 'auto';

    // Lock UI
    isStreaming = true;
    sendBtn.disabled = true;
    input.disabled = true;
    setStatus('thinking');

    // Show user message
    appendMessage('user', text);

    // Prepare bot bubble
    const botBubble = appendMessage('bot', '');
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    botBubble.appendChild(cursor);

    let fullText = '';

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(text),
        credentials: 'include',
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status} ${res.statusText}`);
      }

      setStatus('streaming');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        fullText += chunk;
        botBubble.textContent = fullText;
        botBubble.appendChild(cursor); // keep cursor at end
        scrollToBottom();
      }

      // Remove cursor when done
      cursor.remove();
      setStatus('ready');

    } catch (err) {
      cursor.remove();
      botBubble.closest('.msg').classList.add('error');
      botBubble.textContent = `Error: ${err.message}`;
      setStatus('error');
      setTimeout(() => setStatus('ready'), 3000);
    } finally {
      isStreaming = false;
      sendBtn.disabled = false;
      input.disabled = false;
      input.focus();
    }
  }

  // Focus input on load
  input.focus();
})();