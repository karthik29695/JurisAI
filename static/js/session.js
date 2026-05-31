    // ─── Session Management ──────────────────────────────────────────────────

    // Poll session TTL every 30 seconds and display countdown in nav
    let _sessionPollTimer = null;

    function formatTTL(seconds) {
        if (seconds < 0) return 'Expired';
        const m = String(Math.floor(seconds / 60)).padStart(2, '0');
        const s = String(seconds % 60).padStart(2, '0');
        return `${m}:${s}`;
    }

    async function pollSessionInfo() {
        try {
            const res  = await fetch('/session/info');
            const data = await res.json();
            const badge = document.getElementById('session-badge');
            const label = document.getElementById('session-ttl-label');
            if (!badge || !label) return;

            const ttl = data.ttl_seconds ?? -1;
            label.textContent = formatTTL(ttl);
            badge.classList.remove('hidden');
            badge.classList.add('flex');

            // Warn user when session is about to expire (< 3 minutes)
            if (ttl > 0 && ttl < 180) {
                badge.classList.add('border-orange-400', 'text-orange-500');
                badge.classList.remove('border-slate-200', 'dark:border-slate-700');
                if (ttl < 60) {
                    badge.classList.add('border-red-400', 'text-red-500');
                    badge.classList.remove('border-orange-400', 'text-orange-500');
                }
            }

            // Session expired — clear UI and notify
            if (ttl === -1 || ttl === 0) {
                clearInterval(_sessionPollTimer);
                badge.classList.remove('flex');
                badge.classList.add('hidden');
                addBotMessage('⏰ <strong>Your session has expired.</strong> All document data has been automatically deleted for your privacy. Please refresh the page to start a new session.');
                disablePanels();
                resetUpload();
            }
        } catch (e) {
            // silently ignore network errors during polling
        }
    }

    function startSessionPoller() {
        pollSessionInfo();                              // immediate first check
        _sessionPollTimer = setInterval(pollSessionInfo, 30_000);  // every 30s
    }

    async function endSession() {
        const confirmed = confirm(
            '🔒 Clear Session\n\n' +
            'This will permanently delete all uploaded document data from the server.\n\n' +
            'Are you sure?'
        );
        if (!confirmed) return;

        try {
            await fetch('/session/end', { method: 'POST' });
        } catch (e) { /* ignore */ }

        // Reset UI
        clearInterval(_sessionPollTimer);
        resetUpload();
        disablePanels();

        const badge = document.getElementById('session-badge');
        if (badge) { badge.classList.remove('flex'); badge.classList.add('hidden'); }

        addBotMessage('🛡️ <strong>Session cleared.</strong> All document embeddings and data have been securely deleted from the server. Refresh the page to start a new session.');

        // Brief delay then reload to get a fresh session cookie
        setTimeout(() => window.location.reload(), 3000);
    }

    function disablePanels() {
        const summaryPanel = document.getElementById('summary-panel');
        const riskPanel    = document.getElementById('risk-panel');
        if (summaryPanel) summaryPanel.classList.add('opacity-40', 'pointer-events-none');
        if (riskPanel)    riskPanel.classList.add('opacity-40', 'pointer-events-none');
    }

    // Start polling after DOM ready
    startSessionPoller();

