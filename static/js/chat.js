    // ─── Risk Analysis ───────────────────────────────────────────────
    async function fetchRisk() {
        const findingsEl = document.getElementById('risk-findings');
        findingsEl.innerHTML = `<div class="flex items-center gap-2 text-neon-purple text-sm py-4"><i class="fa-solid fa-circle-notch spin"></i> Running NLP risk analysis…</div>`;
        resetRiskGauge();

        try {
            const res = await fetch(`${BASE_URL}/risk?lang=${currentLanguageCode}`);
            const data = await res.json();

            if (data.error) {
                findingsEl.innerHTML = `<p class="text-sm text-red-400">⚠️ ${data.error}</p>`;
                return;
            }

            renderRiskGauge(data.risk_score, data.risk_level);
            renderRiskFindings(data.findings);
            document.getElementById('risk-summary-text').innerText = data.summary || '';

            addBotMessage(`🛡️ Risk analysis complete. Overall risk: <strong>${data.risk_level}</strong> (score ${data.risk_score}/100). Found ${data.findings.length} clause(s) flagged.`);
        } catch (err) {
            findingsEl.innerHTML = `<p class="text-sm text-red-400">⚠️ Could not reach the server.</p>`;
        }
    }

    function resetRiskGauge() {
        document.getElementById('risk-circle').style.strokeDashoffset = '251.2';
        document.getElementById('risk-score-text').innerText = '--';
        document.getElementById('risk-level-label').innerText = '';
        document.getElementById('risk-summary-text').innerText = '';
    }

    function renderRiskGauge(score, level) {
        const colorMap = { High: '#ef4444', Medium: '#eab308', Low: '#22c55e' };
        const circle = document.getElementById('risk-circle');
        const scoreEl = document.getElementById('risk-score-text');
        const levelEl = document.getElementById('risk-level-label');

        circle.setAttribute('stroke', colorMap[level] || '#eab308');
        const offset = 251.2 * ((100 - score) / 100);
        circle.style.strokeDashoffset = offset;

        // Count-up animation
        let count = 0;
        const timer = setInterval(() => {
            count = Math.min(count + 2, score);
            scoreEl.innerText = count;
            if (count >= score) clearInterval(timer);
        }, 20);

        levelEl.innerText = level + ' Risk';
        levelEl.className = 'mt-2 text-sm font-semibold ' +
            (level === 'High' ? 'text-red-400' : level === 'Medium' ? 'text-yellow-400' : 'text-green-400');
    }

    function renderRiskFindings(findings) {
        const el = document.getElementById('risk-findings');
        if (!findings || findings.length === 0) {
            el.innerHTML = '<p class="text-sm text-slate-500 dark:text-slate-400">No risk clauses detected.</p>';
            return;
        }
        el.innerHTML = findings.map(f => {
            const badgeClass = `badge-${f.level}`;
            const borderClass = `risk-${f.level}`;
            const excerpt = f.excerpt ? `<p class="text-xs text-slate-400 mt-1 italic truncate" title="${escHtml(f.excerpt)}">${escHtml(f.excerpt.slice(0, 100))}…</p>` : '';
            return `
            <div class="p-3 rounded-lg bg-white dark:bg-slate-800 border-l-4 ${borderClass} flex flex-col gap-1 shadow-sm hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors">
                <div class="flex justify-between items-start gap-2">
                    <h3 class="text-sm font-semibold text-slate-800 dark:text-slate-200">${escHtml(f.label)}</h3>
                    <span class="px-2 py-0.5 rounded text-[10px] font-bold whitespace-nowrap ${badgeClass}">${f.level.toUpperCase()}</span>
                </div>
                <p class="text-xs text-slate-600 dark:text-slate-400">${escHtml(f.explanation)}</p>
                ${excerpt}
                <span class="text-[10px] text-slate-400 mt-1">Detected by: ${f.detected_by}</span>
            </div>`;
        }).join('');
    }

    function escHtml(str) {
        return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    // ─── Chat ────────────────────────────────────────────────────────
    const chatMessages = document.getElementById('chat-messages');
    const chatInput    = document.getElementById('chat-input');

    function fillInput(text) { chatInput.value = text; chatInput.focus(); }

    chatInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleChatSubmit(); } });

    async function handleChatSubmit() {
        const text = chatInput.value.trim();
        if (!text) return;
        addUserMessage(text);
        chatInput.value = '';

        showTypingIndicator();
        document.getElementById('chat-status').innerText = 'Thinking…';

        try {
            const res = await fetch(`${BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, lang: currentLanguageCode })
            });
            const data = await res.json();
            removeTypingIndicator();
            document.getElementById('chat-status').innerText = 'Online & Ready';

            if (data.error) {
                addBotMessage(`⚠️ ${data.error}`);
            } else {
                addBotMessage(data.reply || 'No response received.');
            }
        } catch (err) {
            removeTypingIndicator();
            document.getElementById('chat-status').innerText = 'Online & Ready';
            addBotMessage('⚠️ Could not reach the server. Is Flask running on port 5000?');
        }
    }

    function addUserMessage(text) {
        chatMessages.insertAdjacentHTML('beforeend', `
            <div class="flex gap-3 justify-end">
                <div class="bg-neon-blue text-white p-3 rounded-2xl rounded-tr-none max-w-[80%] shadow-md">
                    <p class="text-sm">${escHtml(text)}</p>
                </div>
            </div>`);
        scrollToBottom();
    }

    function addBotMessage(html) {
        chatMessages.insertAdjacentHTML('beforeend', `
            <div class="flex gap-3">
                <div class="w-8 h-8 rounded-full bg-slate-100 dark:bg-slate-700 flex items-center justify-center flex-shrink-0">
                    <i class="fa-solid fa-scale-balanced text-xs text-neon-blue dark:text-neon-cyan"></i>
                </div>
                <div class="bg-white dark:bg-slate-800 p-3 rounded-2xl rounded-tl-none border border-slate-200 dark:border-slate-700 max-w-[80%] shadow-sm">
                    <p class="text-sm text-slate-700 dark:text-slate-200">${html}</p>
                    <div class="mt-2 flex justify-end gap-2">
                        <button onclick="speakText(this)" class="text-slate-400 hover:text-neon-blue dark:hover:text-neon-cyan transition-colors opacity-60 hover:opacity-100" title="Read aloud">
                            <i class="fa-solid fa-volume-high text-xs"></i>
                        </button>
                    </div>
                </div>
            </div>`);
        scrollToBottom();
    }

    function showTypingIndicator() {
        chatMessages.insertAdjacentHTML('beforeend', `
            <div id="typing-indicator" class="flex gap-3">
                <div class="w-8 h-8 rounded-full bg-slate-100 dark:bg-slate-700 flex items-center justify-center flex-shrink-0">
                    <i class="fa-solid fa-scale-balanced text-xs text-neon-blue dark:text-neon-cyan"></i>
                </div>
                <div class="bg-white dark:bg-slate-800 p-3 rounded-2xl rounded-tl-none border border-slate-200 dark:border-slate-700 flex items-center gap-1 shadow-sm">
                    <div class="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
                    <div class="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
                    <div class="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
                </div>
            </div>`);
        scrollToBottom();
    }
    function removeTypingIndicator() { document.getElementById('typing-indicator')?.remove(); }

    function clearChat() {
        stopTTS();
        chatMessages.innerHTML = '';
        addBotMessage("Hello! I'm JurisAI. Upload a document or ask me a legal question to get started.");
    }

    function scrollToBottom() { chatMessages.scrollTop = chatMessages.scrollHeight; }

