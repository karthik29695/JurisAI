    // ─── TTS via backend /tts (gTTS) ────────────────────────────────
    let currentAudio = null;

    function stopTTS() {
        if (currentAudio) { currentAudio.pause(); currentAudio = null; }
        window.speechSynthesis?.cancel();
        document.getElementById('stop-tts-btn')?.classList.add('hidden');
        document.querySelectorAll('#chat-messages .fa-stop').forEach(icon => {
            icon.classList.remove('fa-stop','text-neon-blue','dark:text-neon-cyan');
            icon.classList.add('fa-volume-high');
        });
    }

    async function speakText(btnEl) {
        const icon = btnEl.querySelector('i');
        const isPlaying = icon.classList.contains('fa-stop');
        stopTTS();
        if (isPlaying) return;

        // Extract plain text from the sibling <p>
        const textEl = btnEl.closest('div.bg-white, div.bg-neon-blue')?.querySelector('p');
        if (!textEl) return;
        const rawText = textEl.innerText.trim();
        if (!rawText) return;

        icon.classList.remove('fa-volume-high');
        icon.classList.add('fa-stop','text-neon-blue');
        document.getElementById('stop-tts-btn')?.classList.remove('hidden');

        try {
            const res = await fetch(`${BASE_URL}/tts`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: rawText, lang: currentLanguageCode })
            });
            if (!res.ok) throw new Error('TTS failed');
            const blob = await res.blob();
            const url  = URL.createObjectURL(blob);
            currentAudio = new Audio(url);
            currentAudio.onended = stopTTS;
            currentAudio.onerror = stopTTS;
            currentAudio.play();
        } catch (e) {
            // Fallback: browser speechSynthesis
            const utt = new SpeechSynthesisUtterance(rawText);
            const langMap = { en:'en-US', hi:'hi-IN', te:'te-IN', bn:'bn-IN', mr:'mr-IN', ta:'ta-IN', gu:'gu-IN', kn:'kn-IN', ml:'ml-IN' };
            utt.lang = langMap[currentLanguageCode] || 'en-US';
            utt.onend = stopTTS;
            utt.onerror = stopTTS;
            window.speechSynthesis.speak(utt);
        }
    }

    // ─── Voice Input (Speech-to-Text) ────────────────────────────────
    function toggleVoice() {
        isRecording = !isRecording;
        const overlay  = document.getElementById('voice-overlay');
        const micBtn   = document.getElementById('mic-btn');

        if (isRecording) {
            overlay.classList.remove('hidden');
            overlay.classList.add('flex');
            micBtn?.classList.add('text-red-500');

            // Use Web Speech API if available
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (SpeechRecognition) {
                recognition = new SpeechRecognition();
                recognition.lang = currentLanguageCode === 'en' ? 'en-US' : currentLanguageCode;
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;

                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    chatInput.value = transcript;
                    stopVoice();
                };
                recognition.onerror = () => stopVoice();
                recognition.onend   = () => { if (isRecording) stopVoice(); };
                recognition.start();
            } else {
                // No speech API — auto-stop after 3s
                setTimeout(stopVoice, 3000);
            }
        } else {
            stopVoice();
        }
    }

    function stopVoice() {
        isRecording = false;
        recognition?.stop();
        document.getElementById('voice-overlay').classList.add('hidden');
        document.getElementById('voice-overlay').classList.remove('flex');
        document.getElementById('mic-btn')?.classList.remove('text-red-500');
    }


