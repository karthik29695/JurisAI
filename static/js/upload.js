    // ─── Upload & File Handling ─────────────────────────────────────
    const dropZone   = document.getElementById('drop-zone');
    const fileInput  = document.getElementById('file-input');
    const uploadContent  = document.getElementById('upload-content');
    const uploadProgress = document.getElementById('upload-progress');
    const progressBar    = document.getElementById('progress-bar');
    const uploadStatus   = document.getElementById('upload-status');
    const fileInfo   = document.getElementById('file-info');
    const docActions = document.getElementById('doc-actions');

    ['dragenter','dragover','dragleave','drop'].forEach(ev => {
        dropZone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); });
        document.body.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); });
    });
    ['dragenter','dragover'].forEach(ev => dropZone.addEventListener(ev, () => dropZone.classList.add('border-neon-cyan')));
    ['dragleave','drop'].forEach(ev => dropZone.addEventListener(ev, () => dropZone.classList.remove('border-neon-cyan')));
    dropZone.addEventListener('drop', e => { handleFiles(e.dataTransfer.files); });
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function() { handleFiles(this.files); });

    // Accepted types
    const ACCEPTED_TYPES = {
        pdf:   ['.pdf'],
        docx:  ['.doc', '.docx'],
        image: ['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.tif', '.bmp']
    };

    function getFileCategory(filename) {
        const ext = '.' + filename.split('.').pop().toLowerCase();
        if (ACCEPTED_TYPES.pdf.includes(ext))   return 'pdf';
        if (ACCEPTED_TYPES.docx.includes(ext))  return 'docx';
        if (ACCEPTED_TYPES.image.includes(ext)) return 'image';
        return null;
    }

    function getFileIcon(category) {
        if (category === 'pdf')   return '<i class="fa-solid fa-file-pdf text-red-500 dark:text-red-400 text-xl"></i>';
        if (category === 'docx')  return '<i class="fa-solid fa-file-word text-blue-500 dark:text-blue-400 text-xl"></i>';
        if (category === 'image') return '<i class="fa-solid fa-file-image text-green-500 dark:text-green-400 text-xl"></i>';
        return '<i class="fa-solid fa-file text-slate-400 text-xl"></i>';
    }

    function handleFiles(files) {
        if (!files || files.length === 0) return;
        const file = files[0];
        const category = getFileCategory(file.name);

        if (!category) {
            addBotMessage('⚠️ Unsupported file type. Please upload a <strong>PDF</strong>, <strong>DOCX/DOC</strong>, or an image (<strong>JPG, PNG, TIFF, WEBP, BMP</strong>).');
            return;
        }

        uploadToBackend(file, category);
    }

    async function uploadToBackend(file, category) {
        // Show progress UI
        uploadContent.classList.add('hidden');
        uploadProgress.classList.remove('hidden');
        uploadProgress.classList.add('flex');
        progressBar.style.width = '10%';

        // Status messages differ by type
        const statusMessages = {
            pdf:   ['Uploading PDF…', 'Extracting text…', 'Indexing chunks…'],
            docx:  ['Uploading document…', 'Parsing DOCX…', 'Indexing chunks…'],
            image: ['Uploading image…', 'Running OCR…', 'Indexing text…']
        };
        const msgs = statusMessages[category];
        uploadStatus.innerText = msgs[0];

        const formData = new FormData();
        formData.append('file', file);

        // Animate progress bar while waiting, cycling status messages
        let fakeProgress = 10;
        let msgIdx = 0;
        const fakeTimer = setInterval(() => {
            if (fakeProgress < 80) {
                fakeProgress += 4;
                progressBar.style.width = fakeProgress + '%';
                if (fakeProgress > 35 && msgIdx < 1) { msgIdx = 1; uploadStatus.innerText = msgs[1]; }
                if (fakeProgress > 60 && msgIdx < 2) { msgIdx = 2; uploadStatus.innerText = msgs[2]; }
            }
        }, 350);

        // Choose the correct backend endpoint
        const endpoint = category === 'pdf' ? '/upload' : '/upload_file';

        try {
            const res = await fetch(`${BASE_URL}${endpoint}`, { method: 'POST', body: formData });
            const data = await res.json();
            clearInterval(fakeTimer);

            if (!res.ok || data.error) {
                showUploadError(data.error || 'Upload failed.');
                return;
            }

            // Success
            progressBar.style.width = '100%';
            uploadStatus.innerText = '✓ Done!';
            currentFilename = data.filename;

            setTimeout(() => {
                uploadProgress.classList.add('hidden');
                uploadProgress.classList.remove('flex');
                uploadContent.classList.remove('hidden');

                // Show file info with correct icon
                document.getElementById('file-info-icon').innerHTML = getFileIcon(category);
                document.getElementById('file-name').innerText = file.name;
                document.getElementById('file-size').innerText = formatBytes(file.size);
                fileInfo.classList.remove('hidden');
                fileInfo.classList.add('flex');

                docActions.classList.remove('hidden');
                docActions.classList.add('flex');
                enablePanels();

                const typeLabel = category === 'image' ? 'image (OCR applied)' : category.toUpperCase();
                addBotMessage(`✅ <strong>${file.name}</strong> uploaded as <em>${typeLabel}</em> and indexed successfully! You can now ask questions, get a summary, or run risk analysis.`);
            }, 600);

        } catch (err) {
            clearInterval(fakeTimer);
            showUploadError('Network error — is the Flask server running?');
        }
    }

    function showUploadError(msg) {
        uploadProgress.classList.add('hidden');
        uploadProgress.classList.remove('flex');
        uploadContent.classList.remove('hidden');
        progressBar.style.width = '0%';
        addBotMessage(`⚠️ ${msg}`);
    }

    function resetUpload() {
        if (currentFilename) {
            fetch(`${BASE_URL}/remove`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: currentFilename })
            }).catch(() => {});
            currentFilename = null;
        }
        fileInfo.classList.add('hidden');
        fileInfo.classList.remove('flex');
        docActions.classList.add('hidden');
        docActions.classList.remove('flex');
        uploadContent.classList.remove('hidden');
        progressBar.style.width = '0%';
        fileInput.value = '';
        disablePanels();
        resetRiskGauge();
        document.getElementById('summary-content').innerText = 'Upload a document and click Get Summary to see an AI-generated overview here.';
        addBotMessage('Document removed. Upload a new PDF to continue.');
    }

    function enablePanels() {
        document.getElementById('risk-dashboard').classList.remove('opacity-50', 'pointer-events-none');
        document.getElementById('summary-panel').classList.remove('opacity-50', 'pointer-events-none');
    }
    function disablePanels() {
        document.getElementById('risk-dashboard').classList.add('opacity-50', 'pointer-events-none');
        document.getElementById('summary-panel').classList.add('opacity-50', 'pointer-events-none');
    }
    function formatBytes(b) {
        if (b < 1024) return b + ' B';
        if (b < 1024*1024) return (b/1024).toFixed(1) + ' KB';
        return (b/(1024*1024)).toFixed(1) + ' MB';
    }

    // ─── Summary ─────────────────────────────────────────────────────
    async function fetchSummary() {
        const summaryLoading = document.getElementById('summary-loading');
        const summaryContent = document.getElementById('summary-content');
        summaryLoading.classList.remove('hidden');
        summaryLoading.classList.add('flex');
        summaryContent.innerText = '';

        try {
            const res = await fetch(`${BASE_URL}/summary?lang=${currentLanguageCode}`);
            const data = await res.json();
            summaryLoading.classList.add('hidden');
            summaryLoading.classList.remove('flex');

            if (data.error) {
                summaryContent.innerText = '⚠️ ' + data.error;
                return;
            }

            summaryContent.innerText = data.summary || 'No summary available.';
            // Also post in chat
            addBotMessage('📄 <strong>Summary ready!</strong> Check the Summary panel above or ask me specific questions.');
        } catch (err) {
            summaryLoading.classList.add('hidden');
            summaryLoading.classList.remove('flex');
            summaryContent.innerText = '⚠️ Could not reach the server.';
        }
    }

