document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const classifyBtn = document.getElementById('classifyBtn');
    const clearBtn = document.getElementById('clearBtn');
    const resultContainer = document.getElementById('resultContainer');
    const resultPlaceholder = document.getElementById('resultPlaceholder');

    let currentFile = null;

    if (!dropzone || !fileInput) return;

    // Trigger file chooser
    dropzone.addEventListener('click', (e) => {
        if (e.target !== classifyBtn && e.target !== clearBtn) {
            fileInput.click();
        }
    });

    // Drag & Drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelected(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelected(e.target.files[0]);
        }
    });

    function handleFileSelected(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WEBP).');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewContainer.style.display = 'block';
            classifyBtn.disabled = false;
            classifyBtn.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        };
        reader.readAsDataURL(file);
    }

    if (classifyBtn) {
        classifyBtn.addEventListener('click', async () => {
            if (!currentFile) return;
            
            classifyBtn.disabled = true;
            classifyBtn.innerHTML = `
                <svg class="animate-spin" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 1s linear infinite;">
                    <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                    <path d="M12 2a10 10 0 0 1 10 10"></path>
                </svg> Classifying with CNN...
            `;

            const formData = new FormData();
            formData.append('file', currentFile);

            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.success) {
                    displayResult(data.result);
                } else {
                    alert('Classification failed: ' + (data.error || 'Unknown error'));
                }
            } catch (err) {
                console.error(err);
                alert('Server connection error. Please ensure Flask app is running.');
            } finally {
                classifyBtn.disabled = false;
                classifyBtn.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                    Classify Image
                `;
            }
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            currentFile = null;
            fileInput.value = '';
            previewContainer.style.display = 'none';
            previewImage.src = '';
            if (resultContainer) resultContainer.style.display = 'none';
            if (resultPlaceholder) resultPlaceholder.style.display = 'block';
        });
    }

    function displayResult(result) {
        if (resultPlaceholder) resultPlaceholder.style.display = 'none';
        if (resultContainer) {
            resultContainer.style.display = 'block';
            
            const isRecyclable = result.category.toLowerCase() === 'recyclable';
            const badgeClass = isRecyclable ? 'badge-recyclable' : 'badge-hazardous';
            const iconSvg = isRecyclable 
                ? `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>`
                : `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;

            resultContainer.innerHTML = `
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: var(--text-dim); margin-bottom: 0.4rem; letter-spacing: 0.05em;">Predicted Classification</div>
                    <div class="badge ${badgeClass}" style="font-size: 1.15rem; padding: 0.6rem 1.5rem; margin-bottom: 0.75rem;">
                        ${iconSvg}
                        <span>${result.title}</span>
                    </div>
                    <div style="color: var(--text-muted); font-size: 0.9rem;">
                        Recommended Bin: <strong style="color: #fff;">${result.bin_type}</strong>
                    </div>
                </div>

                <div class="confidence-bar-container">
                    <div class="confidence-header">
                        <span>Classification Confidence</span>
                        <span style="color: ${result.color}; font-weight: 700;">${result.confidence}%</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" id="confFill" style="width: 0%; background: ${result.color};"></div>
                    </div>
                </div>

                <div class="disposal-guide-card" style="border-left-color: ${result.color};">
                    <div class="guide-title" style="color: #fff; display: flex; align-items: center; gap: 0.5rem;">
                        <span>📋 Proper Disposal Guidance</span>
                    </div>
                    <p class="guide-text">${result.disposal_instructions}</p>
                    <div style="margin-top: 0.75rem; font-size: 0.82rem; color: var(--text-dim);">
                        🌱 <em>${result.environmental_impact}</em>
                    </div>
                </div>

                <div style="margin-top: 1.25rem; display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-dim); border-top: 1px solid var(--border-card); padding-top: 0.75rem;">
                    <span>⚡ Inference Speed: <strong>${result.inference_time_ms} ms</strong></span>
                    <span>🕒 ${result.timestamp}</span>
                </div>
            `;

            // Trigger smooth width transition
            setTimeout(() => {
                const fill = document.getElementById('confFill');
                if (fill) fill.style.width = `${result.confidence}%`;
            }, 100);

            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
});

// CSS spin keyframe injection
const styleSheet = document.createElement("style");
styleSheet.innerText = `@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`;
document.head.appendChild(styleSheet);
