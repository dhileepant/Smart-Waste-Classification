// EcoSort AI - Core Dashboard & Classification Logic
document.addEventListener('DOMContentLoaded', () => {
    // ----------------- Classification Dashboard Elements -----------------
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('fileInput');
    const previewCard = document.getElementById('previewCard');
    const previewImage = document.getElementById('previewImage');
    const scanLine = document.getElementById('scanLine');
    const classifyBtn = document.getElementById('classifyBtn');
    const clearBtn = document.getElementById('clearBtn');
    const resultStudio = document.getElementById('resultStudio');
    const idleState = document.getElementById('idleState');
    const sampleChips = document.querySelectorAll('.sample-chip');

    let currentFile = null;
    let currentSamplePath = null;

    // Trigger file chooser on dropzone click (unless button clicked)
    if (dropzone && fileInput) {
        dropzone.addEventListener('click', (e) => {
            if (!e.target.closest('.sample-chip') && !e.target.closest('#classifyBtn') && !e.target.closest('#clearBtn')) {
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
            if (files && files.length > 0) {
                handleFileSelected(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelected(e.target.files[0]);
            }
        });
    }

    // Handle Local File Selection
    function handleFileSelected(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WEBP).');
            return;
        }
        currentFile = file;
        currentSamplePath = null;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewCard.style.display = 'block';
            if (classifyBtn) classifyBtn.disabled = false;
            previewCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        };
        reader.readAsDataURL(file);
    }

    // Handle Quick Sample Chip Clicks
    if (sampleChips) {
        sampleChips.forEach(chip => {
            chip.addEventListener('click', async (e) => {
                e.stopPropagation();
                const samplePath = chip.getAttribute('data-sample');
                if (!samplePath) return;

                currentSamplePath = samplePath;
                currentFile = null;
                previewImage.src = samplePath;
                previewCard.style.display = 'block';
                if (classifyBtn) classifyBtn.disabled = false;

                // Auto-run classification for instant demo delight
                runClassification();
            });
        });
    }

    // Clear Preview
    if (clearBtn) {
        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            currentFile = null;
            currentSamplePath = null;
            if (fileInput) fileInput.value = '';
            if (previewCard) previewCard.style.display = 'none';
            if (previewImage) previewImage.src = '';
            if (resultStudio) resultStudio.style.display = 'none';
            if (idleState) idleState.style.display = 'block';
            if (scanLine) scanLine.style.display = 'none';
        });
    }

    // Trigger Classification
    if (classifyBtn) {
        classifyBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            runClassification();
        });
    }

    async function runClassification() {
        if (!currentFile && !currentSamplePath) return;

        // Visual loading state
        if (classifyBtn) {
            classifyBtn.disabled = true;
            classifyBtn.innerHTML = `
                <svg class="spin-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                    <path d="M12 2a10 10 0 0 1 10 10"></path>
                </svg> Analyzing Tensor...
            `;
        }
        if (scanLine) scanLine.style.display = 'block';

        try {
            let response;
            if (currentFile) {
                const formData = new FormData();
                formData.append('file', currentFile);
                response = await fetch('/api/classify', {
                    method: 'POST',
                    body: formData
                });
            } else if (currentSamplePath) {
                response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sample_path: currentSamplePath })
                });
            }

            const data = await response.json();
            if (data.success && data.result) {
                renderClassificationResult(data.result);
                updateLiveStats();
            } else {
                alert('Classification error: ' + (data.error || 'Server error'));
            }
        } catch (err) {
            console.error('Classification request failed:', err);
            alert('Failed to connect to Flask inference backend.');
        } finally {
            if (classifyBtn) {
                classifyBtn.disabled = false;
                classifyBtn.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg> Run AI Classification
                `;
            }
            if (scanLine) scanLine.style.display = 'none';
        }
    }

    // Render Polished AI Results
    function renderClassificationResult(result) {
        if (idleState) idleState.style.display = 'none';
        if (!resultStudio) return;

        resultStudio.style.display = 'block';

        const isRecyclable = result.category.toLowerCase() === 'recyclable';
        const badgeClass = isRecyclable ? 'badge-recyclable' : 'badge-hazardous';
        const iconSvg = isRecyclable
            ? `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>`
            : `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;

        resultStudio.innerHTML = `
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 0.76rem; text-transform: uppercase; color: var(--text-tertiary); font-weight: 700; letter-spacing: 0.08em; margin-bottom: 0.5rem;">
                    Inference Output
                </div>
                <div class="badge ${badgeClass}" style="font-size: 1.15rem; padding: 0.65rem 1.6rem; margin-bottom: 0.85rem;">
                    ${iconSvg}
                    <span>${result.title}</span>
                </div>
                <div style="font-size: 0.94rem; color: var(--text-secondary);">
                    Target Sorting Destination: <strong style="color: #fff;">${result.bin_type}</strong>
                </div>
            </div>

            <!-- Confidence Progress Studio -->
            <div class="confidence-studio-card">
                <div class="progress-header-wrap">
                    <span>Model Confidence</span>
                    <span style="color: ${result.color}; font-size: 1.1rem; font-weight: 800;">${result.confidence}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-bar-glow" id="confProgressFill" style="background: ${result.color}; width: 0%;"></div>
                </div>
            </div>

            <!-- Actionable Disposal Guide -->
            <div class="guide-box" style="border-left-color: ${result.color};">
                <div class="guide-box-title">
                    <span>📋 Disposal Protocol & Segregation</span>
                </div>
                <p class="guide-box-body">${result.disposal_instructions}</p>
                <div style="margin-top: 0.75rem; font-size: 0.82rem; color: var(--text-tertiary);">
                    🌿 <strong>Eco Impact:</strong> ${result.environmental_impact}
                </div>
            </div>

            <!-- Telemetry Footer -->
            <div class="meta-footer-row">
                <span>⚡ Latency: <strong style="color: var(--secondary);">${result.inference_time_ms} ms</strong></span>
                <span>🕒 ${result.timestamp}</span>
            </div>
        `;

        // Trigger smooth progress bar width animation
        setTimeout(() => {
            const fill = document.getElementById('confProgressFill');
            if (fill) fill.style.width = `${result.confidence}%`;
        }, 80);

        resultStudio.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Dynamic KPI stats updater
    async function updateLiveStats() {
        try {
            const res = await fetch('/api/stats');
            const stats = await res.json();
            const totalEl = document.getElementById('statTotal');
            const recEl = document.getElementById('statRecyclable');
            const hazEl = document.getElementById('statHazardous');
            const latEl = document.getElementById('statLatency');

            if (totalEl) totalEl.innerText = stats.total_classified;
            if (recEl) recEl.innerText = stats.recyclable_count;
            if (hazEl) hazEl.innerText = stats.hazardous_count;
            if (latEl) latEl.innerText = stats.average_latency_ms + ' ms';
        } catch (e) {
            console.error('Stats refresh error:', e);
        }
    }

    // ----------------- History Page Search & Filter -----------------
    const searchInput = document.getElementById('historySearch');
    const filterChips = document.querySelectorAll('.filter-chip');
    const historyRows = document.querySelectorAll('.history-row');

    if (searchInput && historyRows.length > 0) {
        searchInput.addEventListener('input', applyHistoryFilters);
    }

    if (filterChips && filterChips.length > 0) {
        filterChips.forEach(chip => {
            chip.addEventListener('click', () => {
                filterChips.forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
                applyHistoryFilters();
            });
        });
    }

    function applyHistoryFilters() {
        const query = (searchInput ? searchInput.value : '').toLowerCase().trim();
        const activeFilterChip = document.querySelector('.filter-chip.active');
        const filterVal = activeFilterChip ? activeFilterChip.getAttribute('data-filter') : 'all';

        historyRows.forEach(row => {
            const category = (row.getAttribute('data-category') || '').toLowerCase();
            const text = row.innerText.toLowerCase();

            const matchesQuery = !query || text.includes(query);
            const matchesCategory = filterVal === 'all' || category.includes(filterVal);

            row.style.display = (matchesQuery && matchesCategory) ? '' : 'none';
        });
    }
});

// Utility: Export History as JSON / CSV
function exportHistoryJSON() {
    fetch('/api/history')
        .then(r => r.json())
        .then(data => {
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            downloadBlob(blob, 'ecosort_classification_history.json');
        });
}

function exportHistoryCSV() {
    fetch('/api/history')
        .then(r => r.json())
        .then(data => {
            if (!data || data.length === 0) {
                alert('No history records to export.');
                return;
            }
            const headers = ['ID', 'Filename', 'Category', 'Confidence', 'Bin_Type', 'Latency_ms', 'Timestamp'];
            const rows = data.map(d => [
                d.id,
                `"${d.filename}"`,
                `"${d.category}"`,
                d.confidence,
                `"${d.bin_type}"`,
                d.inference_time_ms,
                `"${d.timestamp}"`
            ]);
            const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
            const blob = new Blob([csvContent], { type: 'text/csv' });
            downloadBlob(blob, 'ecosort_classification_history.csv');
        });
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Keyframe helper
const keyframeStyle = document.createElement('style');
keyframeStyle.innerHTML = `
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .spin-icon { animation: spin 0.9s linear infinite; }
`;
document.head.appendChild(keyframeStyle);
