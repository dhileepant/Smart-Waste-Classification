// EcoSort AI - Real-time Computer Vision & Webcam Engine
document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcamVideo');
    const startCamBtn = document.getElementById('startCamBtn');
    const stopCamBtn = document.getElementById('stopCamBtn');
    const captureBtn = document.getElementById('captureBtn');
    const autoToggleBtn = document.getElementById('autoToggleBtn');
    const browserPrompt = document.getElementById('browserPrompt');
    const cvDetectedCategory = document.getElementById('cvDetectedCategory');
    const cvConfidenceVal = document.getElementById('cvConfidenceVal');
    const cvProgressFill = document.getElementById('cvProgressFill');
    const cvLatencyVal = document.getElementById('cvLatencyVal');
    const cvBinVal = document.getElementById('cvBinVal');
    const snapshotsReel = document.getElementById('snapshotsReel');

    const canvas = document.createElement('canvas');
    let stream = null;
    let autoInterval = null;
    let isAutoDetecting = false;

    if (!video) return;

    // Start Browser Webcam
    if (startCamBtn) {
        startCamBtn.addEventListener('click', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' }
                });
                video.srcObject = stream;
                video.style.display = 'block';
                if (browserPrompt) browserPrompt.style.display = 'none';
                startCamBtn.style.display = 'none';
                stopCamBtn.style.display = 'inline-flex';
                captureBtn.style.display = 'inline-flex';
                if (autoToggleBtn) autoToggleBtn.style.display = 'inline-flex';
            } catch (err) {
                console.error('Camera access error:', err);
                alert('Camera access denied or unavailable. Please enable browser camera permissions.');
            }
        });
    }

    // Stop Browser Webcam
    if (stopCamBtn) {
        stopCamBtn.addEventListener('click', () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            if (autoInterval) {
                clearInterval(autoInterval);
                autoInterval = null;
                isAutoDetecting = false;
                if (autoToggleBtn) autoToggleBtn.innerText = '⚡ Enable Auto-Detect';
            }
            video.style.display = 'none';
            if (browserPrompt) browserPrompt.style.display = 'block';
            startCamBtn.style.display = 'inline-flex';
            stopCamBtn.style.display = 'none';
            captureBtn.style.display = 'none';
            if (autoToggleBtn) autoToggleBtn.style.display = 'none';
        });
    }

    // Snapshot & Classify
    if (captureBtn) {
        captureBtn.addEventListener('click', () => {
            processWebcamFrame();
        });
    }

    // Continuous Auto-Detect Toggle
    if (autoToggleBtn) {
        autoToggleBtn.addEventListener('click', () => {
            if (isAutoDetecting) {
                clearInterval(autoInterval);
                autoInterval = null;
                isAutoDetecting = false;
                autoToggleBtn.innerText = '⚡ Enable Auto-Detect';
                autoToggleBtn.className = 'btn btn-secondary';
            } else {
                isAutoDetecting = true;
                autoToggleBtn.innerText = '🛑 Stop Auto-Detect';
                autoToggleBtn.className = 'btn btn-danger';
                processWebcamFrame();
                autoInterval = setInterval(processWebcamFrame, 1500);
            }
        });
    }

    async function processWebcamFrame() {
        if (!stream || video.videoWidth === 0) return;

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const base64Data = canvas.toDataURL('image/jpeg', 0.85);

        // Highlight pipeline step 3 & 4
        highlightPipelineStage(3);

        try {
            const res = await fetch('/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Data, filename: 'live_stream_frame.jpg' })
            });
            const data = await res.json();

            if (data.success && data.result) {
                highlightPipelineStage(4);
                updateCVTelemetry(data.result);
                appendSnapshotToReel(base64Data, data.result);
            }
        } catch (e) {
            console.error('Frame processing failed:', e);
        }
    }

    function updateCVTelemetry(result) {
        if (cvDetectedCategory) {
            const isRec = result.category.toLowerCase() === 'recyclable';
            cvDetectedCategory.className = `badge ${isRec ? 'badge-recyclable' : 'badge-hazardous'}`;
            cvDetectedCategory.innerText = result.title;
        }
        if (cvConfidenceVal) {
            cvConfidenceVal.innerText = `${result.confidence}%`;
            cvConfidenceVal.style.color = result.color;
        }
        if (cvProgressFill) {
            cvProgressFill.style.width = `${result.confidence}%`;
            cvProgressFill.style.background = result.color;
        }
        if (cvLatencyVal) {
            cvLatencyVal.innerText = `${result.inference_time_ms} ms`;
        }
        if (cvBinVal) {
            cvBinVal.innerText = result.bin_type;
        }
    }

    function highlightPipelineStage(stageNum) {
        const steps = document.querySelectorAll('.pipeline-step-item');
        steps.forEach((s, idx) => {
            if (idx + 1 === stageNum) {
                s.classList.add('active');
            } else {
                s.classList.remove('active');
            }
        });
    }

    function appendSnapshotToReel(imgUri, result) {
        if (!snapshotsReel) return;
        const img = document.createElement('img');
        img.src = imgUri;
        img.className = 'thumb-preview-img';
        img.title = `${result.title} (${result.confidence}%)`;
        img.style.cursor = 'pointer';
        img.style.borderColor = result.color;
        
        snapshotsReel.insertBefore(img, snapshotsReel.firstChild);
        if (snapshotsReel.children.length > 8) {
            snapshotsReel.removeChild(snapshotsReel.lastChild);
        }
    }
});

// Viewport Switcher between Server OpenCV MJPEG and Browser Camera
function switchMode(mode) {
    const opencvDiv = document.getElementById('opencvContainer');
    const browserDiv = document.getElementById('browserContainer');
    const tabOpencv = document.getElementById('tabOpencv');
    const tabBrowser = document.getElementById('tabBrowser');

    if (mode === 'opencv') {
        if (opencvDiv) opencvDiv.style.display = 'flex';
        if (browserDiv) browserDiv.style.display = 'none';
        if (tabOpencv) { tabOpencv.className = 'btn btn-primary'; }
        if (tabBrowser) { tabBrowser.className = 'btn btn-secondary'; }
    } else {
        if (opencvDiv) opencvDiv.style.display = 'none';
        if (browserDiv) browserDiv.style.display = 'flex';
        if (tabOpencv) { tabOpencv.className = 'btn btn-secondary'; }
        if (tabBrowser) { tabBrowser.className = 'btn btn-primary'; }
    }
}
