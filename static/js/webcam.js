document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcamVideo');
    const startCamBtn = document.getElementById('startCamBtn');
    const stopCamBtn = document.getElementById('stopCamBtn');
    const captureBtn = document.getElementById('captureBtn');
    const webcamResult = document.getElementById('webcamResult');
    const canvas = document.createElement('canvas');

    let stream = null;
    let autoInterval = null;

    if (!video || !startCamBtn) return;

    startCamBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' }
            });
            video.srcObject = stream;
            video.style.display = 'block';
            startCamBtn.style.display = 'none';
            stopCamBtn.style.display = 'inline-flex';
            captureBtn.style.display = 'inline-flex';
        } catch (err) {
            console.error('Camera access denied:', err);
            alert('Unable to access camera. Please allow camera permissions in your browser or try image upload.');
        }
    });

    stopCamBtn.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        if (autoInterval) {
            clearInterval(autoInterval);
            autoInterval = null;
        }
        video.style.display = 'none';
        startCamBtn.style.display = 'inline-flex';
        stopCamBtn.style.display = 'none';
        captureBtn.style.display = 'none';
    });

    captureBtn.addEventListener('click', async () => {
        if (!stream) return;

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const base64Data = canvas.toDataURL('image/jpeg', 0.85);

        captureBtn.disabled = true;
        captureBtn.innerText = 'Analyzing Frame...';

        try {
            const response = await fetch('/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Data })
            });
            const data = await response.json();
            if (data.success) {
                renderWebcamResult(data.result);
            }
        } catch (err) {
            console.error('Webcam inference error:', err);
        } finally {
            captureBtn.disabled = false;
            captureBtn.innerText = '📸 Capture & Classify';
        }
    });

    function renderWebcamResult(result) {
        if (!webcamResult) return;
        webcamResult.style.display = 'block';
        const isRecyclable = result.category.toLowerCase() === 'recyclable';
        const badgeClass = isRecyclable ? 'badge-recyclable' : 'badge-hazardous';

        webcamResult.innerHTML = `
            <div class="glass-card" style="border-left: 4px solid ${result.color}; margin-top: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <div class="badge ${badgeClass}">${result.title}</div>
                    <span style="font-weight: 700; color: ${result.color}; font-size: 1.1rem;">${result.confidence}% Match</span>
                </div>
                <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.5rem;">
                    <strong>Recommended Disposal:</strong> ${result.disposal_instructions}
                </p>
                <div style="font-size: 0.8rem; color: var(--text-dim);">
                    ⚡ Inference: ${result.inference_time_ms} ms | Bin: ${result.bin_type}
                </div>
            </div>
        `;
        webcamResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
