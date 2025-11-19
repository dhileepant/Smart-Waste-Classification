import cv2
import time
import numpy as np
from src.evaluate import load_inference_model, predict_single_image
from src.utils import load_class_indices

class RealtimeWasteDetector:
    """
    OpenCV Real-Time Video Stream and Image Processing Pipeline for Waste Detection.
    """
    def __init__(self, model_path="models/waste_classifier.h5"):
        self.class_indices = load_class_indices()
        self.model = None
        self.model_path = model_path
        self._load_model()
        self.last_prediction = None
        self.last_inference_time = 0
        self.fps = 0
        self.prev_frame_time = 0

    def _load_model(self):
        try:
            self.model = load_inference_model(self.model_path)
            print(f"[OpenCV Stream] Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"[OpenCV Stream] Warning: Could not load model ({e}). Will run in simulation mode.")
            self.model = None

    def process_frame(self, frame):
        """
        Process a single OpenCV video frame, draw center ROI, compute inference,
        and overlay detection bounding box and analytics HUD.
        """
        h, w, _ = frame.shape
        curr_time = time.time()
        
        # Calculate FPS
        if self.prev_frame_time > 0:
            self.fps = 1.0 / (curr_time - self.prev_frame_time)
        self.prev_frame_time = curr_time

        # Define Center Region of Interest (ROI)
        roi_size = min(h, w) // 2
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        roi = frame[y1:y2, x1:x2]
        
        # Run inference periodically (every 0.25 seconds) to maintain smooth 30+ FPS
        if curr_time - self.last_inference_time > 0.25:
            if self.model is not None:
                try:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    self.last_prediction = predict_single_image(self.model, roi_rgb, self.class_indices)
                except Exception as e:
                    print(f"Inference error: {e}")
            else:
                # Simulation fallback if model file is not present yet
                self.last_prediction = {
                    "category": "recyclable",
                    "confidence": 94.5,
                    "title": "Recyclable Waste"
                }
            self.last_inference_time = curr_time

        # Overlay styles based on category
        if self.last_prediction:
            category = self.last_prediction.get("category", "recyclable").lower()
            conf = self.last_prediction.get("confidence", 0.0)
            
            if "recyclable" in category and "hazardous" not in category:
                box_color = (0, 200, 80) # Bright green in BGR
                label = f"RECYCLABLE ({conf:.1f}%)"
            else:
                box_color = (40, 40, 240) # Bright red in BGR
                label = f"HAZARDOUS / NON-RECYCLABLE ({conf:.1f}%)"
        else:
            box_color = (200, 200, 200)
            label = "Scanning waste..."

        # Draw ROI Bounding Box with rounded style corners
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        corner_len = 25
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 6)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 6)
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, 6)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, 6)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, 6)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, 6)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 6)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 6)

        # Header Badge
        badge_y = max(35, y1 - 15)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, badge_y - text_h - 10), (x1 + text_w + 20, badge_y + 6), box_color, -1)
        cv2.putText(frame, label, (x1 + 10, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # HUD Top Banner
        cv2.rectangle(frame, (10, 10), (260, 50), (20, 24, 33), -1)
        cv2.rectangle(frame, (10, 10), (260, 50), (60, 70, 90), 1)
        cv2.putText(frame, f"OpenCV Pipeline | FPS: {self.fps:.1f}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        return frame

def generate_video_stream(detector):
    """
    Generator function yielding multipart JPEG frames for Flask MJPEG video feed.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[OpenCV Stream] Warning: Camera (index 0) not available. Generating synthetic test stream.")
        # Synthetic video loop
        while True:
            synth_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(synth_frame, "Webcam Not Connected", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(synth_frame, "Use Image Upload / Browser Camera Mode", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 80), 2)
            processed = detector.process_frame(synth_frame)
            ret, buffer = cv2.imencode('.jpg', processed)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Mirror frame horizontally for natural webcam feel
        frame = cv2.flip(frame, 1)
        processed_frame = detector.process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

    cap.release()
