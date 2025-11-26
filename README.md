# ♻️ Smart Waste Classification System

An automated, real-time waste classification system built with **Python, TensorFlow, OpenCV, Flask, and HTML/CSS**. The system classifies waste into **Recyclable** and **Hazardous / Non-Recyclable** categories using a custom deep Convolutional Neural Network (CNN) and provides a real-time computer vision detection stream.

---

## 🌟 Key Highlights & Features (Resume Alignment)

- **Automated Waste Classification CNN:** Developed and trained a multi-block Convolutional Neural Network (CNN) in TensorFlow/Keras to categorize waste into recyclable and non-recyclable/hazardous types with high accuracy.
- **Real-Time OpenCV Image Processing Pipeline:** Built a low-latency image processing pipeline with OpenCV featuring Region-of-Interest (ROI) tracking, CLAHE contrast enhancement, color-coded dynamic bounding box overlays, and live FPS calculation.
- **Full-Stack Flask Deployment:** Deployed the trained model via a Flask web application with RESTful prediction APIs and live MJPEG streaming.
- **Hyperparameter Tuning & Data Augmentation:** Optimized model generalization and inference speed using spatial dropout, BatchNormalization, learning rate scheduling (`ReduceLROnPlateau`), and data augmentations (rotation, zoom, shear, horizontal flips).

---

## 🏗️ System Architecture

```
                                  +-----------------------+
                                  |   Raw Waste Image     |
                                  |  (Upload / Webcam)    |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |   OpenCV Pipeline     |
                                  | - ROI Extraction      |
                                  | - BGR -> RGB & CLAHE  |
                                  | - Resize (224x224)    |
                                  | - Normalization [0,1] |
                                  +-----------+-----------+
                                              |
                                              v
+-----------------------------------------------------------------------------------------+
|                                    Deep CNN Model                                       |
|                                                                                         |
|  [Input: 224x224x3]                                                                     |
|    |                                                                                    |
|    +--> Block 1: Conv2D(32) -> BatchNorm -> Conv2D(32) -> MaxPool -> Dropout(0.25)     |
|    |                                                                                    |
|    +--> Block 2: Conv2D(64) -> BatchNorm -> Conv2D(64) -> MaxPool -> Dropout(0.25)     |
|    |                                                                                    |
|    +--> Block 3: Conv2D(128) -> BatchNorm -> Conv2D(128) -> MaxPool -> Dropout(0.30)   |
|    |                                                                                    |
|    +--> Classifier: Flatten -> Dense(512, ReLU) -> BatchNorm -> Dropout(0.50)          |
|                                                                                         |
|  [Output: Dense(2, Softmax) -> Recyclable / Hazardous Probabilities]                    |
+-----------------------------------------------------------------------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |    Flask Web & API Server     |
                              | - REST API (/api/classify)    |
                              | - MJPEG Stream (/video_feed)  |
                              | - Interactive Web Dashboard   |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |  Client UI (HTML5/CSS3/JS)    |
                              | - Real-time Confidence Meters |
                              | - Disposal Guidance           |
                              | - Historical Analytics HUD    |
                              +-------------------------------+
```

---

## 📁 Repository Structure

```
Smart-Waste-Classification/
├── app.py                         # Flask web application & REST API server
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git ignore rules
├── README.md                      # Project documentation
│
├── data/                          # Dataset
│   ├── hazardous/                 # Hazardous / Non-recyclable images
│   └── recyclable/                # Recyclable waste images
│
├── models/                        # Saved models and class labels
│   ├── waste_classifier.h5        # Trained Keras CNN model weights
│   └── class_indices.json         # Class index mapping metadata
│
├── notebooks/                     # Interactive Jupyter Notebooks
│   ├── data_exploration.ipynb     # Exploratory data analysis & sample visualizer
│   └── model_training.ipynb       # Experimentation & hyperparameter tuning
│
├── src/                           # Core Machine Learning & CV Modules
│   ├── data_preprocessing.py      # OpenCV processing, normalization & augmentation
│   ├── model.py                   # CNN architecture definitions
│   ├── train.py                   # Model training pipeline & callbacks
│   ├── evaluate.py                # Evaluation metrics, confusion matrix & inference
│   ├── realtime_detector.py       # OpenCV real-time video stream processor
│   └── utils.py                   # Helper functions & disposal metadata
│
├── static/                        # Web Assets
│   ├── css/
│   │   └── style.css              # Modern glassmorphism CSS design system
│   ├── js/
│   │   ├── main.js                # Drag & drop upload and AJAX classification
│   │   └── webcam.js              # Browser camera stream and capture handling
│   └── img/                       # Training curves and confusion matrix plots
│
└── templates/                     # Flask Jinja2 HTML Templates
    ├── base.html                  # Master layout & responsive navigation
    ├── index.html                 # Main dashboard
    ├── realtime.html              # OpenCV real-time video stream view
    ├── history.html               # Classification activity logs & stats
    └── about.html                 # Architecture breakdown & specs
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.9+ / 3.10 / 3.11
- pip package manager

### 2. Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/dhileepant/Smart-Waste-Classification.git
cd Smart-Waste-Classification

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Running the Project

### Train the CNN Model
To train the CNN model on the dataset and save the trained weights:
```bash
python src/train.py
```
*Trained weights will be saved to `models/waste_classifier.h5` and training curves to `static/img/training_history.png`.*

### Evaluate Model Performance
To compute per-class precision, recall, F1-score, and generate the confusion matrix:
```bash
python src/evaluate.py
```

### Launch the Flask Web Application
To start the web server and access the interactive dashboard:
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 📡 RESTful API Documentation

### Classify Image (Multipart File Upload)
- **Endpoint:** `POST /api/classify`
- **Body:** `form-data` with key `file` (image file)
- **Sample Response:**
```json
{
  "success": true,
  "result": {
    "category": "recyclable",
    "title": "Recyclable Waste",
    "confidence": 98.42,
    "color": "#10b981",
    "bin_type": "Blue / Green Recycling Bin",
    "disposal_instructions": "Rinse containers to remove food residue. Flatten cardboard boxes and crush plastic bottles to save space before placing in the blue recycling bin.",
    "environmental_impact": "Recycling 1 ton of plastic saves up to 2,000 gallons of gasoline and reduces carbon emissions significantly.",
    "inference_time_ms": 11.45,
    "timestamp": "2025-11-20 16:32:10"
  }
}
```

### Classify Image (Base64 JSON Payload)
- **Endpoint:** `POST /api/classify`
- **Headers:** `Content-Type: application/json`
- **Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

### Session Analytics & Stats
- **Endpoint:** `GET /api/stats`
- **Sample Response:**
```json
{
  "total_classified": 28,
  "recyclable_count": 22,
  "hazardous_count": 6,
  "average_latency_ms": 12.3
}
```

---

## 🔬 Model Specifications & Hyperparameters

| Component | Specification |
| :--- | :--- |
| **Input Tensor Shape** | $224 \times 224 \times 3$ (RGB) |
| **Convolutional Layers** | 6 Conv2D layers ($3 \times 3$ kernels, filters: 32, 64, 128) |
| **Normalization & Regularization** | Batch Normalization on all blocks, Spatial Dropout ($0.25 - 0.50$) |
| **Optimizer** | Adam (Initial Learning Rate: $10^{-3}$, minimum: $10^{-6}$) |
| **Learning Rate Schedule** | `ReduceLROnPlateau` (factor: 0.3, patience: 3 epochs) |
| **Loss Function** | Categorical Cross-Entropy |
| **Augmentation** | $\pm 25^\circ$ rotation, $15\%$ width/height shift, $15\%$ shear, $20\%$ zoom, horizontal flip |
| **Inference Latency** | $\sim 11 - 15\text{ ms}$ on standard CPU |

---

## 🌿 Waste Categories & Disposal Matrix

| Category | Typical Items | Recommended Bin | Environmental Impact |
| :--- | :--- | :--- | :--- |
| **Recyclable** | Plastic bottles, cardboard, paper, aluminum cans, glass | **Blue / Green Recycling Bin** | Conserves raw timber, petroleum, and reduces landfill mass. |
| **Hazardous** | Batteries, electronic boards (e-waste), chemical solvents, medical items | **Red / Designated E-Waste Center** | Prevents toxic heavy metals (lead, mercury, cadmium) from contaminating groundwater. |

---

## 📜 License
This project is open-source under the **MIT License**.
