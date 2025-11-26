import os
import sys
# Ensure project root is in sys.path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.data_preprocessing import preprocess_image_cv, load_and_preprocess_data
from src.utils import load_class_indices, format_prediction_result

def load_inference_model(model_path="models/waste_classifier.h5"):
    """
    Load saved Keras model for evaluation or real-time inference.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = load_model(model_path)
    return model

def predict_single_image(model, image_input, class_indices=None, img_size=(224, 224)):
    """
    Perform inference on a single image (file path or numpy array).
    Returns formatted prediction dictionary with disposal guidance and latency.
    """
    if class_indices is None:
        class_indices = load_class_indices()
        
    start_time = time.time()
    
    # Preprocess with OpenCV
    processed_img = preprocess_image_cv(image_input, target_size=img_size)
    img_batch = np.expand_dims(processed_img, axis=0)
    
    # Model forward pass
    preds = model.predict(img_batch, verbose=0)[0]
    inference_time = (time.time() - start_time) * 1000  # in ms
    
    predicted_idx = int(np.argmax(preds))
    predicted_class = class_indices.get(predicted_idx, "recyclable")
    confidence = float(preds[predicted_idx])
    
    probabilities = {class_indices.get(i, f"class_{i}"): float(prob) for i, prob in enumerate(preds)}
    
    result = format_prediction_result(
        category=predicted_class,
        confidence=confidence,
        all_probabilities=probabilities,
        inference_time_ms=inference_time
    )
    return result

def evaluate_model_performance(model_path="models/waste_classifier.h5", data_dir="data", output_dir="static/img"):
    """
    Compute comprehensive classification metrics and save confusion matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("[Evaluation] Loading dataset for performance benchmarking...")
    X, y_true_one_hot, class_names = load_and_preprocess_data(data_dir, max_samples_per_class=100)
    y_true = np.argmax(y_true_one_hot, axis=1)
    
    print(f"[Evaluation] Loading model from {model_path}...")
    model = load_inference_model(model_path)
    
    print("[Evaluation] Running batch predictions...")
    y_pred_probs = model.predict(X, batch_size=32, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\n" + "=" * 50)
    print("[Evaluation] Classification Report:")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix - Smart Waste Classification", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Category", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Category", fontsize=11, fontweight="bold")
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[Evaluation] Confusion matrix saved to: {cm_path}")
    
    return report

if __name__ == "__main__":
    evaluate_model_performance()