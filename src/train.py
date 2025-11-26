import os
import sys
# Ensure project root is in sys.path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

from src.data_preprocessing import load_and_preprocess_data, create_data_generator
from src.model import build_cnn_model, build_mobilenet_model
from src.utils import save_class_indices

def train_model(data_dir="data", output_model_path="models/waste_classifier.h5", 
                img_size=(224, 224), epochs=8, batch_size=16, max_samples_per_class=137, use_transfer_learning=True):
    """
    Complete training pipeline for the Smart Waste Classification model.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("static/img", exist_ok=True)
    
    print("=" * 60)
    print("[Training] Starting Smart Waste Classification Model Training")
    print("=" * 60)
    
    # 1. Load and preprocess dataset via OpenCV pipeline
    X, y, class_names = load_and_preprocess_data(
        data_dir=data_dir, 
        img_size=img_size, 
        max_samples_per_class=max_samples_per_class
    )
    
    print(f"\n[Dataset] Total samples loaded: {len(X)}")
    print(f"[Dataset] Target classes: {class_names}")
    
    # Save class indices mapping
    save_class_indices(class_names, "models/class_indices.json")
    
    # 2. Stratified train-test split (80% Train, 20% Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    print(f"[Split] Training set: {len(X_train)} samples")
    print(f"[Split] Validation set: {len(X_val)} samples")
    
    # 3. Model Architecture Selection
    if use_transfer_learning:
        print("\n[Model] Building MobileNetV2 Transfer Learning Architecture...")
        model = build_mobilenet_model(input_shape=(*img_size, 3), num_classes=len(class_names), learning_rate=0.001)
    else:
        print("\n[Model] Building Custom 3-Block CNN Architecture...")
        model = build_cnn_model(input_shape=(*img_size, 3), num_classes=len(class_names), learning_rate=0.0003)
        
    model.summary()
    
    # 4. Callbacks for hyperparameter tuning & regularization
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', 
            patience=5, 
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.3, 
            patience=2, 
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=output_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 5. Model Training Loop
    print("\n[Training] Commencing training loop...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(output_model_path)
    print(f"\n[Training] Model successfully trained and saved to: {output_model_path}")
    
    # 6. Plot & Save Training History Curves
    plot_training_history(history, "static/img/training_history.png")
    
    return model, history

def plot_training_history(history, save_path="static/img/training_history.png"):
    """
    Plot and save training accuracy & loss curves.
    """
    plt.figure(figsize=(12, 4.5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Training Accuracy', color='#10b981', linewidth=2.2)
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy', color='#3b82f6', linewidth=2.2, linestyle='--')
    plt.title('Model Accuracy vs Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Training Loss', color='#ef4444', linewidth=2.2)
    plt.plot(history.history.get('val_loss', []), label='Validation Loss', color='#f59e0b', linewidth=2.2, linestyle='--')
    plt.title('Loss Convergence vs Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Training] Training curves saved to: {save_path}")

if __name__ == "__main__":
    train_model(epochs=6, max_samples_per_class=137, use_transfer_learning=True)