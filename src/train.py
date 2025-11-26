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
from src.model import build_cnn_model
from src.utils import save_class_indices

def train_model(data_dir="data", output_model_path="models/waste_classifier.h5", 
                img_size=(224, 224), epochs=10, batch_size=32, max_samples_per_class=None):
    """
    Complete training pipeline for the Smart Waste Classification CNN model.
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
    
    # 3. Data Augmentation
    train_gen = create_data_generator(X_train, y_train, batch_size=batch_size)
    
    # 4. Build CNN Architecture
    model = build_cnn_model(
        input_shape=(*img_size, 3), 
        num_classes=len(class_names),
        learning_rate=0.001
    )
    model.summary()
    
    # 5. Callbacks for hyperparameter tuning & regularization
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
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
    
    # 6. Fit model
    steps_per_epoch = max(1, len(X_train) // batch_size)
    print("\n[Training] Commencing CNN training loop with augmentation...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(output_model_path)
    print(f"\n[Training] Model successfully trained and saved to: {output_model_path}")
    
    # 7. Generate & Save Training Plots
    plot_training_curves(history, "static/img/training_history.png")
    
    return model, history

def plot_training_curves(history, output_path="static/img/training_history.png"):
    """
    Generate clean accuracy and loss curves for documentation and web dashboard.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    # Accuracy Curve
    ax1.plot(epochs_range, history.history['accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs_range, history.history['val_accuracy'], 'g-s', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Classification Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Loss Curve
    ax2.plot(epochs_range, history.history['loss'], 'r-o', label='Training Loss', linewidth=2)
    ax2.plot(epochs_range, history.history['val_loss'], 'm-s', label='Validation Loss', linewidth=2)
    ax2.set_title('Cross-Entropy Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Training] Training curves saved to: {output_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_directory = os.path.join(project_root, "data")
    model_output = os.path.join(project_root, "models", "waste_classifier.h5")
    train_model(data_dir=data_directory, output_model_path=model_output, epochs=10, batch_size=32)