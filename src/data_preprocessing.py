import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image_cv(image, target_size=(224, 224), apply_clahe=False):
    """
    Apply OpenCV preprocessing pipeline to a single image.
    Supports input as file path or existing numpy array.
    """
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image not found at path: {image}")
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Failed to read image at: {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        img = image.copy()
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        raise TypeError("Image must be a file path string or a numpy array.")

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) if requested
    if apply_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Resize to target CNN input dimensions
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Normalize pixel intensity to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized

def load_and_preprocess_data(data_dir, img_size=(224, 224), max_samples_per_class=None):
    """
    Load and preprocess images from structured class folders.
    
    Args:
        data_dir: Path to directory containing class subfolders ('hazardous', 'recyclable')
        img_size: Tuple (width, height)
        max_samples_per_class: Optional limit on samples per class for quick training
    
    Returns:
        X: numpy array of preprocessed images
        y: one-hot encoded labels
        class_names: list of class names in order
    """
    images = []
    labels = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")
        
    class_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print(f"[OpenCV Pipeline] Scanning classes: {class_folders}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_name in class_folders:
        class_path = os.path.join(data_dir, class_name)
        file_list = [f for f in os.listdir(class_path) if os.path.splitext(f.lower())[1] in valid_extensions]
        
        if max_samples_per_class is not None:
            file_list = file_list[:max_samples_per_class]

        print(f"[OpenCV Pipeline] Loading {len(file_list)} images for class '{class_name}'...")
        count = 0
        for img_name in file_list:
            img_path = os.path.join(class_path, img_name)
            try:
                processed_img = preprocess_image_cv(img_path, target_size=img_size)
                images.append(processed_img)
                labels.append(class_name)
                count += 1
            except Exception as e:
                print(f"Warning: Skipping corrupted image {img_path}: {e}")

        print(f"[OpenCV Pipeline] Successfully processed {count} images for class '{class_name}'.")

    X = np.array(images, dtype=np.float32)
    
    # Label Encoding & One-Hot representation
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y_one_hot = to_categorical(y_encoded, num_classes=len(class_folders))
    
    return X, y_one_hot, list(le.classes_)

def create_data_generator(X_train, y_train, batch_size=32):
    """
    Create an augmented data generator for CNN training.
    """
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    return datagen.flow(X_train, y_train, batch_size=batch_size)