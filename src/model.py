import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

def build_cnn_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=0.001, dropout_rate=0.25):
    """
    Construct a customized Convolutional Neural Network (CNN) for waste classification.
    
    Architecture:
      - Block 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool -> Dropout
      - Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool -> Dropout
      - Block 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool -> Dropout
      - Classifier: Flatten -> Dense(512) -> BatchNorm -> Dropout(0.5) -> Dense(num_classes, softmax)
    """
    model = Sequential([
        # Block 1
        Input(shape=input_shape),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        
        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        
        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate + 0.05),
        
        # Classification Head
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name="SmartWaste_CNN")
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=0.0001):
    """
    Transfer learning model utilizing lightweight MobileNetV2 for real-time edge inference.
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    # Freeze base model layers initially
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="SmartWaste_MobileNetV2")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model