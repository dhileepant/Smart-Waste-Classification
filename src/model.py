import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, Input, GlobalAveragePooling2D, Rescaling
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

def build_cnn_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=0.001, dropout_rate=0.25):
    """
    Builds a multi-block Convolutional Neural Network (CNN) for waste classification.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Block 1 - Low Level Feature Extraction
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        
        # Block 2 - Mid Level Textures
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        
        # Block 3 - High Level Semantic Shapes
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate + 0.05),
        
        # Classification Head
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ], name="SmartWaste_CNN")
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=0.001):
    """
    Transfer learning model utilizing lightweight MobileNetV2 with ImageNet features for high-accuracy edge inference.
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    scaled = Rescaling(scale=2.0, offset=-1.0)(inputs)
    x = base_model(scaled, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="SmartWaste_MobileNetV2")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model