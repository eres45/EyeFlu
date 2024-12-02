import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# GPU Memory Management
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define paths
base_dataset_path = r'C:\Users\Ronit\Downloads\EyeVision\Dataset'
train_dir = os.path.join(base_dataset_path, 'train')
model_save_path = os.path.join(base_dataset_path, 'model.keras')
best_model_path = os.path.join(base_dataset_path, 'best_model.keras')

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,  # Use 20% of data for validation
    fill_mode='nearest'
)

# Create augmented datasets
train_gen = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Transfer Learning with MobileNetV2
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze the base model

# Custom Model Architecture
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-6
)

model_checkpoint = ModelCheckpoint(
    best_model_path, 
    monitor='val_loss', 
    save_best_only=True
)

# Train Model
initial_epochs = 30
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=initial_epochs,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Fine-tuning: Unfreeze Base Model Layers
base_model.trainable = True
fine_tune_at = len(base_model.layers) // 2  # Unfreeze the last half of layers

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile Model with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Fine-tune Model
fine_tune_epochs = 20
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Save final model
model.save(model_save_path)
print("Fine-tuned model saved successfully.")

# Evaluate Model
print("\nModel Evaluation:")
evaluation = model.evaluate(val_gen)
print("Validation Metrics:", dict(zip(model.metrics_names, evaluation)))
