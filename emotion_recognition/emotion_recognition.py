import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define paths for training and testing directories
train_dir = "datasets/fer2013/train"
test_dir = "datasets/fer2013/test"

# Define image dimensions and batch size
img_height, img_width = 48, 48
batch_size = 32

# Data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,       # Normalize pixel values
    rotation_range=10,     # Random rotation
    width_shift_range=0.1, # Horizontal shift
    height_shift_range=0.1,# Vertical shift
    zoom_range=0.1,        # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode="nearest"    # Fill missing pixels
)

test_datagen = ImageDataGenerator(rescale=1.0/255)  # Only rescale for testing

# Generate batches of augmented data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

# Define the CNN model
model = Sequential([
    # Convolutional layers
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten the output and feed into dense layers
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),  # Dropout for regularization
    Dense(7, activation="softmax")  # 7 classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=50,
    callbacks=[early_stopping]
)
model.save("emotion_recognition_model.h5")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


