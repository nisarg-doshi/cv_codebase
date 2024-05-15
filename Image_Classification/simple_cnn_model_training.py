import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import tensorflow as tf
import math

def remove_corrupted_images(directory):
    """
    Removes corrupted images (non-JFIF format) from the specified directory.

    Args:
    - directory (str): Path to the directory containing images.

    Returns:
    None
    """
    num_skipped = 0
    for folder_name in ("with_object", "without_object"):
        folder_path = os.path.join(directory, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print(f"Deleted {num_skipped} images.")

def load_and_preprocess_data(directory, image_size=(256, 256), batch_size=16):
    """
    Loads and preprocesses images from the specified directory.

    Args:
    - directory (str): Path to the directory containing images.
    - image_size (tuple): Size of the images (default is (256, 256)).
    - batch_size (int): Batch size for training (default is 16).

    Returns:
    - train_ds: Training dataset.
    - val_ds: Validation dataset.
    """
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    return train_ds, val_ds

def visualize_dataset(train_ds):
    """
    Visualizes a sample of the training dataset.

    Args:
    - train_ds: Training dataset.

    Returns:
    None
    """
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")

def create_data_augmentation_layers():
    """
    Creates data augmentation layers.

    Returns:
    - data_augmentation_layers: List of data augmentation layers.
    """
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
    return data_augmentation_layers

def visualize_augmented_data(train_ds, data_augmentation_layers):
    """
    Visualizes a sample of the augmented training dataset.

    Args:
    - train_ds: Training dataset.
    - data_augmentation_layers: List of data augmentation layers.

    Returns:
    None
    """
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[0]).astype("uint8"))
            plt.axis("off")

def create_model(input_shape, num_classes):
    """
    Creates a CNN model.

    Args:
    - input_shape (tuple): Shape of the input images.
    - num_classes (int): Number of output classes.

    Returns:
    - model: CNN model.
    """
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    model = keras.Model(inputs, outputs)
    return model

def train_model(model, train_ds, val_ds, epochs, callbacks):
    """
    Trains the CNN model.

    Args:
    - model: CNN model.
    - train_ds: Training dataset.
    - val_ds: Validation dataset.
    - epochs (int): Number of training epochs.
    - callbacks: List of callbacks.

    Returns:
    None
    """
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

def save_model(model, path):
    """
    Saves the trained model to the specified path.

    Args:
    - model: Trained CNN model.
    - path (str): Path to save the model.

    Returns:
    None
    """
    model.save(path)

if __name__ == "__main__":
    # Remove corrupted images
    current_dir = os.getcwd()
    remove_corrupted_images(f"/{current_dir}/image_data")

    # Load and preprocess data
    train_ds, val_ds = load_and_preprocess_data("/{current_dir}/image_data")

    # Visualize dataset
    visualize_dataset(train_ds)

    # Data augmentation
    data_augmentation = create_data_augmentation_layers()

    # Visualize augmented data
    visualize_augmented_data(train_ds, data_augmentation)

    # Create CNN model
    input_shape = (256, 256, 3)
    num_classes = 2
    model = create_model(input_shape, num_classes)

    # Define training parameters
    epochs = 100
    
    checkpoint_path = "{current_dir}/object_diff/ckpt/model_{epoch}.ckpt"
    n_batches = len(train_ds) / 16
    n_batches = math.ceil(n_batches)
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]

    # Train the model
    train_model(model, train_ds, val_ds, epochs, callbacks)

    # Save the trained model
    save_model(model, f'{current_dir}/object_diff/cnn_rgb_model/my_model')
