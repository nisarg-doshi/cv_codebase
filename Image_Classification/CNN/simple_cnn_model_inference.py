import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import keras

def load_model(model_path):
    """
    Loads the trained CNN model.

    Args:
    - model_path (str): Path to the trained model.

    Returns:
    - model: Trained CNN model.
    """
    model = keras.models.load_model(model_path)
    return model

def preprocess_image(image_path):
    """
    Preprocesses the input image.

    Args:
    - image_path (str): Path to the input image.

    Returns:
    - image: Preprocessed image.
    """
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def inference(model, image):
    """
    Performs inference using the trained model.

    Args:
    - model: Trained CNN model.
    - image: Preprocessed input image.

    Returns:
    - prediction: Model prediction.
    """
    prediction = model.predict(image)
    return prediction



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Inference Script")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    args = parser.parse_args()
    # Load the trained model
    model = load_model(args.model_path)

    # Preprocess the input image
    image = preprocess_image(args.image_path)

    # Perform inference
    prediction = inference(model, image)

    # Print the prediction
    print("Prediction:", prediction)

