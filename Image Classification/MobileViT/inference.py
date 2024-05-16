import tensorflow as tf
import numpy as np
import argparse
import cv2




class DatasetPreprocessor:
    """
    Class to preprocess and prepare the dataset for training.
    """

    def __init__(self, image_size, resize_bigger, num_classes=5, batch_size=64):
        """
        Initializes the DatasetPreprocessor class.

        Args:
        - image_size (int): Size of the input image.
        - resize_bigger (int): Size for resizing the input image to a bigger spatial resolution.
        - num_classes (int): Number of classes for classification.
        - batch_size (int): Batch size for training.
        """
        self.image_size = image_size
        self.resize_bigger = resize_bigger
        self.num_classes = num_classes
        self.batch_size = batch_size

    def preprocess_dataset(self, image, label = None, is_training=True):
        """
        Preprocesses the dataset by resizing and augmenting the images and one-hot encoding the labels.

        Args:
        - image (tensor): Input image.
        - label (int): Corresponding label.
        - is_training (bool): Boolean flag indicating whether it's training or not.

        Returns:
        - tuple: Tuple containing preprocessed image and one-hot encoded label.
        """
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            image = tf.image.resize(image, (self.resize_bigger, self.resize_bigger))
            image = tf.image.random_crop(image, (self.image_size, self.image_size, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (self.image_size, self.image_size))
        label = tf.one_hot(label, depth=self.num_classes)
        return image, label

    def prepare_dataset(self, dataset, is_training=True):
        """
        Prepares the dataset for training by applying preprocessing operations, batching, and prefetching.

        Args:
        - dataset (tf.data.Dataset): Input dataset.
        - is_training (bool): Boolean flag indicating whether it's training or not.

        Returns:
        - tf.data.Dataset: Prepared dataset.
        """
        if is_training:
            dataset = dataset.shuffle(self.batch_size * 10)
        dataset = dataset.map(lambda x, y: self.preprocess_dataset(x, y, is_training), num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)






class MobileViTInference:
    """
    Class for performing inference using the trained MobileViT model.
    """

    def __init__(self, model_path):
        """
        Initializes the MobileViTInference class with the path to the saved model.

        Args:
        - model_path (str): Path to the saved model.
        """
        
        self.model = tf.keras.models.load_model(model_path)


    def predict(self, image_array):
        """
        Performs inference on the input image and predicts the class label.

        Args:
        - image_path (ndarray): Image array.

        Returns:
        - int: Predicted class label.
        """
        img_array = self.preprocess_image(image_path, self.model.input_shape[1])
        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction)
        return predicted_class

# Example usage:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileViT Inference Script")
    parser.add_argument("--model_path", type=str, default="saved_model/mobileVIT", help="Path to the saved model")
    parser.add_argument("--image_path", type=str, default="path/to/your/image.jpg", help="Path to the input image")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the input image")
    parser.add_argument("--resize_bigger", type=int, default=280, help="Size for resizing the input image")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes for classification")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for preprocessing")
    args = parser.parse_args()

    inference_model = MobileViTInference(args.model_path)
    preprocessor = DatasetPreprocessor(
        image_size=args.image_size,
        resize_bigger=args.resize_bigger,
        num_classes=args.num_classes,
        batch_size=args.batch_size
    )

    image = cv2.imread(args.image_path)
    preprocessed_image, _ = preprocessor.preprocess_dataset(image, label=0, is_training=False)
    predicted_class = inference_model.predict(preprocessed_image)
    print("Predicted class label:", predicted_class)

