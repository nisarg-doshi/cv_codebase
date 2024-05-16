

import os
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, backend
from tensorflow import keras



class MobileViT:
    """
    MobileViT class implements the MobileViT model for image classification.

    It contains methods for building different blocks of the MobileViT architecture,
    such as convolutional blocks, inverted residual blocks, transformer blocks, and mobilevit blocks.
    """


    def __init__(self, image_size=256, expansion_factor=2, patch_size=4, num_classes=5):
        """
        Initializes the MobileViT model with default parameters.

        Args:
        - image_size (int): Size of the input image.
        - expansion_factor (int): Expansion factor for the MobileNetV2 blocks.
        - patch_size (int): Size of the patches used in the Transformer blocks.
        - num_classes (int): Number of classes for classification.
        """
        self.image_size = image_size
        self.expansion_factor = expansion_factor
        self.patch_size = patch_size
        self.num_classes = num_classes

    def conv_block(self, x, filters=16, kernel_size=3, strides=2):
        """
        Constructs a convolutional block with specified parameters.

        Args:
        - x (tensor): Input tensor.
        - filters (int): Number of filters in the convolutional layer.
        - kernel_size (int): Size of the convolutional kernel.
        - strides (int): Stride size for the convolution operation.

        Returns:
        - tensor: Output tensor after applying the convolutional block.
        """
        conv_layer = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            activation=tf.keras.activations.swish,
            padding="same",
        )
        return conv_layer(x)

    def correct_pad(self, inputs, kernel_size):
        """
        Calculates the correct padding for convolutional layers.

        Args:
        - inputs (tensor): Input tensor.
        - kernel_size (int): Size of the convolutional kernel.

        Returns:
        - tuple: Tuple containing the padding values for height and width.
        """
        img_dim = 2 if backend.image_data_format() == "channels_first" else 1
        input_size = inputs.shape[img_dim : (img_dim + 2)]
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if input_size[0] is None:
            adjust = (1, 1)
        else:
            adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
        correct = (kernel_size[0] // 2, kernel_size[1] // 2)
        return (
            (correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]),
        )

    def inverted_residual_block(self, x, expanded_channels, output_channels, strides=1):
        """
        Constructs an inverted residual block.

        Args:
        - x (tensor): Input tensor.
        - expanded_channels (int): Number of channels after the expansion layer.
        - output_channels (int): Number of output channels.
        - strides (int): Stride size for the convolutional operation.

        Returns:
        - tensor: Output tensor after applying the inverted residual block.
        """
        m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
        m = layers.BatchNormalization()(m)
        m = tf.keras.activations.swish(m)

        if strides == 2:
            m = layers.ZeroPadding2D(padding=self.correct_pad(m, 3))(m)
        m = layers.DepthwiseConv2D(
            3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
        )(m)
        m = layers.BatchNormalization()(m)
        m = tf.keras.activations.swish(m)

        m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
        m = layers.BatchNormalization()(m)

        if keras.ops.equal(x.shape[-1], output_channels) and strides == 1:
            return layers.Add()([m, x])
        return m

    def mlp(self, x, hidden_units, dropout_rate):
        """
        Constructs a multi-layer perceptron (MLP) block.

        Args:
        - x (tensor): Input tensor.
        - hidden_units (list): List containing the number of units in each hidden layer.
        - dropout_rate (float): Dropout rate for regularization.

        Returns:
        - tensor: Output tensor after applying the MLP block.
        """
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.keras.activations.swish)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def transformer_block(self, x, transformer_layers, projection_dim, num_heads=2):
        """
        Constructs a transformer block.

        Args:
        - x (tensor): Input tensor.
        - transformer_layers (int): Number of transformer layers.
        - projection_dim (int): Dimensionality of the projected embeddings.
        - num_heads (int): Number of attention heads.

        Returns:
        - tensor: Output tensor after applying the transformer block.
        """
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(x)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, x])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(
                x3,
                hidden_units=[x.shape[-1] * 2, x.shape[-1]],
                dropout_rate=0.1,
            )
            # Skip connection 2.
            x = layers.Add()([x3, x2])

        return x

    def mobilevit_block(self, x, num_blocks, projection_dim, strides=1):
        """
        Constructs a MobileViT block.

        Args:
        - x (tensor): Input tensor.
        - num_blocks (int): Number of MobileViT blocks.
        - projection_dim (int): Dimensionality of the projected embeddings.
        - strides (int): Stride size for the convolutional operation.

        Returns:
        - tensor: Output tensor after applying the MobileViT block.
        """
        # Local projection with convolutions.
        local_features = self.conv_block(x, filters=projection_dim, strides=strides)
        local_features = self.conv_block(
            local_features, filters=projection_dim, kernel_size=1, strides=strides
        )

        # Unfold into patches and then pass through Transformers.
        num_patches = int((local_features.shape[1] * local_features.shape[2]) / self.patch_size)
        non_overlapping_patches = layers.Reshape((self.patch_size, num_patches, projection_dim))(
            local_features
        )
        global_features = self.transformer_block(
            non_overlapping_patches, num_blocks, projection_dim
        )

        # Fold into conv-like feature-maps.
        folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
            global_features
        )

        # Apply point-wise conv -> concatenate with the input features.
        folded_feature_map = self.conv_block(
            folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
        )
        local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

        # Fuse the local and global features using a convoluion layer.
        local_global_features = self.conv_block(
            local_global_features, filters=projection_dim, strides=strides
        )

        return local_global_features

    def create_mobilevit(self):
        """
        Creates the MobileViT model architecture.

        Returns:
        - Model: MobileViT model instance.
        """
        inputs = tf.keras.Input((self.image_size, self.image_size, 3))
        x = layers.Rescaling(scale=1.0 / 255)(inputs)

        # Initial conv-stem -> MV2 block.
        x = self.conv_block(x, filters=16)
        x = self.inverted_residual_block(
            x, expanded_channels=16 * self.expansion_factor, output_channels=16
        )

        # Downsampling with MV2 block.
        x = self.inverted_residual_block(
            x, expanded_channels=16 * self.expansion_factor, output_channels=24, strides=2
        )
        x = self.inverted_residual_block(
            x, expanded_channels=24 * self.expansion_factor, output_channels=24
        )
        x = self.inverted_residual_block(
            x, expanded_channels=24 * self.expansion_factor, output_channels=24
        )

        # First MV2 -> MobileViT block.
        x = self.inverted_residual_block(
            x, expanded_channels=24 * self.expansion_factor, output_channels=48, strides=2
        )
        x = self.mobilevit_block(x, num_blocks=2, projection_dim=64)

        # Second MV2 -> MobileViT block.
        x = self.inverted_residual_block(
            x, expanded_channels=64 * self.expansion_factor, output_channels=64, strides=2
        )
        x = self.mobilevit_block(x, num_blocks=4, projection_dim=80)

        # Third MV2 -> MobileViT block.
        x = self.inverted_residual_block(
            x, expanded_channels=80 * self.expansion_factor, output_channels=80, strides=2
        )
        x = self.mobilevit_block(x, num_blocks=3, projection_dim=96)
        x = self.conv_block(x, filters=320, kernel_size=1, strides=1)

        # Classification head.
        x = layers.GlobalAvgPool2D()(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        return tf.keras.Model(inputs, outputs)

    
def run_experiment(self, train_dataset, val_dataset, num_classes, image_size, learning_rate, label_smoothing_factor, epochs=30):
    """
    Runs the training experiment for the MobileViT model.

    Args:
    - train_dataset (tf.data.Dataset): Training dataset.
    - val_dataset (tf.data.Dataset): Validation dataset.
    - num_classes (int): Number of classes for classification.
    - epochs (int): Number of epochs for training.

    Returns:
    - Model: Trained MobileViT model instance.
    """
    mobilevit_model = MobileViT(image_size=self.image_size, num_classes=self.num_classes).create_mobilevit()
    mobilevit_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_factor),
                            metrics=["accuracy"])

    # Define checkpoint callback
    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    # Train the model
    mobilevit_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback],
    )
    mobilevit_model.load_weights(checkpoint_filepath)
    _, accuracy = mobilevit_model.evaluate(val_dataset)
    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")
    mobilevit_model.save('saved_model/mobileVIT')
    # tf.saved_model.save(mobilevit_xxs, "mobilevit_xxs")
    return "model_trained sucessfully!"



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

    def preprocess_dataset(self, image, label, is_training=True):
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileViT Training Script")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the input image")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--resize_bigger", type=int, default=280, help="Size for resizing the input image")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes for classification")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate for training")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training")
    args = parser.parse_args()

    train_dataset, val_dataset = tfds.load(
        "tf_flowers", split=["train[:90%]", "train[90%:]"], as_supervised=True
    )
     
    preprocessor = DatasetPreprocessor(resize_bigger=args.resize_bigger, num_classes = args.num_classes, batch_size=args.batch_size)
    train_dataset = preprocessor.prepare_dataset(train_dataset, is_training=True)
    val_dataset = preprocessor.prepare_dataset(val_dataset, is_training=False)

    model = MobileViT()
    model.run_experiment(train_dataset, val_dataset, num_classes = args.num_classes, epochs=args.epochs, image_size = args.image_size, learning_rate = args.learning_rate, label_smoothing_factor = args.label_smoothing_factor)

    
