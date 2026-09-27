from typing import Tuple

from keras import layers, models
import keras


def compile_cnn_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """Compile the CNN classifier with Adam optimizer and sparse categorical crossentropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_model(
    input_shape: Tuple[int, int] = (128, 9),
    n_classes: int = 6,
    filters_1: int = 64,
    filters_2: int = 128,
    kernel_size: int = 5,
    dropout_conv: float = 0.3,
    dropout_dense: float = 0.5,
    dense_units: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    name: str = "har_1dcnn_classifier",
) -> keras.Model:
    """Build and compile the 1D-CNN classifier for HAR.

    Architecture summary:
        1. Input: (seq_len, num_features) -> e.g. (128, 9)
        2. Conv1D(filters_1, kernel_size) -> BatchNorm -> ReLU   (x2)
        3. MaxPooling1D(pool_size=2) -> Dropout(dropout_conv)
        4. Conv1D(filters_2, kernel_size=3) -> BatchNorm -> ReLU
        5. GlobalAveragePooling1D
        6. Dense(dense_units, relu) -> Dropout(dropout_dense)
        7. Dense(n_classes, softmax) -> Activity probabilities

    Parameters:
        input_shape: (seq_len, num_features) of each window (default: (128, 9)).
        n_classes: Number of target activity classes (default: 6).
        filters_1: Filters in the first two Conv1D blocks (default: 64).
        filters_2: Filters in the third Conv1D block (default: 128).
        kernel_size: Kernel size for the first two Conv1D blocks (default: 5).
        dropout_conv: Dropout rate after pooling (default: 0.3).
        dropout_dense: Dropout rate before the output layer (default: 0.5).
        dense_units: Units in the pre-classification dense layer (default: 128).
        learning_rate: Adam learning rate (default: 1e-3).
        seed: Seed for weight initialization and dropout masks (default: 42).
        name: Name of the Keras model.

    Returns:
        keras.Model: Compiled Keras Functional model.
    """
    # Makes weight initialization reproducible for a given seed
    keras.utils.set_random_seed(int(seed))

    # 1. Input Layer
    inputs = layers.Input(shape=tuple(input_shape), dtype="float32", name="sensor_window_input")

    # 2. First Conv1D block
    x = layers.Conv1D(int(filters_1), int(kernel_size), padding="same", name="conv1d_1")(inputs)
    x = layers.BatchNormalization(name="batchnorm_1")(x)
    x = layers.ReLU(name="relu_1")(x)

    # Second Conv1D block
    x = layers.Conv1D(int(filters_1), int(kernel_size), padding="same", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="batchnorm_2")(x)
    x = layers.ReLU(name="relu_2")(x)

    # 3. Pooling + Dropout
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)
    x = layers.Dropout(float(dropout_conv), name="conv_dropout")(x)

    # 4. Third Conv1D block (higher-level patterns)
    x = layers.Conv1D(int(filters_2), 3, padding="same", name="conv1d_3")(x)
    x = layers.BatchNormalization(name="batchnorm_3")(x)
    x = layers.ReLU(name="relu_3")(x)

    # 5. Global Average Pooling — collapses time dimension into one summary vector
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # 6. Classification head
    x = layers.Dense(int(dense_units), activation="relu", name="dense_classification_head")(x)
    x = layers.Dropout(float(dropout_dense), name="dense_dropout")(x)

    # 7. Activity Class Probabilities Output
    outputs = layers.Dense(int(n_classes), activation="softmax", name="activity_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return compile_cnn_model(model, learning_rate=learning_rate)

def load_cnn_model(filepath: str) -> keras.Model:
    """Load a saved .keras CNN model (built-in layers only, kept for API symmetry).

    Parameters:
        filepath: Path to the saved .keras model file.

    Returns:
        keras.Model: Loaded Keras model ready for inference or fine-tuning.
    """
    return models.load_model(filepath)