"""Hybrid CNN-LSTM Component for Human Activity Recognition (HAR).

Author: Monal (Member 4)
Project: SE4050 Deep Learning Assignment - Four-member HAR architecture comparison.
Architecture:
    Input: (128, 9)
    1D-CNN Feature Extraction:
        - Conv1D (64 filters, kernel_size=3, padding='same', relu)
        - BatchNormalization
        - Dropout (0.3)
        - Conv1D (64 filters, kernel_size=3, padding='same', relu)
        - BatchNormalization
        - MaxPooling1D (pool_size=2) -> Temporal downsampling (64, 64)
        - Dropout (0.3)
    LSTM Sequence Modeling:
        - LSTM (64 units, return_sequences=False) -> (64,)
        - Dropout (0.3)
    Classification Head:
        - Dense (64 units, relu)
        - Dropout (0.3)
        - Dense (6 units, softmax) -> Activity probabilities

Design notes:
    - 1D-CNN layers extract salient local temporal features and sensor channel correlations
      (such as sudden acceleration spikes and periodic gait sub-phases).
    - MaxPooling reduces temporal resolution while retaining dominant activations,
      improving computational efficiency for the downstream LSTM.
    - The LSTM layer models long-term sequential dependencies and macro-activity dynamics.
    - Adam optimizer with gradient clipping (clipnorm=1.0) ensures stable backpropagation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

# Compatible import of Keras / TensorFlow Keras
try:
    import keras
    from keras import layers, models
except ImportError:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models


def build_cnn_lstm_model(
    input_shape: Tuple[int, int] = (128, 9),
    n_classes: int = 6,
    cnn_filters: Union[List[int], Tuple[int, ...]] = (64, 64),
    kernel_size: int = 3,
    pool_size: int = 2,
    lstm_units: int = 64,
    n_lstm_layers: int = 1,
    dropout: float = 0.3,
    dense_units: int = 64,
    learning_rate: float = 5e-4,
    seed: int = 42,
    name: str = "har_cnn_lstm_classifier",
) -> keras.Model:
    """Build and compile the hybrid CNN-LSTM classifier for HAR.

    Architecture summary:
        1. Input: (seq_len, num_features) -> e.g. (128, 9)
        2. Convolutional blocks: Conv1D -> BatchNorm -> ReLU -> Dropout
        3. Temporal downsampling: MaxPooling1D (pool_size)
        4. Recurrent sequence modeling: n_lstm_layers x LSTM -> Dropout
        5. Dense classification head: Dense(dense_units, relu) -> Dropout -> Dense(n_classes, softmax)

    Parameters:
        input_shape: (seq_len, num_features) of each window (default: (128, 9)).
        n_classes: Number of target activity classes (default: 6).
        cnn_filters: List or tuple of filter counts for Conv1D layers (default: (64, 64)).
        kernel_size: 1D convolution kernel length (default: 3).
        pool_size: Max pooling window size (default: 2).
        lstm_units: Number of hidden units in LSTM layer(s) (default: 64).
        n_lstm_layers: Number of stacked LSTM layers (default: 1).
        dropout: Dropout rate applied across conv, recurrent, and dense layers (default: 0.3).
        dense_units: Units in pre-classification dense layer (default: 64).
        learning_rate: Initial Adam learning rate (default: 5e-4).
        seed: Random seed for initialization (default: 42).
        name: Name of the Keras model.

    Returns:
        keras.Model: Compiled Keras Functional model.
    """
    if n_lstm_layers < 1:
        raise ValueError(f"n_lstm_layers must be at least 1, got {n_lstm_layers}.")
    if not cnn_filters:
        raise ValueError("cnn_filters must contain at least one filter size.")

    # Set random seed for reproducibility
    keras.utils.set_random_seed(int(seed))

    # 1. Input Layer
    inputs = layers.Input(shape=tuple(input_shape), dtype="float32", name="sensor_window_input")

    # 2. 1D Convolutional Feature Extraction Blocks
    x = inputs
    for idx, filters in enumerate(cnn_filters):
        x = layers.Conv1D(
            filters=int(filters),
            kernel_size=int(kernel_size),
            padding="same",
            activation="relu",
            name=f"conv1d_block_{idx + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"conv1d_bn_{idx + 1}")(x)
        if idx < len(cnn_filters) - 1:
            x = layers.Dropout(float(dropout), name=f"conv1d_dropout_{idx + 1}")(x)

    # Temporal Downsampling
    if pool_size > 1:
        x = layers.MaxPooling1D(pool_size=int(pool_size), name="conv1d_maxpool")(x)
    x = layers.Dropout(float(dropout), name="conv1d_pool_dropout")(x)

    # 3. Recurrent LSTM Sequence Modeling
    for i in range(n_lstm_layers):
        is_last = (i == n_lstm_layers - 1)
        x = layers.LSTM(
            units=int(lstm_units),
            return_sequences=not is_last,
            name=f"lstm_layer_{i + 1}",
        )(x)
        x = layers.Dropout(float(dropout), name=f"lstm_dropout_{i + 1}")(x)

    # 4. Dense Classification Head
    x = layers.Dense(int(dense_units), activation="relu", name="dense_classification_head")(x)
    x = layers.Dropout(float(dropout), name="head_dropout")(x)

    # 5. Activity Probabilities Output
    outputs = layers.Dense(int(n_classes), activation="softmax", name="activity_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return compile_cnn_lstm_model(model, learning_rate=learning_rate)


def compile_cnn_lstm_model(
    model: keras.Model,
    learning_rate: float = 5e-4,
) -> keras.Model:
    """Compile the CNN-LSTM model with Adam (clipnorm=1.0), sparse categorical crossentropy and accuracy.

    Uses identical evaluation metrics and gradient clipping for fair benchmark comparison across all models.

    Parameters:
        model: Uncompiled Keras model.
        learning_rate: Initial learning rate (default: 5e-4).

    Returns:
        Compiled Keras model.
    """
    optimizer = keras.optimizers.Adam(learning_rate=float(learning_rate), clipnorm=1.0)
    loss = keras.losses.SparseCategoricalCrossentropy()
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def load_cnn_lstm_model(filepath: str) -> keras.Model:
    """Load a saved .keras CNN-LSTM model.

    Parameters:
        filepath: Path to saved .keras file.

    Returns:
        keras.Model: Loaded Keras model.
    """
    return models.load_model(filepath)


def predict_cnn_lstm(
    model: keras.Model,
    x: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Generate activity probabilities using the trained CNN-LSTM model.

    Parameters:
        model: Trained Keras model.
        x: Input sensor windows shaped (N, 128, 9).
        batch_size: Inference batch size.

    Returns:
        np.ndarray: Class probabilities shaped (N, 6) summing to 1.0.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim != 3 or x_arr.shape[1:] != (128, 9):
        raise ValueError(
            f"Expected input array shaped (N, 128, 9), but received shape {x_arr.shape}."
        )
    return model.predict(x_arr, batch_size=batch_size, verbose=0)
