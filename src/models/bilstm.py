"""Bidirectional LSTM Component for Human Activity Recognition (HAR).

Author: Member 3
Project: SE4050 Deep Learning Assignment - Four-member HAR architecture comparison.
Architecture:
    Input: (128, 9)
    Bidirectional LSTM: 64 units per direction, returns full sequence -> (128, 128)
    Dropout: 0.4
    Bidirectional LSTM: 64 units per direction, returns last state -> (128,)
    Dropout: 0.4
    Dense layer: 64 units, ReLU
    Softmax output: 6 units (classes 0-5)

Pooling options (how the last BiLSTM layer summarises the window):
    "last"      final states only (default, the architecture above)
    "avg"       average over all 128 time steps (no extra parameters)
    "attention" learned softmax weights over all 128 time steps (+129 parameters)

Design notes:
    - Activities are defined by how motion evolves over time, which is what a
      recurrent network's gated memory models.
    - Reading the window in both directions is legitimate because each 2.56 s
      window is classified as a whole, not streamed in real time.
    - Adam with gradient clipping (clipnorm=1.0) keeps recurrent training stable;
      the learning rate (5e-4) is lower than the Transformer's (1e-3) because
      LSTMs are more sensitive to it.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# Compatible import of Keras / TensorFlow Keras
try:
    import keras
    from keras import layers, models
except ImportError:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models

POOLING_OPTIONS = ("last", "avg", "attention")


# Register serializable decorator helper
def _register_serializable(cls):
    """Register custom layer for Keras serialization across Keras 2/3 and TF."""
    try:
        if hasattr(keras, "saving") and hasattr(keras.saving, "register_keras_serializable"):
            keras.saving.register_keras_serializable(package="har_bilstm")(cls)
    except Exception:
        pass

    try:
        import tensorflow as tf
        if hasattr(tf.keras.utils, "register_keras_serializable"):
            tf.keras.utils.register_keras_serializable(package="har_bilstm")(cls)
    except Exception:
        pass
    return cls


@_register_serializable
class TemporalAttentionPooling(layers.Layer):
    """Attention pooling over time: a learned weighted average of all time steps.

    A Dense(1) layer scores each time step, a softmax over time turns the scores
    into weights that sum to 1, and the output is the weighted sum of the steps.
    Input (N, T, F) -> output (N, F). Adds F + 1 parameters (129 for F = 128).
    """

    def build(self, input_shape) -> None:
        """Create the per-time-step scoring layer."""
        self.score_dense = layers.Dense(1, name="attention_score")
        self.score_dense.build(input_shape)
        super().build(input_shape)

    def attention_weights(self, inputs):
        """Return the softmax weights over time, shaped (N, T)."""
        scores = self.score_dense(inputs)  # (N, T, 1)
        weights = keras.ops.softmax(scores, axis=1)
        return keras.ops.squeeze(weights, axis=-1)

    def call(self, inputs):
        """Weighted sum of the time steps."""
        weights = keras.ops.expand_dims(self.attention_weights(inputs), axis=-1)
        return keras.ops.sum(inputs * weights, axis=1)

    def compute_output_shape(self, input_shape):
        """(N, T, F) -> (N, F)."""
        return (input_shape[0], input_shape[-1])

    def get_config(self) -> Dict[str, Any]:
        """Serialize layer configuration (no extra constructor arguments)."""
        return super().get_config()


def build_bilstm_model(
    input_shape: Tuple[int, int] = (128, 9),
    n_classes: int = 6,
    lstm_units: int = 64,
    n_layers: int = 2,
    dropout: float = 0.4,
    dense_units: int = 64,
    learning_rate: float = 5e-4,
    seed: int = 42,
    pooling: str = "last",
    name: str = "har_bilstm_classifier",
) -> keras.Model:
    """Build and compile the stacked BiLSTM classifier for HAR.

    Architecture summary:
        1. Input: (seq_len, num_features) -> e.g. (128, 9)
        2. n_layers x [Bidirectional(LSTM(lstm_units)) -> Dropout(dropout)]
           All but the last layer return the full sequence (128, 2 * lstm_units).
           With pooling="last" the last layer returns only its final states
           (2 * lstm_units,); otherwise it returns the full sequence, which is
           then pooled over time (average or attention) to (2 * lstm_units,).
        3. Dense(dense_units, relu)
        4. Dense(n_classes, softmax) -> Activity probabilities

    Parameters:
        input_shape: (seq_len, num_features) of each window (default: (128, 9)).
        n_classes: Number of target activity classes (default: 6).
        lstm_units: LSTM units per direction (default: 64).
        n_layers: Number of stacked bidirectional LSTM layers (default: 2).
        dropout: Dropout rate after each recurrent layer (default: 0.4).
        dense_units: Units in the pre-classification dense layer (default: 64).
        learning_rate: Adam learning rate (default: 5e-4).
        seed: Seed for weight initialization and dropout masks (default: 42).
        pooling: How the last layer summarises the window: "last", "avg" or
            "attention" (default: "last").
        name: Name of the Keras model.

    Returns:
        keras.Model: Compiled Keras Functional model.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}.")
    if pooling not in POOLING_OPTIONS:
        raise ValueError(f"pooling must be one of {POOLING_OPTIONS}, got '{pooling}'.")

    # Makes weight initialization reproducible for a given seed
    keras.utils.set_random_seed(int(seed))

    # 1. Input Layer
    inputs = layers.Input(shape=tuple(input_shape), dtype="float32", name="sensor_window_input")

    # 2. Stacked Bidirectional LSTM layers
    x = inputs
    for i in range(n_layers):
        is_last = i == n_layers - 1
        x = layers.Bidirectional(
            layers.LSTM(int(lstm_units), return_sequences=not is_last or pooling != "last"),
            merge_mode="concat",
            name=f"bilstm_{i + 1}",
        )(x)
        x = layers.Dropout(float(dropout), name=f"bilstm_dropout_{i + 1}")(x)

    # Pool the last layer's sequence over time
    if pooling == "avg":
        x = layers.GlobalAveragePooling1D(name="temporal_avg_pool")(x)
    elif pooling == "attention":
        x = TemporalAttentionPooling(name="temporal_attention_pool")(x)

    # 3. Classification Head
    x = layers.Dense(int(dense_units), activation="relu", name="dense_classification_head")(x)

    # 4. Activity Class Probabilities Output
    outputs = layers.Dense(int(n_classes), activation="softmax", name="activity_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return compile_bilstm_model(model, learning_rate=learning_rate)


def compile_bilstm_model(
    model: keras.Model,
    learning_rate: float = 5e-4,
) -> keras.Model:
    """Compile the BiLSTM with Adam (clipnorm=1.0), sparse categorical crossentropy and accuracy.

    Uses the same loss and metric as the Transformer so the two are directly comparable.

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


def load_bilstm_model(filepath: str) -> keras.Model:
    """Load a saved .keras BiLSTM model, including the custom attention pooling layer.

    Parameters:
        filepath: Path to the saved .keras model file.

    Returns:
        keras.Model: Loaded Keras model ready for inference or fine-tuning.
    """
    return models.load_model(
        filepath,
        custom_objects={"TemporalAttentionPooling": TemporalAttentionPooling},
    )


def predict_bilstm(
    model: keras.Model,
    x: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Generate activity probabilities using the trained BiLSTM model.

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
