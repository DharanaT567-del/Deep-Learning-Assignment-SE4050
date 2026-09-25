"""Transformer Encoder Component for Human Activity Recognition (HAR).

Author: Dharana
Project: SE4050 Deep Learning Assignment - Four-member HAR architecture comparison.
Architecture:
    Input: (128, 9)
    Dense projection: 64 dimensions
    Trainable positional embeddings: (128, 64)
    2 Transformer encoder blocks:
        - 4 attention heads, key_dim=16 per head
        - Pre-LayerNormalization before MHA and FFN
        - Residual connections
        - Feed-forward: 128 units (GELU) -> 64 units
        - Dropout: 0.2
    Global average pooling over time (GlobalAveragePooling1D)
    Dense layer: 64 units, ReLU
    Softmax output: 6 units (classes 0-5)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Compatible import of Keras / TensorFlow Keras
try:
    import keras
    from keras import layers, models, ops
except ImportError:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models

# Register serializable decorator helper
def _register_serializable(cls):
    """Register custom layer for Keras serialization across Keras 2/3 and TF."""
    try:
        if hasattr(keras, "saving") and hasattr(keras.saving, "register_keras_serializable"):
            keras.saving.register_keras_serializable(package="har_transformer")(cls)
    except Exception:
        pass

    try:
        import tensorflow as tf
        if hasattr(tf.keras.utils, "register_keras_serializable"):
            tf.keras.utils.register_keras_serializable(package="har_transformer")(cls)
    except Exception:
        pass
    return cls


@_register_serializable
class TrainablePositionalEmbedding(layers.Layer):
    """Trainable 1D positional embedding layer for time-series sequences.

    Adds learnable position vectors of shape (seq_len, d_model) to the
    input sequence projections of shape (batch_size, seq_len, d_model).

    Parameters:
        seq_len: Number of time steps in the input window (default: 128).
        d_model: Dimensionality of the projected feature space (default: 64).
    """

    def __init__(self, seq_len: int = 128, d_model: int = 64, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.pos_embedding = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Initialize trainable positional weights."""
        self.pos_embedding = self.add_weight(
            name="position_embeddings",
            shape=(self.seq_len, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            dtype=self.dtype,
        )
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Add positional embedding to inputs."""
        return inputs + self.pos_embedding

    def get_config(self) -> Dict[str, Any]:
        """Serialize configuration for model saving."""
        config = super().get_config()
        config.update(
            {
                "seq_len": self.seq_len,
                "d_model": self.d_model,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> TrainablePositionalEmbedding:
        """Instantiate layer from configuration dict."""
        return cls(**config)


@_register_serializable
class TransformerEncoderBlock(layers.Layer):
    """Pre-LayerNormalization Transformer Encoder Block for sensor time-series.

    Architecture (Pre-LN):
        1. x_norm1 = LayerNorm(x)
        2. attn = MultiHeadAttention(query=x_norm1, key=x_norm1, value=x_norm1)
        3. x = x + Dropout(attn)
        4. x_norm2 = LayerNorm(x)
        5. ffn = Dense(d_ff, activation=ffn_activation)(x_norm2)
        6. ffn = Dropout(ffn)
        7. ffn = Dense(d_model)(ffn)
        8. out = x + Dropout(ffn)

    Parameters:
        d_model: Latent feature dimension (default: 64).
        num_heads: Number of attention heads (default: 4).
        key_dim: Key dimension per attention head (default: 16).
        d_ff: Number of hidden units in the feed-forward network (default: 128).
        ffn_activation: Activation function for the first FFN dense layer (default: "gelu").
        dropout_rate: Dropout rate for attention and FFN sublayers (default: 0.2).
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        key_dim: int = 16,
        d_ff: int = 128,
        ffn_activation: str = "gelu",
        dropout_rate: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.d_ff = int(d_ff)
        self.ffn_activation = str(ffn_activation)
        self.dropout_rate = float(dropout_rate)

        # Pre-attention normalization & MHA
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name="pre_mha_ln")
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
            name="multi_head_attention",
        )
        self.attn_dropout = layers.Dropout(self.dropout_rate, name="attn_dropout")

        # Pre-FFN normalization & Feed-Forward Network
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name="pre_ffn_ln")
        self.ffn_dense1 = layers.Dense(
            self.d_ff, activation=self.ffn_activation, name="ffn_dense_1"
        )
        self.ffn_dropout1 = layers.Dropout(self.dropout_rate, name="ffn_dropout_1")
        self.ffn_dense2 = layers.Dense(self.d_model, name="ffn_dense_2")
        self.ffn_dropout2 = layers.Dropout(self.dropout_rate, name="ffn_dropout_2")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build sublayers for strict Keras 3 compatibility."""
        self.ln1.build(input_shape)
        self.mha.build(input_shape, input_shape, input_shape)
        self.ln2.build(input_shape)
        self.ffn_dense1.build(input_shape)
        ffn_mid_shape = (
            input_shape[0],
            input_shape[1],
            self.d_ff,
        ) if len(input_shape) == 3 else (None, self.d_ff)
        self.ffn_dense2.build(ffn_mid_shape)
        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Execute forward pass with Pre-LN residual connections."""
        # 1. Multi-Head Self-Attention Sublayer (Pre-LN)
        norm1 = self.ln1(inputs)
        attn_output = self.mha(
            query=norm1, key=norm1, value=norm1, training=training
        )
        attn_output = self.attn_dropout(attn_output, training=training)
        x = inputs + attn_output

        # 2. Feed-Forward Sublayer (Pre-LN)
        norm2 = self.ln2(x)
        ffn_output = self.ffn_dense1(norm2)
        ffn_output = self.ffn_dropout1(ffn_output, training=training)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.ffn_dropout2(ffn_output, training=training)
        out = x + ffn_output

        return out

    def get_config(self) -> Dict[str, Any]:
        """Serialize layer configuration."""
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "d_ff": self.d_ff,
                "ffn_activation": self.ffn_activation,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> TransformerEncoderBlock:
        """Instantiate layer from configuration dict."""
        return cls(**config)


def build_transformer_classifier(
    seq_len: int = 128,
    num_features: int = 9,
    d_model: int = 64,
    num_heads: int = 4,
    key_dim: int = 16,
    num_layers: int = 2,
    d_ff: int = 128,
    ffn_activation: str = "gelu",
    dropout_rate: float = 0.2,
    dense_units: int = 64,
    num_classes: int = 6,
    config: Optional[Union[Dict[str, Any], str]] = None,
    name: str = "har_transformer_classifier",
) -> keras.Model:
    """Build and return the configurable Transformer classifier for HAR.

    Architecture summary:
        1. Input: (seq_len, num_features) -> e.g. (128, 9)
        2. Dense Projection: (seq_len, d_model) -> (128, 64)
        3. Trainable Positional Embedding: adds (seq_len, d_model)
        4. N Transformer Encoder Blocks (Pre-LN, MHA, GELU FFN, Residuals)
        5. Final Layer Normalization
        6. Global Average Pooling (128, 64) -> (64,)
        7. Dense Classification Head: Dense(64, relu) -> Dropout(0.2)
        8. Output Softmax: Dense(6, softmax) -> Activity probabilities

    Parameters:
        seq_len: Sequence length of input window (default: 128).
        num_features: Number of input sensor channels (default: 9).
        d_model: Latent projection and transformer dimension (default: 64).
        num_heads: Number of attention heads (default: 4).
        key_dim: Attention key dimensionality per head (default: 16).
        num_layers: Number of stacked transformer encoder blocks (default: 2).
        d_ff: Hidden units in the feed-forward network (default: 128).
        ffn_activation: Activation in FFN intermediate layer (default: "gelu").
        dropout_rate: Dropout rate (default: 0.2).
        dense_units: Units in pre-classification dense layer (default: 64).
        num_classes: Number of target activity classes (default: 6).
        config: Optional dict or path to JSON file overriding parameters.
        name: Name of the Keras model.

    Returns:
        keras.Model: Configured, uncompiled Keras Functional model.
    """
    if config is not None:
        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # Extract nested 'model' dictionary if present
            cfg = loaded.get("model", loaded)
        elif isinstance(config, dict):
            cfg = config.get("model", config)
        else:
            raise TypeError("config must be a dict or a file path string.")

        seq_len = int(cfg.get("seq_len", seq_len))
        num_features = int(cfg.get("num_features", num_features))
        d_model = int(cfg.get("d_model", d_model))
        num_heads = int(cfg.get("num_heads", num_heads))
        key_dim = int(cfg.get("key_dim", key_dim))
        num_layers = int(cfg.get("num_layers", num_layers))
        d_ff = int(cfg.get("d_ff", d_ff))
        ffn_activation = str(cfg.get("ffn_activation", ffn_activation))
        dropout_rate = float(cfg.get("dropout_rate", dropout_rate))
        dense_units = int(cfg.get("dense_units", dense_units))
        num_classes = int(cfg.get("num_classes", num_classes))

    # 1. Input Layer
    inputs = layers.Input(
        shape=(seq_len, num_features),
        dtype="float32",
        name="sensor_window_input",
    )

    # 2. Linear / Dense Projection to d_model
    x = layers.Dense(d_model, name="input_projection")(inputs)

    # 3. Trainable Positional Embedding
    x = TrainablePositionalEmbedding(
        seq_len=seq_len, d_model=d_model, name="trainable_positional_embedding"
    )(x)
    x = layers.Dropout(dropout_rate, name="pos_dropout")(x)

    # 4. Stacked Transformer Encoder Blocks
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            key_dim=key_dim,
            d_ff=d_ff,
            ffn_activation=ffn_activation,
            dropout_rate=dropout_rate,
            name=f"transformer_encoder_block_{i+1}",
        )(x)

    # 5. Final Layer Normalization before pooling (Pre-LN standard)
    x = layers.LayerNormalization(epsilon=1e-6, name="encoder_final_ln")(x)

    # 6. Global Average Pooling over time dimension
    x = layers.GlobalAveragePooling1D(name="global_average_pooling_1d")(x)

    # 7. Classification Head
    x = layers.Dense(dense_units, activation="relu", name="dense_classification_head")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    # 8. Activity Class Probabilities Output
    outputs = layers.Dense(
        num_classes, activation="softmax", name="activity_probabilities"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def compile_transformer_model(
    model: keras.Model,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Compile the Transformer model with standard training settings.

    Optimizer: Adam
    Loss: SparseCategoricalCrossentropy
    Metrics: Accuracy

    Parameters:
        model: Uncompiled Keras model.
        learning_rate: Initial learning rate (default: 0.001).

    Returns:
        Compiled Keras model.
    """
    optimizer = keras.optimizers.Adam(learning_rate=float(learning_rate))
    loss = keras.losses.SparseCategoricalCrossentropy()
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def load_transformer_model(filepath: str) -> keras.Model:
    """Safely load a serialized .keras Transformer model with custom layer registry.

    Parameters:
        filepath: Path to the saved .keras model file.

    Returns:
        keras.Model: Loaded Keras model ready for inference or fine-tuning.
    """
    custom_objects = {
        "TrainablePositionalEmbedding": TrainablePositionalEmbedding,
        "TransformerEncoderBlock": TransformerEncoderBlock,
    }
    return models.load_model(filepath, custom_objects=custom_objects)


def predict_transformer(
    model: keras.Model,
    x: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Generate activity probabilities using the trained Transformer model.

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


def predict_single_window(
    model: keras.Model,
    window: np.ndarray,
) -> Tuple[int, str, float, np.ndarray]:
    """Classify a single (128, 9) sensor window and return prediction details.

    Parameters:
        model: Trained Keras Transformer model.
        window: Array shaped (128, 9) or (1, 128, 9).

    Returns:
        Tuple containing:
            - predicted_class: int class index (0-5)
            - activity_name: str descriptive name
            - confidence: float probability (0.0 - 1.0)
            - probabilities: np.ndarray 1D probability distribution of shape (6,)
    """
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape != (128, 9):
            raise ValueError(f"Expected 2D window shaped (128, 9), got {arr.shape}.")
        arr = np.expand_dims(arr, axis=0)
    elif arr.ndim == 3:
        if arr.shape != (1, 128, 9):
            raise ValueError(f"Expected single 3D window shaped (1, 128, 9), got {arr.shape}.")
    else:
        raise ValueError(f"Window must be 2D (128, 9) or 3D (1, 128, 9), got shape {arr.shape}.")

    from src.data_contract import ACTIVITY_LABEL_MAPPING

    probs = predict_transformer(model, arr, batch_size=1)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    act_name = ACTIVITY_LABEL_MAPPING.get(pred_idx, f"CLASS_{pred_idx}")
    return pred_idx, act_name, conf, probs

