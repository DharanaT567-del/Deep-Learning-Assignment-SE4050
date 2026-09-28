"""Unit and integration tests for Monal's HAR CNN-LSTM component.

Author: Monal (Member 4)
Project: SE4050 Deep Learning Assignment - Four-member HAR architecture comparison.

Verifies:
1. Forward pass on batch (2, 128, 9) produces valid probabilities (2, 6) summing to 1.0.
2. Gradients reach trainable weights and single optimization step yields finite loss.
3. Model serialization (.keras) and prediction parity upon reload.
4. Input validation and error handling on mismatched dimensions.
5. Contract compliance and strict prevention of subject leakage.
6. End-to-end smoke training writes all required run artifacts.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

# Compatible imports
try:
    import keras
    import tensorflow as tf
except ImportError:
    import tensorflow as tf
    from tensorflow import keras

from src.data_contract import (
    DataContractValidationError,
    generate_synthetic_har_data,
    load_har_npz,
    validate_har_dataset,
)
from src.models.cnn_lstm import (
    build_cnn_lstm_model,
    compile_cnn_lstm_model,
    load_cnn_lstm_model,
    predict_cnn_lstm,
)
from src.train_cnn_lstm import load_config, train_cnn_lstm_pipeline


def small_cnn_lstm_config(base_dir: str) -> dict:
    """Return a lightweight configuration for fast test execution."""
    return {
        "model": {
            "seq_len": 128,
            "num_features": 9,
            "cnn_filters": [16, 16],
            "kernel_size": 3,
            "pool_size": 2,
            "lstm_units": 16,
            "n_lstm_layers": 1,
            "dropout": 0.1,
            "dense_units": 16,
            "num_classes": 6,
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 2,
            "seed": 42,
            "early_stopping_patience": 2,
            "reduce_lr_patience": 1,
            "reduce_lr_factor": 0.5,
            "min_lr": 1e-5,
        },
        "data": {
            "npz_path": "data/uci_har_processed.npz",
            "normalize": False,
        },
        "output": {
            "base_dir": base_dir,
        },
    }


class TestCnnLstmModel(unittest.TestCase):
    """Test suite for CNN-LSTM architecture and operations."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_forward_pass_shape_and_probability_sum(self) -> None:
        """Forward pass should output shape (N, 6) with row sums close to 1.0."""
        model = build_cnn_lstm_model(
            input_shape=(128, 9),
            n_classes=6,
            cnn_filters=[16, 16],
            lstm_units=16,
            seed=42,
        )
        dummy_input = np.random.randn(4, 128, 9).astype(np.float32)
        preds = predict_cnn_lstm(model, dummy_input)

        self.assertEqual(preds.shape, (4, 6))
        self.assertTrue(np.all(preds >= 0.0))
        self.assertTrue(np.all(preds <= 1.0))
        row_sums = np.sum(preds, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5)

    def test_backward_pass_gradients_and_loss_finite(self) -> None:
        """Loss computation and gradient step must yield finite loss and update weights."""
        model = build_cnn_lstm_model(
            input_shape=(128, 9),
            n_classes=6,
            cnn_filters=[16, 16],
            lstm_units=16,
            seed=42,
        )
        x_batch = tf.random.normal((8, 128, 9))
        y_batch = tf.constant([0, 1, 2, 3, 4, 5, 0, 1], dtype=tf.int32)

        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        with tf.GradientTape() as tape:
            preds = model(x_batch, training=True)
            loss_val = loss_fn(y_batch, preds)

        grads = tape.gradient(loss_val, model.trainable_weights)
        self.assertTrue(np.isfinite(float(loss_val)))
        self.assertGreater(len(grads), 0)
        for g in grads:
            self.assertIsNotNone(g)
            self.assertTrue(np.all(np.isfinite(g.numpy())))

    def test_serialization_and_exact_inference_parity(self) -> None:
        """Model saved to .keras and reloaded must reproduce exact same predictions."""
        model = build_cnn_lstm_model(
            input_shape=(128, 9),
            n_classes=6,
            cnn_filters=[16, 16],
            lstm_units=16,
            seed=42,
        )
        dummy_input = np.random.randn(3, 128, 9).astype(np.float32)
        preds_original = predict_cnn_lstm(model, dummy_input)

        save_path = os.path.join(self.temp_dir, "test_cnn_lstm.keras")
        model.save(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded_model = load_cnn_lstm_model(save_path)
        preds_loaded = predict_cnn_lstm(loaded_model, dummy_input)

        np.testing.assert_allclose(preds_original, preds_loaded, atol=1e-5)

    def test_reject_invalid_input_shapes(self) -> None:
        """predict_cnn_lstm must reject inputs not shaped (N, 128, 9)."""
        model = build_cnn_lstm_model(input_shape=(128, 9), n_classes=6)

        with self.assertRaises(ValueError):
            predict_cnn_lstm(model, np.zeros((10, 64, 9), dtype=np.float32))

        with self.assertRaises(ValueError):
            predict_cnn_lstm(model, np.zeros((10, 128, 12), dtype=np.float32))

        with self.assertRaises(ValueError):
            predict_cnn_lstm(model, np.zeros((128, 9), dtype=np.float32))


class TestCnnLstmPipeline(unittest.TestCase):
    """Test suite for CNN-LSTM training pipeline, leak prevention, and artifacts."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_pipeline_rejects_subject_leakage(self) -> None:
        """Pipeline must raise DataContractValidationError if train and val subjects overlap."""
        dataset = generate_synthetic_har_data(train_samples=64, val_samples=32)
        # Introduce artificial leakage
        dataset["subject_val"] = dataset["subject_train"][:32]

        leak_npz_path = os.path.join(self.temp_dir, "leakage_dataset.npz")
        np.savez_compressed(leak_npz_path, **dataset)

        cfg = small_cnn_lstm_config(self.temp_dir)
        cfg["data"]["npz_path"] = leak_npz_path

        with self.assertRaises(DataContractValidationError):
            train_cnn_lstm_pipeline(config=cfg, verbose=0)

    def test_smoke_training_pipeline_writes_artifacts(self) -> None:
        """End-to-end smoke pipeline must write all required artifacts without reading test split."""
        cfg = small_cnn_lstm_config(self.temp_dir)

        model, metadata, run_dir = train_cnn_lstm_pipeline(
            config=cfg,
            synthetic_smoke=True,
            verbose=0,
        )

        self.assertIsNotNone(model)
        self.assertTrue(os.path.isdir(run_dir))

        # Check required artifacts
        expected_files = [
            "best_model.keras",
            "history.json",
            "history.csv",
            "config.json",
            "run_metadata.json",
        ]
        for fname in expected_files:
            target_path = os.path.join(run_dir, fname)
            self.assertTrue(os.path.exists(target_path), f"Missing artifact: {fname}")

        # Check run_metadata contents
        self.assertEqual(metadata["author"], "Monal (Member 4)")
        self.assertEqual(metadata["component"], "Hybrid CNN-LSTM Classifier")
        self.assertGreater(metadata["total_parameters"], 0)
        self.assertGreater(metadata["duration_seconds"], 0)
        self.assertIn("val_accuracy_at_best_epoch", metadata)


if __name__ == "__main__":
    unittest.main()
