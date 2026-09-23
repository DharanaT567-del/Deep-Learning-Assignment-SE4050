"""Unit and integration tests for Dharana's HAR Transformer component.

Verifies:
1. Model forward pass on synthetic batch (2, 128, 9).
2. Output shape (2, 6), finite values, and valid probability simplex (sum to 1.0).
3. Synthetic training step with gradient computation.
4. Model serialization (.keras) and exact prediction matching upon reloading.
5. Strict rejection of malformed inputs, NaNs, and overlapping train/val subjects.
6. End-to-end smoke training run generating all specified artifacts.
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
    apply_training_standardization,
    generate_synthetic_har_data,
    validate_har_dataset,
    validate_har_split,
)
from src.models.transformer import (
    TrainablePositionalEmbedding,
    TransformerEncoderBlock,
    build_transformer_classifier,
    compile_transformer_model,
    load_transformer_model,
    predict_transformer,
)
from src.train_transformer import train_transformer_pipeline


class TestTransformerModel(unittest.TestCase):
    """Test suite for Transformer architecture and inference."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.seq_len = 128
        self.num_features = 9
        self.num_classes = 6
        self.batch_size = 2

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_model_forward_pass_and_output_contract(self) -> None:
        """1 & 2: Check input batch (2, 128, 9) and valid probability output (2, 6)."""
        model = build_transformer_classifier(
            seq_len=self.seq_len,
            num_features=self.num_features,
            d_model=64,
            num_heads=4,
            key_dim=16,
            num_layers=2,
            d_ff=128,
            dropout_rate=0.2,
            dense_units=64,
            num_classes=self.num_classes,
        )

        dummy_batch = np.random.randn(self.batch_size, self.seq_len, self.num_features).astype(np.float32)
        preds = model(dummy_batch, training=False).numpy()

        # Check shape
        self.assertEqual(preds.shape, (self.batch_size, self.num_classes))

        # Check finiteness (no NaNs or Infs)
        self.assertTrue(np.all(np.isfinite(preds)), "Model output contains NaN or Inf.")

        # Check non-negativity
        self.assertTrue(np.all(preds >= 0.0), "Probabilities must be non-negative.")

        # Check probabilities sum to 1.0 per sample
        sums = np.sum(preds, axis=-1)
        np.testing.assert_allclose(sums, np.ones(self.batch_size), rtol=1e-5, atol=1e-5)

    def test_02_gradient_and_training_step(self) -> None:
        """3: Run one synthetic training step to verify loss and backprop."""
        model = build_transformer_classifier(
            seq_len=self.seq_len,
            num_features=self.num_features,
            d_model=64,
            num_heads=4,
            key_dim=16,
            num_layers=2,
            d_ff=128,
            num_classes=self.num_classes,
        )
        model = compile_transformer_model(model, learning_rate=0.001)

        x_batch = np.random.randn(self.batch_size, self.seq_len, self.num_features).astype(np.float32)
        y_batch = np.array([0, 5], dtype=np.int64)

        # Execute single gradient update step via train_on_batch
        loss_before = model.evaluate(x_batch, y_batch, verbose=0)[0]
        self.assertTrue(np.isfinite(loss_before), "Initial loss must be finite.")

        result = model.train_on_batch(x_batch, y_batch)
        train_loss = result[0] if isinstance(result, (list, tuple)) else result
        self.assertTrue(np.isfinite(train_loss), "Train step loss must be finite.")

    def test_03_serialization_and_reload(self) -> None:
        """4: Save and reload .keras model; verify identical inference."""
        model = build_transformer_classifier(
            seq_len=self.seq_len,
            num_features=self.num_features,
            d_model=64,
            num_heads=4,
            key_dim=16,
            num_layers=2,
            num_classes=self.num_classes,
        )
        model = compile_transformer_model(model)

        test_input = np.random.randn(4, self.seq_len, self.num_features).astype(np.float32)
        original_preds = model(test_input, training=False).numpy()

        save_path = os.path.join(self.temp_dir, "test_transformer.keras")
        model.save(save_path)
        self.assertTrue(os.path.exists(save_path), "Model file .keras was not saved.")

        reloaded_model = load_transformer_model(save_path)
        reloaded_preds = reloaded_model(test_input, training=False).numpy()

        np.testing.assert_allclose(
            original_preds,
            reloaded_preds,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Reloaded model predictions do not match original model.",
        )

        # Also test predict_transformer helper
        helper_preds = predict_transformer(reloaded_model, test_input)
        np.testing.assert_allclose(original_preds, helper_preds, rtol=1e-5, atol=1e-5)


class TestDataContractAndLeakagePrevention(unittest.TestCase):
    """Test suite for data integrity and subject leakage prevention."""

    def test_04_synthetic_data_generation_contract(self) -> None:
        """Verify synthetic generator meets all contract criteria."""
        dataset = generate_synthetic_har_data(
            n_train_windows_per_class=10,
            n_val_windows_per_class=5,
            n_test_windows_per_class=5,
            seed=42,
        )
        # Should pass without raising error
        validate_har_dataset(dataset, check_test=True)

        train_subs = set(np.unique(dataset["subject_train"]))
        val_subs = set(np.unique(dataset["subject_val"]))
        self.assertEqual(len(train_subs.intersection(val_subs)), 0, "Subjects must not overlap.")

    def test_05_rejection_of_subject_leakage(self) -> None:
        """5: Verify overlapping train and val subjects are strictly rejected."""
        dataset = generate_synthetic_har_data(n_train_windows_per_class=5, n_val_windows_per_class=5)
        # Introduce intentional subject overlap leakage
        dataset["subject_val"][0] = dataset["subject_train"][0]

        with self.assertRaises(DataContractValidationError) as ctx:
            validate_har_dataset(dataset)
        self.assertIn("CRITICAL DATA LEAKAGE", str(ctx.exception))

    def test_06_rejection_of_malformed_inputs(self) -> None:
        """5: Verify malformed shapes, NaNs, and out-of-range labels are rejected."""
        # 1. Incorrect shape (e.g. 10 channels instead of 9)
        with self.assertRaises(DataContractValidationError):
            validate_har_split(
                x=np.zeros((10, 128, 10), dtype=np.float32),
                y=np.zeros(10, dtype=int),
                subject=np.ones(10, dtype=int),
            )

        # 2. Contains NaNs
        nan_x = np.zeros((10, 128, 9), dtype=np.float32)
        nan_x[0, 0, 0] = np.nan
        with self.assertRaises(DataContractValidationError):
            validate_har_split(
                x=nan_x,
                y=np.zeros(10, dtype=int),
                subject=np.ones(10, dtype=int),
            )

        # 3. Invalid class label (e.g. 6 or negative)
        with self.assertRaises(DataContractValidationError):
            validate_har_split(
                x=np.zeros((10, 128, 9), dtype=np.float32),
                y=np.array([0, 1, 2, 3, 4, 6, 0, 1, 2, 3], dtype=int),
                subject=np.ones(10, dtype=int),
            )

    def test_07_normalization_isolation(self) -> None:
        """Verify normalization statistics are fitted strictly on X_train."""
        X_tr = np.full((10, 128, 9), 10.0, dtype=np.float32)
        X_val = np.full((5, 128, 9), 20.0, dtype=np.float32)

        X_tr_norm, X_val_norm, _, stats = apply_training_standardization(X_tr, X_val)

        # Train mean is 10.0
        np.testing.assert_allclose(stats["channel_means"], [10.0] * 9)
        # Train should become zeros (since std is 0 -> default 1.0)
        np.testing.assert_allclose(X_tr_norm, np.zeros((10, 128, 9)))
        # Val should be (20 - 10) / 1.0 = 10.0 (not scaled to val mean)
        np.testing.assert_allclose(X_val_norm, np.full((5, 128, 9), 10.0))


class TestTrainingPipeline(unittest.TestCase):
    """End-to-end test of training script artifact generation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_08_end_to_end_smoke_training(self) -> None:
        """6: Verify training pipeline runs and saves all required artifacts."""
        config = {
            "model": {
                "seq_len": 128,
                "num_features": 9,
                "d_model": 32,
                "num_heads": 2,
                "key_dim": 16,
                "num_layers": 1,
                "d_ff": 64,
                "dropout_rate": 0.1,
                "dense_units": 32,
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
                "min_lr": 1e-6,
            },
            "data": {},
            "output": {
                "base_dir": self.temp_dir,
            },
        }

        data = generate_synthetic_har_data(
            n_train_windows_per_class=6,
            n_val_windows_per_class=3,
            n_test_windows_per_class=3,
            seed=42,
        )

        run_dir = os.path.join(self.temp_dir, "run_test")
        model, metadata, saved_run_dir = train_transformer_pipeline(
            config=config,
            data=data,
            run_dir=run_dir,
            verbose=0,
        )

        # Check that artifacts exist
        self.assertTrue(os.path.exists(os.path.join(run_dir, "best_model.keras")))
        self.assertTrue(os.path.exists(os.path.join(run_dir, "history.json")))
        self.assertTrue(os.path.exists(os.path.join(run_dir, "history.csv")))
        self.assertTrue(os.path.exists(os.path.join(run_dir, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(run_dir, "run_metadata.json")))

        # Check metadata contents
        with open(os.path.join(run_dir, "run_metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["author"], "Dharana")
        self.assertIn("total_parameters", meta)
        self.assertIn("train_subjects", meta)
        self.assertIn("val_subjects", meta)
        self.assertIn("duration_seconds", meta)


if __name__ == "__main__":
    unittest.main()
