"""Unit and integration tests for Member 3's HAR BiLSTM component.

Mirrors tests/test_transformer.py. Verifies:
1. Model forward pass on a synthetic batch (2, 128, 9) gives valid probabilities (2, 6).
2. Gradients reach every trainable weight and a training step gives a finite loss.
3. Model serialization (.keras) and exact prediction matching upon reloading.
4. Contract compliance of the pipeline's input data.
5. The training pipeline rejects overlapping train/val subjects.
6. Malformed inputs are rejected by predict_bilstm and the pipeline.
7. NPZ normalization is fitted on the train split only.
8. End-to-end smoke training writes all artifacts and never reads X_test.
9. Attention pooling builds, adds 129 parameters, and reloads with identical predictions.
10. Average pooling adds no parameters; an unknown pooling option is rejected.
11. augment_batch keeps shape and is the identity when all settings are 0.
12. Augmented training never changes validation data and never reads X_test.
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
from src.models.bilstm import (
    build_bilstm_model,
    load_bilstm_model,
    predict_bilstm,
)
from src.train_bilstm import augment_batch, load_config, train_bilstm_pipeline


def small_pipeline_config(base_dir: str) -> dict:
    """Return a tiny, fast BiLSTM config for pipeline tests."""
    return {
        "model": {
            "seq_len": 128,
            "num_features": 9,
            "lstm_units": 8,
            "n_layers": 1,
            "dropout": 0.1,
            "dense_units": 8,
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
            "base_dir": base_dir,
        },
    }


class TestBiLSTMModel(unittest.TestCase):
    """Test suite for BiLSTM architecture and inference."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.seq_len = 128
        self.num_features = 9
        self.num_classes = 6
        self.batch_size = 2

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_model_forward_pass_and_output_contract(self) -> None:
        """Input batch (2, 128, 9) gives a valid probability output (2, 6)."""
        model = build_bilstm_model(
            input_shape=(self.seq_len, self.num_features),
            n_classes=self.num_classes,
            lstm_units=64,
            n_layers=2,
            dropout=0.4,
            dense_units=64,
        )

        dummy_batch = np.random.randn(self.batch_size, self.seq_len, self.num_features).astype(np.float32)
        preds = model(dummy_batch, training=False).numpy()

        self.assertEqual(preds.shape, (self.batch_size, self.num_classes))
        self.assertTrue(np.all(np.isfinite(preds)), "Model output contains NaN or Inf.")
        self.assertTrue(np.all(preds >= 0.0), "Probabilities must be non-negative.")
        np.testing.assert_allclose(np.sum(preds, axis=-1), np.ones(self.batch_size), rtol=1e-5, atol=1e-5)

        # Two stacked bidirectional layers: first returns sequences, last returns the final state
        self.assertEqual(tuple(model.get_layer("bilstm_1").output.shape), (None, self.seq_len, 128))
        self.assertEqual(tuple(model.get_layer("bilstm_2").output.shape), (None, 128))

    def test_02_gradient_and_training_step(self) -> None:
        """Gradients reach every trainable weight and one training step gives a finite loss."""
        model = build_bilstm_model(lstm_units=16, n_layers=2, dense_units=16)

        x_batch = np.random.randn(self.batch_size, self.seq_len, self.num_features).astype(np.float32)
        y_batch = np.array([0, 5], dtype=np.int64)

        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        with tf.GradientTape() as tape:
            loss = loss_fn(y_batch, model(x_batch, training=True))
        grads = tape.gradient(loss, model.trainable_weights)

        self.assertTrue(np.isfinite(float(loss)), "Initial loss must be finite.")
        for weight, grad in zip(model.trainable_weights, grads):
            self.assertIsNotNone(grad, f"No gradient for '{weight.name}'.")
            self.assertTrue(np.all(np.isfinite(grad)), f"Non-finite gradient for '{weight.name}'.")
            self.assertGreater(float(tf.reduce_sum(tf.abs(grad))), 0.0, f"Zero gradient for '{weight.name}'.")

        result = model.train_on_batch(x_batch, y_batch)
        train_loss = result[0] if isinstance(result, (list, tuple)) else result
        self.assertTrue(np.isfinite(train_loss), "Train step loss must be finite.")

        # Gradient clipping is configured on the optimizer
        self.assertEqual(model.optimizer.clipnorm, 1.0)

    def test_03_serialization_and_reload(self) -> None:
        """Save and reload .keras model; verify identical inference."""
        model = build_bilstm_model(lstm_units=16, n_layers=2, dense_units=16)

        test_input = np.random.randn(4, self.seq_len, self.num_features).astype(np.float32)
        original_preds = model(test_input, training=False).numpy()

        save_path = os.path.join(self.temp_dir, "test_bilstm.keras")
        model.save(save_path)
        self.assertTrue(os.path.exists(save_path), "Model file .keras was not saved.")

        reloaded_model = load_bilstm_model(save_path)
        reloaded_preds = reloaded_model(test_input, training=False).numpy()

        np.testing.assert_allclose(
            original_preds,
            reloaded_preds,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Reloaded model predictions do not match original model.",
        )

        # Also test predict_bilstm helper
        helper_preds = predict_bilstm(reloaded_model, test_input)
        np.testing.assert_allclose(original_preds, helper_preds, rtol=1e-5, atol=1e-5)

    def test_09_attention_pooling_forward_and_reload(self) -> None:
        """Attention pooling builds, adds 129 parameters, and reloads with identical predictions."""
        last_model = build_bilstm_model(lstm_units=64, n_layers=2, dense_units=64)
        model = build_bilstm_model(lstm_units=64, n_layers=2, dense_units=64, pooling="attention")

        # Dense(1) scorer over 2 * 64 = 128 features: 128 weights + 1 bias
        self.assertEqual(model.count_params(), last_model.count_params() + 129)

        test_input = np.random.randn(4, self.seq_len, self.num_features).astype(np.float32)
        original_preds = model(test_input, training=False).numpy()
        self.assertEqual(original_preds.shape, (4, self.num_classes))
        np.testing.assert_allclose(np.sum(original_preds, axis=-1), np.ones(4), rtol=1e-5, atol=1e-5)

        # Attention weights are a distribution over the 128 time steps
        pool = model.get_layer("temporal_attention_pool")
        sequence = keras.Model(model.input, model.get_layer("bilstm_dropout_2").output)(test_input)
        weights = keras.ops.convert_to_numpy(pool.attention_weights(sequence))
        self.assertEqual(weights.shape, (4, self.seq_len))
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(4), rtol=1e-5, atol=1e-5)

        save_path = os.path.join(self.temp_dir, "test_bilstm_attention.keras")
        model.save(save_path)
        reloaded_preds = load_bilstm_model(save_path)(test_input, training=False).numpy()
        np.testing.assert_allclose(original_preds, reloaded_preds, rtol=1e-5, atol=1e-5)

    def test_10_avg_pooling_and_invalid_option(self) -> None:
        """Average pooling adds no parameters; an unknown pooling option is rejected."""
        last_model = build_bilstm_model(lstm_units=16, n_layers=2, dense_units=16)
        avg_model = build_bilstm_model(lstm_units=16, n_layers=2, dense_units=16, pooling="avg")
        self.assertEqual(avg_model.count_params(), last_model.count_params())
        self.assertEqual(tuple(avg_model.get_layer("bilstm_2").output.shape), (None, self.seq_len, 32))

        with self.assertRaises(ValueError):
            build_bilstm_model(lstm_units=8, n_layers=1, dense_units=8, pooling="max")


class TestBiLSTMDataContract(unittest.TestCase):
    """Test suite for data integrity and leakage prevention in the BiLSTM pipeline."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_04_contract_compliance(self) -> None:
        """Pipeline input passes the contract and the config file matches the spec."""
        dataset = generate_synthetic_har_data(
            n_train_windows_per_class=10,
            n_val_windows_per_class=5,
            n_test_windows_per_class=5,
            seed=42,
        )
        validate_har_dataset(dataset, check_test=True)

        config = load_config("configs/bilstm.json")
        self.assertEqual(config["model"]["lstm_units"], 64)
        self.assertEqual(config["model"]["n_layers"], 2)
        self.assertEqual(config["model"]["dropout"], 0.4)
        self.assertEqual(config["training"]["learning_rate"], 0.0005)
        self.assertEqual(config["output"]["base_dir"], "outputs/bilstm")
        # Improvements are opt-in: the default config is the original baseline
        self.assertEqual(config["model"]["pooling"], "last")
        self.assertFalse(config["training"]["augment"])

    def test_05_rejection_of_subject_leakage(self) -> None:
        """The training pipeline refuses to train on overlapping train/val subjects."""
        dataset = generate_synthetic_har_data(n_train_windows_per_class=5, n_val_windows_per_class=5)
        dataset["subject_val"][0] = dataset["subject_train"][0]

        with self.assertRaises(DataContractValidationError) as ctx:
            train_bilstm_pipeline(
                small_pipeline_config(self.temp_dir),
                data=dataset,
                run_dir=os.path.join(self.temp_dir, "leaky_run"),
                verbose=0,
            )
        self.assertIn("CRITICAL DATA LEAKAGE", str(ctx.exception))

    def test_06_rejection_of_malformed_inputs(self) -> None:
        """predict_bilstm and the pipeline reject wrong shapes and NaNs."""
        model = build_bilstm_model(lstm_units=8, n_layers=1, dense_units=8)

        for bad_shape in [(4, 128, 10), (4, 64, 9), (128, 9)]:
            with self.assertRaises(ValueError, msg=f"shape {bad_shape}"):
                predict_bilstm(model, np.zeros(bad_shape, dtype=np.float32))

        dataset = generate_synthetic_har_data(n_train_windows_per_class=5, n_val_windows_per_class=5)
        dataset["X_train"][0, 0, 0] = np.nan
        with self.assertRaises(DataContractValidationError):
            train_bilstm_pipeline(
                small_pipeline_config(self.temp_dir),
                data=dataset,
                run_dir=os.path.join(self.temp_dir, "nan_run"),
                verbose=0,
            )

    def test_07_normalization_isolation(self) -> None:
        """NPZ normalization used by the pipeline is fitted on X_train only."""
        dataset = generate_synthetic_har_data(n_train_windows_per_class=5, n_val_windows_per_class=5)
        dataset["X_val"] = dataset["X_val"] + 3.0
        npz_path = os.path.join(self.temp_dir, "shifted.npz")
        np.savez(npz_path, **dataset)

        loaded = load_har_npz(npz_path, normalize=True)

        np.testing.assert_allclose(loaded["X_train"].mean(axis=(0, 1)), np.zeros(9), atol=1e-4)
        # Val was shifted +3 before scaling, so with train statistics it cannot be centred on 0
        self.assertTrue(np.all(np.abs(loaded["X_val"].mean(axis=(0, 1))) > 1.0))


class TestBiLSTMTrainingPipeline(unittest.TestCase):
    """End-to-end test of training script artifact generation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_08_end_to_end_smoke_training(self) -> None:
        """Pipeline runs, saves all artifacts, and never reads the test split."""
        data = generate_synthetic_har_data(
            n_train_windows_per_class=6,
            n_val_windows_per_class=3,
            n_test_windows_per_class=3,
            seed=42,
        )
        # Poison the test split: if training touched it, the loss would go NaN
        data["X_test"] = np.full_like(data["X_test"], np.nan)

        run_dir = os.path.join(self.temp_dir, "run_test")
        model, metadata, saved_run_dir = train_bilstm_pipeline(
            config=small_pipeline_config(self.temp_dir),
            data=data,
            run_dir=run_dir,
            verbose=0,
        )

        self.assertEqual(saved_run_dir, run_dir)
        for artifact in ["best_model.keras", "history.json", "history.csv", "config.json", "run_metadata.json"]:
            self.assertTrue(os.path.exists(os.path.join(run_dir, artifact)), f"Missing {artifact}")

        with open(os.path.join(run_dir, "history.json"), "r", encoding="utf-8") as f:
            history = json.load(f)
        self.assertTrue(np.all(np.isfinite(history["loss"])))
        self.assertTrue(np.all(np.isfinite(history["val_loss"])))

        with open(os.path.join(run_dir, "run_metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["author"], "Member 3")
        self.assertIn("total_parameters", meta)
        self.assertIn("duration_seconds", meta)
        self.assertIn("seconds_per_epoch", meta)
        self.assertEqual(meta["train_subjects"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(meta["val_subjects"], [7, 8])

        # Reloaded best model still predicts valid probabilities
        preds = predict_bilstm(model, data["X_val"])
        self.assertEqual(preds.shape, (len(data["X_val"]), 6))

    def test_11_augment_batch_properties(self) -> None:
        """augment_batch keeps shape and dtype, and is the identity when all settings are 0."""
        x = np.random.randn(5, 128, 9).astype(np.float32)

        unchanged = augment_batch(x, jitter_std=0.0, scale_std=0.0, max_shift=0).numpy()
        np.testing.assert_array_equal(unchanged, x)

        augmented = augment_batch(x, jitter_std=0.05, scale_std=0.1, max_shift=8).numpy()
        self.assertEqual(augmented.shape, x.shape)
        self.assertEqual(augmented.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(augmented)))
        self.assertFalse(np.allclose(augmented, x), "Augmentation should change the batch.")

        # A pure time shift only reorders readings within each window
        shifted = augment_batch(x, jitter_std=0.0, scale_std=0.0, max_shift=8).numpy()
        np.testing.assert_allclose(np.sort(shifted, axis=1), np.sort(x, axis=1), rtol=1e-6)

    def test_12_augmented_training_is_train_only(self) -> None:
        """Augmented training never changes validation data and never reads X_test."""
        data = generate_synthetic_har_data(
            n_train_windows_per_class=6,
            n_val_windows_per_class=3,
            n_test_windows_per_class=3,
            seed=42,
        )
        data["X_test"] = np.full_like(data["X_test"], np.nan)

        config = small_pipeline_config(self.temp_dir)
        config["model"]["pooling"] = "attention"
        config["training"]["augment"] = True

        run_dir = os.path.join(self.temp_dir, "run_aug")
        model, metadata, _ = train_bilstm_pipeline(config=config, data=data, run_dir=run_dir, verbose=0)

        with open(os.path.join(run_dir, "history.json"), "r", encoding="utf-8") as f:
            history = json.load(f)
        self.assertTrue(np.all(np.isfinite(history["loss"])))
        self.assertTrue(metadata["augment"])
        self.assertEqual(metadata["pooling"], "attention")

        # If validation windows had been augmented, the recorded best val_loss would not
        # match a clean evaluation of the restored best model on the raw validation data
        clean_val_loss = model.evaluate(data["X_val"], data["y_val"], verbose=0)[0]
        self.assertAlmostEqual(clean_val_loss, min(history["val_loss"]), places=4)


if __name__ == "__main__":
    unittest.main()
