"""Training pipeline for Monal's CNN-LSTM Component on HAR dataset.

Author: Monal (Member 4)
Project: SE4050 Deep Learning Assignment - Four-member HAR architecture comparison.

This script manages:
1. Seed setting for reproducibility (Python, NumPy, TensorFlow).
2. Data contract validation and leakage prevention.
3. Hybrid CNN-LSTM model construction and compilation.
4. Keras callbacks: EarlyStopping, ModelCheckpoint (.keras), ReduceLROnPlateau, CSVLogger.
5. Structured run outputs: best_model.keras, history.json, history.csv, config.json, run_metadata.json.
6. Support for both real NPZ data and synthetic smoke validation.

Rule strictly followed: The test split is never used during training or hyperparameter tuning.
Model selection and early stopping use train and validation splits only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Compatible import of Keras / TensorFlow Keras
try:
    import keras
    import tensorflow as tf
except ImportError:
    import tensorflow as tf
    from tensorflow import keras

# Adjust Python path for direct script execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_contract import (
    DataContractValidationError,
    generate_synthetic_har_data,
    load_har_npz,
    validate_har_dataset,
)
from src.models.cnn_lstm import build_cnn_lstm_model, load_cnn_lstm_model, predict_cnn_lstm


def set_seed(seed: int = 42) -> None:
    """Set seeds across Python random, NumPy, and TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_system_metadata() -> Dict[str, Any]:
    """Capture environment, framework versions, and hardware device information."""
    devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "python_version": sys.version,
        "tensorflow_version": getattr(tf, "__version__", "unknown"),
        "keras_version": getattr(keras, "__version__", "unknown"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_devices": [d.name for d in devices],
        "gpu_available": len(gpus) > 0,
        "gpu_devices": [g.name for g in gpus],
    }


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load JSON config or fall back to default configuration values."""
    default_config: Dict[str, Any] = {
        "model": {
            "seq_len": 128,
            "num_features": 9,
            "cnn_filters": [64, 64],
            "kernel_size": 3,
            "pool_size": 2,
            "lstm_units": 64,
            "n_lstm_layers": 1,
            "dropout": 0.3,
            "dense_units": 64,
            "num_classes": 6,
        },
        "training": {
            "learning_rate": 0.0005,
            "batch_size": 64,
            "epochs": 60,
            "seed": 42,
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
            "reduce_lr_factor": 0.5,
            "min_lr": 1e-6,
        },
        "data": {
            "npz_path": "data/uci_har_processed.npz",
            "normalize": False,
        },
        "output": {
            "base_dir": "outputs/cnn_lstm",
        },
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        for section, values in user_config.items():
            if section in default_config and isinstance(values, dict):
                default_config[section].update(values)
            else:
                default_config[section] = values

    return default_config


def train_cnn_lstm_pipeline(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    synthetic_smoke: bool = False,
    override_epochs: Optional[int] = None,
    override_batch_size: Optional[int] = None,
    override_lr: Optional[float] = None,
    override_seed: Optional[int] = None,
    override_output_dir: Optional[str] = None,
    verbose: int = 1,
) -> Tuple[keras.Model, Dict[str, Any], str]:
    """Execute the end-to-end CNN-LSTM training pipeline.

    Parameters:
        config: In-memory configuration dictionary (takes priority if provided).
        config_path: Path to config JSON file.
        synthetic_smoke: If True, uses synthetic data to verify pipeline functionality.
        override_epochs: Override number of training epochs.
        override_batch_size: Override mini-batch size.
        override_lr: Override Adam learning rate.
        override_seed: Override random seed.
        override_output_dir: Override output destination directory.
        verbose: Keras verbosity level (0, 1, or 2).

    Returns:
        Tuple of (trained_model, run_metadata_dict, run_directory_path)
    """
    # 1. Resolve configuration and overrides
    cfg = load_config(config_path) if config is None else config
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    out_cfg = cfg.get("output", {})

    seed = int(override_seed if override_seed is not None else train_cfg.get("seed", 42))
    set_seed(seed)

    base_dir = override_output_dir or out_cfg.get("base_dir", "outputs/cnn_lstm")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f" Member 4: Hybrid CNN-LSTM Training Pipeline")
    print(f" Run Directory: {run_dir}")
    print(f" Random Seed:   {seed}")
    print(f"=======================================================\n")

    # 2. Data Loading and Validation
    if synthetic_smoke:
        print("[Data] Generating synthetic smoke HAR dataset...")
        har_dataset = generate_synthetic_har_data(
            train_samples=256,
            val_samples=64,
            test_samples=64,
            seq_len=model_cfg.get("seq_len", 128),
            channels=model_cfg.get("num_features", 9),
            num_classes=model_cfg.get("num_classes", 6),
            seed=seed,
        )
        data_source = "synthetic_smoke"
    else:
        npz_path = data_cfg.get("npz_path", "data/uci_har_processed.npz")
        # Ensure path resolution relative to project root
        if not os.path.isabs(npz_path):
            candidate = os.path.join(PROJECT_ROOT, npz_path)
            if os.path.exists(candidate):
                npz_path = candidate

        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Processed UCI HAR dataset not found at '{npz_path}'. "
                f"Please run 'python -m src.data.uci_har_loader --download' or supply synthetic_smoke=True."
            )

        print(f"[Data] Loading shared dataset from: {npz_path}")
        har_dataset = load_har_npz(npz_path)
        data_source = npz_path

    # Strictly validate data contract and subject leakage
    validate_har_dataset(har_dataset)

    X_train = har_dataset["X_train"]
    y_train = har_dataset["y_train"]
    subject_train = har_dataset["subject_train"]

    X_val = har_dataset["X_val"]
    y_val = har_dataset["y_val"]
    subject_val = har_dataset["subject_val"]

    # Subject leakage safety assertion
    train_subs = set(np.unique(subject_train))
    val_subs = set(np.unique(subject_val))
    overlap = train_subs.intersection(val_subs)
    if overlap:
        raise DataContractValidationError(f"CRITICAL: Data leakage detected! Overlapping subjects: {overlap}")

    train_subjects_list = [int(s) if hasattr(s, "item") else s for s in sorted(list(train_subs))]
    val_subjects_list = [int(s) if hasattr(s, "item") else s for s in sorted(list(val_subs))]

    print(f"[Data] Training split:   {X_train.shape[0]} windows from {len(train_subs)} subjects ({train_subjects_list})")
    print(f"[Data] Validation split: {X_val.shape[0]} windows from {len(val_subs)} subjects ({val_subjects_list})")
    print(f"[Data] Data contract confirmed: 0% subject leakage.")

    # 3. Model Construction and Compilation
    learning_rate = float(override_lr if override_lr is not None else train_cfg.get("learning_rate", 5e-4))
    cnn_filters = model_cfg.get("cnn_filters", [64, 64])
    kernel_size = int(model_cfg.get("kernel_size", 3))
    pool_size = int(model_cfg.get("pool_size", 2))
    lstm_units = int(model_cfg.get("lstm_units", 64))
    n_lstm_layers = int(model_cfg.get("n_lstm_layers", 1))
    dropout = float(model_cfg.get("dropout", 0.3))
    dense_units = int(model_cfg.get("dense_units", 64))
    num_classes = int(model_cfg.get("num_classes", 6))

    model = build_cnn_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        n_classes=num_classes,
        cnn_filters=cnn_filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        lstm_units=lstm_units,
        n_lstm_layers=n_lstm_layers,
        dropout=dropout,
        dense_units=dense_units,
        learning_rate=learning_rate,
        seed=seed,
    )

    total_params = int(model.count_params())
    trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable_params = total_params - trainable_params

    print(f"[Model] CNN-LSTM Model Built:")
    print(f"  CNN filters:          {cnn_filters}, kernel_size={kernel_size}, pool_size={pool_size}")
    print(f"  LSTM units:           {lstm_units}, layers={n_lstm_layers}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 4. Callbacks Configuration
    best_model_path = os.path.join(run_dir, "best_model.keras")
    history_csv_path = os.path.join(run_dir, "history.csv")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(train_cfg.get("early_stopping_patience", 10)),
            restore_best_weights=True,
            verbose=1 if verbose > 0 else 0,
            mode="min",
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1 if verbose > 0 else 0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=int(train_cfg.get("reduce_lr_patience", 5)),
            factor=float(train_cfg.get("reduce_lr_factor", 0.5)),
            min_lr=float(train_cfg.get("min_lr", 1e-6)),
            mode="min",
            verbose=1 if verbose > 0 else 0,
        ),
        keras.callbacks.CSVLogger(history_csv_path),
    ]

    # 5. Training Execution
    batch_size = int(override_batch_size if override_batch_size is not None else train_cfg.get("batch_size", 64))
    epochs = int(override_epochs if override_epochs is not None else train_cfg.get("epochs", 60))

    start_time = time.time()
    start_iso = datetime.datetime.now().isoformat()

    print(f"[Training] Starting: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}...")
    history_obj = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    end_time = time.time()
    end_iso = datetime.datetime.now().isoformat()
    duration_seconds = round(end_time - start_time, 2)

    # 6. Save Artifacts
    history_dict = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}
    history_json_path = os.path.join(run_dir, "history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    config_json_path = os.path.join(run_dir, "config.json")
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    if not os.path.exists(best_model_path):
        model.save(best_model_path)

    loaded_best_model = load_cnn_lstm_model(best_model_path)

    val_loss_hist = history_dict.get("val_loss", [])
    val_acc_hist = history_dict.get("val_accuracy", [])
    best_val_loss = float(min(val_loss_hist)) if val_loss_hist else None
    best_val_loss_epoch = int(np.argmin(val_loss_hist) + 1) if val_loss_hist else None
    best_val_acc = float(val_acc_hist[best_val_loss_epoch - 1]) if val_acc_hist else None
    final_val_acc = float(val_acc_hist[-1]) if val_acc_hist else None
    epochs_trained = len(history_obj.epoch)

    metadata: Dict[str, Any] = {
        "author": "Monal (Member 4)",
        "component": "Hybrid CNN-LSTM Classifier",
        "timestamp_start": start_iso,
        "timestamp_end": end_iso,
        "duration_seconds": duration_seconds,
        "seconds_per_epoch": round(duration_seconds / max(epochs_trained, 1), 2),
        "epochs_trained": epochs_trained,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "val_accuracy_at_best_epoch": best_val_acc,
        "final_val_accuracy": final_val_acc,
        "seed": seed,
        "data_source": data_source,
        "train_subjects": train_subjects_list,
        "val_subjects": val_subjects_list,
        "system_metadata": get_system_metadata(),
    }

    metadata_path = os.path.join(run_dir, "run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Training Complete]")
    print(f"  Duration:         {duration_seconds}s across {epochs_trained} epochs")
    print(f"  Best Val Loss:    {best_val_loss:.4f} (epoch {best_val_loss_epoch})")
    print(f"  Val Acc at Best:  {best_val_acc:.4%}")
    print(f"  Artifacts saved:  {run_dir}")

    return loaded_best_model, metadata, run_dir


def evaluate_cnn_lstm_on_test(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute final test set evaluation strictly once.

    Parameters:
        model: Trained Keras model.
        X_test: Test features (N, 128, 9).
        y_test: Test integer labels (N,).
        run_dir: Directory where test_metrics.json will be saved.

    Returns:
        Dictionary of test metrics (accuracy, macro_f1, weighted_f1).
    """
    from sklearn.metrics import accuracy_score, f1_score

    print(f"\n[Final Evaluation] Evaluating strictly once on unseen test set...")
    probabilities = predict_cnn_lstm(model, X_test)
    y_pred = np.argmax(probabilities, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    results = {
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "test_samples": int(len(y_test)),
    }

    print(f"  Test Accuracy:    {acc:.4%}")
    print(f"  Test Macro F1:    {macro_f1:.4f}")
    print(f"  Test Weighted F1: {weighted_f1:.4f}")

    if run_dir:
        test_path = os.path.join(run_dir, "test_metrics.json")
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved test metrics to {test_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Monal's CNN-LSTM HAR Classifier.")
    parser.add_argument("--config", type=str, default="configs/cnn_lstm.json", help="Path to config JSON")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--data-path", type=str, default=None, help="Path to data npz")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run quick synthetic smoke test")
    parser.add_argument("--evaluate-test", action="store_true", help="Run single test evaluation after training")
    args = parser.parse_args()

    cfg = load_config(args.config if os.path.exists(args.config) else None)
    if args.data_path:
        cfg["data"]["npz_path"] = args.data_path

    model, metadata, run_dir = train_cnn_lstm_pipeline(
        config=cfg,
        synthetic_smoke=args.synthetic_smoke,
        override_epochs=args.epochs,
        override_batch_size=args.batch_size,
        override_lr=args.lr,
        override_seed=args.seed,
        override_output_dir=args.output_dir,
    )

    if args.evaluate_test:
        if args.synthetic_smoke:
            har_dataset = generate_synthetic_har_data(test_samples=64)
        else:
            npz_path = cfg["data"]["npz_path"]
            if not os.path.isabs(npz_path):
                npz_path = os.path.join(PROJECT_ROOT, npz_path)
            har_dataset = load_har_npz(npz_path)
        evaluate_cnn_lstm_on_test(model, har_dataset["X_test"], har_dataset["y_test"], run_dir=run_dir)


if __name__ == "__main__":
    main()
