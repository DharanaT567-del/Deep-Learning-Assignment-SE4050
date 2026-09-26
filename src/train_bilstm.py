"""Training pipeline for Member 3's BiLSTM Component on HAR dataset.

This script manages:
1. Seed setting for reproducibility (Python, NumPy, TensorFlow).
2. Data contract validation and leakage prevention.
3. BiLSTM model construction and compilation.
4. Keras callbacks: EarlyStopping, ModelCheckpoint (.keras), ReduceLROnPlateau, CSVLogger.
5. Structured run outputs: best_model.keras, history.json, history.csv, config.json, run_metadata.json.
6. Support for both real NPZ data and synthetic smoke validation.
7. Optional train-only data augmentation (jitter, per-channel scaling, time shift).

The test split is never used here: training and model selection use train/val only.
Augmentation is applied to training batches only; validation windows are never changed.
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
    generate_synthetic_har_data,
    load_har_npz,
    validate_har_dataset,
)
from src.models.bilstm import build_bilstm_model, load_bilstm_model


def set_seed(seed: int = 42) -> None:
    """Set seeds across Python random, NumPy, and TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def augment_batch(
    x: tf.Tensor,
    jitter_std: float = 0.05,
    scale_std: float = 0.1,
    max_shift: int = 8,
) -> tf.Tensor:
    """Randomly perturb a batch of sensor windows, keeping its shape (N, T, C).

    - Jitter: add Gaussian noise with std `jitter_std` to every reading.
    - Scaling: multiply each channel of each window by a factor drawn from N(1, scale_std).
    - Time shift: circularly shift each window by a random number of steps in [-max_shift, max_shift].

    Inputs are already standardized, so the stds are in units of one channel std.
    Setting all three to 0 returns the input unchanged.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    shape = tf.shape(x)
    n, t, c = shape[0], shape[1], shape[2]

    if jitter_std > 0:
        x = x + tf.random.normal(shape, stddev=jitter_std)
    if scale_std > 0:
        x = x * tf.random.normal((n, 1, c), mean=1.0, stddev=scale_std)
    if max_shift > 0:
        shifts = tf.random.uniform((n, 1), minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32)
        idx = tf.math.floormod(tf.range(t)[tf.newaxis, :] - shifts, t)
        x = tf.gather(x, idx, batch_dims=1)
    return x


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
            "lstm_units": 64,
            "n_layers": 2,
            "dropout": 0.4,
            "dense_units": 64,
            "num_classes": 6,
            "pooling": "last",
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
            "augment": False,
            "aug_jitter_std": 0.05,
            "aug_scale_std": 0.1,
            "aug_max_shift": 8,
        },
        "data": {
            "npz_path": "data/uci_har_processed.npz",
            "normalize": False,
        },
        "output": {
            "base_dir": "outputs/bilstm",
        },
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        for section in ["model", "training", "data", "output"]:
            if section in user_config and isinstance(user_config[section], dict):
                default_config[section].update(user_config[section])
            elif section in user_config:
                default_config[section] = user_config[section]

    return default_config


def train_bilstm_pipeline(
    config: Dict[str, Any],
    data: Optional[Dict[str, np.ndarray]] = None,
    run_dir: Optional[str] = None,
    verbose: int = 1,
) -> Tuple[keras.Model, Dict[str, Any], str]:
    """Execute complete training pipeline for the BiLSTM model.

    Parameters:
        config: Full configuration dictionary.
        data: Optional preloaded data dictionary matching the HAR contract.
        run_dir: Explicit output directory (defaults to timestamped folder under base_dir).
        verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch).

    Returns:
        (best_model, run_metadata, run_dir_path)
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    # 1. Setup Output Directory
    if run_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_out = output_cfg.get("base_dir", "outputs/bilstm")
        run_dir = os.path.join(base_out, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 2. Data Loading & Validation
    data_source = "provided"
    if data is None:
        npz_path = data_cfg.get("npz_path", "data/uci_har_processed.npz")
        if os.path.exists(npz_path):
            print(f"[Data] Loading dataset from '{npz_path}'...")
            data = load_har_npz(npz_path, normalize=bool(data_cfg.get("normalize", False)))
            data_source = npz_path
        else:
            print(f"[Data] No dataset found at '{npz_path}'. Generating synthetic HAR data for smoke test...")
            data = generate_synthetic_har_data(seed=seed)
            data_source = "synthetic"

    # Ensure dataset strictly meets the shared data contract
    validate_har_dataset(data)

    # Only train/val are read below; X_test is reserved for the final evaluation
    X_train = data["X_train"]
    y_train = data["y_train"]
    subject_train = data["subject_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]
    subject_val = data["subject_val"]

    train_subjects_list = sorted([int(s) for s in np.unique(subject_train)])
    val_subjects_list = sorted([int(s) for s in np.unique(subject_val)])

    print(f"[Data Contract] Validated successfully.")
    print(f"  Train windows: {len(X_train)} (Subjects: {train_subjects_list})")
    print(f"  Val windows:   {len(X_val)} (Subjects: {val_subjects_list})")
    print(f"  Input window shape: {X_train.shape[1:]}")

    # 3. Build & Compile Model
    learning_rate = float(train_cfg.get("learning_rate", 0.0005))
    model = build_bilstm_model(
        input_shape=(int(model_cfg.get("seq_len", 128)), int(model_cfg.get("num_features", 9))),
        n_classes=int(model_cfg.get("num_classes", 6)),
        lstm_units=int(model_cfg.get("lstm_units", 64)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.4)),
        dense_units=int(model_cfg.get("dense_units", 64)),
        learning_rate=learning_rate,
        seed=seed,
        pooling=str(model_cfg.get("pooling", "last")),
    )

    total_params = int(model.count_params())
    trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable_params = total_params - trainable_params

    print(f"[Model Architecture] Built '{model.name}'")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 4. Configure Callbacks
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

    # 5. Execute Training
    batch_size = int(train_cfg.get("batch_size", 64))
    epochs = int(train_cfg.get("epochs", 60))
    augment = bool(train_cfg.get("augment", False))

    start_time = time.time()
    start_iso = datetime.datetime.now().isoformat()

    print(
        f"[Training] Starting training: epochs={epochs}, batch_size={batch_size}, "
        f"lr={learning_rate}, augment={augment}..."
    )
    if augment:
        # Augment training batches only; validation data is passed through unchanged
        jitter_std = float(train_cfg.get("aug_jitter_std", 0.05))
        scale_std = float(train_cfg.get("aug_scale_std", 0.1))
        max_shift = int(train_cfg.get("aug_max_shift", 8))
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train))
            .shuffle(len(X_train), seed=seed, reshuffle_each_iteration=True)
            .batch(batch_size)
            .map(
                lambda xb, yb: (augment_batch(xb, jitter_std, scale_std, max_shift), yb),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .prefetch(tf.data.AUTOTUNE)
        )
        history_obj = model.fit(
            train_ds,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
    else:
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

    # 6. Save Training Artifacts
    # Save History JSON
    history_dict = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}
    history_json_path = os.path.join(run_dir, "history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    # Save Resolved Config JSON
    config_json_path = os.path.join(run_dir, "config.json")
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Verify best model was saved; if early stopping happened before checkpoint, save current best
    if not os.path.exists(best_model_path):
        model.save(best_model_path)

    # Test loading saved model
    loaded_best_model = load_bilstm_model(best_model_path)

    # Compile run metadata
    val_loss_hist = history_dict.get("val_loss", [])
    val_acc_hist = history_dict.get("val_accuracy", [])
    best_val_loss = float(min(val_loss_hist)) if val_loss_hist else None
    best_val_loss_epoch = int(np.argmin(val_loss_hist) + 1) if val_loss_hist else None
    best_val_acc = float(val_acc_hist[best_val_loss_epoch - 1]) if val_acc_hist else None
    final_val_acc = float(val_acc_hist[-1]) if val_acc_hist else None
    epochs_trained = len(history_obj.epoch)

    metadata: Dict[str, Any] = {
        "author": "Member 3",
        "component": "Bidirectional LSTM Classifier",
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
        "pooling": str(model_cfg.get("pooling", "last")),
        "augment": augment,
        "data_source": data_source,
        "train_subjects": train_subjects_list,
        "val_subjects": val_subjects_list,
        "num_train_samples": len(X_train),
        "num_val_samples": len(X_val),
        "system_info": get_system_metadata(),
        "model_file": os.path.basename(best_model_path),
        "history_file": os.path.basename(history_json_path),
        "config_file": os.path.basename(config_json_path),
    }

    metadata_path = os.path.join(run_dir, "run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Run Completed] Duration: {duration_seconds}s | Best Val Loss: {best_val_loss} (Epoch {best_val_loss_epoch})")
    print(f"  Artifacts saved to: {run_dir}")
    print(f"  - Model:     {best_model_path}")
    print(f"  - History:   {history_json_path}, {history_csv_path}")
    print(f"  - Metadata:  {metadata_path}")

    return loaded_best_model, metadata, run_dir


def main() -> None:
    """CLI entrypoint for training the BiLSTM component."""
    parser = argparse.ArgumentParser(
        description="Train Bidirectional LSTM for Human Activity Recognition (HAR)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bilstm.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to preprocessed HAR NPZ dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom directory to store run artifacts",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run short synthetic verification smoke test (2 epochs)",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # CLI Overrides
    if args.data_path:
        config["data"]["npz_path"] = args.data_path
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.output_dir:
        config["output"]["base_dir"] = args.output_dir

    if args.smoke_test:
        print("[Smoke Test] Running lightweight verification with synthetic data (2 epochs)...")
        config["training"]["epochs"] = 2
        config["training"]["early_stopping_patience"] = 2
        data = generate_synthetic_har_data(
            n_train_windows_per_class=10,
            n_val_windows_per_class=5,
            n_test_windows_per_class=5,
            seed=config["training"]["seed"],
        )
    else:
        data = None

    train_bilstm_pipeline(config, data=data, run_dir=args.output_dir)


if __name__ == "__main__":
    main()
