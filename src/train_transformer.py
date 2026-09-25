"""Training pipeline for Dharana's Transformer Component on HAR dataset.

This script manages:
1. Seed setting for reproducibility (Python, NumPy, TensorFlow).
2. Data contract validation and leakage prevention.
3. Transformer model construction and compilation.
4. Keras callbacks: EarlyStopping, ModelCheckpoint (.keras), ReduceLROnPlateau, CSVLogger.
5. Structured run outputs: best_model.keras, history.json, history.csv, config.json, run_metadata.json.
6. Support for both real NPZ data and synthetic smoke validation.
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
from src.models.transformer import (
    build_transformer_classifier,
    compile_transformer_model,
    load_transformer_model,
    predict_transformer,
)


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
            "d_model": 64,
            "num_heads": 4,
            "key_dim": 16,
            "num_layers": 2,
            "d_ff": 128,
            "ffn_activation": "gelu",
            "dropout_rate": 0.2,
            "dense_units": 64,
            "num_classes": 6,
        },
        "training": {
            "learning_rate": 0.001,
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
            "base_dir": "outputs/transformer",
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


def train_transformer_pipeline(
    config: Dict[str, Any],
    data: Optional[Dict[str, np.ndarray]] = None,
    run_dir: Optional[str] = None,
    verbose: int = 1,
) -> Tuple[keras.Model, Dict[str, Any], str]:
    """Execute complete training pipeline for the Transformer model.

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
        base_out = output_cfg.get("base_dir", "outputs/transformer")
        run_dir = os.path.join(base_out, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 2. Data Loading & Validation
    if data is None:
        npz_path = data_cfg.get("npz_path", "data/uci_har_processed.npz")
        if os.path.exists(npz_path):
            print(f"[Data] Loading dataset from '{npz_path}'...")
            data = load_har_npz(npz_path, normalize=bool(data_cfg.get("normalize", False)))
        else:
            print(f"[Data] No dataset found at '{npz_path}'. Generating synthetic HAR data for smoke test...")
            data = generate_synthetic_har_data(seed=seed)

    # Ensure dataset strictly meets the shared data contract
    validate_har_dataset(data)

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
    model = build_transformer_classifier(config=model_cfg)
    learning_rate = float(train_cfg.get("learning_rate", 0.001))
    model = compile_transformer_model(model, learning_rate=learning_rate)

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

    start_time = time.time()
    start_iso = datetime.datetime.now().isoformat()

    print(f"[Training] Starting training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}...")
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
    loaded_best_model = load_transformer_model(best_model_path)

    # Compile run metadata
    val_loss_hist = history_dict.get("val_loss", [])
    val_acc_hist = history_dict.get("val_accuracy", [])
    best_val_loss = float(min(val_loss_hist)) if val_loss_hist else None
    best_val_loss_epoch = int(np.argmin(val_loss_hist) + 1) if val_loss_hist else None
    final_val_acc = float(val_acc_hist[-1]) if val_acc_hist else None

    epochs_trained = len(history_obj.epoch)
    seconds_per_epoch = round(duration_seconds / max(1, epochs_trained), 3)

    metadata: Dict[str, Any] = {
        "author": "Dharana",
        "component": "Transformer Encoder Classifier",
        "timestamp_start": start_iso,
        "timestamp_end": end_iso,
        "duration_seconds": duration_seconds,
        "epochs_trained": epochs_trained,
        "seconds_per_epoch": seconds_per_epoch,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "final_val_accuracy": final_val_acc,
        "seed": seed,
        "train_subjects": train_subjects_list,
        "val_subjects": val_subjects_list,
        "num_train_samples": len(X_train),
        "num_val_samples": len(X_val),
        "data_source": "real_npz" if data.get("subject_test", None) is not None and len(data.get("subject_test", [])) == 2947 else "custom_or_synthetic",
        "system_info": get_system_metadata(),
        "model_file": os.path.basename(best_model_path),
        "history_file": os.path.basename(history_json_path),
        "config_file": os.path.basename(config_json_path),
    }

    metadata_path = os.path.join(run_dir, "run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Run Completed] Duration: {duration_seconds}s ({seconds_per_epoch}s/epoch) | Best Val Loss: {best_val_loss} (Epoch {best_val_loss_epoch})")
    print(f"  Artifacts saved to: {run_dir}")
    print(f"  - Model:     {best_model_path}")
    print(f"  - History:   {history_json_path}, {history_csv_path}")
    print(f"  - Metadata:  {metadata_path}")

    return loaded_best_model, metadata, run_dir


def evaluate_transformer_on_test(
    model: keras.Model,
    data: Dict[str, np.ndarray],
    run_dir: Optional[str] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Evaluate trained Transformer on held-out test split.

    Calculates accuracy, macro F1, weighted F1, per-class metrics, and confusion matrix.
    Saves test_metrics.json and predictions.npz into run_dir.

    Parameters:
        model: Trained Keras Transformer model.
        data: Data dictionary containing 'X_test', 'y_test', and optionally 'subject_test'.
        run_dir: Optional directory where evaluation artifacts are saved.
        save_artifacts: Whether to write test_metrics.json and predictions.npz.

    Returns:
        Dictionary of calculated evaluation metrics.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from src.data_contract import ACTIVITY_LABEL_MAPPING

    if "X_test" not in data or "y_test" not in data:
        raise ValueError("Data dictionary must contain 'X_test' and 'y_test' for evaluation.")

    X_test = data["X_test"]
    y_test = data["y_test"]
    subject_test = data.get("subject_test", None)

    y_probs = predict_transformer(model, X_test)
    y_pred = np.argmax(y_probs, axis=-1)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    class_names = [ACTIVITY_LABEL_MAPPING[i] for i in range(len(ACTIVITY_LABEL_MAPPING))]
    cm = confusion_matrix(y_test, y_pred)
    clf_report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, digits=4
    )
    clf_report_str = classification_report(
        y_test, y_pred, target_names=class_names, digits=4
    )

    metrics = {
        "author": "Dharana",
        "component": "Transformer Encoder",
        "num_test_samples": int(len(y_test)),
        "test_subjects": sorted(int(s) for s in np.unique(subject_test)) if subject_test is not None else [],
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "classification_report": clf_report_dict,
    }

    print("\n" + "=" * 60)
    print("TRANSFORMER HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy:  {acc * 100:.2f}%")
    print(f"Test Macro F1:  {macro_f1:.4f}")
    print(f"Test Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:\n")
    print(clf_report_str)

    if save_artifacts and run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        metrics_path = os.path.join(run_dir, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        preds_npz_path = os.path.join(run_dir, "predictions.npz")
        np.savez_compressed(
            preds_npz_path,
            y_test=y_test,
            y_pred=y_pred,
            y_probs=y_probs,
            subject_test=subject_test if subject_test is not None else np.array([]),
        )
        print(f"[Artifacts] Saved evaluation results to {run_dir}:")
        print(f"  - {metrics_path}")
        print(f"  - {preds_npz_path}")

    return metrics


def main() -> None:
    """CLI entrypoint for training Dharana's Transformer component."""
    parser = argparse.ArgumentParser(
        description="Train Transformer Encoder for Human Activity Recognition (HAR)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/transformer.json",
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
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Run evaluation on test split after training completes",
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
        npz_path = config.get("data", {}).get("npz_path", "data/uci_har_processed.npz")
        if os.path.exists(npz_path):
            data = load_har_npz(npz_path, normalize=bool(config.get("data", {}).get("normalize", False)))
        else:
            data = None

    best_model, meta, run_dir = train_transformer_pipeline(config, data=data, run_dir=args.output_dir)

    if args.eval_test and data is not None and "X_test" in data:
        evaluate_transformer_on_test(best_model, data=data, run_dir=run_dir)


if __name__ == "__main__":
    main()
