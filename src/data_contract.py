"""HAR Shared Data Contract and Validation Interface.

This module formalizes the data interface agreement between all four project members.
It validates data integrity, dimensional contracts, and enforces strict subject-wise
data leakage prevention.

Data Contract:
    - X_train, X_val, (X_test): float arrays shaped (N, 128, 9), finite (no NaNs or Infs).
    - y_train, y_val, (y_test): integer arrays shaped (N,), values in {0, 1, 2, 3, 4, 5}.
    - subject_train, subject_val, (subject_test): integer/string subject IDs per window.
    - Leakage constraint: set(subject_train) ∩ set(subject_val) == ∅.
    - Stratification check: Both train and val splits contain all 6 activity classes.
    - Normalization rule: Fitted strictly on training set only. Never fit on val/test.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Canonical UCI HAR Activity Label Mapping
ACTIVITY_LABEL_MAPPING = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}

# Canonical 9-channel sensor representation
SENSOR_CHANNEL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


class DataContractValidationError(ValueError):
    """Raised when data violates the four-member shared HAR data contract."""
    pass


def validate_har_split(
    x: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    split_name: str = "train",
    expected_seq_len: int = 128,
    expected_channels: int = 9,
    num_classes: int = 6,
    require_all_classes: bool = True,
) -> None:
    """Validate a single split for correct shapes, finite values, and label bounds.

    Parameters:
        x: Feature array of sensor windows.
        y: Activity class labels.
        subject: Subject identifier array.
        split_name: Name of split (e.g. 'train', 'val', 'test').
        expected_seq_len: Required time window length (default: 128).
        expected_channels: Required sensor channel count (default: 9).
        num_classes: Number of distinct classes (default: 6).
        require_all_classes: Whether all classes {0..num_classes-1} must be present.
    """
    if not isinstance(x, np.ndarray):
        raise DataContractValidationError(
            f"[{split_name}] 'X' must be a numpy.ndarray, got {type(x)}."
        )
    if not isinstance(y, np.ndarray):
        raise DataContractValidationError(
            f"[{split_name}] 'y' must be a numpy.ndarray, got {type(y)}."
        )
    if not isinstance(subject, np.ndarray):
        raise DataContractValidationError(
            f"[{split_name}] 'subject' must be a numpy.ndarray, got {type(subject)}."
        )

    # Dimensional checks
    if x.ndim != 3:
        raise DataContractValidationError(
            f"[{split_name}] X must be 3-dimensional (N, {expected_seq_len}, {expected_channels}), got shape {x.shape}."
        )
    n_samples, seq_len, n_channels = x.shape

    if seq_len != expected_seq_len:
        raise DataContractValidationError(
            f"[{split_name}] Window sequence length must be {expected_seq_len}, got {seq_len}."
        )
    if n_channels != expected_channels:
        raise DataContractValidationError(
            f"[{split_name}] Channel count must be {expected_channels}, got {n_channels}."
        )

    if y.ndim != 1 or len(y) != n_samples:
        raise DataContractValidationError(
            f"[{split_name}] y shape must be ({n_samples},), got {y.shape}."
        )

    if subject.ndim != 1 or len(subject) != n_samples:
        raise DataContractValidationError(
            f"[{split_name}] subject shape must be ({n_samples},), got {subject.shape}."
        )

    # Numerical validity checks
    if not np.all(np.isfinite(x)):
        raise DataContractValidationError(
            f"[{split_name}] X contains NaN or infinite values."
        )

    if not np.issubdtype(y.dtype, np.integer):
        raise DataContractValidationError(
            f"[{split_name}] y must contain integer class indices, got dtype {y.dtype}."
        )

    unique_labels = np.unique(y)
    if np.any(unique_labels < 0) or np.any(unique_labels >= num_classes):
        raise DataContractValidationError(
            f"[{split_name}] y labels must be integers in range [0, {num_classes - 1}], found: {unique_labels.tolist()}."
        )

    if require_all_classes and len(unique_labels) < num_classes:
        missing = set(range(num_classes)) - set(unique_labels)
        raise DataContractValidationError(
            f"[{split_name}] Missing activity classes {missing}. All {num_classes} classes must be represented."
        )


def validate_har_dataset(data: Dict[str, np.ndarray], check_test: bool = False) -> None:
    """Validate full dataset dictionary against the shared contract.

    Enforces:
        1. Presence of required keys: X_train, y_train, subject_train, X_val, y_val, subject_val.
        2. Finite numeric values and valid shapes (N, 128, 9).
        3. Zero subject overlap between training and validation (Data Leakage Prevention).
        4. Presence of all 6 classes in train and val.
        5. (Optional) Validity of X_test, y_test, subject_test if present.

    Parameters:
        data: Dictionary mapping key names to numpy arrays.
        check_test: If True, also strictly validates test split if present.
    """
    required_keys = ["X_train", "y_train", "subject_train", "X_val", "y_val", "subject_val"]
    for k in required_keys:
        if k not in data:
            raise DataContractValidationError(
                f"Missing required key '{k}' in dataset. Expected keys: {required_keys}"
            )

    # Validate individual splits
    validate_har_split(data["X_train"], data["y_train"], data["subject_train"], split_name="train")
    validate_har_split(data["X_val"], data["y_val"], data["subject_val"], split_name="val")

    # Strict Subject-Wise Data Leakage Check
    train_subjects = set(np.unique(data["subject_train"]))
    val_subjects = set(np.unique(data["subject_val"]))
    overlap = train_subjects.intersection(val_subjects)
    if overlap:
        raise DataContractValidationError(
            f"CRITICAL DATA LEAKAGE: Overlapping subjects detected between train and validation splits: {overlap}. "
            f"Subject-wise split must be strictly disjoint."
        )

    if "X_test" in data and check_test:
        validate_har_split(
            data["X_test"],
            data["y_test"],
            data.get("subject_test", np.zeros(len(data["y_test"]), dtype=int)),
            split_name="test",
            require_all_classes=False,
        )
        if "subject_test" in data:
            test_subjects = set(np.unique(data["subject_test"]))
            train_test_overlap = train_subjects.intersection(test_subjects)
            if train_test_overlap:
                raise DataContractValidationError(
                    f"CRITICAL DATA LEAKAGE: Overlapping subjects between train and test: {train_test_overlap}."
                )


def apply_training_standardization(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Standardize sensor channels using statistics fitted STRICTLY on training data only.

    Calculates mean and standard deviation across (N_train, 128) per sensor channel (axis=9),
    then applies the identical transformation to validation and test arrays without recomputing.

    Parameters:
        X_train: Training windows shaped (N_train, 128, 9).
        X_val: Validation windows shaped (N_val, 128, 9).
        X_test: Optional test windows shaped (N_test, 128, 9).

    Returns:
        (X_train_norm, X_val_norm, X_test_norm, stats_dict)
    """
    # Channel-wise mean and std computed strictly on train split
    # Shape of mean and std: (1, 1, 9)
    train_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    train_std = np.std(X_train, axis=(0, 1), keepdims=True)
    # Avoid zero-division on flat sensor channels
    train_std = np.where(train_std < 1e-8, 1.0, train_std)

    X_train_norm = (X_train - train_mean) / train_std
    X_val_norm = (X_val - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std if X_test is not None else None

    stats = {
        "channel_means": train_mean.squeeze().tolist(),
        "channel_stds": train_std.squeeze().tolist(),
    }
    return X_train_norm, X_val_norm, X_test_norm, stats


def load_har_npz(
    npz_path: str,
    normalize: bool = False,
) -> Dict[str, np.ndarray]:
    """Load and validate HAR dataset from a standard NPZ file.

    Parameters:
        npz_path: Path to the .npz archive.
        normalize: If True and data is unnormalized, fit standard scaler strictly on X_train.

    Returns:
        Dictionary containing verified arrays.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"HAR NPZ data file not found at '{npz_path}'.")

    with np.load(npz_path, allow_pickle=False) as npz_data:
        data = {k: npz_data[k] for k in npz_data.files}

    # Validate the data contract
    validate_har_dataset(data)

    if normalize:
        x_tr, x_v, x_te, stats = apply_training_standardization(
            data["X_train"],
            data["X_val"],
            data.get("X_test", None),
        )
        data["X_train"] = x_tr
        data["X_val"] = x_v
        if x_te is not None:
            data["X_test"] = x_te
        data["normalization_stats"] = stats

    return data


def generate_synthetic_har_data(
    n_train_windows_per_class: int = 40,
    n_val_windows_per_class: int = 15,
    n_test_windows_per_class: int = 15,
    seq_len: int = 128,
    num_channels: int = 9,
    num_classes: int = 6,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate realistic synthetic HAR dataset for pipeline verification and smoke testing.

    Ensures zero subject overlap:
        Train subjects: [1, 2, 3, 4, 5, 6]
        Val subjects: [7, 8]
        Test subjects: [9, 10]
    """
    rng = np.random.RandomState(seed)

    def _generate_split(windows_per_class: int, subject_pool: list[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, subjects = [], [], []
        t = np.linspace(0, 2 * np.pi, seq_len)

        for label in range(num_classes):
            for _ in range(windows_per_class):
                # Sample a subject from the split-specific pool
                sub = rng.choice(subject_pool)
                # Create distinct frequency/offset signature per activity
                freq = 0.5 + label * 0.4
                phase = rng.uniform(0, np.pi)
                base_signal = np.sin(freq * t + phase)[:, None]  # (128, 1)

                # Broadcast across 9 channels with varying amplitudes and sensor noise
                channel_weights = rng.uniform(0.5, 1.5, size=(1, num_channels))
                sensor_noise = rng.normal(0, 0.05, size=(seq_len, num_channels))
                window = base_signal * channel_weights + sensor_noise

                xs.append(window.astype(np.float32))
                ys.append(label)
                subjects.append(sub)

        # Shuffle split
        idx = rng.permutation(len(ys))
        return (
            np.array(xs, dtype=np.float32)[idx],
            np.array(ys, dtype=np.int64)[idx],
            np.array(subjects, dtype=np.int64)[idx],
        )

    train_x, train_y, train_sub = _generate_split(n_train_windows_per_class, [1, 2, 3, 4, 5, 6])
    val_x, val_y, val_sub = _generate_split(n_val_windows_per_class, [7, 8])
    test_x, test_y, test_sub = _generate_split(n_test_windows_per_class, [9, 10])

    dataset = {
        "X_train": train_x,
        "y_train": train_y,
        "subject_train": train_sub,
        "X_val": val_x,
        "y_val": val_y,
        "subject_val": val_sub,
        "X_test": test_x,
        "y_test": test_y,
        "subject_test": test_sub,
    }

    validate_har_dataset(dataset, check_test=True)
    return dataset
