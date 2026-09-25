"""Shared UCI HAR dataset loader for the four-member HAR project.

Author: Member 3
Project: SE4050 Deep Learning Assignment - Four-member HAR architecture comparison.

Converts the raw UCI HAR Smartphones dataset into the shared data contract
(`src/data_contract.py`) and saves it as `data/uci_har_processed.npz`, which
every member's training pipeline loads.

Pipeline:
    1. Download and unzip the dataset (the archive contains a nested zip).
    2. Stack the 9 raw inertial signal files into windows shaped (N, 128, 9).
    3. Convert labels from 1..6 to 0..5.
    4. Split the official training subjects into train/val BY SUBJECT. Windows
       overlap by 50%, so a random split would leak near-identical windows.
    5. Standardize per channel using train-split statistics only.
    6. Validate against the data contract and save the NPZ.

Dataset citation:
    Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013).
    Human Activity Recognition Using Smartphones [Dataset]. UCI Machine Learning
    Repository. https://doi.org/10.24432/C54S4K  (CC BY 4.0)

Usage (from the repo root):
    python -m src.data.uci_har_loader --download
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

# Adjust Python path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_contract import (
    ACTIVITY_LABEL_MAPPING,
    SENSOR_CHANNEL_NAMES,
    apply_training_standardization,
    validate_har_dataset,
)

PathLike = Union[str, Path]

UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
DATASET_FOLDER_NAME = "UCI HAR Dataset"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_DIR = DEFAULT_DATA_DIR / DATASET_FOLDER_NAME
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "uci_har_processed.npz"

# Channel order of the stacked windows; identical to the shared contract's order
SIGNAL_NAMES: List[str] = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
assert SIGNAL_NAMES == SENSOR_CHANNEL_NAMES, "Loader channel order must match the data contract."

VALID_SPLITS = ("train", "test")
NUM_CLASSES = 6
WINDOW_LENGTH = 128
MAX_SPLIT_ATTEMPTS = 100

# Exactly the keys written to the NPZ (the data contract keys)
CONTRACT_KEYS = [
    "X_train", "y_train", "subject_train",
    "X_val", "y_val", "subject_val",
    "X_test", "y_test", "subject_test",
]


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw UCI HAR folder is missing, with download instructions."""


def _missing_dataset_message(dataset_dir: Path) -> str:
    """Build an actionable error message for a missing or incomplete dataset."""
    return (
        f"UCI HAR dataset not found at '{dataset_dir}'.\n"
        f"  Option 1: run  python -m src.data.uci_har_loader --download\n"
        f"  Option 2: download it manually from {UCI_HAR_URL}\n"
        f"            unzip it twice, and place the folder at '{DEFAULT_DATASET_DIR}'."
    )


def _check_split(split: str) -> None:
    """Reject split names other than 'train' and 'test'."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'.")


def _require_file(path: Path, dataset_dir: Path) -> Path:
    """Return path if it exists, otherwise raise DatasetNotFoundError."""
    if not path.is_file():
        raise DatasetNotFoundError(
            f"Expected file '{path}' is missing.\n" + _missing_dataset_message(dataset_dir)
        )
    return path


def download_uci_har(dest_dir: PathLike = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    """Download and double-unzip UCI HAR into dest_dir; return the dataset folder path."""
    dest_dir = Path(dest_dir)
    dataset_dir = dest_dir / DATASET_FOLDER_NAME
    if dataset_dir.is_dir() and not force:
        print(f"[Download] Dataset already present at '{dataset_dir}'.")
        return dataset_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "har.zip"

    # A partial download leaves a corrupt archive behind; fetch it again in that case
    if force or not zipfile.is_zipfile(archive_path):
        tmp_path = archive_path.with_suffix(".zip.part")
        print(f"[Download] Fetching {UCI_HAR_URL} ...")
        try:
            req = urllib.request.Request(UCI_HAR_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_file:
                shutil.copyfileobj(resp, out_file)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download UCI HAR from {UCI_HAR_URL} ({exc}).\n"
                f"Download it manually, unzip it twice, and place the folder at '{dataset_dir}'."
            ) from exc
        if not zipfile.is_zipfile(tmp_path):
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file from {UCI_HAR_URL} is not a valid zip archive.\n"
                f"Download it manually and place the unzipped folder at '{dataset_dir}'."
            )
        tmp_path.replace(archive_path)

    # Outer archive contains 'UCI HAR Dataset.zip', which contains the actual folder
    print(f"[Download] Extracting '{archive_path}' ...")
    with zipfile.ZipFile(archive_path) as outer:
        outer.extractall(dest_dir)
    inner_zip = dest_dir / f"{DATASET_FOLDER_NAME}.zip"
    if inner_zip.is_file():
        with zipfile.ZipFile(inner_zip) as inner:
            inner.extractall(dest_dir)
        inner_zip.unlink()
    shutil.rmtree(dest_dir / "__MACOSX", ignore_errors=True)

    if not dataset_dir.is_dir():
        raise DatasetNotFoundError(_missing_dataset_message(dataset_dir))
    print(f"[Download] Dataset ready at '{dataset_dir}'.")
    return dataset_dir


def load_inertial_signals(dataset_dir: PathLike, split: str) -> np.ndarray:
    """Stack the 9 raw inertial signal files into a float32 array (N, 128, 9)."""
    _check_split(split)
    dataset_dir = Path(dataset_dir)
    signal_dir = dataset_dir / split / "Inertial Signals"

    channels = []
    for name in SIGNAL_NAMES:
        path = _require_file(signal_dir / f"{name}_{split}.txt", dataset_dir)
        channel = np.loadtxt(path, dtype=np.float32, ndmin=2)
        if channel.shape[1] != WINDOW_LENGTH:
            raise ValueError(
                f"'{path.name}' has {channel.shape[1]} readings per window, expected {WINDOW_LENGTH}."
            )
        channels.append(channel)

    n_windows = {c.shape[0] for c in channels}
    if len(n_windows) != 1:
        raise ValueError(f"Inertial signal files for '{split}' disagree on window count: {n_windows}.")
    # List of 9 arrays (N, 128) -> (N, 128, 9), channel index follows SIGNAL_NAMES
    return np.stack(channels, axis=-1)


def load_engineered_features(dataset_dir: PathLike, split: str) -> np.ndarray:
    """Load the 561 engineered features (N, 561) float32, used by the classical SVM baseline."""
    _check_split(split)
    dataset_dir = Path(dataset_dir)
    path = _require_file(dataset_dir / split / f"X_{split}.txt", dataset_dir)
    return np.loadtxt(path, dtype=np.float32, ndmin=2)


def load_labels(dataset_dir: PathLike, split: str) -> np.ndarray:
    """Load activity labels and convert them from 1..6 to 0..5 (int64)."""
    _check_split(split)
    dataset_dir = Path(dataset_dir)
    path = _require_file(dataset_dir / split / f"y_{split}.txt", dataset_dir)
    raw = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if raw.min() < 1 or raw.max() > NUM_CLASSES:
        raise ValueError(f"'{path.name}' labels must be in 1..{NUM_CLASSES}, found {np.unique(raw).tolist()}.")
    return raw - 1


def load_subjects(dataset_dir: PathLike, split: str) -> np.ndarray:
    """Load one subject ID per window (N,) int64."""
    _check_split(split)
    dataset_dir = Path(dataset_dir)
    path = _require_file(dataset_dir / split / f"subject_{split}.txt", dataset_dir)
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def split_train_val_by_subject(
    subjects: np.ndarray,
    labels: np.ndarray,
    n_val_subjects: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick n_val_subjects whole subjects for validation; return (train_mask, val_mask).

    Re-draws deterministically (seed, seed+1, ...) until every class appears in both
    splits, and raises after MAX_SPLIT_ATTEMPTS tries.
    """
    subjects = np.asarray(subjects)
    labels = np.asarray(labels)
    if subjects.shape != labels.shape:
        raise ValueError(f"subjects {subjects.shape} and labels {labels.shape} must have the same shape.")

    unique_subjects = np.unique(subjects)
    if not 1 <= n_val_subjects < len(unique_subjects):
        raise ValueError(
            f"n_val_subjects must be between 1 and {len(unique_subjects) - 1}, got {n_val_subjects}."
        )

    all_classes = set(range(NUM_CLASSES))
    for attempt in range(MAX_SPLIT_ATTEMPTS):
        rng = np.random.default_rng(seed + attempt)
        val_subjects = rng.choice(unique_subjects, size=n_val_subjects, replace=False)
        val_mask = np.isin(subjects, val_subjects)
        train_mask = ~val_mask

        # Leakage guard: a subject may only ever be on one side of the split
        train_set = set(np.unique(subjects[train_mask]).tolist())
        val_set = set(np.unique(subjects[val_mask]).tolist())
        assert train_set.isdisjoint(val_set), f"Subject leakage between train and val: {train_set & val_set}"

        if set(labels[train_mask].tolist()) >= all_classes and set(labels[val_mask].tolist()) >= all_classes:
            return train_mask, val_mask

    raise RuntimeError(
        f"Could not find a subject split with all {NUM_CLASSES} classes in both train and val "
        f"after {MAX_SPLIT_ATTEMPTS} attempts (n_val_subjects={n_val_subjects}, seed={seed})."
    )


def build_processed_dataset(
    dataset_dir: PathLike = DEFAULT_DATASET_DIR,
    n_val_subjects: int = 4,
    seed: int = 42,
    standardize: bool = True,
) -> Dict[str, np.ndarray]:
    """Load raw UCI HAR, split train/val by subject, standardize, validate, and return the contract dict."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise DatasetNotFoundError(_missing_dataset_message(dataset_dir))

    # 1. Load the official train (21 subjects) and test (9 subjects) splits
    X_full = load_inertial_signals(dataset_dir, "train")
    y_full = load_labels(dataset_dir, "train")
    subject_full = load_subjects(dataset_dir, "train")
    X_test = load_inertial_signals(dataset_dir, "test")
    y_test = load_labels(dataset_dir, "test")
    subject_test = load_subjects(dataset_dir, "test")

    # 2. Carve validation subjects out of the official training subjects
    train_mask, val_mask = split_train_val_by_subject(subject_full, y_full, n_val_subjects, seed)
    X_train, y_train, subject_train = X_full[train_mask], y_full[train_mask], subject_full[train_mask]
    X_val, y_val, subject_val = X_full[val_mask], y_full[val_mask], subject_full[val_mask]

    # 3. Standardize with statistics fitted on the train split only
    if standardize:
        X_train, X_val, X_test, _ = apply_training_standardization(X_train, X_val, X_test)

    data = {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "subject_train": subject_train.astype(np.int64),
        "X_val": X_val.astype(np.float32),
        "y_val": y_val.astype(np.int64),
        "subject_val": subject_val.astype(np.int64),
        "X_test": X_test.astype(np.float32),
        "y_test": y_test.astype(np.int64),
        "subject_test": subject_test.astype(np.int64),
    }

    # 4. Enforce the shared contract, including train/val and train/test leakage checks
    validate_har_dataset(data, check_test=True)
    return data


def save_processed_dataset(data: Dict[str, np.ndarray], out_path: PathLike = DEFAULT_OUTPUT_PATH) -> Path:
    """Save exactly the contract keys to a compressed NPZ and return its path."""
    missing = [k for k in CONTRACT_KEYS if k not in data]
    if missing:
        raise KeyError(f"Cannot save dataset, missing contract keys: {missing}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **{k: data[k] for k in CONTRACT_KEYS})

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[Save] Wrote '{out_path}' ({size_mb:.1f} MB).")
    return out_path


def print_dataset_summary(data: Dict[str, np.ndarray]) -> None:
    """Print array shapes, subject IDs, and class counts per split (the leakage audit trail)."""
    print("\n[Summary] Array shapes:")
    for key in CONTRACT_KEYS:
        print(f"  {key:<14} {tuple(data[key].shape)}  {data[key].dtype}")

    print("\n[Summary] Subjects per split:")
    split_subjects = {}
    for split in ("train", "val", "test"):
        ids = sorted(int(s) for s in np.unique(data[f"subject_{split}"]))
        split_subjects[split] = set(ids)
        print(f"  {split:<5} ({len(ids):>2} subjects): {ids}")

    overlaps = {
        "train/val": split_subjects["train"] & split_subjects["val"],
        "train/test": split_subjects["train"] & split_subjects["test"],
        "val/test": split_subjects["val"] & split_subjects["test"],
    }
    for pair, overlap in overlaps.items():
        status = "none" if not overlap else f"LEAKAGE {sorted(overlap)}"
        print(f"  overlap {pair:<10} {status}")

    print("\n[Summary] Class counts per split:")
    header = "  " + f"{'class':<22}" + "".join(f"{s:>8}" for s in ("train", "val", "test"))
    print(header)
    for label, name in ACTIVITY_LABEL_MAPPING.items():
        counts = "".join(f"{int(np.sum(data[f'y_{s}'] == label)):>8}" for s in ("train", "val", "test"))
        print(f"  {label} {name:<20}{counts}")


def main() -> None:
    """CLI entry point: optionally download, then build, validate, save, and summarize."""
    parser = argparse.ArgumentParser(
        description="Build the processed UCI HAR dataset (data/uci_har_processed.npz)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Path to the unzipped 'UCI HAR Dataset' folder",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for the processed NPZ",
    )
    parser.add_argument(
        "--n-val-subjects",
        type=int,
        default=4,
        help="Number of training subjects held out for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the subject-wise train/val split",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset first if it is missing",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Skip train-only per-channel standardization",
    )

    args = parser.parse_args()
    dataset_dir = Path(args.data_dir)

    if args.download:
        dataset_dir = download_uci_har(dest_dir=dataset_dir.parent)

    try:
        data = build_processed_dataset(
            dataset_dir=dataset_dir,
            n_val_subjects=args.n_val_subjects,
            seed=args.seed,
            standardize=not args.no_standardize,
        )
    except DatasetNotFoundError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)

    save_processed_dataset(data, args.out)
    print_dataset_summary(data)


if __name__ == "__main__":
    main()
