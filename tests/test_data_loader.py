"""Unit tests for Member 3's shared UCI HAR loader.

Verifies:
1. Inertial signal files are stacked in SIGNAL_NAMES channel order.
2. Labels are converted from 1..6 to 0..5.
3. The train/val split is disjoint by subject (no leakage).
4. Every class 0..5 appears in both train and val.
5. Standardization statistics come from the train split only.
6. The built dataset passes the shared data contract.
7. NPZ save/load round-trips exactly.
8. A missing dataset folder raises a clear, actionable error.

All tests run on a small synthetic copy of the UCI HAR folder layout, so the real
dataset is not required. One extra test runs on the real dataset when it is present.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_contract import validate_har_dataset
from src.data.uci_har_loader import (
    CONTRACT_KEYS,
    DEFAULT_DATASET_DIR,
    SIGNAL_NAMES,
    UCI_HAR_URL,
    DatasetNotFoundError,
    build_processed_dataset,
    load_inertial_signals,
    load_labels,
    load_subjects,
    save_processed_dataset,
    split_train_val_by_subject,
)


def write_fake_uci_har(
    root: Path,
    train_subjects: list[int],
    test_subjects: list[int],
    windows_per_subject_class: int = 2,
    seed: int = 0,
) -> Path:
    """Write a tiny dataset with the real UCI HAR folder layout and return its path.

    Channel k is filled with values centred on (k + 1) * 10, so stacking order is checkable.
    Test windows get an extra +5 offset so standardization leakage is detectable.
    """
    rng = np.random.default_rng(seed)
    dataset_dir = root / "UCI HAR Dataset"

    for split, subjects in (("train", train_subjects), ("test", test_subjects)):
        split_dir = dataset_dir / split
        (split_dir / "Inertial Signals").mkdir(parents=True, exist_ok=True)

        labels, subject_ids = [], []
        for subject in subjects:
            for label in range(1, 7):
                labels += [label] * windows_per_subject_class
                subject_ids += [subject] * windows_per_subject_class
        n = len(labels)

        offset = 5.0 if split == "test" else 0.0
        for k, name in enumerate(SIGNAL_NAMES):
            channel = (k + 1) * 10.0 + offset + rng.normal(0, 1, size=(n, 128))
            np.savetxt(split_dir / "Inertial Signals" / f"{name}_{split}.txt", channel, fmt="%.6f")
        np.savetxt(split_dir / f"y_{split}.txt", labels, fmt="%d")
        np.savetxt(split_dir / f"subject_{split}.txt", subject_ids, fmt="%d")

    return dataset_dir


class TestUciHarLoader(unittest.TestCase):
    """Test suite for loading, splitting, standardizing and saving UCI HAR."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.train_subjects = [1, 3, 5, 6, 7, 8, 11, 14]
        cls.test_subjects = [2, 4, 9]
        cls.dataset_dir = write_fake_uci_har(cls.temp_dir, cls.train_subjects, cls.test_subjects)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_signal_stacking_order(self) -> None:
        """Channels are stacked on the last axis in SIGNAL_NAMES order."""
        X = load_inertial_signals(self.dataset_dir, "train")

        self.assertEqual(X.shape, (len(self.train_subjects) * 6 * 2, 128, 9))
        self.assertEqual(X.dtype, np.float32)
        channel_means = X.mean(axis=(0, 1))
        np.testing.assert_allclose(channel_means, [(k + 1) * 10.0 for k in range(9)], atol=0.5)

    def test_02_labels_are_zero_indexed(self) -> None:
        """Labels come back in 0..5, never 1..6."""
        y = load_labels(self.dataset_dir, "train")

        self.assertTrue(np.issubdtype(y.dtype, np.integer))
        self.assertEqual(sorted(np.unique(y).tolist()), [0, 1, 2, 3, 4, 5])
        self.assertNotIn(6, y)

    def test_03_subject_split_is_disjoint(self) -> None:
        """No subject ID appears in both train and val."""
        subjects = load_subjects(self.dataset_dir, "train")
        labels = load_labels(self.dataset_dir, "train")
        train_mask, val_mask = split_train_val_by_subject(subjects, labels, n_val_subjects=3, seed=42)

        self.assertFalse(np.any(train_mask & val_mask), "A window cannot be in both splits.")
        self.assertTrue(np.all(train_mask | val_mask), "Every window must land in a split.")
        train_subs = set(np.unique(subjects[train_mask]).tolist())
        val_subs = set(np.unique(subjects[val_mask]).tolist())
        self.assertEqual(len(val_subs), 3)
        self.assertEqual(train_subs & val_subs, set())

        # Same seed gives the same split
        train_mask_2, _ = split_train_val_by_subject(subjects, labels, n_val_subjects=3, seed=42)
        np.testing.assert_array_equal(train_mask, train_mask_2)

    def test_04_all_classes_present_in_both_splits(self) -> None:
        """Every class 0..5 appears in train and val, re-drawing when a draw misses one."""
        subjects = load_subjects(self.dataset_dir, "train")
        labels = load_labels(self.dataset_dir, "train")
        train_mask, val_mask = split_train_val_by_subject(subjects, labels, n_val_subjects=3, seed=7)
        self.assertEqual(set(labels[train_mask].tolist()), set(range(6)))
        self.assertEqual(set(labels[val_mask].tolist()), set(range(6)))

        # Subject 99 only ever performs class 0, so any split holding it out as the
        # only val subject is missing classes 1..5 and must be rejected.
        lonely_subjects = np.array([99] * 6 + [1] * 6)
        lonely_labels = np.array([0] * 6 + list(range(6)))
        with self.assertRaises(RuntimeError):
            split_train_val_by_subject(lonely_subjects, lonely_labels, n_val_subjects=1, seed=0)

    def test_05_standardization_uses_train_statistics_only(self) -> None:
        """Val/test are scaled with the train mean/std, not their own."""
        data = build_processed_dataset(self.dataset_dir, n_val_subjects=3, seed=42)

        # Train is centred on zero by construction
        np.testing.assert_allclose(data["X_train"].mean(axis=(0, 1)), np.zeros(9), atol=1e-3)
        # Test raw values sit +5 above train; with train statistics they stay clearly off zero.
        # If test were standardized with its own statistics its mean would be ~0.
        test_means = data["X_test"].mean(axis=(0, 1))
        self.assertTrue(np.all(np.abs(test_means) > 0.5), f"Test means look self-normalized: {test_means}")

        # Val is transformed with train statistics too: re-derive it by hand and compare
        raw = build_processed_dataset(self.dataset_dir, n_val_subjects=3, seed=42, standardize=False)
        mean = raw["X_train"].mean(axis=(0, 1))
        std = raw["X_train"].std(axis=(0, 1))
        np.testing.assert_allclose(data["X_val"], (raw["X_val"] - mean) / std, rtol=1e-4, atol=1e-4)

    def test_06_output_passes_data_contract(self) -> None:
        """The built dict passes validate_har_dataset, including the test split."""
        data = build_processed_dataset(self.dataset_dir, n_val_subjects=3, seed=42)

        validate_har_dataset(data, check_test=True)
        self.assertEqual(set(data.keys()), set(CONTRACT_KEYS))
        self.assertEqual(data["X_test"].shape[0], len(self.test_subjects) * 6 * 2)

    def test_07_npz_roundtrip(self) -> None:
        """Saving then loading returns identical keys and arrays."""
        data = build_processed_dataset(self.dataset_dir, n_val_subjects=3, seed=42)
        out_path = save_processed_dataset(data, self.temp_dir / "nested" / "processed.npz")

        self.assertTrue(out_path.is_file())
        with np.load(out_path, allow_pickle=False) as loaded:
            self.assertEqual(sorted(loaded.files), sorted(CONTRACT_KEYS))
            for key in CONTRACT_KEYS:
                np.testing.assert_array_equal(loaded[key], data[key], err_msg=key)
                self.assertEqual(loaded[key].dtype, data[key].dtype)

    def test_08_missing_dataset_raises_clear_error(self) -> None:
        """A missing folder raises DatasetNotFoundError naming the URL and location."""
        missing_dir = self.temp_dir / "does_not_exist"
        with self.assertRaises(DatasetNotFoundError) as ctx:
            build_processed_dataset(missing_dir)

        message = str(ctx.exception)
        self.assertIn(UCI_HAR_URL, message)
        self.assertIn(str(missing_dir), message)
        self.assertIn("--download", message)
        # Still a FileNotFoundError for callers that catch the builtin
        self.assertIsInstance(ctx.exception, FileNotFoundError)


@unittest.skipUnless(DEFAULT_DATASET_DIR.is_dir(), "Real UCI HAR dataset not downloaded")
class TestRealUciHarDataset(unittest.TestCase):
    """Checks against the real dataset, skipped when it is not present."""

    def test_09_real_dataset_shapes_and_subjects(self) -> None:
        """Real data has the documented sizes and 21 train/val + 9 test disjoint subjects."""
        data = build_processed_dataset(DEFAULT_DATASET_DIR, n_val_subjects=4, seed=42)

        self.assertEqual(data["X_train"].shape[0] + data["X_val"].shape[0], 7352)
        self.assertEqual(data["X_test"].shape, (2947, 128, 9))
        train_subs = set(np.unique(data["subject_train"]).tolist())
        val_subs = set(np.unique(data["subject_val"]).tolist())
        test_subs = set(np.unique(data["subject_test"]).tolist())
        self.assertEqual(len(train_subs | val_subs), 21)
        self.assertEqual(len(test_subs), 9)
        self.assertEqual(train_subs & val_subs, set())
        self.assertEqual((train_subs | val_subs) & test_subs, set())


if __name__ == "__main__":
    unittest.main()
