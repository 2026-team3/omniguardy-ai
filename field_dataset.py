"""Label field WAV recordings by their normal/abnormal parent directory."""

from pathlib import Path
import csv
import warnings

import numpy as np
from sklearn.model_selection import train_test_split


def split_field_files(dataset_dir: Path, validation_size=0.2, seed=42):
    paths, labels = [], []
    for folder, label in (("normal", 0), ("abnormal", 1)):
        wav_files = sorted((dataset_dir / folder).glob("*.wav"))
        if len(wav_files) < 2:
            raise ValueError(f"Expected at least two WAV files in {dataset_dir / folder}")
        paths.extend(wav_files)
        labels.extend([label] * len(wav_files))

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, np.asarray(labels, dtype=np.int32),
        test_size=validation_size, random_state=seed, stratify=labels,
    )
    return train_paths, val_paths, train_labels, val_labels


def split_field_three_way(dataset_dir: Path, split_csv=None, seed=42):
    """Return original-file train/validation/test splits before windowing.

    A manifest with filename,split,group_id is the safe way to keep a session
    or source recording in one split. Without it, file-level separation only
    is possible and a warning is emitted.
    """
    files = [(path, label)
             for folder, label in (("normal", 0), ("abnormal", 1))
             for path in sorted((dataset_dir / folder).glob("*.wav"))]
    if min(sum(label == value for _, label in files) for value in (0, 1)) < 3:
        raise ValueError("At least three original WAVs per label are required")

    if split_csv is None:
        warnings.warn("No field split manifest: recording-session leakage "
                      "cannot be ruled out", RuntimeWarning)
        paths, labels = zip(*files)
        train, remainder, train_y, remainder_y = train_test_split(
            paths, labels, test_size=0.4, stratify=labels, random_state=seed)
        val, test, val_y, test_y = train_test_split(
            remainder, remainder_y, test_size=0.5,
            stratify=remainder_y, random_state=seed)
        splits = {"train": list(zip(train, train_y)),
                  "validation": list(zip(val, val_y)),
                  "test": list(zip(test, test_y))}
    else:
        with Path(split_csv).open(newline="", encoding="utf-8-sig") as source:
            rows = list(csv.DictReader(source))
        by_name = {}
        groups = {}
        for row in rows:
            name, split, group = row["filename"], row["split"], row["group_id"]
            if name in by_name or split not in {"train", "validation", "test"} or not group:
                raise ValueError("Invalid or duplicate field split manifest row")
            if group in groups and groups[group] != split:
                raise ValueError(f"Recording group {group} crosses splits")
            by_name[name] = split
            groups[group] = split
        expected = {f"{path.parent.name}/{path.name}" for path, _ in files}
        if set(by_name) != expected:
            raise ValueError("Field split manifest must list each WAV exactly once")
        splits = {name: [] for name in ("train", "validation", "test")}
        for path, label in files:
            key = f"{path.parent.name}/{path.name}"
            splits[by_name[key]].append((path, label))

    for name, items in splits.items():
        if {label for _, label in items} != {0, 1}:
            raise ValueError(f"Field {name} split must include both labels")
    return splits
