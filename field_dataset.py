"""Label field WAV recordings by their normal/abnormal parent directory."""

from pathlib import Path

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
