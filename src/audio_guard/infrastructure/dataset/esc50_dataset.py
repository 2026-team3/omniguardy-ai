"""ESC-50을 Door Event class 후보로 변환합니다."""

from pathlib import Path

import pandas as pd


def load_metadata(root: Path):
    return pd.read_csv(root / "meta" / "esc50.csv")


def examples_for_folds(metadata, audio_dir: Path, folds):
    """door knock은 knock, 나머지는 hard-negative background로 반환합니다."""
    return [
        (audio_dir / row.filename,
         "knock" if row.category == "door_wood_knock" else "background")
        for row in metadata[metadata["fold"].isin(folds)].itertuples()
    ]
