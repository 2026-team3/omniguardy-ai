"""ESC-50 메타데이터와 오디오 파일 목록을 제공합니다."""

from pathlib import Path

import pandas as pd


def load_metadata(root: Path):
    return pd.read_csv(root / "meta" / "esc50.csv")


def examples_for_folds(metadata, audio_dir: Path, folds, abnormal_categories):
    """지정 fold의 ``(경로, 이진 라벨)`` 목록을 반환합니다."""
    return [
        (audio_dir / row.filename, int(row.category in abnormal_categories))
        for row in metadata[metadata["fold"].isin(folds)].itertuples()
    ]
