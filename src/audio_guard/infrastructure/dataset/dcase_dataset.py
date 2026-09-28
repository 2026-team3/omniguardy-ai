"""DCASE 메타데이터를 통합 manifest 후보로 변환하는 도우미입니다."""

import csv
from pathlib import Path


def load_dcase_rows(metadata_path: Path, audio_dir: Path, event_mapping=None):
    """정규화된 filename/event_label CSV를 읽습니다.

    DCASE task별 원본 schema는 준비 단계에서 이 형식으로 정규화하며,
    mapping에 없는 이벤트는 hard-negative background로 취급합니다.
    """
    event_mapping = event_mapping or {"knock": "knock", "door_knock": "knock"}
    with Path(metadata_path).open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows or not {"filename", "event_label"}.issubset(rows[0]):
        raise ValueError("DCASE metadata requires filename and event_label columns")
    return [
        (
            Path(audio_dir) / row["filename"],
            event_mapping.get(row["event_label"], "background"),
            row,
        )
        for row in rows
    ]
