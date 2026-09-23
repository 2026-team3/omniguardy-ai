"""현장 normal/abnormal WAV와 녹음 세션 단위 분할을 읽습니다."""

import csv
from pathlib import Path


LABELS_BY_DIRECTORY = {"normal": 0, "abnormal": 1}
SPLITS = frozenset({"train", "validation", "test"})


def _field_files(dataset_dir: Path):
    files = []
    for directory, label in LABELS_BY_DIRECTORY.items():
        files.extend(
            (path, label)
            for path in sorted((dataset_dir / directory).rglob("*.wav"))
        )
    return files


def split_field_three_way(dataset_dir: Path, split_csv: Path):
    """현장 WAV를 manifest에 따라 train/validation/test로 분리합니다.

    manifest는 ``filename,split,group_id`` 열을 반드시 포함해야 합니다.
    filename은 dataset_dir 기준의 상대 경로이며, 같은 group_id는 하나의 split에만
    존재해야 녹음 세션 누수를 막을 수 있습니다.
    """
    dataset_dir = Path(dataset_dir)
    split_csv = Path(split_csv)
    files = _field_files(dataset_dir)
    if not files:
        raise ValueError("No field WAV files found in normal/ or abnormal/")

    with split_csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    required = {"filename", "split", "group_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Field split manifest requires filename,split,group_id columns")

    split_by_name = {}
    split_by_group = {}
    for row in rows:
        filename = row["filename"].replace("\\", "/")
        split = row["split"]
        group_id = row["group_id"]
        if not filename or split not in SPLITS or not group_id:
            raise ValueError("Invalid field split manifest row")
        if filename in split_by_name:
            raise ValueError(f"Duplicate field split manifest filename: {filename}")
        previous_split = split_by_group.get(group_id)
        if previous_split is not None and previous_split != split:
            raise ValueError(f"Recording group crosses splits: {group_id}")
        split_by_name[filename] = split
        split_by_group[group_id] = split

    expected_names = {
        path.relative_to(dataset_dir).as_posix()
        for path, _ in files
    }
    if set(split_by_name) != expected_names:
        missing = sorted(expected_names - set(split_by_name))
        extra = sorted(set(split_by_name) - expected_names)
        raise ValueError(
            f"Field split manifest must list each WAV exactly once "
            f"(missing={missing[:3]}, extra={extra[:3]})"
        )

    splits = {name: [] for name in SPLITS}
    for path, label in files:
        filename = path.relative_to(dataset_dir).as_posix()
        splits[split_by_name[filename]].append((path, label))

    for name, examples in splits.items():
        labels = {label for _, label in examples}
        if labels != {0, 1}:
            raise ValueError(f"Field {name} split must contain normal and abnormal WAVs")
    return splits
