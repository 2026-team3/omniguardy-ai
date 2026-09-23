"""현장에서 수집한 abnormal WAV의 녹음 세션 단위 분할을 읽습니다."""

import csv
from dataclasses import dataclass
from pathlib import Path


ACTIVE_SPLITS = frozenset({"train", "validation", "test"})
ALLOWED_SPLITS = ACTIVE_SPLITS | {"ignored"}
EVENT_TYPES = frozenset({
    "knock",
    "door_handle",
    "forced_door",
    "impact",
    "glass_breaking",
    "siren",
    "footsteps",
    "other",
    "doorbell",
})


@dataclass(frozen=True)
class FieldExample:
    path: Path
    label: int
    event_type: str


def _field_files(dataset_dir: Path):
    return [
        (path, 1)
        for path in sorted((dataset_dir / "abnormal").rglob("*.wav"))
    ]


def split_field_three_way(dataset_dir: Path, split_csv: Path):
    """현장 WAV를 manifest에 따라 train/validation/test로 분리합니다.

    manifest는 ``filename,split,group_id`` 열을 반드시 포함하고 ``event_type``을
    선택적으로 포함할 수 있습니다. event_type이 없으면 ``other``로 처리합니다.
    filename은 dataset_dir 기준의 상대 경로이며, 같은 group_id는 하나의 split에만
    존재해야 녹음 세션 누수를 막을 수 있습니다.
    """
    dataset_dir = Path(dataset_dir)
    split_csv = Path(split_csv)
    files = _field_files(dataset_dir)
    if not files:
        raise ValueError("No field WAV files found in abnormal/")

    with split_csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    required = {"filename", "split", "group_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Field split manifest requires filename,split,group_id columns")

    row_by_name = {}
    split_by_group = {}
    event_type_by_group = {}
    active_groups_by_event = {}
    for row in rows:
        filename = row["filename"].strip().replace("\\", "/")
        split = row["split"].strip()
        group_id = row["group_id"].strip()
        event_type = (row.get("event_type") or "other").strip()
        if not filename or split not in ALLOWED_SPLITS or not group_id:
            raise ValueError("Invalid field split manifest row")
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported field event_type: {event_type}")
        if event_type == "doorbell" and split != "ignored":
            raise ValueError("doorbell field files must use the ignored split")
        if filename in row_by_name:
            raise ValueError(f"Duplicate field split manifest filename: {filename}")
        previous_split = split_by_group.get(group_id)
        if previous_split is not None and previous_split != split:
            raise ValueError(f"Recording group crosses splits: {group_id}")
        previous_event_type = event_type_by_group.get(group_id)
        if previous_event_type is not None and previous_event_type != event_type:
            raise ValueError(f"Recording group crosses event types: {group_id}")
        row_by_name[filename] = (split, event_type)
        split_by_group[group_id] = split
        event_type_by_group[group_id] = event_type
        if split in ACTIVE_SPLITS:
            active_groups_by_event.setdefault(event_type, {})[group_id] = split

    for event_type, group_splits in active_groups_by_event.items():
        expected_split_count = min(len(group_splits), len(ACTIVE_SPLITS))
        actual_splits = set(group_splits.values())
        if len(actual_splits) < expected_split_count:
            raise ValueError(
                f"Field event_type must span {expected_split_count} splits when "
                f"it has {len(group_splits)} recording groups: {event_type}"
            )

    expected_names = {
        path.relative_to(dataset_dir).as_posix()
        for path, _ in files
    }
    if set(row_by_name) != expected_names:
        missing = sorted(expected_names - set(row_by_name))
        extra = sorted(set(row_by_name) - expected_names)
        raise ValueError(
            f"Field split manifest must list each WAV exactly once "
            f"(missing={missing[:3]}, extra={extra[:3]})"
        )

    splits = {name: [] for name in ACTIVE_SPLITS}
    for path, label in files:
        filename = path.relative_to(dataset_dir).as_posix()
        split, event_type = row_by_name[filename]
        if split == "ignored":
            continue
        splits[split].append(FieldExample(path, label, event_type))

    for name, examples in splits.items():
        if not examples:
            raise ValueError(f"Field {name} split must contain abnormal WAVs")
    return splits
