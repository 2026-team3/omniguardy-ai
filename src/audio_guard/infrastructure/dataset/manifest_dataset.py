"""누수 없는 Door Event dataset manifest 로딩과 검증입니다."""

import csv
from dataclasses import dataclass
from pathlib import Path

from audio_guard.labels import LABELS

SPLITS = frozenset({"train", "validation", "test"})
REQUIRED_COLUMNS = frozenset({
    "path", "class_name", "dataset", "source_id",
    "session_id", "group_id", "split",
})


@dataclass(frozen=True)
class ManifestExample:
    path: Path
    class_name: str
    label: int
    dataset: str
    source_id: str
    session_id: str
    group_id: str
    split: str
    event_start_seconds: float | None = None
    event_end_seconds: float | None = None


def _optional_float(value):
    value = (value or "").strip()
    return float(value) if value else None


def load_manifest(manifest_path: Path, require_files=True):
    manifest_path = Path(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
            raise ValueError(f"Dataset manifest is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Dataset manifest is empty")

    identity_splits = {name: {} for name in ("source_id", "session_id", "group_id")}
    examples = []
    for row_number, row in enumerate(rows, start=2):
        values = {key: (row.get(key) or "").strip() for key in REQUIRED_COLUMNS}
        if any(not value for value in values.values()):
            raise ValueError(f"Manifest row {row_number} has an empty required value")
        if values["class_name"] not in LABELS:
            raise ValueError(f"Unknown class at row {row_number}: {values['class_name']}")
        if values["split"] not in SPLITS:
            raise ValueError(f"Unknown split at row {row_number}: {values['split']}")

        for identity in identity_splits:
            key = (values["dataset"], values[identity])
            previous = identity_splits[identity].setdefault(key, values["split"])
            if previous != values["split"]:
                raise ValueError(
                    f"{identity} crosses splits: {values[identity]} "
                    f"({previous}, {values['split']})"
                )

        path = Path(values["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        if require_files and not path.is_file():
            raise ValueError(f"Audio file does not exist: {path}")
        start = _optional_float(row.get("event_start_seconds"))
        end = _optional_float(row.get("event_end_seconds"))
        if (start is None) != (end is None):
            raise ValueError(f"Incomplete event interval at row {row_number}")
        if start is not None and (start < 0 or end <= start):
            raise ValueError(f"Invalid event interval at row {row_number}")
        if values["class_name"] == "background" and start is not None:
            raise ValueError("Background rows must not contain an event interval")
        examples.append(ManifestExample(
            path=path,
            class_name=values["class_name"],
            label=LABELS[values["class_name"]],
            dataset=values["dataset"],
            source_id=values["source_id"],
            session_id=values["session_id"],
            group_id=values["group_id"],
            split=values["split"],
            event_start_seconds=start,
            event_end_seconds=end,
        ))

    present_splits = {example.split for example in examples}
    if present_splits != SPLITS:
        raise ValueError(f"Manifest must contain all splits: {sorted(SPLITS)}")
    return examples


def examples_for_split(examples, split):
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    selected = [example for example in examples if example.split == split]
    if not selected:
        raise ValueError(f"No examples for split: {split}")
    return selected
