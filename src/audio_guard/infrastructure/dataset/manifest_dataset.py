

import csv
from dataclasses import dataclass
from pathlib import Path

from audio_guard.labels import LABELS

SPLITS = frozenset({"train", "validation", "test"})

#녹음 파일 자체 정보
RECORDING_REQUIRED_COLUMNS = frozenset({
    "recording_id",
    "path",
    "dataset",
    "source_id",
    "session_id",
    "group_id",
    "split",
})

#녹음 안에서 발생한 이벤트 정보
EVENT_REQUIRED_COLUMNS = frozenset({
    "recording_id",
    "class_name",
    "event_start_seconds",
    "event_end_seconds",
})

#녹음 파일 내 단일 문 이벤트의 클래스와 발생 구간
@dataclass(frozen=True)
class EventAnnotation:
    class_name: str
    label: int
    start_seconds: float
    end_seconds: float

#하나의 녹음 파일과 해당 파일에 포함된 여러 문 이벤트
@dataclass(frozen=True)
class RecordingExample:
    recording_id: str
    path: Path
    dataset: str
    source_id: str
    session_id: str
    group_id: str
    split: str
    events: tuple[EventAnnotation, ...]



#녹음 manifest와 이벤트 annotation을 읽어 녹음 단위 데이터로 반환
def load_manifest(
    manifest_path: Path,
    events_path: Path,
    require_files=True,
):
   

    manifest_path = Path(manifest_path)
    events_path = Path(events_path)

    # 녹음 파일 정보 로드
    with manifest_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)

        if (
            not reader.fieldnames
            or not RECORDING_REQUIRED_COLUMNS.issubset(reader.fieldnames)
        ):
            missing = sorted(
                RECORDING_REQUIRED_COLUMNS - set(reader.fieldnames or ())
            )
            raise ValueError(
                f"Recording manifest is missing columns: {missing}"
            )

        recording_rows = list(reader)

    if not recording_rows:
        raise ValueError("Recording manifest is empty")

    # 이벤트 annotation 로드
    with events_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)

        if (
            not reader.fieldnames
            or not EVENT_REQUIRED_COLUMNS.issubset(reader.fieldnames)
        ):
            missing = sorted(
                EVENT_REQUIRED_COLUMNS - set(reader.fieldnames or ())
            )
            raise ValueError(
                f"Event manifest is missing columns: {missing}"
            )

        event_rows = list(reader)

    identity_splits = {
        name: {}
        for name in ("source_id", "session_id", "group_id")
    }

    recording_by_id = {}

    # 녹음 파일 검증
    for row_number, row in enumerate(recording_rows, start=2):
        values = {
            key: (row.get(key) or "").strip()
            for key in RECORDING_REQUIRED_COLUMNS
        }

        if any(not value for value in values.values()):
            raise ValueError(
                f"Recording manifest row {row_number} "
                "has an empty required value"
            )

        recording_id = values["recording_id"]

        if recording_id in recording_by_id:
            raise ValueError(
                f"Duplicate recording_id: {recording_id}"
            )

        if values["split"] not in SPLITS:
            raise ValueError(
                f"Unknown split at row {row_number}: "
                f"{values['split']}"
            )

        # 같은 source/session/group이 서로 다른 split에 들어가는 것을 방지
        for identity in identity_splits:
            key = (values["dataset"], values[identity])

            previous = identity_splits[identity].setdefault(
                key,
                values["split"],
            )

            if previous != values["split"]:
                raise ValueError(
                    f"{identity} crosses splits: "
                    f"{values[identity]} "
                    f"({previous}, {values['split']})"
                )

        path = Path(values["path"])

        if not path.is_absolute():
            path = manifest_path.parent / path

        path = path.resolve()

        if require_files and not path.is_file():
            raise ValueError(
                f"Audio file does not exist: {path}"
            )

        recording_by_id[recording_id] = {
            "recording_id": recording_id,
            "path": path,
            "dataset": values["dataset"],
            "source_id": values["source_id"],
            "session_id": values["session_id"],
            "group_id": values["group_id"],
            "split": values["split"],
            "events": [],
        }

    # 이벤트를 recording_id 기준으로 녹음 파일에 연결
    for row_number, row in enumerate(event_rows, start=2):
        values = {
            key: (row.get(key) or "").strip()
            for key in EVENT_REQUIRED_COLUMNS
        }

        if any(not value for value in values.values()):
            raise ValueError(
                f"Event manifest row {row_number} "
                "has an empty required value"
            )

        recording_id = values["recording_id"]
        class_name = values["class_name"]

        if recording_id not in recording_by_id:
            raise ValueError(
                f"Unknown recording_id at event row "
                f"{row_number}: {recording_id}"
            )

        if class_name not in LABELS:
            raise ValueError(
                f"Unknown event class at row "
                f"{row_number}: {class_name}"
            )

        if class_name == "background":
            raise ValueError(
                "Background must not be explicitly annotated as an event"
            )

        try:
            start = float(values["event_start_seconds"])
            end = float(values["event_end_seconds"])
        except ValueError as exc:
            raise ValueError(
                f"Invalid event interval at row {row_number}"
            ) from exc

        if start < 0 or end <= start:
            raise ValueError(
                f"Invalid event interval at row {row_number}"
            )

        recording_by_id[recording_id]["events"].append(
            EventAnnotation(
                class_name=class_name,
                label=LABELS[class_name],
                start_seconds=start,
                end_seconds=end,
            )
        )

    examples = []

    for recording in recording_by_id.values():
        # 이벤트를 발생 시간 순으로 정렬
        events = tuple(
            sorted(
                recording["events"],
                key=lambda event: event.start_seconds,
            )
        )

        examples.append(
            RecordingExample(
                recording_id=recording["recording_id"],
                path=recording["path"],
                dataset=recording["dataset"],
                source_id=recording["source_id"],
                session_id=recording["session_id"],
                group_id=recording["group_id"],
                split=recording["split"],
                events=events,
            )
        )

    present_splits = {
        example.split
        for example in examples
    }

    if present_splits != SPLITS:
        raise ValueError(
            f"Manifest must contain all splits: {sorted(SPLITS)}"
        )

    return examples

def examples_for_split(examples, split):
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    selected = [example for example in examples if example.split == split]
    if not selected:
        raise ValueError(f"No examples for split: {split}")
    return selected
