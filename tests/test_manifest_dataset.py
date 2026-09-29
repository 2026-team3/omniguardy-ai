import csv

import pytest

from audio_guard.infrastructure.dataset.manifest_dataset import (
    examples_for_split,
    load_manifest,
)
from audio_guard.labels import LABELS


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _create_audio_files(tmp_path):
    for filename in (
        "knock.wav",
        "handle.wav",
        "background.wav",
    ):
        (tmp_path / filename).touch()


def _recording_rows():
    return [
        {
            "recording_id": "K001",
            "path": "knock.wav",
            "dataset": "direct",
            "source_id": "K001",
            "session_id": "S01",
            "group_id": "G01",
            "split": "train",
        },
        {
            "recording_id": "H001",
            "path": "handle.wav",
            "dataset": "direct",
            "source_id": "H001",
            "session_id": "S02",
            "group_id": "G02",
            "split": "validation",
        },
        {
            "recording_id": "B001",
            "path": "background.wav",
            "dataset": "direct",
            "source_id": "B001",
            "session_id": "S03",
            "group_id": "G03",
            "split": "test",
        },
    ]


def _event_rows():
    return [
        {
            "recording_id": "K001",
            "class_name": "knock",
            "event_start_seconds": "5.2",
            "event_end_seconds": "5.6",
        },
        {
            "recording_id": "K001",
            "class_name": "knock",
            "event_start_seconds": "13.4",
            "event_end_seconds": "13.8",
        },
        {
            "recording_id": "K001",
            "class_name": "knock",
            "event_start_seconds": "21.8",
            "event_end_seconds": "22.2",
        },
        {
            "recording_id": "H001",
            "class_name": "handle",
            "event_start_seconds": "7.1",
            "event_end_seconds": "7.8",
        },
        {
            "recording_id": "H001",
            "class_name": "handle",
            "event_start_seconds": "15.2",
            "event_end_seconds": "15.9",
        },
    ]


def _prepare_manifests(tmp_path, recording_rows=None, event_rows=None):
    _create_audio_files(tmp_path)

    manifest_path = tmp_path / "dataset_manifest.csv"
    events_path = tmp_path / "events.csv"

    _write_csv(
        manifest_path,
        [
            "recording_id",
            "path",
            "dataset",
            "source_id",
            "session_id",
            "group_id",
            "split",
        ],
        recording_rows or _recording_rows(),
    )

    _write_csv(
        events_path,
        [
            "recording_id",
            "class_name",
            "event_start_seconds",
            "event_end_seconds",
        ],
        _event_rows() if event_rows is None else event_rows,
    )

    return manifest_path, events_path


def test_load_manifest_supports_multiple_events_per_recording(tmp_path):
    manifest_path, events_path = _prepare_manifests(tmp_path)

    examples = load_manifest(
        manifest_path,
        events_path,
    )

    assert len(examples) == 3

    knock = next(
        example
        for example in examples
        if example.recording_id == "K001"
    )

    assert len(knock.events) == 3

    assert all(
        event.class_name == "knock"
        for event in knock.events
    )

    assert all(
        event.label == LABELS["knock"]
        for event in knock.events
    )

    assert knock.events[0].start_seconds == pytest.approx(5.2)
    assert knock.events[1].start_seconds == pytest.approx(13.4)
    assert knock.events[2].start_seconds == pytest.approx(21.8)


def test_background_recording_has_no_events(tmp_path):
    manifest_path, events_path = _prepare_manifests(tmp_path)

    examples = load_manifest(
        manifest_path,
        events_path,
    )

    background = next(
        example
        for example in examples
        if example.recording_id == "B001"
    )

    assert background.events == ()


def test_examples_for_split_returns_recordings(tmp_path):
    manifest_path, events_path = _prepare_manifests(tmp_path)

    examples = load_manifest(
        manifest_path,
        events_path,
    )

    train = examples_for_split(
        examples,
        "train",
    )

    assert len(train) == 1
    assert train[0].recording_id == "K001"


def test_unknown_recording_id_in_events_is_rejected(tmp_path):
    event_rows = _event_rows()

    event_rows.append(
        {
            "recording_id": "UNKNOWN",
            "class_name": "knock",
            "event_start_seconds": "30.0",
            "event_end_seconds": "30.5",
        }
    )

    manifest_path, events_path = _prepare_manifests(
        tmp_path,
        event_rows=event_rows,
    )

    with pytest.raises(
        ValueError,
        match="Unknown recording_id",
    ):
        load_manifest(
            manifest_path,
            events_path,
        )


def test_invalid_event_interval_is_rejected(tmp_path):
    event_rows = [
        {
            "recording_id": "K001",
            "class_name": "knock",
            "event_start_seconds": "10.0",
            "event_end_seconds": "9.0",
        }
    ]

    manifest_path, events_path = _prepare_manifests(
        tmp_path,
        event_rows=event_rows,
    )

    with pytest.raises(
        ValueError,
        match="Invalid event interval",
    ):
        load_manifest(
            manifest_path,
            events_path,
        )


def test_background_cannot_be_annotated_as_event(tmp_path):
    event_rows = [
        {
            "recording_id": "K001",
            "class_name": "background",
            "event_start_seconds": "5.0",
            "event_end_seconds": "6.0",
        }
    ]

    manifest_path, events_path = _prepare_manifests(
        tmp_path,
        event_rows=event_rows,
    )

    with pytest.raises(
        ValueError,
        match="Background",
    ):
        load_manifest(
            manifest_path,
            events_path,
        )