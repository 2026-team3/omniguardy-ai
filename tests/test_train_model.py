import numpy as np

from audio_guard.application.train_model import _window_labels
from audio_guard.infrastructure.dataset.manifest_dataset import (
    EventAnnotation,
    RecordingExample,
)
from audio_guard.labels import LABELS


def _recording(events):
    return RecordingExample(
        recording_id="K001",
        path=None,
        dataset="direct",
        source_id="K001",
        session_id="S01",
        group_id="G01",
        split="train",
        events=tuple(events),
    )


def test_multiple_knock_events_are_labeled():
    recording = _recording([
        EventAnnotation(
            class_name="knock",
            label=LABELS["knock"],
            start_seconds=1.2,
            end_seconds=1.5,
        ),
        EventAnnotation(
            class_name="knock",
            label=LABELS["knock"],
            start_seconds=3.2,
            end_seconds=3.5,
        ),
    ])

    offsets = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    labels = _window_labels(
        recording,
        offsets,
        window_seconds=1.0,
    )

    assert labels.tolist() == [
        LABELS["background"],
        LABELS["knock"],
        LABELS["background"],
        LABELS["knock"],
        LABELS["background"],
    ]


def test_handle_event_is_labeled():
    recording = _recording([
        EventAnnotation(
            class_name="handle",
            label=LABELS["handle"],
            start_seconds=2.2,
            end_seconds=2.7,
        ),
    ])

    offsets = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    labels = _window_labels(
        recording,
        offsets,
        window_seconds=1.0,
    )

    assert labels.tolist() == [
        LABELS["background"],
        LABELS["background"],
        LABELS["handle"],
        LABELS["background"],
    ]


def test_recording_without_events_is_all_background():
    recording = _recording([])

    offsets = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    labels = _window_labels(
        recording,
        offsets,
        window_seconds=1.0,
    )

    assert labels.tolist() == [
        LABELS["background"],
        LABELS["background"],
        LABELS["background"],
        LABELS["background"],
    ]