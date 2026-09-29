#Validation set만 사용해 Knock/Handle threshold를 독립적으로 확정

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

from audio_guard.domain.pipeline import create_pipeline
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from audio_guard.infrastructure.dataset.manifest_dataset import examples_for_split, load_manifest
from audio_guard.labels import EVENT_CLASSES, LABELS

 #녹음 단위 실제 클래스와 최대 class probability를 반환
def clip_probabilities(model, examples, pipeline):
   

    labels, scores = [], []

    for example in examples:
        audio, sample_rate = load_audio(example.path)

        model_input = pipeline.transform(
            audio,
            sample_rate,
        )

        probabilities = np.asarray(
            model.predict(
                model_input,
                verbose=0,
            )
        )

        if (
            probabilities.ndim != 2
            or probabilities.shape[1] != len(LABELS)
        ):
            raise ValueError(
                "Door Event model must return "
                "three probabilities per window"
            )

        event_labels = {
            event.label
            for event in example.events
        }

        if not event_labels:
            actual_label = LABELS["background"]

        elif len(event_labels) == 1:
            actual_label = next(iter(event_labels))

        else:
            raise ValueError(
                f"Recording contains multiple event classes: "
                f"{example.recording_id}"
            )

        labels.append(actual_label)

        scores.append(
            probabilities.max(axis=0)
        )

    return (
        np.asarray(
            labels,
            dtype=np.int32,
        ),
        np.asarray(
            scores,
            dtype=np.float32,
        ),
    )

def threshold_metrics(labels, class_scores, class_index, threshold):
    actual = labels == class_index
    predicted = class_scores >= threshold
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "false_positives": int(np.sum(predicted & ~actual)),
        "background_false_positives": int(
            np.sum(predicted & (labels == LABELS["background"]))
        ),
    }


def choose_class_threshold(
    labels,
    class_scores,
    class_index,
    candidates=None,
    min_precision=None,
    min_recall=None,
):
    candidates = candidates if candidates is not None else np.linspace(0.05, 0.95, 91)
    rows = [
        threshold_metrics(labels, class_scores, class_index, threshold)
        for threshold in candidates
    ]
    eligible = [
        row for row in rows
        if (min_precision is None or row["precision"] >= min_precision)
        and (min_recall is None or row["recall"] >= min_recall)
    ]
    if not eligible:
        raise ValueError("No threshold meets the validation precision/recall constraints")
    return max(eligible, key=lambda row: (
        row["f1"],
        row["precision"],
        row["recall"],
        -row["background_false_positives"],
        row["threshold"],
    ))


def calibrate_thresholds(
    manifest_path: Path,
    events_path: Path,
    model_path: Path,
    config_out: Path,
    min_precision=None,
    min_recall=None,
    window_seconds=1.0,
    hop_seconds=0.2,
    cooldown_seconds=1.0,
):
    examples = load_manifest(
    manifest_path,
    events_path,
)
    validation = examples_for_split(examples, "validation")
    model = tf.keras.models.load_model(model_path, compile=False)
    pipeline = create_pipeline("door_event", window_seconds, hop_seconds)
    labels, probabilities = clip_probabilities(model, validation, pipeline)

    selected = {}
    for event_class in EVENT_CLASSES:
        index = LABELS[event_class]
        selected[event_class] = choose_class_threshold(
            labels,
            probabilities[:, index],
            index,
            min_precision=min_precision,
            min_recall=min_recall,
        )
    manifest = {
        "model_path": str(Path(model_path).name),
        "pipeline": "door_event",
        "sample_rate": 22050,
        "labels": LABELS,
        "thresholds": {
            name: selected[name]["threshold"] for name in EVENT_CLASSES
        },
        "window_seconds": float(window_seconds),
        "hop_seconds": float(hop_seconds),
        "cooldown_seconds": float(cooldown_seconds),
        "calibration": {"split": "validation", "metrics": selected},
    }
    config_out = Path(config_out)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    with config_out.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    return manifest
