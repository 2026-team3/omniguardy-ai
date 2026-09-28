"""확정된 모델과 threshold로 잠긴 test split을 최종 평가합니다."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from audio_guard.domain.event_policy import EventPolicy
from audio_guard.domain.pipeline import create_pipeline
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from audio_guard.infrastructure.dataset.manifest_dataset import examples_for_split, load_manifest
from audio_guard.labels import CLASS_NAMES, LABELS


def predictions_from_probabilities(probabilities, thresholds):
    policy = EventPolicy(thresholds)
    predictions = []
    for row in probabilities:
        class_name, _ = policy.decide(row)
        predictions.append(LABELS[class_name])
    return np.asarray(predictions, dtype=np.int32)


def runtime_predictions(model, examples, pipeline, thresholds):
    """Cooldown 없이 각 clip의 첫 threshold 초과 window를 재현합니다."""
    policy = EventPolicy(thresholds)
    labels, predictions = [], []
    for example in examples:
        audio, sample_rate = load_audio(example.path)
        model_input = pipeline.transform(audio, sample_rate)
        probabilities = np.asarray(model.predict(model_input, verbose=0))
        predicted_class = "background"
        for window_probabilities in probabilities:
            candidate, _ = policy.decide(window_probabilities)
            if candidate != "background":
                predicted_class = candidate
                break
        labels.append(example.label)
        predictions.append(LABELS[predicted_class])
    return np.asarray(labels, dtype=np.int32), np.asarray(predictions, dtype=np.int32)


def classification_metrics(labels, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    matrix = confusion_matrix(
        labels, predictions, labels=list(range(len(CLASS_NAMES)))
    )
    by_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    background = LABELS["background"]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "classes": by_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(CLASS_NAMES),
        "background_to_knock_false_positives": int(
            np.sum((labels == background) & (predictions == LABELS["knock"]))
        ),
        "background_to_handle_false_positives": int(
            np.sum((labels == background) & (predictions == LABELS["handle"]))
        ),
    }


def evaluate_model(
    manifest_path: Path,
    model_path: Path,
    runtime_config: Path,
    results_dir=Path("results"),
):
    with Path(runtime_config).open(encoding="utf-8") as source:
        config = json.load(source)
    if config.get("calibration", {}).get("split") != "validation":
        raise ValueError("Runtime config must come from validation calibration")
    examples = load_manifest(manifest_path)
    test = examples_for_split(examples, "test")
    model = tf.keras.models.load_model(model_path, compile=False)
    pipeline = create_pipeline(
        "door_event", config["window_seconds"], config["hop_seconds"]
    )
    labels, predictions = runtime_predictions(
        model, test, pipeline, config["thresholds"]
    )
    result = {
        "split": "test",
        "thresholds": config["thresholds"],
        **classification_metrics(labels, predictions),
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "test_evaluation.json"
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    result["result_path"] = str(output_path)
    return result
