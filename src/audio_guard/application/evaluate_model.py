"""v3/v4 모델을 ESC-50으로 평가하고 판정 임계치를 선택합니다."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from audio_guard.domain.pipeline import create_pipeline
from audio_guard.domain.pipeline.base import SAMPLE_RATE
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from audio_guard.infrastructure.dataset.esc50_dataset import (
    examples_for_folds,
    load_metadata,
)
from audio_guard.labels import LABELS, TARGET_CLASSES


def metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }


def choose_threshold(labels, scores, min_recall=None):
    """검증 데이터에서 abnormal 클래스의 판정 임계치를 선택합니다."""
    candidates = np.unique(np.clip(np.concatenate([
        scores, np.array([0.001, 0.999])]), 0.001, 0.999))
    rows = [metrics(labels, scores, threshold) for threshold in candidates]
    if min_recall is not None:
        rows = [row for row in rows if row["recall"] >= min_recall]
        if not rows:
            raise ValueError("No threshold meets validation recall target")
        return max(rows, key=lambda row: (row["precision"], row["f1"]))
    return max(rows, key=lambda row: (row["f1"], row["recall"]))


def scores_for_files(model, examples, pipeline):
    labels, scores = [], []
    for path, label in examples:
        audio, sample_rate = load_audio(path)
        model_input = pipeline.transform(audio, sample_rate)
        prediction = model.predict(model_input, verbose=0)
        scores.append(float(np.asarray(prediction).reshape(-1).max()))
        labels.append(label)
    return np.asarray(labels), np.asarray(scores)


def _write_v3_analysis(metadata, labels, scores, results_dir):
    threshold_rows = [
        metrics(labels, scores, threshold)
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7)
    ]
    columns = {
        "threshold": "Threshold",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "tp": "TP",
        "fn": "FN",
        "fp": "FP",
        "tn": "TN",
    }
    pd.DataFrame(threshold_rows).rename(columns=columns).to_csv(
        results_dir / "threshold_results.csv", index=False, encoding="utf-8-sig")

    predictions = (scores >= 0.5).astype(int)
    error_rows = []
    rows = list(metadata.itertuples())
    for index, (actual, predicted) in enumerate(zip(labels, predictions)):
        if actual == predicted:
            continue
        error_rows.append({
            "filename": rows[index].filename,
            "category": rows[index].category,
            "true_label": int(actual),
            "predicted_label": int(predicted),
            "abnormal_probability": float(scores[index]),
            "error_type": "FP" if actual == 0 else "FN",
        })
    errors = pd.DataFrame(error_rows)
    errors.to_csv(results_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    if errors.empty:
        summary = pd.DataFrame(columns=["category", "error_type", "count"])
    else:
        summary = errors.groupby(
            ["category", "error_type"]).size().reset_index(name="count")
    summary.to_csv(results_dir / "error_summary.csv", index=False, encoding="utf-8-sig")
    return max(threshold_rows, key=lambda row: row["f1"])


def evaluate_model(
    pipeline_name: str,
    esc50_dir: Path,
    model_path: Path,
    results_dir=Path("results"),
    min_recall=None,
):
    """선택한 파이프라인의 모델을 ESC-50 검증 및 테스트 fold로 평가합니다."""
    metadata = load_metadata(esc50_dir)
    audio_dir = esc50_dir / "audio"
    model = tf.keras.models.load_model(model_path, compile=False)
    pipeline = create_pipeline(pipeline_name)
    results_dir.mkdir(parents=True, exist_ok=True)

    if pipeline_name == "v3":
        rows = list(metadata.itertuples())
        examples = [
            (audio_dir / row.filename, int(row.category in TARGET_CLASSES))
            for row in rows
        ]
        labels, scores = scores_for_files(model, examples, pipeline)
        return _write_v3_analysis(metadata, labels, scores, results_dir)

    validation = examples_for_folds(metadata, audio_dir, {4}, TARGET_CLASSES)
    validation_y, validation_scores = scores_for_files(model, validation, pipeline)
    selected = choose_threshold(validation_y, validation_scores, min_recall)

    test = examples_for_folds(metadata, audio_dir, {5}, TARGET_CLASSES)
    test_y, test_scores = scores_for_files(model, test, pipeline)
    test_metrics = metrics(test_y, test_scores, selected["threshold"])

    manifest = {
        "model_path": model_path.name,
        "pipeline": "v4",
        "sample_rate": SAMPLE_RATE,
        "labels": LABELS,
        "esc_abnormal_categories": sorted(TARGET_CLASSES),
        "threshold": selected["threshold"],
    }
    with model_path.with_suffix(".config.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)

    return {"validation": selected, "test": test_metrics}
