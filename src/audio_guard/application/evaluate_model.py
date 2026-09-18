"""v3/v4/v5 모델 평가와 임계치 선택을 통합합니다."""

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
from audio_guard.infrastructure.dataset.field_dataset import (
    split_field_files,
    split_field_three_way,
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
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def choose_threshold(esc_y, esc_scores, field_y, field_scores, min_recall=None):
    """검증 점수에서 현장 재현율 조건을 반영한 임계치를 선택합니다."""
    all_scores = np.concatenate([esc_scores, field_scores])
    candidates = np.unique(np.clip(np.concatenate([
        all_scores, np.array([0.001, 0.999])]), 0.001, 0.999))
    rows = []
    for threshold in candidates:
        esc = metrics(esc_y, esc_scores, threshold)
        field = metrics(field_y, field_scores, threshold)
        combined = metrics(np.concatenate([esc_y, field_y]), all_scores, threshold)
        rows.append((threshold, esc, field, combined))
    if min_recall is not None:
        rows = [row for row in rows if row[2]["recall"] >= min_recall]
        if not rows:
            raise ValueError("No threshold meets field validation recall target")
        return max(rows, key=lambda row: (row[2]["precision"], row[2]["f1"], row[3]["f1"]))
    return max(rows, key=lambda row: (row[3]["f1"], row[2]["recall"]))


def scores_for_files(model, examples, pipeline):
    labels, scores = [], []
    for path, label in examples:
        audio, sample_rate = load_audio(path)
        model_input = pipeline.transform(audio, sample_rate)
        scores.append(float(np.asarray(model.predict(model_input, verbose=0)).reshape(-1).max()))
        labels.append(label)
    return np.asarray(labels), np.asarray(scores)


def _best_f1_threshold(labels, scores):
    candidates = np.arange(0.05, 0.951, 0.05)
    return max((metrics(labels, scores, value) for value in candidates),
               key=lambda row: row["f1"])


def evaluate_model(
    pipeline_name: str,
    esc50_dir: Path,
    field_data_dir: Path,
    model_path: Path,
    results_dir=Path("results"),
    max_windows=8,
    field_splits=None,
    min_field_recall=None,
):
    """선택한 파이프라인으로 검증 임계치와 테스트 지표를 계산합니다."""
    metadata = load_metadata(esc50_dir)
    audio_dir = esc50_dir / "audio"
    model = tf.keras.models.load_model(model_path, compile=False)
    pipeline = create_pipeline(pipeline_name, max_windows)
    results_dir.mkdir(parents=True, exist_ok=True)

    if pipeline_name == "v3":
        rows = list(metadata.itertuples())
        examples = [(audio_dir / row.filename,
                     int(row.category in TARGET_CLASSES)) for row in rows]
        labels, scores = scores_for_files(model, examples, pipeline)
        threshold_rows = [metrics(labels, scores, threshold)
                          for threshold in (0.3, 0.4, 0.5, 0.6, 0.7)]
        columns = {
            "threshold": "Threshold", "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall", "f1": "F1", "tp": "TP",
            "fn": "FN", "fp": "FP", "tn": "TN",
        }
        pd.DataFrame(threshold_rows).rename(columns=columns).to_csv(
            results_dir / "threshold_results.csv", index=False,
            encoding="utf-8-sig")
        predictions = (scores >= 0.5).astype(int)
        error_rows = []
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
        errors.to_csv(results_dir / "error_analysis.csv", index=False,
                      encoding="utf-8-sig")
        if errors.empty:
            summary = pd.DataFrame(columns=["category", "error_type", "count"])
        else:
            summary = errors.groupby(
                ["category", "error_type"]).size().reset_index(name="count")
        summary.to_csv(results_dir / "error_summary.csv", index=False,
                       encoding="utf-8-sig")
        return max(threshold_rows, key=lambda row: row["f1"])

    if pipeline_name == "v4":
        esc_val = examples_for_folds(metadata, audio_dir, {4}, TARGET_CLASSES)
        _, field_val, _, field_val_y = split_field_files(field_data_dir)
        field_val = list(zip(field_val, field_val_y))
        esc_y, esc_scores = scores_for_files(model, esc_val, pipeline)
        field_y, field_scores = scores_for_files(model, field_val, pipeline)
        selected = _best_f1_threshold(
            np.concatenate([esc_y, field_y]),
            np.concatenate([esc_scores, field_scores]),
        )
        test = examples_for_folds(metadata, audio_dir, {5}, TARGET_CLASSES)
        test_y, test_scores = scores_for_files(model, test, pipeline)
        return {"validation": selected,
                "test": metrics(test_y, test_scores, selected["threshold"])}

    if model.input_shape[1] != max_windows:
        raise ValueError("--max-windows must match the trained model input")
    field = split_field_three_way(field_data_dir, field_splits)
    esc_val = examples_for_folds(metadata, audio_dir, {4}, TARGET_CLASSES)
    esc_y, esc_scores = scores_for_files(model, esc_val, pipeline)
    field_y, field_scores = scores_for_files(model, field["validation"], pipeline)
    threshold, esc_metrics, field_metrics, combined = choose_threshold(
        esc_y, esc_scores, field_y, field_scores, min_field_recall)
    esc_test = examples_for_folds(metadata, audio_dir, {5}, TARGET_CLASSES)
    esc_test_y, esc_test_scores = scores_for_files(model, esc_test, pipeline)
    field_test_y, field_test_scores = scores_for_files(model, field["test"], pipeline)
    manifest = {
        "model_path": model_path.name,
        "pipeline": "v5",
        "sample_rate": SAMPLE_RATE,
        "labels": LABELS,
        "esc_abnormal_categories": sorted(TARGET_CLASSES),
        "threshold": float(threshold),
        "max_windows": max_windows,
    }
    with model_path.with_suffix(".config.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    return {
        "validation": {"esc": esc_metrics, "field": field_metrics, "combined": combined},
        "test": {
            "esc": metrics(esc_test_y, esc_test_scores, threshold),
            "field": metrics(field_test_y, field_test_scores, threshold),
        },
    }
