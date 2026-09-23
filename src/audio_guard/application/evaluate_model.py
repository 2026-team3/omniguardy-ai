"""v3/v4 모델 평가, threshold 비교, 런타임 설정 생성을 제공합니다."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from audio_guard.domain.pipeline import create_pipeline
from audio_guard.domain.pipeline.base import SAMPLE_RATE
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from audio_guard.infrastructure.dataset.esc50_dataset import examples_for_folds, load_metadata
from audio_guard.infrastructure.dataset.field_dataset import split_field_three_way
from audio_guard.labels import LABELS, TARGET_CLASSES


DEFAULT_THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
LEGACY_V3_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def field_recall_metrics(labels, scores, threshold):
    """abnormal-only 현장 데이터에서 의미 있는 탐지 지표만 계산합니다."""
    if len(labels) == 0 or not np.all(labels == 1):
        raise ValueError("Field evaluation requires abnormal-only labels")
    predictions = (scores >= threshold).astype(int)
    tp = int(np.sum(predictions == 1))
    fn = int(np.sum(predictions == 0))
    return {
        "threshold": float(threshold),
        "recall": float(tp / len(labels)),
        "tp": tp,
        "fn": fn,
        "total": int(len(labels)),
    }


def choose_threshold(labels, scores, min_recall=None, candidates=None):
    """Recall 제약을 만족하는 후보 중 F1이 가장 높은 threshold를 선택합니다."""
    if candidates is None:
        candidates = np.unique(np.clip(np.concatenate([
            scores, np.array([0.001, 0.999])]), 0.001, 0.999))
    rows = [metrics(labels, scores, threshold) for threshold in candidates]
    if min_recall is not None:
        rows = [row for row in rows if row["recall"] >= min_recall]
        if not rows:
            raise ValueError("No threshold meets validation recall target")
    return max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))


def scores_for_files(model, examples, pipeline):
    labels, scores = [], []
    for path, label in examples:
        audio, sample_rate = load_audio(path)
        model_input = pipeline.transform(audio, sample_rate)
        prediction = model.predict(model_input, verbose=0)
        scores.append(float(np.asarray(prediction).reshape(-1).max()))
        labels.append(label)
    return np.asarray(labels, dtype=np.int32), np.asarray(scores, dtype=np.float32)


def _threshold_rows(dataset, split, labels, scores, thresholds):
    return [
        {"dataset": dataset, "split": split, **metrics(labels, scores, threshold)}
        for threshold in thresholds
    ]


def _field_threshold_rows(split, labels, scores, thresholds):
    return [
        {"dataset": "field", "split": split,
         **field_recall_metrics(labels, scores, threshold)}
        for threshold in thresholds
    ]


def _validate_thresholds(thresholds):
    values = sorted({float(value) for value in thresholds})
    if not values or any(not 0 < value < 1 for value in values):
        raise ValueError("thresholds must contain values between 0 and 1")
    return values


def _write_v3_analysis(metadata, labels, scores, results_dir):
    # v3의 기존 평가 산출물과 threshold 후보는 바꾸지 않습니다.
    rows = [metrics(labels, scores, threshold) for threshold in LEGACY_V3_THRESHOLDS]
    pd.DataFrame(rows).to_csv(
        results_dir / "threshold_results.csv", index=False, encoding="utf-8-sig")

    predictions = (scores >= 0.5).astype(int)
    errors = []
    for row, actual, predicted, score in zip(metadata.itertuples(), labels, predictions, scores):
        if actual == predicted:
            continue
        errors.append({
            "filename": row.filename,
            "category": row.category,
            "true_label": int(actual),
            "predicted_label": int(predicted),
            "abnormal_probability": float(score),
            "error_type": "FP" if actual == 0 else "FN",
        })
    error_frame = pd.DataFrame(errors)
    error_frame.to_csv(results_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    if error_frame.empty:
        summary = pd.DataFrame(columns=["category", "error_type", "count"])
    else:
        summary = error_frame.groupby(
            ["category", "error_type"]).size().reset_index(name="count")
    summary.to_csv(results_dir / "error_summary.csv", index=False, encoding="utf-8-sig")
    return max(rows, key=lambda row: row["f1"])


def evaluate_model(
    pipeline_name: str,
    esc50_dir: Path,
    model_path: Path,
    results_dir=Path("results"),
    min_recall=None,
    field_data_dir=None,
    field_splits=None,
    thresholds=DEFAULT_THRESHOLDS,
    min_field_recall=None,
):
    """ESC-50와 선택적 현장 test set을 분리 평가하고 config를 생성합니다."""
    thresholds = _validate_thresholds(thresholds)
    if (field_data_dir is None) != (field_splits is None):
        raise ValueError("field_data_dir and field_splits must be supplied together")
    if field_data_dir is None and min_field_recall is not None:
        raise ValueError("min_field_recall requires field data")

    metadata = load_metadata(esc50_dir)
    audio_dir = Path(esc50_dir) / "audio"
    model = tf.keras.models.load_model(model_path, compile=False)
    pipeline = create_pipeline(pipeline_name)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if pipeline_name == "v3":
        rows = list(metadata.itertuples())
        examples = [(audio_dir / row.filename, int(row.category in TARGET_CLASSES))
                    for row in rows]
        labels, scores = scores_for_files(model, examples, pipeline)
        return _write_v3_analysis(metadata, labels, scores, results_dir)

    esc_validation = examples_for_folds(metadata, audio_dir, {4}, TARGET_CLASSES)
    esc_test = examples_for_folds(metadata, audio_dir, {5}, TARGET_CLASSES)
    esc_validation_y, esc_validation_scores = scores_for_files(model, esc_validation, pipeline)
    esc_test_y, esc_test_scores = scores_for_files(model, esc_test, pipeline)
    rows = _threshold_rows("esc50", "validation", esc_validation_y,
                           esc_validation_scores, thresholds)
    rows.extend(_threshold_rows("esc50", "test", esc_test_y, esc_test_scores, thresholds))

    field_data = None
    field_csv_path = None
    if field_data_dir is not None:
        field = split_field_three_way(field_data_dir, field_splits)
        field_validation_y, field_validation_scores = scores_for_files(
            model, field["validation"], pipeline)
        field_test_y, field_test_scores = scores_for_files(model, field["test"], pipeline)
        field_rows = _field_threshold_rows(
            "validation", field_validation_y, field_validation_scores, thresholds)
        field_rows.extend(_field_threshold_rows(
            "test", field_test_y, field_test_scores, thresholds))
        field_csv_path = results_dir / "field_recall_comparison.csv"
        pd.DataFrame(field_rows).to_csv(
            field_csv_path, index=False, encoding="utf-8-sig")

        eligible_thresholds = thresholds
        if min_field_recall is not None:
            eligible_thresholds = [
                row["threshold"] for row in field_rows
                if row["split"] == "validation"
                and row["recall"] >= min_field_recall
            ]
            if not eligible_thresholds:
                raise ValueError("No threshold meets field validation recall target")
        selected = choose_threshold(
            esc_validation_y, esc_validation_scores, min_recall,
            candidates=eligible_thresholds)
        selected_from = "esc50_validation_with_field_recall_constraint"
        field_data = {
            "validation": (field_validation_y, field_validation_scores),
            "test": (field_test_y, field_test_scores),
        }
    else:
        selected = choose_threshold(
            esc_validation_y, esc_validation_scores, min_recall,
            candidates=thresholds)
        selected_from = "esc50_validation"

    selected_threshold = selected["threshold"]
    rows.extend(_threshold_rows("selected", selected_from, esc_validation_y,
                                esc_validation_scores, [selected_threshold]))
    csv_path = results_dir / "threshold_comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    validation_result = {
        "esc50": metrics(esc_validation_y, esc_validation_scores, selected_threshold),
    }
    test_result = {
        "esc50": metrics(esc_test_y, esc_test_scores, selected_threshold),
    }
    if field_data is not None:
        validation_result["field"] = field_recall_metrics(
            *field_data["validation"], selected_threshold)
        test_result["field"] = field_recall_metrics(
            *field_data["test"], selected_threshold)

    manifest = {
        "model_path": Path(model_path).name,
        "pipeline": "v4",
        "sample_rate": SAMPLE_RATE,
        "labels": LABELS,
        "esc_abnormal_categories": sorted(TARGET_CLASSES),
        "threshold": selected_threshold,
    }
    with Path(model_path).with_suffix(".config.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    return {
        "selected_from": selected_from,
        "selected_threshold": selected,
        "validation": validation_result,
        "test": test_result,
        "threshold_csv": str(csv_path),
        "field_recall_csv": str(field_csv_path) if field_csv_path else None,
    }
