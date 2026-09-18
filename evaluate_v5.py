"""검증셋에서 V5 임계치를 선택한 뒤 ESC와 현장 데이터를 따로 테스트합니다."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from audio_runtime import LABELS, TARGET_CLASSES, predict_score
from audio_guard.domain.pipeline.base import SAMPLE_RATE
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from field_dataset import split_field_three_way
from train_v5 import examples_from_esc


def scores_for_files(model, examples, config):
    labels, scores = [], []
    for path, label in examples:
        audio, sr = load_audio(path)
        scores.append(predict_score(model, audio, sr, config))
        labels.append(label)
    return np.asarray(labels), np.asarray(scores)


def metrics(labels, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold),
            "precision": precision_score(labels, pred, zero_division=0),
            "recall": recall_score(labels, pred, zero_division=0),
            "f1": f1_score(labels, pred, zero_division=0),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def choose_threshold(esc_y, esc_scores, field_y, field_scores, min_recall=None):
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
        # 현장 재현율 조건을 만족하는 후보 중 현장 오탐을 줄이는 임계치를 선택합니다.
        return max(rows, key=lambda row: (row[2]["precision"],
                                          row[2]["f1"], row[3]["f1"]))
    return max(rows, key=lambda row: (row[3]["f1"], row[2]["recall"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50-dir", type=Path, required=True)
    parser.add_argument("--field-data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--field-splits", type=Path)
    parser.add_argument("--model", type=Path,
                        default=Path("models/audio_model_v5.keras"))
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--min-field-recall", type=float)
    args = parser.parse_args()
    if args.min_field_recall is not None and not 0 <= args.min_field_recall <= 1:
        parser.error("--min-field-recall must be between 0 and 1")

    df = pd.read_csv(args.esc50_dir / "meta" / "esc50.csv")
    audio_dir = args.esc50_dir / "audio"
    field = split_field_three_way(args.field_data_dir, args.field_splits)
    model = tf.keras.models.load_model(args.model, compile=False)
    if model.input_shape[1] != args.max_windows:
        raise ValueError("--max-windows must match the trained model input")
    config = {"pipeline": "v5", "sample_rate": SAMPLE_RATE,
              "max_windows": args.max_windows}
    esc_val_y, esc_val_scores = scores_for_files(
        model, examples_from_esc(df, audio_dir, {4}), config)
    field_val_y, field_val_scores = scores_for_files(
        model, field["validation"], config)
    threshold, esc_val, field_val, combined_val = choose_threshold(
        esc_val_y, esc_val_scores, field_val_y, field_val_scores,
        args.min_field_recall)
    print("ESC validation:", esc_val)
    print("field validation:", field_val)
    print("combined validation:", combined_val)

    esc_test_y, esc_test_scores = scores_for_files(
        model, examples_from_esc(df, audio_dir, {5}), config)
    field_test_y, field_test_scores = scores_for_files(
        model, field["test"], config)
    print("ESC held-out test:", metrics(esc_test_y, esc_test_scores, threshold))
    print("field file-held-out test (session independence requires a manifest):",
          metrics(field_test_y, field_test_scores, threshold))

    config_path = args.model.with_suffix(".config.json")
    manifest = {"model_path": args.model.name, "pipeline": "v5",
                "sample_rate": SAMPLE_RATE, "labels": LABELS,
                "esc_abnormal_categories": sorted(TARGET_CLASSES),
                "threshold": float(threshold), "max_windows": args.max_windows}
    with config_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    print("Saved runtime config:", config_path)
    print("Set AUDIO_CONFIG_PATH to this manifest for both API and CLI.")


if __name__ == "__main__":
    main()
