"""Fold 4에서 임계치를 선택하고 사용하지 않은 fold 5 결과를 따로 출력합니다."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from audio_guard.domain.pipeline.v4_windowed import audio_to_mel_windows
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from field_dataset import split_field_files
from audio_labels import TARGET_CLASSES


def predict_fold(model, df, audio_dir, fold):
    labels, scores, categories = [], [], []
    for row in df[df["fold"] == fold].itertuples():
        audio, sr = load_audio(audio_dir / row.filename)
        windows = audio_to_mel_windows(audio, sr)[..., None]
        window_scores = model.predict(windows, verbose=0).reshape(-1)
        # 윈도우 하나라도 이상 이벤트를 포함하면 해당 클립을 비정상으로 판정합니다.
        scores.append(float(window_scores.max()))
        labels.append(int(row.category in TARGET_CLASSES))
        categories.append(row.category)
    return np.asarray(labels), np.asarray(scores), categories


def predict_field_files(model, paths, labels):
    scores = []
    for path in paths:
        audio, sr = load_audio(path)
        windows = audio_to_mel_windows(audio, sr)[..., None]
        scores.append(float(model.predict(windows, verbose=0).reshape(-1).max()))
    return np.asarray(labels), np.asarray(scores)


def metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50-dir", type=Path, required=True)
    parser.add_argument("--field-data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=Path,
                        default=Path("models/audio_model_v4.keras"))
    args = parser.parse_args()
    df = pd.read_csv(args.esc50_dir / "meta" / "esc50.csv")
    model = tf.keras.models.load_model(args.model, compile=False)
    audio_dir = args.esc50_dir / "audio"

    val_y, val_scores, _ = predict_fold(model, df, audio_dir, 4)
    _, field_val, _, field_val_y = split_field_files(args.field_data_dir)
    field_y, field_scores = predict_field_files(model, field_val, field_val_y)
    print("field door-lock validation clips:", len(field_y))
    print("field validation at 0.5:", metrics(field_y, field_scores, 0.5))
    val_y = np.concatenate([val_y, field_y])
    val_scores = np.concatenate([val_scores, field_scores])
    candidates = [metrics(val_y, val_scores, float(t))
                  for t in np.arange(0.05, 0.951, 0.05)]
    # 임계치는 검증 데이터에서만 선택합니다. 제품의 최소 재현율 기준이
    # 필요하다면 이 단계에 안전 조건을 추가할 수 있습니다.
    selected = max(candidates, key=lambda item: item["f1"])
    print("validation threshold sweep:")
    print(pd.DataFrame(candidates).to_string(index=False))
    print("selected on validation:", selected)
    print("field validation at selected threshold:",
          metrics(field_y, field_scores, selected["threshold"]))

    test_y, test_scores, test_categories = predict_fold(model, df, audio_dir, 5)
    test_result = metrics(test_y, test_scores, selected["threshold"])
    print("held-out test:", test_result)
    test_pred = (test_scores >= selected["threshold"]).astype(int)
    errors = pd.DataFrame({"category": test_categories, "true": test_y,
                           "predicted": test_pred, "score": test_scores})
    errors = errors[errors["true"] != errors["predicted"]]
    print("test errors by category:")
    print(errors.groupby(["category", "true"]).size().to_string())


if __name__ == "__main__":
    main()
