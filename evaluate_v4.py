"""Select a threshold on fold 4; report untouched fold 5 results separately."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from feature import audio_to_mel_windows, load_audio
from train_v4 import TARGET_CLASSES


def predict_fold(model, df, audio_dir, fold):
    labels, scores, categories = [], [], []
    for row in df[df["fold"] == fold].itertuples():
        audio, sr = load_audio(audio_dir / row.filename)
        windows = audio_to_mel_windows(audio, sr)[..., None]
        window_scores = model.predict(windows, verbose=0).reshape(-1)
        # A clip is abnormal if one of its windows contains the event.
        scores.append(float(window_scores.max()))
        labels.append(int(row.category in TARGET_CLASSES))
        categories.append(row.category)
    return np.asarray(labels), np.asarray(scores), categories


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
    parser.add_argument("--model", type=Path,
                        default=Path("models/audio_model_v4.keras"))
    args = parser.parse_args()
    df = pd.read_csv(args.esc50_dir / "meta" / "esc50.csv")
    model = tf.keras.models.load_model(args.model, compile=False)
    audio_dir = args.esc50_dir / "audio"

    val_y, val_scores, _ = predict_fold(model, df, audio_dir, 4)
    candidates = [metrics(val_y, val_scores, float(t))
                  for t in np.arange(0.05, 0.951, 0.05)]
    # Use validation only for threshold selection; safety constraints can be
    # introduced here if a minimum recall is required by the product.
    selected = max(candidates, key=lambda item: item["f1"])
    print("validation threshold sweep:")
    print(pd.DataFrame(candidates).to_string(index=False))
    print("selected on validation:", selected)

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
