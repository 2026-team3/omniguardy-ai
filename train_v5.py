"""3초 윈도우 수를 제한한 묶음으로 파일 단위 MIL 학습을 수행합니다."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from audio_guard.domain.pipeline.v5_bag import audio_to_mel_bag
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from field_dataset import split_field_three_way
from audio_labels import TARGET_CLASSES


def examples_from_esc(df, audio_dir, folds):
    return [(audio_dir / row.filename, int(row.category in TARGET_CLASSES))
            for row in df[df["fold"].isin(folds)].itertuples()]


def features_from_files(examples, max_windows):
    features, labels = [], []
    for path, label in examples:
        audio, sr = load_audio(path)
        features.append(audio_to_mel_bag(audio, sr, max_windows))
        labels.append(label)
    return np.asarray(features, dtype=np.float32)[..., None], np.asarray(labels)


def build_model(max_windows):
    layers = tf.keras.layers
    window_input = layers.Input(shape=(128, 128, 1))
    x = layers.Conv2D(32, 3, activation="relu")(window_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    window_score = layers.Dense(1, activation="sigmoid")(x)
    encoder = tf.keras.Model(window_input, window_score, name="window_encoder")
    bag_input = layers.Input(shape=(max_windows, 128, 128, 1))
    scores = layers.TimeDistributed(encoder)(bag_input)
    bag_score = layers.GlobalMaxPooling1D()(scores)
    return tf.keras.Model(bag_input, bag_score, name="audio_mil_v5")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50-dir", type=Path, required=True)
    parser.add_argument("--field-data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--field-splits", type=Path)
    parser.add_argument("--model-out", type=Path,
                        default=Path("models/audio_model_v5.keras"))
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--field-weight", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.field_weight <= 0:
        parser.error("--field-weight must be positive")
    if args.max_windows < 1:
        parser.error("--max-windows must be positive")
    np.random.seed(42)
    tf.random.set_seed(42)

    df = pd.read_csv(args.esc50_dir / "meta" / "esc50.csv")
    esc_train = examples_from_esc(df, args.esc50_dir / "audio", {1, 2, 3})
    esc_val = examples_from_esc(df, args.esc50_dir / "audio", {4})
    field = split_field_three_way(args.field_data_dir, args.field_splits)
    field_train, field_val = field["train"], field["validation"]
    print("file counts:", {"esc_train": len(esc_train), "esc_val": len(esc_val),
                           "field_train": len(field_train), "field_val": len(field_val),
                           "field_test_reserved": len(field["test"])})

    x_train, y_train = features_from_files(esc_train + field_train, args.max_windows)
    x_val, y_val = features_from_files(esc_val + field_val, args.max_windows)
    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=y_train)
    sample_weights = class_weights[y_train]
    sample_weights[len(esc_train):] *= args.field_weight
    val_weights = np.ones(len(y_val), dtype=np.float32)
    val_weights[len(esc_val):] *= args.field_weight

    model = build_model(args.max_windows)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    model.fit(x_train, y_train, sample_weight=sample_weights,
              validation_data=(x_val, y_val, val_weights),
              epochs=args.epochs, batch_size=8,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=5, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(
                      monitor="val_loss", factor=0.5, patience=3)])
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    print("Saved:", args.model_out)


if __name__ == "__main__":
    main()
