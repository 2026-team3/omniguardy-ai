"""Train on ESC-50 and labeled field door-lock recordings without file leakage."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from feature import audio_to_mel_windows, load_audio
from field_dataset import split_field_files

TARGET_CLASSES = {
    "door_wood_knock", "door_wood_creaks", "glass_breaking",
    "siren", "chainsaw", "footsteps",
}


def load_fold(df, audio_dir, folds):
    features, labels = [], []
    for row in df[df["fold"].isin(folds)].itertuples():
        audio, sr = load_audio(audio_dir / row.filename)
        windows = audio_to_mel_windows(audio, sr)
        features.extend(windows)
        labels.extend([int(row.category in TARGET_CLASSES)] * len(windows))
    return np.asarray(features, dtype=np.float32)[..., None], np.asarray(labels)


def load_field_files(paths, labels):
    features, window_labels = [], []
    for path, label in zip(paths, labels):
        audio, sr = load_audio(path)
        windows = audio_to_mel_windows(audio, sr)
        features.extend(windows)
        window_labels.extend([int(label)] * len(windows))
    return (np.asarray(features, dtype=np.float32)[..., None],
            np.asarray(window_labels, dtype=np.int32))


def build_model():
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(128, activation="relu"),
        layers.Dropout(0.5), layers.Dense(1, activation="sigmoid"),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50-dir", type=Path, required=True)
    parser.add_argument("--field-data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model-out", type=Path,
                        default=Path("models/audio_model_v4.keras"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)
    df = pd.read_csv(args.esc50_dir / "meta" / "esc50.csv")
    audio_dir = args.esc50_dir / "audio"
    x_train, y_train = load_fold(df, audio_dir, {1, 2, 3})
    x_val, y_val = load_fold(df, audio_dir, {4})
    field_train, field_val, field_train_y, field_val_y = split_field_files(
        args.field_data_dir)
    print("field original train:", len(field_train), np.bincount(field_train_y))
    print("field original validation:", len(field_val), np.bincount(field_val_y))
    field_x_train, field_y_train = load_field_files(field_train, field_train_y)
    field_x_val, field_y_val = load_field_files(field_val, field_val_y)
    x_train = np.concatenate([x_train, field_x_train])
    y_train = np.concatenate([y_train, field_y_train])
    x_val = np.concatenate([x_val, field_x_val])
    y_val = np.concatenate([y_val, field_y_val])
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = dict(enumerate(weights))
    print("train:", x_train.shape, np.bincount(y_train))
    print("validation:", x_val.shape, np.bincount(y_val))

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=args.epochs, batch_size=16, class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=5,
                restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc", mode="max", factor=0.5, patience=3),
        ],
    )
    model.save(args.model_out)
    print("Saved:", args.model_out)


if __name__ == "__main__":
    main()
