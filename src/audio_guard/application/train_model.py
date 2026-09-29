"""Group-safe manifest로 3-class Door Event CNN을 학습합니다."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from audio_guard.application.audio_augmentation import augment_audio
from audio_guard.domain.pipeline import create_pipeline
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from audio_guard.infrastructure.dataset.manifest_dataset import (
    examples_for_split,
    load_manifest,
)
from audio_guard.labels import LABELS

SEED = 42


 #녹음 파일의 여러 이벤트 구간을 각 오디오 window의 라벨로 변환
def _window_labels(recording, offsets, window_seconds):
   

    labels = np.full(
        len(offsets),
        LABELS["background"],
        dtype=np.int32,
    )

    for event in recording.events:
        overlaps = (
            (offsets < event.end_seconds)
            & (offsets + window_seconds > event.start_seconds)
        )

        labels[overlaps] = event.label

    return labels


def features_from_examples(examples, pipeline, augment=False, rng=None):
    features, labels = [], []
    for example in examples:
        audio, sample_rate = load_audio(example.path)
        variants = augment_audio(audio, sample_rate, rng=rng) if augment else [audio]
        for variant in variants:
            transformed, offsets = pipeline.transform_with_offsets(variant, sample_rate)
            features.extend(transformed)
            labels.extend(_window_labels(example, offsets, pipeline.window_seconds))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def build_cnn_model():
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(LABELS), activation="softmax"),
    ])


def train_model(
    manifest_path: Path,
    events_path: Path,
    model_out: Path,
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    window_seconds=1.0,
    hop_seconds=0.2,
):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid training hyperparameters")
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    rng = np.random.default_rng(SEED)

    examples = load_manifest(
    manifest_path,
    events_path,
)
    pipeline = create_pipeline("door_event", window_seconds, hop_seconds)
    train_examples = examples_for_split(examples, "train")
    validation_examples = examples_for_split(examples, "validation")
    x_train, y_train = features_from_examples(
        train_examples, pipeline, augment=True, rng=rng
    )
    x_validation, y_validation = features_from_examples(
        validation_examples, pipeline
    )
    present_classes = np.unique(y_train)
    if set(present_classes) != set(LABELS.values()):
        raise ValueError("Training split must contain background, knock and handle")
    weights = compute_class_weight("balanced", classes=present_classes, y=y_train)
    class_weight = dict(zip(present_classes.tolist(), weights.tolist()))

    model = build_cnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=7, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ],
    )
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    return model_out
