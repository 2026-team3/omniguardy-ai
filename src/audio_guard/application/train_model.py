"""v3/v4 모델 학습을 하나의 파이프라인 진입점으로 제공합니다."""

from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from audio_guard.domain.pipeline import create_pipeline
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from audio_guard.infrastructure.dataset.esc50_dataset import (
    examples_for_folds,
    load_metadata,
)
from audio_guard.labels import TARGET_CLASSES


def features_from_files(examples, pipeline):
    """파일 목록을 선택한 파이프라인의 모델 입력으로 변환합니다."""
    features, labels = [], []
    for path, label in examples:
        audio, sample_rate = load_audio(path)
        transformed = pipeline.transform(audio, sample_rate)
        features.extend(transformed)
        labels.extend([int(label)] * len(transformed))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def v3_features_from_files(examples, pipeline):
    """기존 v3과 동일하게 원본 샘플레이트로 특징을 생성합니다."""
    features, labels = [], []
    for path, label in examples:
        audio, sample_rate = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        transformed = pipeline.transform(audio, sample_rate)
        features.extend(transformed)
        labels.extend([int(label)] * len(transformed))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def build_cnn_model():
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(128, activation="relu"),
        layers.Dropout(0.5), layers.Dense(1, activation="sigmoid"),
    ])


def train_model(
    pipeline_name: str,
    esc50_dir: Path,
    model_out: Path,
    epochs=None,
):
    """선택한 전처리 전략으로 모델을 학습하고 저장합니다."""
    np.random.seed(42)
    tf.random.set_seed(42)
    pipeline = create_pipeline(pipeline_name)
    metadata = load_metadata(esc50_dir)
    audio_dir = esc50_dir / "audio"

    if pipeline_name == "v3":
        examples = examples_for_folds(
            metadata, audio_dir, {1, 2, 3, 4, 5}, TARGET_CLASSES)
        features, labels = v3_features_from_files(examples, pipeline)
        x_train, x_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels)
        sample_weight = None
        validation_data = (x_val, y_val)
        batch_size = 16
        epochs = epochs or 20
    else:
        esc_train = examples_for_folds(metadata, audio_dir, {1, 2, 3}, TARGET_CLASSES)
        esc_val = examples_for_folds(metadata, audio_dir, {4}, TARGET_CLASSES)
        x_train, y_train = features_from_files(esc_train, pipeline)
        x_val, y_val = features_from_files(esc_val, pipeline)
        sample_weight = None
        validation_data = (x_val, y_val)
        batch_size = 16
        epochs = epochs or 30
    model = build_cnn_model()
    learning_rate = 1e-3 if pipeline_name == "v3" else 1e-4
    metrics = ["accuracy"] if pipeline_name == "v3" else [
        tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metrics,
    )
    class_weight = None
    if sample_weight is None:
        weights = compute_class_weight(
            "balanced", classes=np.array([0, 1]), y=y_train)
        class_weight = dict(enumerate(weights))
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[] if pipeline_name == "v3" else [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=5,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc",
                mode="max",
                factor=0.5,
                patience=3,
            ),
        ],
    )
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    return model_out
