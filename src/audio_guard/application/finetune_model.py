"""현장 녹음으로 기존 모델을 추가 학습합니다."""

from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split

from audio_guard.domain.pipeline import create_pipeline


def augment_audio(audio, sample_rate):
    """기존 파인튜닝에서 사용한 다섯 가지 파형을 생성합니다."""
    noise = np.random.normal(0, 0.003, len(audio))
    shift = np.random.randint(-sample_rate // 2, sample_rate // 2)
    return [audio, audio + noise, audio * 0.8, audio * 1.2, np.roll(audio, shift)]


def _load_audio(path):
    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sample_rate


def _field_files(dataset_dir):
    examples = []
    for folder, label in (("normal", 0), ("abnormal", 1)):
        examples.extend((path, label)
                        for path in sorted((dataset_dir / folder).glob("*.wav")))
    return examples


def _build_features(examples, pipeline, augment):
    features, labels = [], []
    for path, label in examples:
        audio, sample_rate = _load_audio(path)
        variants = augment_audio(audio, sample_rate) if augment else [audio]
        for variant in variants:
            transformed = pipeline.transform(variant, sample_rate)
            features.extend(transformed)
            labels.extend([label] * len(transformed))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def finetune_model(
    pipeline_name: str,
    dataset_dir: Path,
    base_model: Path,
    model_out: Path,
    epochs=30,
    max_windows=8,
):
    """파일 단위 분할 후 학습 데이터에만 증강을 적용해 추가 학습합니다."""
    np.random.seed(42)
    tf.random.set_seed(42)
    examples = _field_files(dataset_dir)
    paths = [path for path, _ in examples]
    labels = [label for _, label in examples]
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels)
    pipeline = create_pipeline(pipeline_name, max_windows)
    x_train, y_train = _build_features(
        list(zip(train_paths, train_labels)), pipeline, augment=True)
    x_val, y_val = _build_features(
        list(zip(val_paths, val_labels)), pipeline, augment=False)

    model = tf.keras.models.load_model(base_model, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=8,
        class_weight={0: 1.0, 1: 1.0},
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3),
        ],
    )
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    return model_out
