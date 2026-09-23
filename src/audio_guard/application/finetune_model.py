"""v4 모델에 ESC-50과 현장 데이터를 추가 학습하는 유스케이스입니다."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from audio_guard.application.audio_augmentation import augment_audio
from audio_guard.domain.pipeline import create_pipeline
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from audio_guard.infrastructure.dataset.esc50_dataset import (
    examples_for_folds,
    load_metadata,
)
from audio_guard.infrastructure.dataset.field_dataset import split_field_three_way
from audio_guard.labels import TARGET_CLASSES


def _features_from_files(
    examples,
    pipeline,
    augment=False,
    noise_std=0.001,
    gain_range=(0.9, 1.1),
    max_time_shift_ms=50,
    rng=None,
):
    features, labels = [], []
    for path, label in examples:
        audio, sample_rate = load_audio(path)
        variants = (
            augment_audio(
                audio,
                sample_rate,
                noise_std=noise_std,
                gain_range=gain_range,
                max_time_shift_ms=max_time_shift_ms,
                rng=rng,
            )
            if augment else [audio]
        )
        for variant in variants:
            transformed = pipeline.transform(variant, sample_rate)
            features.extend(transformed)
            labels.extend([int(label)] * len(transformed))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )


def _class_sample_weights(labels, reference_labels, abnormal_weight_multiplier):
    if abnormal_weight_multiplier <= 0:
        raise ValueError("abnormal_weight_multiplier must be positive")
    balanced = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=reference_labels)
    weights = balanced[labels].astype(np.float32)
    weights[labels == 1] *= abnormal_weight_multiplier
    return weights


def finetune_model(
    esc50_dir: Path,
    field_data_dir: Path,
    field_splits: Path,
    base_model: Path,
    model_out: Path,
    epochs=5,
    learning_rate=5e-6,
    abnormal_weight_multiplier=1.0,
    field_weight=1.0,
    noise_std=0.001,
    gain_min=0.9,
    gain_max=1.1,
    max_time_shift_ms=50,
):
    """ESC-50 양 클래스와 field abnormal train으로 v4를 fine-tuning합니다."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if field_weight <= 0:
        raise ValueError("field_weight must be positive")
    if Path(base_model).name != "audio_model_v4.keras":
        raise ValueError("base_model must be models/audio_model_v4.keras")
    if Path(base_model).resolve() == Path(model_out).resolve():
        raise ValueError("model_out must not overwrite the v4 base model")
    if Path(model_out).exists():
        raise FileExistsError(f"model_out already exists: {model_out}")

    np.random.seed(42)
    tf.random.set_seed(42)
    rng = np.random.default_rng(42)

    pipeline = create_pipeline("v4")
    metadata = load_metadata(esc50_dir)
    audio_dir = Path(esc50_dir) / "audio"
    field = split_field_three_way(field_data_dir, field_splits)
    esc_train = examples_for_folds(metadata, audio_dir, {1, 2, 3}, TARGET_CLASSES)
    esc_validation = examples_for_folds(metadata, audio_dir, {4}, TARGET_CLASSES)

    esc_x, esc_y = _features_from_files(esc_train, pipeline)
    field_train_examples = [
        (example.path, example.label) for example in field["train"]
    ]
    field_train_x, field_train_y = _features_from_files(
        field_train_examples,
        pipeline,
        augment=True,
        noise_std=noise_std,
        gain_range=(gain_min, gain_max),
        max_time_shift_ms=max_time_shift_ms,
        rng=rng,
    )
    validation_esc_x, validation_esc_y = _features_from_files(esc_validation, pipeline)

    x_train = np.concatenate([esc_x, field_train_x])
    y_train = np.concatenate([esc_y, field_train_y])
    sample_weight = _class_sample_weights(
        y_train, esc_y, abnormal_weight_multiplier)
    sample_weight[len(esc_x):] *= field_weight

    model = tf.keras.models.load_model(base_model, compile=False)
    if tuple(model.input_shape[1:]) != (128, 128, 1):
        raise ValueError("base_model must use the v4 CNN input shape (128, 128, 1)")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        validation_data=(validation_esc_x, validation_esc_y),
        epochs=epochs,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=2,
                restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc", mode="max", factor=0.5, patience=1),
        ],
    )
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    return model_out
