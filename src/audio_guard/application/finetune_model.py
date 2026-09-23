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


SEED = 42


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
            if augment
            else [audio]
        )

        for variant in variants:
            transformed = pipeline.transform(
                variant,
                sample_rate,
            )

            features.extend(transformed)
            labels.extend(
                [int(label)] * len(transformed)
            )

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )


def _class_sample_weights(
    labels,
    reference_labels,
    abnormal_weight_multiplier,
):
    if abnormal_weight_multiplier <= 0:
        raise ValueError(
            "abnormal_weight_multiplier must be positive"
        )

    balanced = compute_class_weight(
        "balanced",
        classes=np.array([0, 1]),
        y=reference_labels,
    )

    weights = balanced[labels].astype(np.float32)

    weights[labels == 1] *= abnormal_weight_multiplier

    return weights


def _select_field_train_samples(
    field_train,
    max_field_train_samples,
    seed=SEED,
):
    """
    Learning curve 실험을 위해 field train 데이터 수를 제한합니다.

    같은 seed를 사용하므로:
    - 5개 실험에서 사용한 데이터는
    - 10개 실험에도 그대로 포함됩니다.

    max_field_train_samples:
        None -> 전체 사용
        0    -> field train 사용 안 함
        N    -> 고정된 순서에서 N개 사용
    """

    samples = list(field_train)

    if max_field_train_samples is None:
        return samples

    if max_field_train_samples < 0:
        raise ValueError(
            "max_field_train_samples must be >= 0"
        )

    if max_field_train_samples == 0:
        return []

    if max_field_train_samples > len(samples):
        raise ValueError(
            f"요청한 field train 샘플 수가 실제 데이터보다 많습니다. "
            f"requested={max_field_train_samples}, "
            f"available={len(samples)}"
        )

    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(samples))

    selected_indices = indices[:max_field_train_samples]

    return [
        samples[index]
        for index in selected_indices
    ]


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
    max_field_train_samples: int | None = None,
):
    """
    ESC-50 양 클래스와 field abnormal train으로
    v4 모델을 fine-tuning합니다.

    Learning curve 실험에서는
    max_field_train_samples 값만 변경하여
    field 데이터 양에 따른 성능 변화를 비교합니다.
    """

    # -----------------------------------------
    # 입력값 검증
    # -----------------------------------------

    if epochs < 1:
        raise ValueError(
            "epochs must be positive"
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive"
        )

    if field_weight <= 0:
        raise ValueError(
            "field_weight must be positive"
        )

    if Path(base_model).name != "audio_model_v4.keras":
        raise ValueError(
            "base_model must be models/audio_model_v4.keras"
        )

    if Path(base_model).resolve() == Path(model_out).resolve():
        raise ValueError(
            "model_out must not overwrite the v4 base model"
        )

    if Path(model_out).exists():
        raise FileExistsError(
            f"model_out already exists: {model_out}"
        )

    # -----------------------------------------
    # 재현성 고정
    # -----------------------------------------

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    augmentation_rng = np.random.default_rng(SEED)

    # -----------------------------------------
    # Pipeline
    # -----------------------------------------

    pipeline = create_pipeline("v4")

    # -----------------------------------------
    # ESC-50 로드
    # -----------------------------------------

    metadata = load_metadata(
        esc50_dir
    )

    audio_dir = (
        Path(esc50_dir)
        / "audio"
    )

    esc_train = examples_for_folds(
        metadata,
        audio_dir,
        {1, 2, 3},
        TARGET_CLASSES,
    )

    esc_validation = examples_for_folds(
        metadata,
        audio_dir,
        {4},
        TARGET_CLASSES,
    )

    # -----------------------------------------
    # Field 데이터 로드
    # -----------------------------------------

    field = split_field_three_way(
        field_data_dir,
        field_splits,
    )

    all_field_train = list(
        field["train"]
    )

    selected_field_train = _select_field_train_samples(
        all_field_train,
        max_field_train_samples,
        seed=SEED,
    )

    print()
    print("========================================")
    print("Fine-tuning dataset")
    print("========================================")
    print(
        f"ESC-50 train files: "
        f"{len(esc_train)}"
    )
    print(
        f"Field train available: "
        f"{len(all_field_train)}"
    )
    print(
        f"Field train selected: "
        f"{len(selected_field_train)}"
    )
    print(
        f"max_field_train_samples: "
        f"{max_field_train_samples}"
    )
    print("========================================")
    print()

    # 선택된 실제 파일도 출력
    if selected_field_train:
        print("[Selected field train files]")

        for example in selected_field_train:
            print(
                f"- {example.path}"
            )

        print()

    else:
        print(
            "[Selected field train files] 없음"
        )
        print()

    # -----------------------------------------
    # ESC-50 Feature 생성
    # -----------------------------------------

    esc_x, esc_y = _features_from_files(
        esc_train,
        pipeline,
    )

    # -----------------------------------------
    # Field Feature 생성
    # -----------------------------------------

    field_train_examples = [
        (
            example.path,
            example.label,
        )
        for example in selected_field_train
    ]

    # field가 1개 이상일 때만 feature 생성
    if field_train_examples:

        field_train_x, field_train_y = (
            _features_from_files(
                field_train_examples,
                pipeline,
                augment=True,
                noise_std=noise_std,
                gain_range=(
                    gain_min,
                    gain_max,
                ),
                max_time_shift_ms=max_time_shift_ms,
                rng=augmentation_rng,
            )
        )

        # ESC + Field 결합
        x_train = np.concatenate(
            [
                esc_x,
                field_train_x,
            ],
            axis=0,
        )

        y_train = np.concatenate(
            [
                esc_y,
                field_train_y,
            ],
            axis=0,
        )

    else:

        # field 0개 실험
        x_train = esc_x
        y_train = esc_y

        field_train_x = None
        field_train_y = None

    # -----------------------------------------
    # Validation
    # -----------------------------------------

    (
        validation_esc_x,
        validation_esc_y,
    ) = _features_from_files(
        esc_validation,
        pipeline,
    )

    # -----------------------------------------
    # Sample Weight
    # -----------------------------------------

    sample_weight = _class_sample_weights(
        y_train,
        esc_y,
        abnormal_weight_multiplier,
    )

    # field 데이터에 추가 weight 적용
    if field_train_examples:

        field_start_index = len(esc_x)

        sample_weight[
            field_start_index:
        ] *= field_weight

    # -----------------------------------------
    # Dataset 정보 출력
    # -----------------------------------------

    print(
        f"ESC feature samples: "
        f"{len(esc_x)}"
    )

    if field_train_examples:
        print(
            f"Field feature samples: "
            f"{len(field_train_x)}"
        )
    else:
        print(
            "Field feature samples: 0"
        )

    print(
        f"Total training samples: "
        f"{len(x_train)}"
    )

    print()

    # -----------------------------------------
    # v4 모델 로드
    # -----------------------------------------

    model = tf.keras.models.load_model(
        base_model,
        compile=False,
    )

    if tuple(
        model.input_shape[1:]
    ) != (128, 128, 1):

        raise ValueError(
            "base_model must use the v4 CNN "
            "input shape (128, 128, 1)"
        )

    # -----------------------------------------
    # Compile
    # -----------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc",
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )

    # -----------------------------------------
    # Fine-tuning
    # -----------------------------------------

    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        validation_data=(
            validation_esc_x,
            validation_esc_y,
        ),
        epochs=epochs,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=2,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc",
                mode="max",
                factor=0.5,
                patience=1,
                min_lr=1e-6,
            ),
        ],
    )

    # -----------------------------------------
    # 모델 저장
    # -----------------------------------------

    model_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        model_out
    )

    return model_out