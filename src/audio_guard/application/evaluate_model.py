#확정된 모델과 threshold로 잠긴 test split을 최종 평가

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from audio_guard.application.train_model import features_from_examples
from audio_guard.domain.pipeline import create_pipeline
from audio_guard.infrastructure.dataset.manifest_dataset import (
    examples_for_split,
    load_manifest,
)
from audio_guard.labels import CLASS_NAMES, LABELS


def classification_metrics(labels, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    by_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }

    background = LABELS["background"]

    return {
        "sample_count": int(len(labels)),
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            np.mean(f1)
        ),
        "classes": by_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(CLASS_NAMES),
        "background_to_knock_false_positives": int(
            np.sum(
                (labels == background)
                & (predictions == LABELS["knock"])
            )
        ),
        "background_to_handle_false_positives": int(
            np.sum(
                (labels == background)
                & (predictions == LABELS["handle"])
            )
        ),
    }


def print_confusion_matrix(split, metrics):
    matrix = np.asarray(
        metrics["confusion_matrix"]
    )

    print()
    print("=" * 64)
    print(f"{split.upper()} CONFUSION MATRIX")
    print("Window-level / Rows = True, Columns = Predicted")
    print("=" * 64)

    print(
        f"{'':>12}"
        f"{'background':>14}"
        f"{'knock':>12}"
        f"{'handle':>12}"
    )

    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name:>12}"
            f"{matrix[index][0]:>14}"
            f"{matrix[index][1]:>12}"
            f"{matrix[index][2]:>12}"
        )

    print()
    print(
        f"Samples  : {metrics['sample_count']}"
    )
    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )
    print(
        f"Macro F1 : {metrics['macro_f1']:.4f}"
    )

    print()

    for class_name in CLASS_NAMES:
        class_metrics = metrics["classes"][class_name]

        print(
            f"{class_name:>10} | "
            f"P={class_metrics['precision']:.4f} "
            f"R={class_metrics['recall']:.4f} "
            f"F1={class_metrics['f1']:.4f} "
            f"Support={class_metrics['support']}"
        )

    print()
    print(
        "Background -> Knock FP  : "
        f"{metrics['background_to_knock_false_positives']}"
    )
    print(
        "Background -> Handle FP : "
        f"{metrics['background_to_handle_false_positives']}"
    )


def save_confusion_matrix_image(
    split,
    labels,
    predictions,
    results_dir,
):
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        f"{split.capitalize()} Confusion Matrix"
    )
    ax.set_xlabel(
        "Predicted Label"
    )
    ax.set_ylabel(
        "True Label"
    )

    fig.tight_layout()

    output_path = (
        results_dir
        / f"{split}_confusion_matrix.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def evaluate_split(
    split,
    examples,
    model,
    pipeline,
    results_dir,
):
    split_examples = examples_for_split(
        examples,
        split,
    )

    # 기존 학습/평가와 동일한 전처리
    # Validation/Test에는 augmentation을 적용하지 않음
    features, labels = features_from_examples(
        split_examples,
        pipeline,
        augment=False,
    )

    probabilities = np.asarray(
        model.predict(
            features,
            verbose=0,
        )
    )

    # 기존 Confusion Matrix 평가와 동일:
    # threshold가 아닌 raw softmax argmax 사용
    predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int32)

    metrics = classification_metrics(
        labels,
        predictions,
    )

    # Confusion Matrix PNG 저장
    confusion_matrix_path = save_confusion_matrix_image(
        split,
        labels,
        predictions,
        results_dir,
    )

    result = {
        "split": split,
        "evaluation_level": "window",
        "prediction_method": "argmax",
        "window_seconds": pipeline.window_seconds,
        "hop_seconds": pipeline.hop_seconds,
        **metrics,
        "confusion_matrix_image": str(
            confusion_matrix_path
        ),
    }

    output_path = (
        results_dir
        / f"{split}_evaluation.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            result,
            output,
            ensure_ascii=False,
            indent=2,
        )

    result["result_path"] = str(
        output_path
    )

    print_confusion_matrix(
        split,
        metrics,
    )

    print(
        f"Confusion Matrix Image : "
        f"{confusion_matrix_path}"
    )

    return result


def evaluate_model(
    manifest_path: Path,
    events_path: Path,
    model_path: Path,
    runtime_config: Path,
    results_dir=Path("results"),
):
    with Path(runtime_config).open(
        encoding="utf-8"
    ) as source:
        config = json.load(source)

    examples = load_manifest(
        manifest_path,
        events_path,
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    pipeline = create_pipeline(
        "door_event",
        config["window_seconds"],
        config["hop_seconds"],
    )

    results_dir = Path(
        results_dir
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 64)
    print("DOOR EVENT MODEL EVALUATION")
    print("=" * 64)
    print(
        f"Model  : {model_path}"
    )
    print(
        f"Window : "
        f"{config['window_seconds']} sec"
    )
    print(
        f"Hop    : "
        f"{config['hop_seconds']} sec"
    )
    print(
        "Method : window-level argmax"
    )

    validation_result = evaluate_split(
        split="validation",
        examples=examples,
        model=model,
        pipeline=pipeline,
        results_dir=results_dir,
    )

    test_result = evaluate_split(
        split="test",
        examples=examples,
        model=model,
        pipeline=pipeline,
        results_dir=results_dir,
    )

    print()
    print("=" * 64)
    print("EVALUATION COMPLETE")
    print("=" * 64)

    print(
        "Validation result : "
        f"{validation_result['result_path']}"
    )

    print(
        "Test result       : "
        f"{test_result['result_path']}"
    )

    # 기존 evaluate_cli와 호환되도록
    # Test 결과 반환
    return test_result