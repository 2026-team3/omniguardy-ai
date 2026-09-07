"""학습된 XGBoost 모델로 시간 창 단위 행동을 분류한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from camera.features import summarize_pose, summarize_tracking


@dataclass
class WindowAnalysis:
    """한 시간 창의 행동 분류 결과를 표현한다."""

    label: str
    confidence: float
    person_count: int
    detection_confidence: float
    feature_values: dict[str, float]
    status: str


def load_classifier(path: Path) -> dict[str, Any]:
    """학습된 XGBoost 모델과 feature 순서를 불러온다."""
    if not path.is_file():
        raise FileNotFoundError(f"행동 분류 모델이 없습니다: {path}")
    artifact = joblib.load(path)
    required = {"model", "label_encoder", "feature_columns"}
    if not required.issubset(artifact):
        raise ValueError("행동 분류 모델 파일 형식이 올바르지 않습니다.")
    return artifact


def classify_window(
    tracking_records: list[dict[str, float]],
    pose_records: list[dict[str, float]],
    classifier: dict[str, Any],
) -> WindowAnalysis:
    """집계 특징을 학습 모델에 넣고 예측 라벨과 confidence를 반환한다."""
    tracking_features = summarize_tracking(tracking_records)
    pose_features = summarize_pose(pose_records)
    feature_values = {**tracking_features, **pose_features}
    person_count = len({int(record["track_id"]) for record in tracking_records})
    detection_confidence = (
        float(np.mean([record["confidence"] for record in tracking_records]))
        if tracking_records
        else 0.0
    )
    if not tracking_records:
        return WindowAnalysis("N1", 1.0, 0, 0.0, feature_values, "no_person")
    if not tracking_features["has_tracking"]:
        return WindowAnalysis(
            "UNKNOWN",
            0.0,
            person_count,
            detection_confidence,
            feature_values,
            "insufficient_tracking",
        )

    feature_columns = classifier["feature_columns"]
    model_input = pd.DataFrame(
        [{column: feature_values.get(column, 0.0) for column in feature_columns}],
        columns=feature_columns,
    )
    probabilities = classifier["model"].predict_proba(model_input)[0]
    class_index = int(np.argmax(probabilities))
    label = str(classifier["label_encoder"].inverse_transform([class_index])[0])
    return WindowAnalysis(
        label,
        float(probabilities[class_index]),
        person_count,
        detection_confidence,
        {column: float(value) for column, value in feature_values.items()},
        "classified",
    )


def risk_level(label: str, confidence: float) -> str:
    """Agent 정책 전 단계에서 사용할 단순 위험도 수준을 정한다."""
    if label == "UNKNOWN":
        return "UNKNOWN"
    if label == "N1":
        return "NORMAL"
    return "HIGH" if confidence >= 0.75 else "WARNING"
