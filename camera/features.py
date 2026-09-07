"""YOLO 추적 결과와 Pose 결과를 학습 모델용 특징으로 변환한다."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision

from camera.config import POSE_COLUMNS

def create_pose_landmarker(model_path: Path):
    """실시간 VIDEO 모드로 MediaPipe Pose Landmarker를 생성한다."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose 모델이 없습니다: {model_path}")

    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def summarize_tracking(records: list[dict[str, float]]) -> dict[str, float]:
    """학습 파이프라인과 같은 방식으로 시간 창의 추적 특징을 집계한다."""
    if not records:
        return {"has_tracking": 0.0, "tracked_person_count": 0.0}

    frame = pd.DataFrame(records)
    features: list[dict[str, float]] = []
    for _, group in frame.groupby("track_id"):
        group = group.sort_values("frame")
        if len(group) < 10:
            continue
        center_x = (group["x1"].to_numpy() + group["x2"].to_numpy()) / 2
        center_y = (group["y1"].to_numpy() + group["y2"].to_numpy()) / 2
        frame_numbers = group["frame"].to_numpy()
        distances = np.hypot(np.diff(center_x), np.diff(center_y))
        gaps = np.diff(frame_numbers)
        speeds = distances[gaps > 0] / gaps[gaps > 0]
        features.append(
            {
                "tracking_frame_count": float(len(group)),
                "move_distance": float(distances.sum()),
                "avg_speed": float(speeds.mean()) if len(speeds) else 0.0,
                "max_speed": float(speeds.max()) if len(speeds) else 0.0,
                "min_speed": float(speeds.min()) if len(speeds) else 0.0,
                "std_speed": float(speeds.std()) if len(speeds) else 0.0,
                "movement_range": float((center_x.max() - center_x.min()) + (center_y.max() - center_y.min())),
                "trajectory_variance": float(np.var(center_x) + np.var(center_y)),
            }
        )

    if not features:
        return {"has_tracking": 0.0, "tracked_person_count": 0.0}

    summary = {"has_tracking": 1.0, "tracked_person_count": float(len(features))}
    feature_frame = pd.DataFrame(features)
    for column in feature_frame.columns:
        summary[f"{column}_mean"] = float(feature_frame[column].mean())
        summary[f"{column}_max"] = float(feature_frame[column].max())
        summary[f"{column}_std"] = float(feature_frame[column].std(ddof=0))
    return summary


def summarize_pose(records: list[dict[str, float]]) -> dict[str, float]:
    """학습 파이프라인과 같은 방식으로 시간 창의 자세 특징을 집계한다."""
    if not records:
        return {"has_pose": 0.0, "pose_frame_count": 0.0}

    frame = pd.DataFrame(records)
    summary = {"has_pose": 1.0, "pose_frame_count": float(len(frame))}
    for column in POSE_COLUMNS:
        summary[f"{column}_mean"] = float(frame[column].mean())
        summary[f"{column}_max"] = float(frame[column].max())
        summary[f"{column}_std"] = float(frame[column].std(ddof=0))
    return summary


def extract_pose_features(
    frame: np.ndarray,
    landmarker: Any,
    timestamp_ms: int,
    frame_index: int,
    previous: dict[int, tuple[int, float, float, float, float]],
) -> dict[str, float] | None:
    """현재 프레임의 자세와 이전 좌표로 학습용 Pose 특징을 만든다."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), timestamp_ms
    )
    values: list[tuple[float, float, float, float]] = []
    for pose_index, landmarks in enumerate(result.pose_landmarks):
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        right_wrist = landmarks[16]
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y) / 2
        arm_extension = math.dist(
            (right_wrist.x, right_wrist.y), (right_shoulder.x, right_shoulder.y)
        )
        upper_body_angle = math.degrees(
            math.atan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x,
            )
        )
        hand_motion = 0.0
        body_motion = 0.0
        if pose_index in previous:
            previous_frame, previous_wrist_x, previous_wrist_y, previous_center_x, previous_center_y = previous[pose_index]
            frame_gap = frame_index - previous_frame
            if frame_gap > 0:
                hand_motion = math.dist(
                    (right_wrist.x, right_wrist.y), (previous_wrist_x, previous_wrist_y)
                ) / frame_gap
                body_motion = math.dist(
                    (center_x, center_y), (previous_center_x, previous_center_y)
                ) / frame_gap
        previous[pose_index] = (
            frame_index,
            right_wrist.x,
            right_wrist.y,
            center_x,
            center_y,
        )
        values.append((hand_motion, body_motion, arm_extension, upper_body_angle))

    if not values:
        return None

    array = np.asarray(values, dtype=float)
    return {
        "hand_motion": float(array[:, 0].mean()),
        "body_motion": float(array[:, 1].mean()),
        "arm_extension": float(array[:, 2].mean()),
        "upper_body_angle": float(array[:, 3].mean()),
    }
