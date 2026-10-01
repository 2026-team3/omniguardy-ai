"""기존 validation 추출 결과로 10초(300프레임) 단위 성능을 별도 평가한다.

학습 코드나 기존 feature 결과는 수정하지 않는다. AIHub validation annotation 블록 안에서
완전히 포함되는 300프레임 창만 만들고, 각 창의 tracking/pose 특징을 다시 집계한다.
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.extract_behavior_features import summarize_block
from tracking.merge_features import POSE_COLUMNS, pose_summary


WINDOW_FRAMES = 300  # 30 fps 기준 10초
ANNOTATIONS = PROJECT_ROOT / "splits/annotations/valid_annotations.csv"
TRACKING_ROOT = PROJECT_ROOT / "results/tracking/valid"
POSE_ROOT = PROJECT_ROOT / "results/pose/valid"
MODEL_PATH = PROJECT_ROOT / "results/models/xgboost_validation/xgboost_validation.pkl"
OUTPUT_ROOT = PROJECT_ROOT / "results/evaluation/fixed_10s_validation"


def make_windows(annotations: pd.DataFrame) -> pd.DataFrame:
    """라벨 블록을 겹치지 않는 완전한 10초 창으로 나눈다."""
    rows: list[dict] = []
    for _, block in annotations.iterrows():
        start = int(block.start_frame)
        end = int(block.end_frame)
        for window_start in range(start, end - WINDOW_FRAMES + 2, WINDOW_FRAMES):
            row = block.to_dict()
            row["start_frame"] = window_start
            row["end_frame"] = window_start + WINDOW_FRAMES - 1
            rows.append(row)
    return pd.DataFrame(rows)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def extract_features(windows: pd.DataFrame) -> pd.DataFrame:
    """기존 frame-level tracking/pose CSV에서 각 window의 모델 특징을 만든다."""
    rows: list[dict] = []
    for video_path, video_windows in windows.groupby("video_path", sort=False):
        stem = Path(video_path).stem
        tracking = read_csv_or_empty(TRACKING_ROOT / f"{stem}.csv")
        pose = read_csv_or_empty(POSE_ROOT / f"{stem}.csv")
        tracking_columns = {"frame", "track_id", "x1", "y1", "x2", "y2"}
        pose_columns = {"frame", *POSE_COLUMNS}

        for _, window in video_windows.iterrows():
            start, end = int(window.start_frame), int(window.end_frame)
            if tracking_columns.issubset(tracking.columns):
                behavior = summarize_block(
                    tracking[(tracking["frame"] >= start) & (tracking["frame"] <= end)]
                )
            else:
                behavior = {"has_tracking": 0, "tracked_person_count": 0}
            pose_features = (
                pose_summary(pose, start, end)
                if pose_columns.issubset(pose.columns)
                else {"has_pose": 0, "pose_frame_count": 0}
            )
            rows.append({**window.to_dict(), **behavior, **pose_features})
    return pd.DataFrame(rows).fillna(0)


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    encoder = bundle["label_encoder"]
    feature_columns = bundle["feature_columns"]

    windows = make_windows(pd.read_csv(ANNOTATIONS))
    if windows.empty:
        raise ValueError("완전한 10초 validation window가 없습니다.")
    features = extract_features(windows)
    for column in feature_columns:
        if column not in features:
            features[column] = 0.0
    features = features.fillna(0)

    prediction_id = model.predict(features[feature_columns])
    features["prediction"] = encoder.inverse_transform(prediction_id.astype(int))
    features["confidence"] = model.predict_proba(features[feature_columns]).max(axis=1)

    labels = list(encoder.classes_)
    accuracy = accuracy_score(features["label"], features["prediction"])
    report = classification_report(
        features["label"], features["prediction"], labels=labels,
        target_names=labels, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(features["label"], features["prediction"], labels=labels)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_ROOT / "window_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report).transpose().to_csv(OUTPUT_ROOT / "classification_report.csv", encoding="utf-8-sig")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        OUTPUT_ROOT / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    (OUTPUT_ROOT / "summary.txt").write_text(
        f"window_frames={WINDOW_FRAMES}\nwindow_seconds=10.0\n"
        f"window_count={len(features)}\naccuracy={accuracy:.4f}\n",
        encoding="utf-8",
    )
    print(f"10초 window 수: {len(features)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"결과 저장: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
