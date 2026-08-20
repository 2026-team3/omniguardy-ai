"""블록 단위 행동 특징과 pose 프레임 특징을 block_id로 병합한다."""

from pathlib import Path

import pandas as pd


FEATURE_ROOT = Path("results/features")
POSE_ROOT = Path("results/pose")
SPLITS = ("train", "valid")
POSE_COLUMNS = ("hand_motion", "body_motion", "arm_extension", "upper_body_angle")


def pose_summary(pose: pd.DataFrame, start_frame: int, end_frame: int) -> dict:
    """지정한 블록 프레임 범위의 pose 특징을 요약한다."""
    selected = pose[(pose["frame"] >= start_frame) & (pose["frame"] <= end_frame)]
    if selected.empty:
        return {"has_pose": 0, "pose_frame_count": 0}
    result = {"has_pose": 1, "pose_frame_count": len(selected)}
    for column in POSE_COLUMNS:
        result[f"{column}_mean"] = selected[column].mean()
        result[f"{column}_max"] = selected[column].max()
        result[f"{column}_std"] = selected[column].std(ddof=0)
    return result


def main() -> None:
    for split in SPLITS:
        behavior = pd.read_csv(FEATURE_ROOT / f"{split}_behavior_features.csv")
        rows = []
        for video_path, blocks in behavior.groupby("video_path", sort=False):
            pose_path = POSE_ROOT / split / f"{Path(video_path).stem}.csv"
            try:
                pose = pd.read_csv(pose_path) if pose_path.is_file() else pd.DataFrame()
            except pd.errors.EmptyDataError:
                pose = pd.DataFrame()
            for _, block in blocks.iterrows():
                summary = pose_summary(pose, block.start_frame, block.end_frame) if {"frame", *POSE_COLUMNS}.issubset(pose.columns) else {"has_pose": 0, "pose_frame_count": 0}
                rows.append({**block.to_dict(), **summary})
        output = FEATURE_ROOT / f"{split}_merged_features.csv"
        pd.DataFrame(rows).fillna(0).to_csv(output, index=False, encoding="utf-8-sig")
        print(f"{split}: {len(rows)}개 블록 병합 -> {output}")


if __name__ == "__main__":
    main()

