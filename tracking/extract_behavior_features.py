"""tracking 결과를 annotation의 모든 블록 범위로 잘라 행동 특징을 만든다."""

from hashlib import sha1
from pathlib import Path

import numpy as np
import pandas as pd


ANNOTATION_ROOT = Path("splits/annotations")
TRACKING_ROOT = Path("results/tracking")
OUTPUT_ROOT = Path("results/features")
SPLITS = ("train", "valid")


def block_id(row: pd.Series) -> str:
    """영상 경로와 프레임 범위를 사용해 안정적인 블록 식별자를 만든다."""
    value = "|".join(map(str, [row.video_path, row.start_frame, row.end_frame, row.label, row.source]))
    return sha1(value.encode("utf-8")).hexdigest()


def track_features(group: pd.DataFrame) -> dict | None:
    """한 사람 track의 이동 특징을 계산한다."""
    group = group.sort_values("frame")
    if len(group) < 10:
        return None
    x = (group["x1"].to_numpy() + group["x2"].to_numpy()) / 2
    y = (group["y1"].to_numpy() + group["y2"].to_numpy()) / 2
    frames = group["frame"].to_numpy()
    distances = np.hypot(np.diff(x), np.diff(y))
    gaps = np.diff(frames)
    speeds = distances[gaps > 0] / gaps[gaps > 0]
    return {
        "tracking_frame_count": len(group),
        "move_distance": distances.sum(),
        "avg_speed": speeds.mean() if len(speeds) else 0.0,
        "max_speed": speeds.max() if len(speeds) else 0.0,
        "min_speed": speeds.min() if len(speeds) else 0.0,
        "std_speed": speeds.std() if len(speeds) else 0.0,
        "movement_range": (x.max() - x.min()) + (y.max() - y.min()),
        "trajectory_variance": np.var(x) + np.var(y),
    }


def summarize_block(block: pd.DataFrame) -> dict:
    """블록 안의 여러 사람 track을 하나의 특징 행으로 요약한다."""
    features = [feature for _, group in block.groupby("track_id") if (feature := track_features(group))]
    if not features:
        return {"has_tracking": 0, "tracked_person_count": 0}
    frame = pd.DataFrame(features)
    result = {"has_tracking": 1, "tracked_person_count": len(frame)}
    for column in frame.columns:
        result[f"{column}_mean"] = frame[column].mean()
        result[f"{column}_max"] = frame[column].max()
        result[f"{column}_std"] = frame[column].std(ddof=0)
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        annotation_name = "train_annotations_all.csv" if split == "train" else "valid_annotations.csv"
        annotations = pd.read_csv(ANNOTATION_ROOT / annotation_name)
        annotations["block_id"] = annotations.apply(block_id, axis=1)
        rows = []
        for video_path, blocks in annotations.groupby("video_path", sort=False):
            video = Path(video_path).stem
            tracking_path = TRACKING_ROOT / split / f"{video}.csv"
            tracking = pd.read_csv(tracking_path) if tracking_path.is_file() else pd.DataFrame()
            for _, annotation in blocks.iterrows():
                required = {"frame", "track_id", "x1", "y1", "x2", "y2"}
                if required.issubset(tracking.columns):
                    selected = tracking[(tracking["frame"] >= annotation.start_frame) & (tracking["frame"] <= annotation.end_frame)]
                    summary = summarize_block(selected)
                else:
                    summary = {"has_tracking": 0, "tracked_person_count": 0}
                rows.append({
                    "block_id": annotation.block_id, "split": split, "video": annotation.video,
                    "video_path": annotation.video_path, "source": annotation.source,
                    "label": annotation.label, "block_type": annotation.block_type,
                    "start_frame": annotation.start_frame, "end_frame": annotation.end_frame,
                    **summary,
                })
        output = OUTPUT_ROOT / f"{split}_behavior_features.csv"
        pd.DataFrame(rows).fillna(0).to_csv(output, index=False, encoding="utf-8-sig")
        print(f"{split}: {len(rows)}개 블록 저장 -> {output}")


if __name__ == "__main__":
    main()
