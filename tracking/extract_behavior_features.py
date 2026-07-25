import math
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# 설정
# ==========================================

SPLITS = ["train", "valid", "test"]

TRACKING_DIR = Path("./results/tracking")
ANNOTATION_DIR = Path("./splits/annotations")
OUTPUT_DIR = Path("./results/features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Feature 계산 함수
# ==========================================

def extract_track_feature(group):

    group = group.sort_values("frame")

    trajectory = list(
        zip(
            group["frame"],
            group["center_x"],
            group["center_y"]
        )
    )

    frame_count = len(trajectory)

    # 너무 짧은 track 제거
    if frame_count < 10:
        return None

    total_distance = 0
    speeds = []

    for i in range(1, len(trajectory)):

        f1, x1, y1 = trajectory[i - 1]
        f2, x2, y2 = trajectory[i]

        dist = math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

        gap = f2 - f1

        total_distance += dist

        if gap > 0:
            speeds.append(dist / gap)

    xs = group["center_x"].values
    ys = group["center_y"].values

    return {

        "frame_count": frame_count,

        "move_distance": total_distance,

        "avg_speed": np.mean(speeds) if speeds else 0,
        "max_speed": np.max(speeds) if speeds else 0,
        "min_speed": np.min(speeds) if speeds else 0,
        "std_speed": np.std(speeds) if speeds else 0,

        "movement_range":
            (xs.max() - xs.min()) +
            (ys.max() - ys.min()),

        "trajectory_variance":
            np.var(xs) +
            np.var(ys)
    }


# ==========================================
# Split 반복
# ==========================================

for split in SPLITS:

    print("=" * 60)
    print(split)

    tracking = pd.read_csv(
        TRACKING_DIR / f"{split}_tracking.csv"
    )
    print(f"\n[{split}]")
    print("Tracking videos :", tracking["video"].nunique())
    print("Tracking frame range :", tracking["frame"].min(), "~", tracking["frame"].max())

    if split == "train":
        annotations = pd.read_csv(
            ANNOTATION_DIR / "train_annotations_all.csv"
        )
    else:
        annotations = pd.read_csv(
            ANNOTATION_DIR / f"{split}_annotations.csv"
        )
    # action block만 사용
    annotations = annotations[
        annotations["block_type"] == "action"
    ].reset_index(drop=True)
    print("Annotation videos :", annotations["video"].nunique())
    print("Annotation rows :", len(annotations))

    # 중심점 계산
    tracking["center_x"] = (
        tracking["x1"] + tracking["x2"]
    ) / 2

    tracking["center_y"] = (
        tracking["y1"] + tracking["y2"]
    ) / 2

    feature_rows = []

    # ----------------------------------
    # annotation(block) 반복
    # ----------------------------------

    for _, ann in annotations.iterrows():

        block = tracking[
            (tracking["video"] == ann["video"])
            &
            (tracking["frame"] >= ann["start_frame"])
            &
            (tracking["frame"] <= ann["end_frame"])
        ]

        if block.empty:

            video_tracking = tracking[
                tracking["video"] == ann["video"]
            ]

            print("=" * 60)
            print(f"No tracking: {ann['video']}")
            print(f"Action frame : {ann['start_frame']} ~ {ann['end_frame']}")

            if video_tracking.empty:
                print("Video does not exist in tracking.csv")
            else:
                print(
                    "Tracking frame :",
                    video_tracking["frame"].min(),
                    "~",
                    video_tracking["frame"].max()
                )

            break

        # -----------------------------
        # track별 feature 계산
        # -----------------------------

        track_features = []

        for track_id, group in block.groupby("track_id"):

            feature = extract_track_feature(group)

            if feature is None:
                continue

            feature["track_id"] = track_id

            track_features.append(feature)

        if len(track_features) == 0:
            continue
        
        track_df = pd.DataFrame(track_features)

        numeric_cols = track_df.drop(columns="track_id")

        aggregated = {}

        for col in numeric_cols.columns:
            aggregated[f"{col}_mean"] = numeric_cols[col].mean()
            aggregated[f"{col}_max"] = numeric_cols[col].max()
            aggregated[f"{col}_std"] = numeric_cols[col].std()

        feature_rows.append({

            "video": ann["video"],
            "label": ann["label"],
            "block_type": ann["block_type"],
            "start_frame": ann["start_frame"],
            "end_frame": ann["end_frame"],

            **aggregated
        })

    feature_df = pd.DataFrame(feature_rows).fillna(0)

    save_path = (
        OUTPUT_DIR /
        f"{split}_behavior_features.csv"
    )

    feature_df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(feature_df.head())
    print("rows :", len(feature_df))
    print("saved :", save_path)