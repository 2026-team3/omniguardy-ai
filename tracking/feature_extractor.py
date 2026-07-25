import pandas as pd
import math
import numpy as np

# tracking 결과 로드
df = pd.read_csv(
    "./results/after_augmentation/tracking_results(07.12).csv"
)

# 중심점 계산
df["center_x"] = (
    df["x1"] + df["x2"]
) / 2
df["center_y"] = (
    df["y1"] + df["y2"]
) / 2

# feature 저장용
feature_data = []
speeds = []

# trajectory grouping
grouped = df.groupby(
    ["video", "track_id"]
)

for (video, track_id), group in grouped:
    # frame 순 정렬
    group = group.sort_values("frame")
    label = group["label"].iloc[0]
    trajectory = []

    for _, row in group.iterrows():
        trajectory.append(
            (
                row["frame"],
                row["center_x"],
                row["center_y"]
            )
        )
    # =========================
    # 이동거리 계산
    # =========================
    total_distance = 0
    speeds = []

    for i in range(1, len(trajectory)):

        frame1, x1, y1 = trajectory[i - 1]
        frame2, x2, y2 = trajectory[i]

        dist = math.sqrt((x2 - x1) ** 2 +(y2 - y1) ** 2)
        frame_gap = frame2 - frame1
        total_distance += dist

        if frame_gap > 0:
            speed = dist / frame_gap
            speeds.append(speed)

    # =========================
    # 체류 frame 수
    # =========================
    frame_count = len(trajectory)
    if frame_count < 10: 
        continue

    # =========================
    # 평균 속도
    # =========================
    if len(speeds) > 0:
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        min_speed = np.min(speeds)
        std_speed = np.std(speeds)
    else:
        avg_speed = 0
        max_speed = 0
        min_speed = 0
        std_speed = 0

    # =========================
    # 이동 범위
    # =========================
    xs = [p[1] for p in trajectory]
    ys = [p[2] for p in trajectory]

    movement_range = 0
    if len(xs) > 0:
        movement_range = (
            (max(xs) - min(xs)) + (max(ys) - min(ys))
        )

    # =========================
    # trajectory variance
    # =========================
    trajectory_variance = 0

    if len(xs) > 1:
        trajectory_variance = (
            np.var(xs) + np.var(ys)
        )

    # =========================
    # 저장
    # =========================
    feature_data.append({
        "video": video,
        "track_id": track_id,
        "label": label, 
        "frame_count": frame_count,
        "move_distance": total_distance,

        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "min_speed": min_speed,
        "std_speed": std_speed,

        "movement_range": movement_range,
        "trajectory_variance": trajectory_variance
    })

# =========================
# DataFrame 저장
# =========================
feature_df = pd.DataFrame(feature_data)
# 가장 오래 추적된 Track만 사용
feature_df = (
    feature_df
    .sort_values("frame_count", ascending=False)
    .drop_duplicates(subset="video")
)

print(feature_df.head())
feature_df.to_csv(
    "./results/after_augmentation/behavior_features(07.12).csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("behavior_features(07.12).csv 저장 완료")