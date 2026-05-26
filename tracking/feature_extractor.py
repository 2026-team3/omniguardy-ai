import pandas as pd
import math

# tracking 결과 로드
df = pd.read_csv(
    "./results/tracking_results.csv"
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

# trajectory grouping
grouped = df.groupby(
    ["video", "track_id"]
)

for (video, track_id), group in grouped:

    # frame 순 정렬
    group = group.sort_values("frame")

    trajectory = []

    for _, row in group.iterrows():

        trajectory.append(
            (
                row["center_x"],
                row["center_y"]
            )
        )

    # =========================
    # 이동거리 계산
    # =========================
    total_distance = 0

    for i in range(
        1,
        len(trajectory)
    ):

        x1, y1 = trajectory[i - 1]
        x2, y2 = trajectory[i]

        dist = math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

        total_distance += dist

    # =========================
    # 체류 frame 수
    # =========================
    frame_count = len(trajectory)

    feature_data.append({

        "video": video,

        "track_id": track_id,

        "frame_count": frame_count,

        "move_distance": total_distance
    })

# =========================
# DataFrame 저장
# =========================
feature_df = pd.DataFrame(feature_data)

print(feature_df.head())

feature_df.to_csv(
    "./results/behavior_features.csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("behavior_features.csv 저장 완료")