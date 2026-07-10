import pandas as pd

# CSV 로드
behavior_df = pd.read_csv("./results/behavior_features(07.07).csv")
pose_df = pd.read_csv("./results/pose_features(07.07).csv")

# video 이름 통일
pose_df["video"] = (pose_df["video"].str.replace(".mp4", "", regex=False))

print("Behavior:", len(behavior_df))
print("Pose:", len(pose_df))

# feature merge
merged_df = pd.merge(behavior_df, pose_df, on=["video", "label"], how="inner")

# 불필요한 column 제거
drop_columns = ["frame_count_y"]

for col in drop_columns:
    if col in merged_df.columns:
        merged_df.drop(columns=[col], inplace=True)

# column 이름 정의
if "frame_count_x" in merged_df.columns:
    merged_df.rename(columns={
        "frame_count_x": "frame_count"
    }, inplace=True)

print(merged_df["label"].value_counts())
print(merged_df.columns)


# 저장
merged_df.to_csv(
    "./results/merged_features(07.07).csv",
    index=False,
    encoding="utf-8-sig"
)

print("Merged:", len(merged_df))
print()
print("=" * 50)
print("merged_features(07.07).csv 저장 완료")
print()
print(merged_df.head())
print("총 row:", len(merged_df))

