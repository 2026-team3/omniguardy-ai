import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CSV Load
# =========================
df = pd.read_csv(
    "./results/merged_features(train).csv"
)

# =========================
# Label 추출 (비디오명에서)
# =========================
df["label"] = (
    df["video"]
    .str.extract(r"(A\d+)")
)
# =========================
# Label → 행동 이름
# =========================
label_map = {
    "A17": "Doorbell Press",
    "A18": "Door Open Attempt",
    "A19": "Door Kick",
    "A20": "Looking Inside",
    "A21": "Door Knocking",

    # 이후 데이터셋에 맞게 계속 추가
}

df["action"] = df["label"].map(label_map)

# =========================
# Feature 목록
# =========================
feature_cols = [
    "frame_count",
    "move_distance",
    "avg_speed",
    "movement_range",
    "trajectory_variance",
    "hand_motion",
    "body_motion",
    "upper_body_angle",
    "arm_extension"
]

# =========================
# 결과 저장 폴더
# =========================
save_dir = "./results/EDA"

os.makedirs(
    save_dir,
    exist_ok=True
)

# =====================================================
# 1. 전체 데이터 확인
# =====================================================

print("=" * 60)
print("Dataset Info")
print("=" * 60)

print(df.info())

print()

print(df.head())

print()

print("총 데이터 수")
print(len(df))

# =====================================================
# 2. Label 개수 확인
# =====================================================

print()
print("=" * 60)
print("Label Count")
print("=" * 60)

label_count = (
    df.groupby("label")
    .size()
    .sort_values(ascending=False)
)

print(label_count)

label_count.to_csv(
    f"{save_dir}/label_count.csv",
    encoding="utf-8-sig"
)

# =====================================================
# 3. Feature 평균
# =====================================================

print()
print("=" * 60)
print("Feature Mean")
print("=" * 60)

feature_mean = (
    df.groupby("action")[feature_cols]
    .mean()
)

print(feature_mean)

feature_mean.to_csv(
    f"{save_dir}/feature_mean.csv",
    encoding="utf-8-sig"
)

# =====================================================
# 4. Feature 표준편차
# =====================================================

print()
print("=" * 60)
print("Feature Std")
print("=" * 60)

feature_std = (
    df.groupby("action")[feature_cols]
    .std()
)

print(feature_std)

feature_std.to_csv(
    f"{save_dir}/feature_std.csv",
    encoding="utf-8-sig"
)

# =====================================================
# 5. Feature Min / Max
# =====================================================

feature_min = (
    df.groupby("action")[feature_cols]
    .min()
)

feature_max = (
    df.groupby("action")[feature_cols]
    .max()
)

feature_min.to_csv(
    f"{save_dir}/feature_min.csv",
    encoding="utf-8-sig"
)

feature_max.to_csv(
    f"{save_dir}/feature_max.csv",
    encoding="utf-8-sig"
)

# =====================================================
# 6. Correlation
# =====================================================

corr = df[feature_cols].corr()

corr.to_csv(
    f"{save_dir}/feature_corr.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation")

plt.tight_layout()

plt.savefig(
    f"{save_dir}/feature_corr.png",
    dpi=300
)

plt.close()

# =====================================================
# 7. Boxplot
# =====================================================

for feature in feature_cols:

    plt.figure(figsize=(12,6))

    df.boxplot(
        column=feature,
        by="action",
        rot=20
    )

    plt.title(feature)

    plt.suptitle("")

    plt.tight_layout()

    plt.savefig(
        f"{save_dir}/boxplot_{feature}.png",
        dpi=300
    )

    plt.close()

# =====================================================
# 8. Histogram
# =====================================================

for feature in feature_cols:

    plt.figure(figsize=(10,6))

    sns.histplot(
        data=df,
        x=feature,
        hue="action",
        element="step",
        stat="density",
        common_norm=False
    )

    plt.tight_layout()

    plt.savefig(
        f"{save_dir}/hist_{feature}.png",
        dpi=300
    )
    plt.close()

print()
print("=" * 60)
print("EDA 완료")
print(f"저장 위치 : {save_dir}")
print("=" * 60)
print()
print("="*60)
print("Feature Variance")
print("="*60)

print(
    df[feature_cols].var().sort_values(
        ascending=False
    )
)