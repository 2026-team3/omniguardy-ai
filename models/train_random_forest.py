import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

# train feature load
df = pd.read_csv(
    "./results/after_augmentation/merged_features(07.12).csv"
)

# 사용할 Feature
feature_cols = [
    "frame_count",

    "move_distance",
    "avg_speed",
    "max_speed",
    "min_speed",
    "std_speed",

    "movement_range",
    "trajectory_variance",

    "hand_motion_sum",
    "hand_motion_mean",
    "hand_motion_std",
    "hand_motion_max",

    "body_motion_sum",
    "body_motion_mean",
    "body_motion_std",
    "body_motion_max",

    "upper_body_angle_mean",
    "upper_body_angle_std",
    "upper_body_angle_max",
    "upper_body_angle_min",

    "arm_extension_mean",
    "arm_extension_std",
    "arm_extension_max",
    "arm_extension_min"
]

X_train = df[feature_cols]
y_train = df["label"]

print("=" * 60)
print("Train Data")
print("=" * 60)

print("X shape:", X_train.shape)

print("\nClass distribution")
print(y_train.value_counts())

print("\nLabel NaN:")
print(y_train.isna().sum())

print("\nFeature NaN:")
print(X_train.isna().sum())


# RandomForest
model = RandomForestClassifier(
    n_estimators=500,   # 숲을 구성할 나무의 수 (default=10, 많을수록 일반화 but, trade-off 고려)
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("="*60)
print("Training RandomForest...")
print("="*60)

# Train
model.fit(
    X_train,
    y_train
)


save_dir = "./results/after_augmentation/train_random_forest"

os.makedirs(
    save_dir,
    exist_ok=True
)

# Feature Importance
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print()
print("="*60)
print("Feature Importance")
print("="*60)
print(importance)

importance.to_csv(
    f"{save_dir}/train_feature_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

# Feature Importance Plot
plt.figure(figsize=(12,6))
plt.bar(importance["feature"], importance["importance"])
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.savefig(
    f"{save_dir}/train_feature_importance.png",
    dpi=300
)
plt.close()

# Save Model
joblib.dump(
    model,
    f"{save_dir}/train_random_forest.pkl"
)

print("\nTop 10 Features")
print(importance.head(10))

print()
print("="*60)
print("Model Saved")
print(f"{save_dir}/train_random_forest.pkl")
print("="*60)