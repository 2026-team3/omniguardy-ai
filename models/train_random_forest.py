import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "./results/merged_features(07.07).csv"
)

# Label 생성
y = df["label"]

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

X = df[feature_cols]
y = df["label"]

print(X.shape)
print(X.columns)

print("label NaN:", df["label"].isna().sum())

print("\nClass distribution")
print(df["label"].value_counts())

# Train / Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# RandomForest
model = RandomForestClassifier(
    n_estimators=500,   # 숲을 구성할 나무의 수 (default=10, 많을수록 일반화 but, trade-off 고려)
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

print("="*60)
print("Training RandomForest...")
print("="*60)

# Train
model.fit(
    X_train,
    y_train
)

# Prediction
pred = model.predict(X_test)

# Accuracy -> 다중 레이블의 경우, 각 레이블의 ACC 값의 평균으로 평가
acc = accuracy_score(
    y_test,
    pred
)

print()
print("="*60)
print(f"Accuracy : {acc:.4f}")
print("="*60)

print()
print(classification_report(
    y_test,
    pred
))

print(model.classes_)
print(len(model.classes_))

save_dir = "./results/random_forest(07.07)"

os.makedirs(
    save_dir,
    exist_ok=True
)

report = classification_report(
    y_test,
    pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    f"{save_dir}/classification_report.csv",
    encoding="utf-8-sig"
)

labels = model.classes_

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(7,7))
disp.plot(ax=ax)
plt.tight_layout()
plt.savefig(
    f"{save_dir}/confusion_matrix.png",
    dpi=300
)
plt.close()

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
    f"{save_dir}/feature_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(12,6))
plt.bar(
    importance["feature"],
    importance["importance"]
)
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.savefig(
    f"{save_dir}/feature_importance.png",
    dpi=300
)
plt.close()

# Save Model
joblib.dump(
    model,
    f"{save_dir}/random_forest.pkl"
)

print("\nTop 10 Features")
print(importance.head(10))

print()
print("="*60)
print("Model Saved")
print(f"{save_dir}/random_forest.pkl")
print("="*60)