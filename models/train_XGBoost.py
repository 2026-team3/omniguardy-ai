import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


df = pd.read_csv(
    "./results/merged_features(07.07).csv"
)

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
print("\nClass distribution")
print(y.value_counts())

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# XGBoost
model = XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),

    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    eval_metric="mlogloss"
)

print("=" * 60)
print("Training XGBoost...")
print("=" * 60)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    pred
)

print()
print("=" * 60)
print(f"Accuracy : {acc:.4f}")
print("=" * 60)

print()

print(classification_report(
    y_test,
    pred,
    target_names=label_encoder.classes_
))

save_dir = "./results/xgboost(07.07)"

os.makedirs(
    save_dir,
    exist_ok=True
)

report = classification_report(
    y_test,
    pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    f"{save_dir}/classification_report.csv",
    encoding="utf-8-sig"
)

# Confusion Matrix
labels = list(range(len(label_encoder.classes_)))

cm = confusion_matrix(
    y_test,
    pred,
    labels=labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
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
print("=" * 60)
print("Feature Importance")
print("=" * 60)
print(importance)

print("\nTop 10 Features")
print(importance.head(10))

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

plt.xticks(
    rotation=60,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    f"{save_dir}/feature_importance.png",
    dpi=300
)

plt.close()

# Save Model
joblib.dump(
    {
        "model": model,
        "label_encoder": label_encoder
    },
    f"{save_dir}/xgboost.pkl"
)

print()
print("=" * 60)
print("Model Saved")
print(f"{save_dir}/xgboost.pkl")
print("=" * 60)