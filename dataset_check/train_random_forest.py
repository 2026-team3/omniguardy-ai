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

# ===========================================
# Load Feature CSV
# ===========================================
df = pd.read_csv(
    "./results/merged_features(train).csv"
)

# ===========================================
# Label 생성
# ===========================================
df["label"] = (
    df["video"]
    .str.extract(r"(A\d+)")
)

# ===========================================
# 사용할 Feature
# ===========================================
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

X = df[feature_cols]
y = df["label"]

# ===========================================
# Train / Test Split
# ===========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===========================================
# RandomForest
# ===========================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

print("="*60)
print("Training RandomForest...")
print("="*60)

model.fit(
    X_train,
    y_train
)

# ===========================================
# Prediction
# ===========================================
pred = model.predict(X_test)

# ===========================================
# Accuracy
# ===========================================
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

# ===========================================
# Save Report
# ===========================================
save_dir = "./results/random_forest"

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

# ===========================================
# Confusion Matrix
# ===========================================
cm = confusion_matrix(
    y_test,
    pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig, ax = plt.subplots(figsize=(7,7))

disp.plot(ax=ax)

plt.tight_layout()

plt.savefig(
    f"{save_dir}/confusion_matrix.png",
    dpi=300
)

plt.close()

# ===========================================
# Feature Importance
# ===========================================
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

plt.figure(figsize=(8,5))

plt.bar(
    importance["feature"],
    importance["importance"]
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    f"{save_dir}/feature_importance.png",
    dpi=300
)

plt.close()

# ===========================================
# Save Model
# ===========================================
joblib.dump(
    model,
    f"{save_dir}/random_forest.pkl"
)

print()
print("="*60)
print("Model Saved")
print(f"{save_dir}/random_forest.pkl")
print("="*60)