"""train 특징으로 XGBoost를 학습하고 VL/VS validation 성능을 평가한다."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


FEATURE_ROOT = Path("results/features")
OUTPUT_ROOT = Path("results/models/xgboost_validation")
METADATA_COLUMNS = {
    "block_id", "split", "video", "video_path", "source", "label", "block_type",
    "start_frame", "end_frame",
}


def main() -> None:
    train = pd.read_csv(FEATURE_ROOT / "train_merged_features.csv")
    valid = pd.read_csv(FEATURE_ROOT / "valid_merged_features.csv")
    feature_columns = [column for column in train.columns if column not in METADATA_COLUMNS]
    missing = set(feature_columns) - set(valid.columns)
    if missing:
        raise ValueError(f"validation 특징 열이 없습니다: {sorted(missing)}")
    if not feature_columns:
        raise ValueError("학습할 특징 열이 없습니다.")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["label"])
    unknown = set(valid["label"]) - set(encoder.classes_)
    if unknown:
        raise ValueError(f"train에 없는 validation 라벨: {sorted(unknown)}")
    y_valid = encoder.transform(valid["label"])

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(train[feature_columns], y_train, sample_weight=weights)

    prediction = model.predict(valid[feature_columns])
    accuracy = accuracy_score(y_valid, prediction)
    print(f"VL/VS validation accuracy: {accuracy:.4f}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_valid, prediction, target_names=encoder.classes_, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(OUTPUT_ROOT / "validation_report.csv", encoding="utf-8-sig")
    matrix = confusion_matrix(y_valid, prediction, labels=range(len(encoder.classes_)))
    figure, axis = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(matrix, display_labels=encoder.classes_).plot(ax=axis, colorbar=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "validation_confusion_matrix.png", dpi=200)
    plt.close(figure)
    pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(OUTPUT_ROOT / "feature_importance.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"model": model, "label_encoder": encoder, "feature_columns": feature_columns}, OUTPUT_ROOT / "xgboost_validation.pkl")


if __name__ == "__main__":
    main()
