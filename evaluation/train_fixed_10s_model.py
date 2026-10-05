"""기존 frame-level 결과를 재활용해 10초 입력용 모델을 별도 학습·평가한다.

기존 feature CSV, 기존 모델, 기존 학습 스크립트를 수정하지 않는다.
결과는 results/evaluation/fixed_10s_model/ 아래에만 생성한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import optuna
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.extract_behavior_features import summarize_block
from tracking.merge_features import POSE_COLUMNS, pose_summary

optuna.logging.set_verbosity(optuna.logging.WARNING)

WINDOW_FRAMES = 300
# feature CSV는 실험(=output-dir)에 상관없이 항상 이 기본 경로에서 읽고 쓴다.
# (한 번 추출한 10초 feature를 모든 실험이 공유해서 재사용하기 위함)
FEATURE_ROOT = PROJECT_ROOT / "results/evaluation/fixed_10s_model"
# 결과(모델/리포트/optuna 로그)를 저장할 경로. --output-dir로 실험마다 바꿀 수 있고,
# 기본값은 FEATURE_ROOT와 같아 기존 동작과 동일하다.
OUTPUT_ROOT = FEATURE_ROOT
METADATA_COLUMNS = {
    "split", "video", "annotation_video", "video_path", "json_path", "label",
    "block_type", "start_frame", "end_frame", "source", "normal_subtype",
}
# 증강 영상(aug1/aug2...)도 annotation_video가 원본 영상명과 같으므로,
# 이 컬럼으로 그룹을 묶으면 CV 폴드 간에 "같은 원본에서 나온 영상"이 섞이지 않는다.
GROUP_COLUMN = "annotation_video"
FIXED_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": 42,
}
# A18(도어락 조작 의심)/A20(들여다보기)은 시각적으로 구분이 어려워 하나로 합쳐서 학습한다.
LABEL_MAP = {
    "A18": "A18_20",
    "A20": "A18_20",
}


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def make_windows(annotations: pd.DataFrame) -> pd.DataFrame:
    """라벨 블록 내부에서만, 겹치지 않는 완전한 10초 창을 생성한다."""
    windows: list[dict] = []
    for _, block in annotations.iterrows():
        start, end = int(block.start_frame), int(block.end_frame)
        for window_start in range(start, end - WINDOW_FRAMES + 2, WINDOW_FRAMES):
            window = block.to_dict()
            window["start_frame"] = window_start
            window["end_frame"] = window_start + WINDOW_FRAMES - 1
            windows.append(window)
    return pd.DataFrame(windows)


def periodicity_features(pose: pd.DataFrame, start_frame: int, end_frame: int) -> dict:
    """윈도우 내 hand_motion 시계열에서 "반복되는 피크" 패턴을 뽑는다.

    A21(반복 노크)은 평균/최대/표준편차로는 A18(문 조작)과 잘 안 구분되는데,
    실제로는 "짧은 간격을 두고 비슷한 크기의 피크가 규칙적으로 반복"되는 동작이라서
    피크 개수와 피크 간격의 일정함(표준편차)을 직접 특징으로 넣으면 구분에 도움이 될 수 있다.
    results/pose/*.csv 에 이미 10프레임 간격으로 캐시된 hand_motion을 그대로 쓰므로
    영상 재처리가 필요 없다.
    """
    selected = pose[(pose["frame"] >= start_frame) & (pose["frame"] <= end_frame)].sort_values("frame")
    if len(selected) < 3:
        return {"hand_motion_peak_count": 0, "hand_motion_peak_interval_std": 0.0, "hand_motion_peak_interval_mean": 0.0}

    values = selected["hand_motion"].to_numpy()
    frames = selected["frame"].to_numpy()
    threshold = values.mean() + values.std()

    peak_frames = [
        frames[i] for i in range(1, len(values) - 1)
        if values[i] > values[i - 1] and values[i] > values[i + 1] and values[i] > threshold
    ]

    if len(peak_frames) >= 2:
        intervals = np.diff(peak_frames).astype(float)
        interval_mean = float(intervals.mean())
        interval_std = float(intervals.std())
    else:
        interval_mean = 0.0
        interval_std = 0.0

    return {
        "hand_motion_peak_count": len(peak_frames),
        "hand_motion_peak_interval_mean": interval_mean,
        "hand_motion_peak_interval_std": interval_std,
    }


def build_features(split: str, annotation_file: Path) -> pd.DataFrame:
    annotations = pd.read_csv(annotation_file)
    windows = make_windows(annotations)
    tracking_root = PROJECT_ROOT / "results/tracking" / split
    pose_root = PROJECT_ROOT / "results/pose" / split
    tracking_columns = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    pose_columns = {"frame", *POSE_COLUMNS}
    rows: list[dict] = []

    for video_path, video_windows in windows.groupby("video_path", sort=False):
        stem = Path(video_path).stem
        tracking = read_csv_or_empty(tracking_root / f"{stem}.csv")
        pose = read_csv_or_empty(pose_root / f"{stem}.csv")
        for _, window in video_windows.iterrows():
            start, end = int(window.start_frame), int(window.end_frame)
            behavior = (
                summarize_block(tracking[(tracking.frame >= start) & (tracking.frame <= end)])
                if tracking_columns.issubset(tracking.columns)
                else {"has_tracking": 0, "tracked_person_count": 0}
            )
            pose_features = (
                pose_summary(pose, start, end)
                if pose_columns.issubset(pose.columns)
                else {"has_pose": 0, "pose_frame_count": 0}
            )
            # periodic_features = (
            #     periodicity_features(pose, start, end)
            #     if pose_columns.issubset(pose.columns)
            #     else {"hand_motion_peak_count": 0, "hand_motion_peak_interval_mean": 0.0, "hand_motion_peak_interval_std": 0.0}
            # )
            # rows.append({**window.to_dict(), **behavior, **pose_features, **periodic_features})
            rows.append({**window.to_dict(), **behavior, **pose_features})
    return pd.DataFrame(rows).fillna(0)


def make_sample_weight(y, labels: pd.Series, boost_class: str | None, boost_factor: float):
    """balanced 가중치에 특정 클래스만 추가로 더 얹은 sample_weight를 만든다.

    예: boost_class="A21", boost_factor=1.5 면, A21 샘플을 틀렸을 때의 손실(loss)이
    balanced 가중치보다 1.5배 더 크게 계산되어, 모델이 A21을 더 적극적으로 맞히려 한다.
    단, 대신 다른 클래스의 precision/recall이 희생될 수 있다.
    """
    weight = compute_sample_weight("balanced", y=y)
    if boost_class:
        weight = weight.copy()
        weight[labels.to_numpy() == boost_class] *= boost_factor
    return weight


def make_sample_weight_multi(y, labels: pd.Series, multipliers: dict[str, float]):
    """balanced 가중치에 클래스마다 다른 배수(multipliers)를 적용한다."""
    weight = compute_sample_weight("balanced", y=y)
    factor = labels.map(multipliers).fillna(1.0).to_numpy()
    return weight * factor


def split_params(raw_params: dict) -> tuple[dict, dict]:
    """Optuna best_params를 (XGB 파라미터, 클래스별 weight 배수)로 나눈다."""
    xgb_params = {k: v for k, v in raw_params.items() if not k.startswith("weight_")}
    weight_multipliers = {k[len("weight_"):]: v for k, v in raw_params.items() if k.startswith("weight_")}
    return xgb_params, weight_multipliers


def make_objective(
    train: pd.DataFrame, feature_columns: list[str], y_train_all, cv_folds: int,
    boost_class: str | None, boost_factor: float, tune_class_weights: bool,
):
    """train 세트 내부의 group 기반 K-fold로 macro F1을 최적화하는 Optuna objective를 만든다.

    주의: 여기서 쓰는 분할은 전체 train 세트 내부에서만 이루어지며, 공식 VL/VS
    validation 세트는 절대 사용하지 않는다. (반복 시도로 validation에 과적합되는 것을 방지)
    """
    groups = train[GROUP_COLUMN].to_numpy()
    splitter = GroupKFold(n_splits=cv_folds)
    X_all = train[feature_columns]
    labels_all = train["label"]
    class_labels = sorted(labels_all.unique())
    # 기준(anchor) 클래스는 가중치를 1.0으로 고정하고, 나머지 클래스만 그 대비 상대 배수를 탐색한다.
    # (전부 자유롭게 풀어두면 "클래스 간 상대 비율"이 아니라 "전체 가중치 스케일"만 같이
    #  커지는 쪽으로 수렴해 min_child_weight 등과 엉켜서 의미 없는 결과가 나온다.)
    anchor_label = labels_all.value_counts().idxmax()
    tunable_labels = [label for label in class_labels if label != anchor_label]

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        if tune_class_weights:
            # anchor_label은 1.0 고정, 나머지 클래스만 그 대비 상대 배수를 탐색한다.
            multipliers = {anchor_label: 1.0}
            for label in tunable_labels:
                multipliers[label] = trial.suggest_float(f"weight_{label}", 0.3, 3.0)
        else:
            multipliers = None
        fold_scores = []
        for fold_train_idx, fold_valid_idx in splitter.split(X_all, y_train_all, groups):
            X_fold_train, X_fold_valid = X_all.iloc[fold_train_idx], X_all.iloc[fold_valid_idx]
            y_fold_train, y_fold_valid = y_train_all[fold_train_idx], y_train_all[fold_valid_idx]
            fold_labels_train = labels_all.iloc[fold_train_idx]
            fold_model = XGBClassifier(
                **FIXED_PARAMS, num_class=len(set(y_train_all)), **params,
            )
            if tune_class_weights:
                fold_weight = make_sample_weight_multi(y_fold_train, fold_labels_train, multipliers)
            else:
                fold_weight = make_sample_weight(y_fold_train, fold_labels_train, boost_class, boost_factor)
            fold_model.fit(X_fold_train, y_fold_train, sample_weight=fold_weight)
            fold_prediction = fold_model.predict(X_fold_valid)
            fold_scores.append(f1_score(y_fold_valid, fold_prediction, average="macro", zero_division=0))
        return sum(fold_scores) / len(fold_scores)

    return objective


def tune_hyperparameters(
    train: pd.DataFrame, feature_columns: list[str], y_train_all, n_trials: int, cv_folds: int,
    boost_class: str | None, boost_factor: float, tune_class_weights: bool,
) -> tuple[dict, dict]:
    if tune_class_weights:
        anchor_label = train["label"].value_counts().idxmax()
        print(f"[클래스별 가중치 탐색] 기준(anchor) 클래스 = '{anchor_label}' (가중치 1.0 고정), 나머지만 상대 탐색")
    objective = make_objective(
        train, feature_columns, y_train_all, cv_folds, boost_class, boost_factor, tune_class_weights,
    )
    study = optuna.create_study(
        direction="maximize", study_name="xgboost_fixed_10s",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    trials_df = study.trials_dataframe().sort_values("value", ascending=False)
    trials_df.to_csv(OUTPUT_ROOT / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_ROOT / "optuna_best_params.json").write_text(
        json.dumps(study.best_params, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print("\n=== Optuna 상위 10개 시도 (macro F1 기준, train 내부 group K-fold) ===")
    display_columns = ["number", "value"] + [c for c in trials_df.columns if c.startswith("params_")]
    print(trials_df[display_columns].head(10).to_string(index=False))
    print(f"\n최적 파라미터: {study.best_params}")
    print(f"최적 macro F1 (CV): {study.best_value:.4f}")
    return split_params(study.best_params)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-features", action="store_true",
        help="이미 생성된 10초 feature CSV를 재사용해 모델 학습만 수행한다.",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Optuna로 하이퍼파라미터를 탐색한 뒤 최적 파라미터로 최종 모델을 학습한다.",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna 시도 횟수 (기본 50).")
    parser.add_argument("--cv-folds", type=int, default=3, help="train 내부 group K-fold 수 (기본 3).")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="결과를 저장할 디렉터리 (기본값: results/evaluation/fixed_10s_model). "
             "baseline 결과를 보존하려면 실험마다 다른 경로를 지정한다 (예: results/evaluation/fixed_10s_optuna_v1).",
    )
    parser.add_argument(
        "--boost-class", type=str, default=None,
        help="balanced sample_weight에 추가로 가중치를 더 줄 라벨명 (예: A21). 지정 안 하면 기존과 동일.",
    )
    parser.add_argument(
        "--boost-factor", type=float, default=1.5,
        help="--boost-class에 곱할 배수 (기본 1.5). --tune-class-weights를 쓰면 무시된다.",
    )
    parser.add_argument(
        "--tune-class-weights", action="store_true",
        help="클래스별 balanced 가중치 배수를 Optuna가 직접 탐색하게 한다 (--tune과 함께 사용). "
             "--boost-class/--boost-factor보다 우선한다.",
    )
    args = parser.parse_args()
    if args.tune_class_weights and not args.tune:
        raise SystemExit("--tune-class-weights는 --tune과 함께 사용해야 합니다.")
    global OUTPUT_ROOT
    OUTPUT_ROOT = (PROJECT_ROOT / args.output_dir) if args.output_dir else FEATURE_ROOT
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    # feature CSV는 실험 디렉터리와 무관하게 항상 FEATURE_ROOT에서 재사용/저장한다.
    train_file = FEATURE_ROOT / "train_10s_features.csv"
    valid_file = FEATURE_ROOT / "valid_10s_features.csv"
    if args.reuse_features:
        if not train_file.is_file() or not valid_file.is_file():
            raise FileNotFoundError("재사용할 10초 feature CSV가 없습니다.")
        train, valid = pd.read_csv(train_file), pd.read_csv(valid_file)
    else:
        train = build_features("train", PROJECT_ROOT / "splits/annotations/train_annotations_all.csv")
        valid = build_features("valid", PROJECT_ROOT / "splits/annotations/valid_annotations.csv")
        train.to_csv(train_file, index=False, encoding="utf-8-sig")
        valid.to_csv(valid_file, index=False, encoding="utf-8-sig")

    train["label"] = train["label"].replace(LABEL_MAP)
    valid["label"] = valid["label"].replace(LABEL_MAP)

    # mydata(직접 촬영) 행은 annotation_video가 원래 비어있는데(build_features의
    # fillna(0)으로 정수 0이 되어버림), 나머지 행은 문자열 영상명이라 GroupKFold
    # 정렬 시 int/str 비교 에러가 난다. mydata는 어차피 서로 공유할 "원본 영상"이
    # 없으니, 자기 자신의 video 파일명을 그룹으로 써서 독립 그룹으로 만든다.
    for frame in (train, valid):
        broken = frame["annotation_video"].isna() | frame["annotation_video"].astype(str).isin(["0", "0.0"])
        frame.loc[broken, "annotation_video"] = frame.loc[broken, "video"]
        frame["annotation_video"] = frame["annotation_video"].astype(str)

    feature_columns = [column for column in train.columns if column not in METADATA_COLUMNS]
    missing = set(feature_columns) - set(valid.columns)
    if missing:
        raise ValueError(f"validation 특징 열이 없습니다: {sorted(missing)}")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train.label)
    y_valid = encoder.transform(valid.label)

    weight_multipliers: dict[str, float] = {}
    if args.tune:
        best_params, weight_multipliers = tune_hyperparameters(
            train, feature_columns, y_train, args.n_trials, args.cv_folds,
            args.boost_class, args.boost_factor, args.tune_class_weights,
        )
    else:
        best_params = {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
        }

    if weight_multipliers:
        anchor_label = train["label"].value_counts().idxmax()
        print(f"[클래스별 가중치 탐색 결과] {anchor_label}=1.0(고정), {weight_multipliers}")
        final_sample_weight = make_sample_weight_multi(y_train, train["label"], weight_multipliers)
    elif args.boost_class:
        print(f"[가중치 실험] '{args.boost_class}' 클래스에 balanced 가중치 x{args.boost_factor} 적용")
        final_sample_weight = make_sample_weight(y_train, train["label"], args.boost_class, args.boost_factor)
    else:
        final_sample_weight = make_sample_weight(y_train, train["label"], None, 1.0)

    # 최종 모델은 (그룹 K-fold로 고른) 최적 파라미터로 train 전체에 대해 한 번만 학습하고,
    # 공식 VL/VS validation 세트에는 이 최종 모델에 대해서만 한 번 평가한다.
    model = XGBClassifier(**FIXED_PARAMS, num_class=len(encoder.classes_), **best_params)
    model.fit(train[feature_columns], y_train, sample_weight=final_sample_weight)

    prediction_id = model.predict(valid[feature_columns]).astype(int)
    valid["prediction"] = encoder.inverse_transform(prediction_id)
    valid["confidence"] = model.predict_proba(valid[feature_columns]).max(axis=1)
    labels = list(encoder.classes_)
    accuracy = accuracy_score(valid.label, valid.prediction)
    report = classification_report(valid.label, valid.prediction, labels=labels, target_names=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(valid.label, valid.prediction, labels=labels)

    valid.to_csv(OUTPUT_ROOT / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report).transpose().to_csv(OUTPUT_ROOT / "validation_report.csv", encoding="utf-8-sig")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(OUTPUT_ROOT / "validation_confusion_matrix.csv", encoding="utf-8-sig")
    figure, axis = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=axis, colorbar=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "validation_confusion_matrix.png", dpi=200)
    plt.close(figure)
    joblib.dump({"model": model, "label_encoder": encoder, "feature_columns": feature_columns}, OUTPUT_ROOT / "xgboost_fixed_10s.pkl")
    (OUTPUT_ROOT / "summary.txt").write_text(
        f"window_frames={WINDOW_FRAMES}\nwindow_seconds=10.0\n"
        f"tuned={args.tune}\nparams={best_params}\n"
        f"weight_multipliers={weight_multipliers}\n"
        f"boost_class={args.boost_class}\nboost_factor={args.boost_factor if args.boost_class else None}\n"
        f"train_windows={len(train)}\nvalid_windows={len(valid)}\naccuracy={accuracy:.4f}\n",
        encoding="utf-8",
    )
    print(f"train={len(train)}, valid={len(valid)}, accuracy={accuracy:.4f}")
    print(f"사용된 파라미터: {best_params}")
    print(f"결과 저장: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()