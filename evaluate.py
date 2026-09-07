import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from feature import audio_to_mel


# ============================
# Model
# ============================

# 기존 모델
# MODEL_PATH = "models/audio_model.keras"

# Fine-tuning 모델
MODEL_PATH = "models/audio_model_v3_weight_1_1.keras"


# ============================
# ESC-50 Path
# ============================

BASE_DIR = "..\\ESC-50"

CSV_PATH = os.path.join(
    BASE_DIR,
    "meta",
    "esc50.csv"
)

AUDIO_DIR = os.path.join(
    BASE_DIR,
    "audio"
)


# ============================
# Abnormal Classes
# ============================

TARGET_CLASSES = [
    "door_wood_knock",
    "door_wood_creaks",
    "glass_breaking",
    "siren",
    "chainsaw",
    "footsteps"
]


# ============================
# Threshold
# ============================

THRESHOLDS = [
    0.3,
    0.4,
    0.5,
    0.6,
    0.7
]


# ============================
# Dataset Loading
# ============================

def load_dataset():

    X = []
    y = []

    filenames = []
    categories = []

    print("==============================")
    print("Loading ESC-50")
    print("==============================")

    df = pd.read_csv(
        CSV_PATH
    )

    for _, row in df.iterrows():

        audio_path = os.path.join(
            AUDIO_DIR,
            row["filename"]
        )

        if not os.path.exists(
            audio_path
        ):
            continue


        # ============================
        # Audio Load
        # ============================

        audio, sr = librosa.load(
            audio_path,
            sr=22050,
            mono=True
        )


        # ============================
        # Mel Spectrogram
        #
        # 학습과 동일한 feature.py 사용
        # ============================

        mel = audio_to_mel(
            audio,
            sr
        )


        X.append(
            mel
        )


        # ============================
        # Label
        # ============================

        label = (
            1
            if row["category"] in TARGET_CLASSES
            else 0
        )

        y.append(
            label
        )


        # ============================
        # Error Analysis 정보
        # ============================

        filenames.append(
            row["filename"]
        )

        categories.append(
            row["category"]
        )


    # ============================
    # numpy 변환
    # ============================

    X = np.array(
        X,
        dtype=np.float32
    )

    # CNN Input:
    # (samples, 128, 128, 1)

    X = X[..., np.newaxis]


    y = np.array(
        y,
        dtype=np.int32
    )


    return (
        X,
        y,
        filenames,
        categories
    )


# ============================
# Threshold Evaluation
# ============================

def evaluate_threshold(
    y_true,
    probabilities,
    threshold
):

    pred = (
        probabilities >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_true,
        pred
    )


    precision = precision_score(
        y_true,
        pred,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        pred,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        pred,
        zero_division=0
    )


    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    )


    tn, fp, fn, tp = cm.ravel()


    return {
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn
    }


# ============================
# Error Analysis
# ============================

def error_analysis(
    y_true,
    probabilities,
    filenames,
    categories,
    threshold
):

    pred = (
        probabilities >= threshold
    ).astype(int)


    errors = []


    for i in range(
        len(y_true)
    ):

        # ============================
        # False Positive
        #
        # 실제 Normal
        # 예측 Abnormal
        # ============================

        if (
            y_true[i] == 0
            and pred[i] == 1
        ):

            error_type = "FP"


        # ============================
        # False Negative
        #
        # 실제 Abnormal
        # 예측 Normal
        # ============================

        elif (
            y_true[i] == 1
            and pred[i] == 0
        ):

            error_type = "FN"


        else:
            continue


        errors.append({
            "filename":
                filenames[i],

            "category":
                categories[i],

            "true_label":
                int(y_true[i]),

            "predicted_label":
                int(pred[i]),

            "abnormal_probability":
                float(probabilities[i]),

            "error_type":
                error_type
        })


    error_df = pd.DataFrame(
        errors
    )


    return error_df


# ============================
# Main
# ============================

print("==============================")
print("Loading Model")
print("==============================")


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


print("Model loaded")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Input shape:",
    model.input_shape
)


# ============================
# Dataset
# ============================

X_test, y_test, filenames, categories = (
    load_dataset()
)


print("\n")
print("==============================")
print("Dataset")
print("==============================")


print(
    "X shape:",
    X_test.shape
)


print(
    "Total:",
    len(y_test)
)


print(
    "Positive (Abnormal):",
    np.sum(
        y_test == 1
    )
)


print(
    "Negative (Normal):",
    np.sum(
        y_test == 0
    )
)


# ============================
# Prediction
# ============================

print("\n")
print("==============================")
print("Predict")
print("==============================")


prob = model.predict(
    X_test,
    verbose=0
)


# 모델 출력이
# (2000, 1)인 경우
# (2000,)으로 변경

prob = np.asarray(
    prob
).reshape(-1)


print(
    "Prediction shape:",
    prob.shape
)


print(
    "Probability min:",
    np.min(prob)
)


print(
    "Probability max:",
    np.max(prob)
)


print(
    "Probability mean:",
    np.mean(prob)
)


# ============================
# Threshold Sweep
# ============================

print("\n")
print("==============================")
print("THRESHOLD SWEEP")
print("==============================")


results = []


for threshold in THRESHOLDS:

    result = evaluate_threshold(
        y_test,
        prob,
        threshold
    )


    results.append(
        result
    )


    print("\n------------------------------")

    print(
        "Threshold:",
        threshold
    )

    print(
        "Accuracy :",
        f"{result['Accuracy']:.4f}"
    )

    print(
        "Precision:",
        f"{result['Precision']:.4f}"
    )

    print(
        "Recall   :",
        f"{result['Recall']:.4f}"
    )

    print(
        "F1       :",
        f"{result['F1']:.4f}"
    )

    print(
        "TP:",
        result["TP"]
    )

    print(
        "FN:",
        result["FN"]
    )

    print(
        "FP:",
        result["FP"]
    )

    print(
        "TN:",
        result["TN"]
    )


# ============================
# Threshold Result 저장
# ============================

result_df = pd.DataFrame(
    results
)


result_df.to_csv(
    "threshold_results.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\n")
print("==============================")
print("Threshold Result Table")
print("==============================")


print(
    result_df.to_string(
        index=False
    )
)


print(
    "\nSaved: threshold_results.csv"
)


# ============================
# F1 기준 Best Threshold
#
# 참고용일 뿐
# 최종 선택은 Recall/FN 우선
# ============================

best_index = (
    result_df["F1"].idxmax()
)


best_result = (
    result_df.loc[
        best_index
    ]
)


best_threshold = float(
    best_result[
        "Threshold"
    ]
)


print("\n")
print("==============================")
print("BEST THRESHOLD BY F1")
print("==============================")


print(
    "Threshold:",
    best_threshold
)

print(
    "Accuracy:",
    f"{best_result['Accuracy']:.4f}"
)

print(
    "Precision:",
    f"{best_result['Precision']:.4f}"
)

print(
    "Recall:",
    f"{best_result['Recall']:.4f}"
)

print(
    "F1:",
    f"{best_result['F1']:.4f}"
)

print(
    "TP:",
    int(
        best_result["TP"]
    )
)

print(
    "FN:",
    int(
        best_result["FN"]
    )
)

print(
    "FP:",
    int(
        best_result["FP"]
    )
)

print(
    "TN:",
    int(
        best_result["TN"]
    )
)


# ============================
# Error Analysis
#
# 우선 기존 기준 0.5에서 분석
# ============================

ANALYSIS_THRESHOLD = 0.5


error_df = error_analysis(
    y_test,
    prob,
    filenames,
    categories,
    ANALYSIS_THRESHOLD
)


error_df.to_csv(
    "error_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\n")
print("==============================")
print(
    "ERROR ANALYSIS"
)
print(
    "Threshold:",
    ANALYSIS_THRESHOLD
)
print("==============================")


# ============================
# False Positive
# ============================

fp_df = error_df[
    error_df[
        "error_type"
    ] == "FP"
]


print("\n")
print("------------------------------")
print("False Positive")
print("------------------------------")


print(
    "Total FP:",
    len(fp_df)
)


if len(fp_df) > 0:

    fp_count = (
        fp_df["category"]
        .value_counts()
    )

    print("\nFP by category:")

    print(
        fp_count.to_string()
    )


# ============================
# False Negative
# ============================

fn_df = error_df[
    error_df[
        "error_type"
    ] == "FN"
]


print("\n")
print("------------------------------")
print("False Negative")
print("------------------------------")


print(
    "Total FN:",
    len(fn_df)
)


if len(fn_df) > 0:

    fn_count = (
        fn_df["category"]
        .value_counts()
    )

    print("\nFN by category:")

    print(
        fn_count.to_string()
    )


print(
    "\nSaved: error_analysis.csv"
)


# ============================
# Error Summary 저장
# ============================

summary_rows = []


if len(fp_df) > 0:

    for category, count in (
        fp_df[
            "category"
        ]
        .value_counts()
        .items()
    ):

        summary_rows.append({

            "error_type":
                "FP",

            "category":
                category,

            "count":
                count
        })


if len(fn_df) > 0:

    for category, count in (
        fn_df[
            "category"
        ]
        .value_counts()
        .items()
    ):

        summary_rows.append({

            "error_type":
                "FN",

            "category":
                category,

            "count":
                count
        })


summary_df = pd.DataFrame(
    summary_rows,
    columns=[
        "error_type",
        "category",
        "count"
    ]
)


summary_df.to_csv(
    "error_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


print(
    "Saved: error_summary.csv"
)


print("\n")
print("==============================")
print("Evaluation Finished")
print("==============================")