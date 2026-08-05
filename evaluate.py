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


# ============================
# Model
# ============================

# 비교할 모델 변경 가능
# audio_model.keras = 기존 ESC-50 모델
# audio_model_v2.keras = fine tuning 모델

MODEL_PATH = "models/audio_model_v2.keras"


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


TARGET_CLASSES = [
    "door_wood_knock",
    "door_wood_creaks",
    "glass_breaking",
    "siren",
    "chainsaw",
    "footsteps"
]


# ============================
# Feature
# ============================

def audio_to_melspec(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )


    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )


    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )


    mel_db = librosa.util.fix_length(
        mel_db,
        size=128,
        axis=1
    )


    return mel_db



# ============================
# Dataset
# ============================

def load_dataset():

    X = []
    y = []


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


        if not os.path.exists(audio_path):
            continue


        mel = audio_to_melspec(
            audio_path
        )


        mel = np.resize(
            mel,
            (128,128)
        )


        X.append(
            mel
        )


        label = (
            1
            if row["category"] in TARGET_CLASSES
            else 0
        )


        y.append(label)



    X = np.array(
        X,
        dtype=np.float32
    )


    X = X[..., np.newaxis]


    y = np.array(
        y,
        dtype=np.int32
    )


    return X,y



# ============================
# Main
# ============================


print("==============================")
print("Loading model")
print("==============================")


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


print("Model loaded")
print(
    "Input:",
    model.input_shape
)



X_test, y_test = load_dataset()



print("==============================")
print("Dataset")
print("==============================")

print(
    "X shape:",
    X_test.shape
)

print(
    "Positive:",
    np.sum(y_test)
)

print(
    "Negative:",
    len(y_test)-np.sum(y_test)
)



print("==============================")
print("Predict")
print("==============================")


prob = model.predict(
    X_test,
    verbose=0
)


pred = (
    prob >= 0.5
).astype(int)



print("==============================")
print("RESULT")
print("==============================")


print(
    "Accuracy:",
    accuracy_score(
        y_test,
        pred
    )
)


print(
    "Precision:",
    precision_score(
        y_test,
        pred,
        zero_division=0
    )
)


print(
    "Recall:",
    recall_score(
        y_test,
        pred,
        zero_division=0
    )
)


print(
    "F1:",
    f1_score(
        y_test,
        pred,
        zero_division=0
    )
)


print(
    "Confusion Matrix"
)


print(
    confusion_matrix(
        y_test,
        pred
    )
)