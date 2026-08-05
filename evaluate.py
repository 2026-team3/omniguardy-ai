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


MODEL_PATH = "models/audio_model.keras"
CSV_PATH = "meta/esc50.csv"
AUDIO_DIR = "audio"

TARGET_CLASSES = [
    "door_wood_knock",
    "door_wood_creaks",
    "glass_breaking",
    "siren",
    "chainsaw",
    "footsteps"
]


def audio_to_melspec(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050
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


def load_dataset():

    X = []
    y = []

    df = pd.read_csv(CSV_PATH)

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
            (128, 128)
        )

        X.append(mel)

        y.append(
            1 if row["category"] in TARGET_CLASSES else 0
        )


    X = np.array(
        X,
        dtype=np.float32
    )

    X = X[..., np.newaxis]

    y = np.array(
        y,
        dtype=np.int32
    )

    return X, y



model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


X_valid, y_valid = load_dataset()


print("Data shape:", X_valid.shape)
print("Positive:", np.sum(y_valid))
print("Negative:", len(y_valid) - np.sum(y_valid))


y_prob = model.predict(
    X_valid
)


y_pred = (
    y_prob > 0.5
).astype(int)


print("====================")
print("Evaluation Result")
print("====================")

print(
    "Accuracy :",
    accuracy_score(y_valid, y_pred)
)

print(
    "Precision:",
    precision_score(y_valid, y_pred)
)

print(
    "Recall   :",
    recall_score(y_valid, y_pred)
)

print(
    "F1-score :",
    f1_score(y_valid, y_pred)
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_valid,
        y_pred
    )
)

