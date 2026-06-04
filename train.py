import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models

from feature import audio_to_mel


# -----------------------
# 설정
# -----------------------
TARGET_CLASSES = [
    "door_wood_knock",
    "door_wood_creaks",
    "glass_breaking",
    "siren",
    "chainsaw",
    "footsteps"
]

BASE_DIR = Path(r"C:\Users\DS\Downloads\omniguardy-ai-audio\ESC-50")
CSV_PATH = BASE_DIR / "meta" / "esc50.csv"
AUDIO_DIR = BASE_DIR / "audio"


print("Loading CSV...")

df = pd.read_csv(CSV_PATH)

X, y = [], []

for _, row in df.iterrows():

    audio_path = AUDIO_DIR / row["filename"]

    audio, sr = sf.read(audio_path)

    # mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    mel = audio_to_mel(audio, sr)

    # shape 고정
    mel = np.resize(mel, (128, 128))

    X.append(mel)

    y.append(1 if row["category"] in TARGET_CLASSES else 0)


X = np.array(X, dtype=np.float32)[..., np.newaxis]
y = np.array(y, dtype=np.int32)

print("X shape:", X.shape)
print("Positive:", np.sum(y))
print("Negative:", len(y) - np.sum(y))


# -----------------------
# train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------
# class weight
# -----------------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {i: w for i, w in enumerate(weights)}


# -----------------------
# CNN model
# -----------------------
model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)


model.save("models/audio_model.keras")
print("Model saved.")