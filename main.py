from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import numpy as np
import librosa
import subprocess
import tempfile
import os
import tensorflow as tf

keras = tf.keras
app = FastAPI()


# -----------------------------
# Response Schema
# -----------------------------
class PredictResponse(BaseModel):
    status: str
    probability: float


# -----------------------------
# Model Load
# -----------------------------
MODEL_PATH = os.path.join("models", "audio_model.keras")
model = keras.models.load_model(MODEL_PATH, compile=False)


# -----------------------------
# Feature Extraction
# -----------------------------
def audio_to_mel(audio, sr):
    N_MELS = 128
    MAX_LEN = 128

    target_length = sr
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    if mel.shape[1] < MAX_LEN:
        mel = np.pad(mel, ((0, 0), (0, MAX_LEN - mel.shape[1])))
    else:
        mel = mel[:, :MAX_LEN]

    return mel.astype(np.float32)


def preprocess(audio, sr):
    mel = audio_to_mel(audio, sr)

    if mel.shape != (128, 128):
        mel_fixed = np.zeros((128, 128))
        h = min(128, mel.shape[0])
        w = min(128, mel.shape[1])
        mel_fixed[:h, :w] = mel[:h, :w]
        mel = mel_fixed

    return mel.astype(np.float32)


# -----------------------------
# Prediction
# -----------------------------
def predict_risk(audio, sr):
    mel = preprocess(audio, sr)
    mel = np.expand_dims(mel, axis=(0, -1))

    pred = model.predict(mel, verbose=0)
    return float(pred[0][0])


def sliding_predict(audio, sr):
    block_size = sr * 5

    probs = []

    for start in range(0, len(audio), block_size):
        chunk = audio[start:start + block_size]

        if len(chunk) < sr:
            continue

        probs.append(predict_risk(chunk, sr))

    if not probs:
        return {"status": "normal", "final_prob": 0.0}

    final_prob = max(probs)

    return {
        "status": "abnormal" if final_prob >= 0.5 else "normal",
        "final_prob": float(final_prob)
    }


# -----------------------------
# API
# -----------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        suffix = os.path.splitext(file.filename)[-1] or ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        wav_path = os.path.splitext(tmp_path)[0] + ".wav"

        # ✅ FIX: ffmpeg 반드시 PATH 대신 풀경로 사용
        FFMPEG_PATH = r"C:\Users\DS\Downloads\ffmpeg-2026-06-01-git-bf608f16fd-essentials_build\ffmpeg-2026-06-01-git-bf608f16fd-essentials_build\bin\ffmpeg.exe"

        result = subprocess.run([
            FFMPEG_PATH,
            "-y",
            "-i", tmp_path,
            "-ar", "22050",
            "-ac", "1",
            wav_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print("FFMPEG ERROR:", result.stderr)
            return {"status": "error_ffmpeg", "probability": 0.0}

        audio, sr = librosa.load(wav_path, sr=22050)

        result = sliding_predict(audio, sr)

        os.remove(tmp_path)
        os.remove(wav_path)

        return {
            "status": result["status"],
            "probability": result["final_prob"]
        }

    except Exception as e:
        print("API ERROR:", e)
        return {"status": "error", "probability": 0.0}