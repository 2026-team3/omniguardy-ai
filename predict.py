import librosa
import numpy as np
import tensorflow as tf

from feature import audio_to_mel

MODEL_PATH = "models/audio_model.keras"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(audio, sr):
    mel = audio_to_mel(audio, sr)

    # shape 안전 보정 (중요)
    if mel.shape != (128, 128):
        fixed = np.zeros((128, 128))
        h = min(128, mel.shape[0])
        w = min(128, mel.shape[1])
        fixed[:h, :w] = mel[:h, :w]
        mel = fixed

    return mel.astype(np.float32)


def predict_audio(audio_path):

    audio, sr = librosa.load(audio_path, sr=22050)

    # mono 보장
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    mel = preprocess(audio, sr)

    # model input shape
    mel = mel[np.newaxis, ..., np.newaxis]

    score = model.predict(mel, verbose=0)[0][0]

    label = "ABNORMAL" if score >= 0.5 else "NORMAL"

    return label, float(score)