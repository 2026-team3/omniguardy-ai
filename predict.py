import librosa
import numpy as np
import tensorflow as tf
import os

from feature import audio_to_mel


MODEL_PATH = "models/audio_model_v2.keras"


print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded")
print("Input shape:", model.input_shape)



def preprocess(audio, sr):

    mel = audio_to_mel(
        audio,
        sr
    )

    print(
        "Mel shape:",
        mel.shape
    )

    return mel.astype(
        np.float32
    )



def predict_audio(audio_path):

    print(
        "Loading:",
        audio_path
    )


    audio, sr = librosa.load(
        audio_path,
        sr=22050,
        mono=True
    )


    print(
        "Audio length:",
        len(audio)
    )


    mel = preprocess(
        audio,
        sr
    )


    mel = mel[np.newaxis, ..., np.newaxis]


    print(
        "Input:",
        mel.shape
    )


    pred = model.predict(
        mel,
        verbose=0
    )


    score = float(
        pred[0][0]
    )


    label = (
        "ABNORMAL"
        if score >= 0.5
        else "NORMAL"
    )


    return label, score




if __name__ == "__main__":


    path = (
        r"dataset/normal/"
        r"아파트 도어락 현관문.wav"
    )


    if not os.path.exists(path):

        print(
            "File not found:",
            path
        )

        exit()


    label, score = predict_audio(
        path
    )


    print("================")
    print("RESULT")
    print("================")

    print(
        "Label:",
        label
    )

    print(
        "Score:",
        score
    )