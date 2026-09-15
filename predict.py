"""FastAPI와 동일한 설정을 읽어 CLI에서 오디오를 추론합니다."""

import argparse
from pathlib import Path

import tensorflow as tf

from audio_runtime import load_config, predict_label, predict_score
from feature import load_audio


def predict_audio(audio_path, config_path=None):
    config = load_config(config_path)
    model = tf.keras.models.load_model(config["model_path"], compile=False)
    audio, sr = load_audio(audio_path)
    score = predict_score(model, audio, sr, config)
    return predict_label(score, config), score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    label, score = predict_audio(args.audio_path, args.config)
    print("Label:", label)
    print("Abnormal probability:", score)


if __name__ == "__main__":
    main()
