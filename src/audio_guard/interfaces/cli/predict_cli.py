"""단일 오디오 파일 예측 명령행 인터페이스입니다."""

import argparse
from pathlib import Path

from audio_guard.application.analyze_clip import AnalyzeClip
from audio_guard.config import load_config
from audio_guard.domain.audio_clip import AudioClip
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from audio_guard.infrastructure.ml.keras_model_repository import KerasModelRepository


def predict_audio(audio_path, config_path=None):
    config = load_config(config_path)
    repository = KerasModelRepository(config["model_path"])
    analyzer = AnalyzeClip(
        repository,
        config["pipeline"],
        config["thresholds"],
        config["window_seconds"],
        config["hop_seconds"],
        config["cooldown_seconds"],
    )
    audio, sample_rate = load_audio(audio_path, config["sample_rate"])
    return analyzer.execute(AudioClip(audio, sample_rate))


def main():
    parser = argparse.ArgumentParser(description="오디오 파일의 이상 위험도를 예측합니다.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    result = predict_audio(args.audio_path, args.config)
    print("이벤트:", result.status)
    print("클래스:", result.predicted_class)
    print("확률:", result.probabilities)
    print("검출 window 시작:", result.window_start_seconds)


if __name__ == "__main__":
    main()
