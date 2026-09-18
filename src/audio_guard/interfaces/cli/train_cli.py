"""모델 학습 명령행 인터페이스입니다."""

import argparse
from pathlib import Path

from audio_guard.application.train_model import train_model


def main():
    parser = argparse.ArgumentParser(description="오디오 이상 탐지 모델을 학습합니다.")
    parser.add_argument("--pipeline", choices=("v3", "v4", "v5"), required=True)
    parser.add_argument("--esc50-dir", type=Path, default=Path("data/ESC-50"))
    parser.add_argument("--field-data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--field-splits", type=Path)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--field-weight", type=float, default=5.0)
    args = parser.parse_args()
    if args.max_windows < 1:
        parser.error("--max-windows must be positive")
    if args.field_weight <= 0:
        parser.error("--field-weight must be positive")
    default_names = {
        "v3": "audio_model.keras",
        "v4": "audio_model_v4.keras",
        "v5": "audio_model_v5.keras",
    }
    model_out = args.model_out or Path("models") / default_names[args.pipeline]
    saved = train_model(
        args.pipeline, args.esc50_dir, args.field_data_dir, model_out,
        args.epochs, args.max_windows, args.field_weight, args.field_splits)
    print("저장 완료:", saved)


if __name__ == "__main__":
    main()
