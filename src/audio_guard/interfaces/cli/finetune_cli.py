"""v4 기반 현장 데이터 fine-tuning 명령행 인터페이스입니다."""

import argparse
from pathlib import Path

from audio_guard.application.finetune_model import finetune_model


def main():
    parser = argparse.ArgumentParser(description="v4 모델을 현장 데이터로 fine-tuning합니다.")
    parser.add_argument("--esc50-dir", type=Path, default=Path("data/ESC-50"))
    parser.add_argument("--field-data-dir", type=Path, required=True)
    parser.add_argument("--field-splits", type=Path, required=True)
    parser.add_argument(
        "--base-model", type=Path,
        default=Path("models/audio_model_v4.keras"))
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--abnormal-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--field-weight", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--gain-min", type=float, default=0.9)
    parser.add_argument("--gain-max", type=float, default=1.1)
    parser.add_argument("--max-time-shift-ms", type=int, default=50)
    args = parser.parse_args()
    saved = finetune_model(
        esc50_dir=args.esc50_dir,
        field_data_dir=args.field_data_dir,
        field_splits=args.field_splits,
        base_model=args.base_model,
        model_out=args.model_out,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        abnormal_weight_multiplier=args.abnormal_weight_multiplier,
        field_weight=args.field_weight,
        noise_std=args.noise_std,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        max_time_shift_ms=args.max_time_shift_ms,
    )
    print("저장 완료:", saved)


if __name__ == "__main__":
    main()
