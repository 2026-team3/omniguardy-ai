"""v4 기반 현장 데이터 fine-tuning 명령행 인터페이스입니다."""

import argparse
from pathlib import Path

from audio_guard.application.finetune_model import finetune_model


def main():
    parser = argparse.ArgumentParser(description="v4 모델을 현장 데이터로 fine-tuning합니다.")
    parser.add_argument("--esc50-dir", type=Path, default=Path("data/ESC-50"))
    parser.add_argument("--field-data-dir", type=Path, required=True)
    parser.add_argument("--field-splits", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("models/audio_model_v5.keras"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--abnormal-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--field-weight", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.002)
    args = parser.parse_args()
    saved = finetune_model(
        args.esc50_dir, args.field_data_dir, args.field_splits,
        args.base_model, args.model_out, args.epochs, args.learning_rate,
        args.abnormal_weight_multiplier, args.field_weight, args.noise_std)
    print("저장 완료:", saved)


if __name__ == "__main__":
    main()
