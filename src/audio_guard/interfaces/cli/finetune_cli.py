"""모델 추가 학습 명령행 인터페이스입니다."""

import argparse
from pathlib import Path

from audio_guard.application.finetune_model import finetune_model


def main():
    parser = argparse.ArgumentParser(description="현장 녹음으로 모델을 추가 학습합니다.")
    parser.add_argument("--pipeline", choices=("v3", "v4", "v5"), required=True)
    parser.add_argument("--field-data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-windows", type=int, default=8)
    args = parser.parse_args()
    saved = finetune_model(
        args.pipeline, args.field_data_dir, args.base_model,
        args.model_out, args.epochs, args.max_windows)
    print("저장 완료:", saved)


if __name__ == "__main__":
    main()
