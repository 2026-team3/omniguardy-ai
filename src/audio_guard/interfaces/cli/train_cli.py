# 문 이벤트 학습 CLI 모델

import argparse
from pathlib import Path

from audio_guard.application.train_model import train_model


def main():
    parser = argparse.ArgumentParser(description="Door Event CNN을 학습합니다.")

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)

    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/door_event_model.keras"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--hop-seconds", type=float, default=0.2)

    args = parser.parse_args()

    saved = train_model(
        manifest_path=args.manifest,
        events_path=args.events,
        model_out=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
    )

    print("저장 완료:", saved)


if __name__ == "__main__":
    main()
