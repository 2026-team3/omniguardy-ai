"""모델 평가 명령행 인터페이스입니다."""

import argparse
from pathlib import Path

from audio_guard.application.evaluate_model import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="오디오 이상 탐지 모델을 평가합니다.")
    parser.add_argument("--pipeline", choices=("v3", "v4"), required=True)
    parser.add_argument("--esc50-dir", type=Path, default=Path("data/ESC-50"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--min-recall", type=float)
    args = parser.parse_args()
    model = args.model or Path(f"models/audio_model_{args.pipeline}.keras")
    result = evaluate_model(
        args.pipeline, args.esc50_dir, model, args.results_dir,
        args.min_recall)
    print(result)


if __name__ == "__main__":
    main()
