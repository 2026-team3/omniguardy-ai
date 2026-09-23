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
    parser.add_argument("--field-data-dir", type=Path)
    parser.add_argument("--field-splits", type=Path)
    parser.add_argument("--min-field-recall", type=float)
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=(0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90),
    )
    args = parser.parse_args()
    model = args.model or Path(f"models/audio_model_{args.pipeline}.keras")
    result = evaluate_model(
        pipeline_name=args.pipeline,
        esc50_dir=args.esc50_dir,
        model_path=model,
        results_dir=args.results_dir,
        min_recall=args.min_recall,
        field_data_dir=args.field_data_dir,
        field_splits=args.field_splits,
        thresholds=args.thresholds,
        min_field_recall=args.min_field_recall,
    )
    print(result)


if __name__ == "__main__":
    main()
