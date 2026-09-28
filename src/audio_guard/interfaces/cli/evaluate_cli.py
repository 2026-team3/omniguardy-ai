"""잠긴 test split 최종 평가 CLI입니다."""

import argparse
from pathlib import Path

from audio_guard.application.evaluate_model import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="확정된 Door Event 모델을 평가합니다.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    print(evaluate_model(args.manifest, args.model, args.config, args.results_dir))


if __name__ == "__main__":
    main()
