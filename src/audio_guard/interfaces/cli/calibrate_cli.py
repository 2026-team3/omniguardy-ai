"""Validation class별 threshold calibration CLI입니다."""

import argparse
from pathlib import Path

from audio_guard.application.calibrate_thresholds import calibrate_thresholds


def main():
    parser = argparse.ArgumentParser(description="Knock/Handle threshold를 확정합니다.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--min-precision", type=float)
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--hop-seconds", type=float, default=0.2)
    parser.add_argument("--cooldown-seconds", type=float, default=1.0)
    args = parser.parse_args()
    result = calibrate_thresholds(
        args.manifest,
        args.model,
        args.config_out,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        cooldown_seconds=args.cooldown_seconds,
    )
    print(result)


if __name__ == "__main__":
    main()
