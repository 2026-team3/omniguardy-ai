"""Recording group 단위 70/15/15 manifest 생성 CLI입니다."""

import argparse
from pathlib import Path

from audio_guard.application.prepare_dataset import prepare_manifest


def main():
    parser = argparse.ArgumentParser(description="누수 없는 dataset split을 생성합니다.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare_manifest(args.input, args.output, args.seed))


if __name__ == "__main__":
    main()
