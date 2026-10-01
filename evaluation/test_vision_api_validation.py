"""실제 HTTP Vision API로 validation 10초 구간을 소규모 E2E 평가한다.

각 클래스에서 annotation 블록 내부의 완전한 10초 구간을 두 개씩 선택한다.
임시 MP4를 생성해 /analyze/vision에 업로드하며, 원본 영상과 기존 모델은 수정하지 않는다.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time

import cv2
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = PROJECT_ROOT / "splits/annotations/valid_annotations.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/evaluation/vision_api_validation_smoke"
WINDOW_FRAMES = 300
LABELS = ["A17", "A18", "A19", "A20", "A21", "N1"]


def select_windows(per_label: int) -> list[dict]:
    annotations = pd.read_csv(ANNOTATIONS)
    selected: list[dict] = []
    for label in LABELS:
        choices = []
        for _, block in annotations[annotations.label == label].iterrows():
            start, end = int(block.start_frame), int(block.end_frame)
            for window_start in range(start, end - WINDOW_FRAMES + 2, WINDOW_FRAMES):
                choices.append({
                    "label": label, "video": block.video, "video_path": block.video_path,
                    "start_frame": window_start, "end_frame": window_start + WINDOW_FRAMES - 1,
                })
                if len(choices) == per_label:
                    break
            if len(choices) == per_label:
                break
        if len(choices) != per_label:
            raise ValueError(f"{label}: {per_label}개의 완전한 10초 window를 찾지 못했습니다.")
        selected.extend(choices)
    return selected


def write_clip(source: Path, target: Path, start_frame: int) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    written = 0
    while written < WINDOW_FRAMES:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    capture.release()
    if written != WINDOW_FRAMES:
        raise ValueError(f"Expected {WINDOW_FRAMES} frames, got {written}: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8012/analyze/vision")
    parser.add_argument("--per-label", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="vision_api_validation_") as directory:
        temp_root = Path(directory)
        for index, item in enumerate(select_windows(args.per_label), start=1):
            clip = temp_root / f"{index}_{item['label']}.mp4"
            try:
                write_clip(Path(item["video_path"]), clip, item["start_frame"])
                started = time.perf_counter()
                # /analyze/vision이 팀 스키마로 바뀌면서 triggerId/triggerType/triggeredAt이
                # 필수 Form 필드가 됐다. 이 스크립트는 행동분류(AUDIO) 성능만 보므로
                # triggerType=AUDIO 고정으로 보낸다.
                form_data = {
                    "triggerId": f"eval-{index}",
                    "triggerType": "AUDIO",
                    "triggeredAt": datetime.now(timezone.utc).isoformat(),
                }
                with clip.open("rb") as video:
                    response = requests.post(
                        args.url,
                        files={"file": (clip.name, video, "video/mp4")},
                        data=form_data,
                        timeout=240,
                    )
                elapsed = time.perf_counter() - started
                response.raise_for_status()
                behavior = response.json()["result"]["behavior"]
                rows.append({
                    **item, "prediction": behavior["prediction"], "confidence": behavior["confidence"],
                    "duration_seconds": elapsed, "http_status": response.status_code, "error": "",
                })
                print(f"{index}: {item['label']} -> {behavior['prediction']} ({elapsed:.1f}s)")
            except Exception as error:
                rows.append({**item, "prediction": "ERROR", "confidence": 0.0, "duration_seconds": 0.0, "http_status": 0, "error": str(error)})
                print(f"{index}: {item['label']} -> ERROR: {error}")

    results = pd.DataFrame(rows)
    results.to_csv(output_root / "api_predictions.csv", index=False, encoding="utf-8-sig")
    successful = results[results.prediction != "ERROR"]
    if successful.empty:
        raise RuntimeError("All API requests failed.")
    report = classification_report(successful.label, successful.prediction, labels=LABELS, target_names=LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(successful.label, successful.prediction, labels=LABELS)
    accuracy = accuracy_score(successful.label, successful.prediction)
    pd.DataFrame(report).transpose().to_csv(output_root / "api_validation_report.csv", encoding="utf-8-sig")
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(output_root / "api_confusion_matrix.csv", encoding="utf-8-sig")
    (output_root / "summary.txt").write_text(
        f"samples={len(successful)}\naccuracy={accuracy:.4f}\n"
        f"mean_inference_seconds={successful.duration_seconds.mean():.2f}\n",
        encoding="utf-8",
    )
    print(f"accuracy={accuracy:.4f}, mean_inference_seconds={successful.duration_seconds.mean():.2f}")


if __name__ == "__main__":
    main()