"""tracking 결과와 키패드 ROI를 영상 위에 그려 검증한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.keypad_rules import identify_operator_by_roi, load_keypad_roi


def read_tracking(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("tracking 파일은 csv, json/jsonl 또는 parquet여야 합니다.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracking", type=Path)
    parser.add_argument("--roi", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("keypad_roi_preview.mp4"))
    parser.add_argument("--image-every", type=int, default=0, help="N이면 N 프레임마다 PNG도 저장")
    args = parser.parse_args()
    tracking = read_tracking(args.tracking)
    roi = load_keypad_roi(args.roi)
    if roi is None:
        raise SystemExit("유효한 ROI JSON이 없습니다.")
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracking.attrs.update(frame_width=width, frame_height=height)
    operator_id = identify_operator_by_roi(tracking, roi, fps)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    by_frame = {int(frame): group for frame, group in tracking.groupby("frame")}
    roi_start, roi_end = (int(roi["x1"] * width), int(roi["y1"] * height)), (int(roi["x2"] * width), int(roi["y2"] * height))
    image_dir = args.output.with_suffix("")
    if args.image_every:
        image_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.rectangle(frame, roi_start, roi_end, (0, 255, 255), 2)
        cv2.putText(frame, "keypad ROI", (roi_start[0], max(20, roi_start[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows = by_frame.get(index)
        if rows is not None:
            for row in rows.itertuples():
                color = (0, 255, 0) if row.track_id == operator_id else (255, 128, 0)
                cv2.rectangle(frame, (int(row.x1), int(row.y1)), (int(row.x2), int(row.y2)), color, 2)
                label = f"id={row.track_id}" + (" OPERATOR" if row.track_id == operator_id else "")
                cv2.putText(frame, label, (int(row.x1), max(20, int(row.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        writer.write(frame)
        if args.image_every and index % args.image_every == 0:
            cv2.imwrite(str(image_dir / f"frame_{index:05d}.png"), frame)
        index += 1
    capture.release()
    writer.release()
    print(f"Saved {args.output}; ROI operator track: {operator_id}")


if __name__ == "__main__":
    main()
