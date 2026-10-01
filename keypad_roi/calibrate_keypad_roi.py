"""키패드 사각형 ROI를 클릭해 정규화 JSON으로 저장한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "tracking" / "keypad_roi.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Click two diagonal keypad ROI corners.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="보정에 사용할 영상")
    source.add_argument("--camera", type=int, help="웹캠 인덱스")
    parser.add_argument("--frame", type=int, default=0, help="영상 프레임 번호")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if args.video:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit("프레임을 읽을 수 없습니다.")

    points: list[tuple[int, int]] = []
    window = "Keypad ROI: click two diagonal corners (r=reset, q=cancel)"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        preview = frame.copy()
        for point in points:
            cv2.circle(preview, point, 5, (0, 255, 255), -1)
        if len(points) == 2:
            cv2.rectangle(preview, points[0], points[1], (0, 255, 0), 2)
            cv2.putText(preview, "Enter: save", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) == 2:
            break
        if key == ord("r"):
            points.clear()
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()

    height, width = frame.shape[:2]
    x_values = sorted(point[0] / width for point in points)
    y_values = sorted(point[1] / height for point in points)
    payload = {
        "version": 1,
        "type": "rectangle",
        "points": [{"x": round(x_values[0], 6), "y": round(y_values[0], 6)},
                   {"x": round(x_values[1], 6), "y": round(y_values[1], 6)}],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved normalized keypad ROI to {args.output}")


if __name__ == "__main__":
    main()
