"""실시간 Vision 파이프라인의 실행 루프와 명령행 옵션을 관리한다."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from camera.classifier import WindowAnalysis, classify_window, load_classifier
from camera.config import DEFAULT_CLASSIFIER, DEFAULT_EVENT_LOG, DEFAULT_POSE_MODEL, DEFAULT_YOLO_MODEL, PROJECT_ROOT
from camera.events import append_event, build_event, publish_event
from camera.features import create_pose_landmarker, extract_pose_features
from camera.sources import create_frame_source


# Ultralytics 설정을 사용자 프로필이 아닌 프로젝트 내부에 저장한다.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))

from ultralytics import YOLO


def draw_overlay(frame: np.ndarray, fps: float, label: str, confidence: float) -> np.ndarray:
    """시연과 성능 확인을 위해 화면에 핵심 상태를 표시한다."""
    display = frame.copy()
    cv2.putText(display, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, f"Behavior: {label} ({confidence:.2f})", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    return display


def run(args: argparse.Namespace) -> None:
    """프레임 수집, 추론, 이벤트 발행을 반복 실행한다."""
    classifier = load_classifier(Path(args.classifier))
    yolo_path = Path(args.yolo_model)
    if not yolo_path.is_file():
        raise FileNotFoundError(f"YOLO 모델이 없습니다: {yolo_path}")

    source = create_frame_source(args.source, args.width, args.height)
    model = YOLO(str(yolo_path))
    tracking_records: list[dict[str, float]] = []
    pose_records: list[dict[str, float]] = []
    previous_pose: dict[int, tuple[int, float, float, float, float]] = {}
    last_analysis = WindowAnalysis("N1", 0.0, 0, 0.0, {}, "waiting")
    frame_index = 0
    started_at = time.monotonic()
    window_started_at = started_at
    previous_frame_at = started_at
    fps_history: deque[float] = deque(maxlen=30)
    event_path = Path(args.event_log)

    try:
        with create_pose_landmarker(Path(args.pose_model)) as landmarker:
            while True:
                ok, frame = source.read()
                if not ok or frame is None:
                    break
                frame_index += 1
                now = time.monotonic()
                frame_delta = max(now - previous_frame_at, 1e-6)
                previous_frame_at = now
                fps_history.append(1.0 / frame_delta)

                result = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0], conf=args.detection_confidence, device="cpu", verbose=False)[0]
                annotated = result.plot()
                if result.boxes is not None:
                    for box in result.boxes:
                        if box.id is None:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        tracking_records.append({"frame": float(frame_index), "track_id": float(box.id[0]), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "confidence": float(box.conf[0])})

                if frame_index % args.pose_interval == 0:
                    pose_feature = extract_pose_features(frame, landmarker, int((now - started_at) * 1000), frame_index, previous_pose)
                    if pose_feature is not None:
                        pose_records.append(pose_feature)

                if now - window_started_at >= args.window_seconds:
                    inference_started_at = time.perf_counter()
                    last_analysis = classify_window(tracking_records, pose_records, classifier)
                    event = build_event(last_analysis, args.window_seconds, (time.perf_counter() - inference_started_at) * 1000)
                    append_event(event_path, event)
                    print(json.dumps(event, ensure_ascii=False))
                    if last_analysis.status == "classified" and (args.publish_normal or last_analysis.label != "N1"):
                        publish_event(args.agent_url, event)
                    tracking_records.clear()
                    pose_records.clear()
                    window_started_at = now

                if not args.no_display:
                    fps = float(np.mean(fps_history)) if fps_history else 0.0
                    cv2.imshow("Realtime Vision", draw_overlay(annotated, fps, last_analysis.label, last_analysis.confidence))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        source.release()
        cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    """실행 환경별로 바꿀 수 있는 명령행 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description="YOLOv8n 기반 실시간 행동 추론")
    parser.add_argument("--source", default="0", help="웹캠 번호, 영상 경로 또는 picamera")
    parser.add_argument("--width", type=int, default=640, help="카메라 입력 가로 해상도")
    parser.add_argument("--height", type=int, default=480, help="카메라 입력 세로 해상도")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="행동 분류 시간 창 길이")
    parser.add_argument("--pose-interval", type=int, default=10, help="Pose 분석 프레임 간격")
    parser.add_argument("--detection-confidence", type=float, default=0.3, help="YOLO 사람 탐지 confidence 기준")
    parser.add_argument("--yolo-model", default=str(DEFAULT_YOLO_MODEL), help="YOLOv8n 모델 경로")
    parser.add_argument("--pose-model", default=str(DEFAULT_POSE_MODEL), help="MediaPipe Pose 모델 경로")
    parser.add_argument("--classifier", default=str(DEFAULT_CLASSIFIER), help="XGBoost 분류 모델 경로")
    parser.add_argument("--event-log", default=str(DEFAULT_EVENT_LOG), help="이벤트 로그 경로")
    parser.add_argument("--agent-url", help="Agent 이벤트 수신 HTTP URL")
    parser.add_argument("--publish-normal", action="store_true", help="정상(N1) 이벤트도 Agent에 전송")
    parser.add_argument("--no-display", action="store_true", help="화면 창 없이 실행")
    return parser.parse_args()
