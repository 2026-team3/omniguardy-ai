"""10초 MP4를 분석하는 독립 Vision FastAPI.

Spring Boot contract:
    POST /analyze/vision
    multipart/form-data, field name=file
"""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Literal

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import torch
from ultralytics import YOLO
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.extract_behavior_features import summarize_block
from tracking.merge_features import POSE_COLUMNS, pose_summary
from tracking.keypad_rules import (
    ENABLE_WEAPON_DETECTION,
    detect_rear_approach,
    detect_weapon_proximity,
)

# hand_motion_desktop/src는 gesture_rules.py/model_loader.py를 상대 import가 아니라
# 최상위 모듈처럼(from gesture_rules import ...) 작성해뒀기 때문에, 패키지로 import하지
# 않고 sys.path에 그 폴더 자체를 추가해서 detector 모듈을 바로 불러온다.
HAND_MOTION_SRC = PROJECT_ROOT / "hand_motion_desktop" / "src"
if str(HAND_MOTION_SRC) not in sys.path:
    sys.path.insert(0, str(HAND_MOTION_SRC))

from detector import analyze_video_for_silent_signal  # noqa: E402  (경로 삽입 후 import)


MODEL_BUNDLE_PATH = PROJECT_ROOT / "results/evaluation/fixed_10s_model/xgboost_fixed_10s.pkl"
YOLO_PATH = PROJECT_ROOT / "tracking/yolov8n.pt"
POSE_PATH = PROJECT_ROOT / "models/pose_landmarker_lite.task"
FRAME_INTERVAL = 10
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
LABEL_DETAILS = {
    "N1": "normal_entrance_activity",
    "A17": "repeated_doorbell_press",
    "A18": "repeated_door_opening_attempt",
    "A19": "repeated_door_kicking",
    "A20": "attempting_to_look_inside",
    "A21": "long_or_repeated_door_knocking",
}
# REAR_CLOSE_APPROACH_SUSPECTED / WEAPON_PROXIMITY_SUSPECTED(실험적) 판정 로직과
# 그 임계값들은 tracking/keypad_rules.py에 있다 (학습이 필요 없는 순수 규칙 기반이라
# 모델 로딩 담당인 InferenceService와는 분리해뒀다).


class KeypadTriggerRequest(BaseModel):
    """Metadata sent after the backend validates a keypad event.

    The keypad PIN itself must never be sent to this API. Reuse ``triggerId``
    later if the Raspberry Pi uploads a related video.
    """

    triggerId: str = Field(min_length=1, max_length=128)
    triggeredAt: datetime
    triggerType: Literal["KEYPAD"] = "KEYPAD"
    securityEventId: str | None = Field(default=None, max_length=128)
    deviceId: str | None = Field(default=None, max_length=128)


_recent_keypad_triggers: deque[dict] = deque(maxlen=100)


class InferenceService:
    def __init__(self) -> None:
        if not MODEL_BUNDLE_PATH.is_file() or not YOLO_PATH.is_file() or not POSE_PATH.is_file():
            raise FileNotFoundError("Vision inference model file is missing.")
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        self.classifier = bundle["model"]
        self.encoder = bundle["label_encoder"]
        self.feature_columns = bundle["feature_columns"]
        self.yolo = YOLO(str(YOLO_PATH))
        # vid_stride는 절대 1보다 크게 쓰지 않는다. BoT-SORT는 상태를 유지하는
        # tracker라 프레임을 건너뛰면 track이 끊기거나 ID가 바뀌어서, 학습 시
        # "매 프레임 연속 추적" 기준으로 만든 속도/이동거리 feature 분포가 추론
        # 시 분포와 달라진다 (실측: stride3에서 91.7%->58.3%로 폭락).
        # 속도를 올려야 하면 yolo_imgsz(입력 해상도)로만 조절한다 — 프레임을
        # 안 건너뛰므로 tracking 연속성과 feature 분포가 학습 때와 동일하게 유지된다.
        configured_stride = int(os.getenv("VISION_YOLO_VID_STRIDE", "1"))
        if configured_stride != 1:
            print(
                f"[vision_api] WARNING: VISION_YOLO_VID_STRIDE={configured_stride}는 "
                "무시하고 1로 강제합니다 (tracking 기반 feature 불일치 방지)."
            )
        self.yolo_vid_stride = 1
        # 640 -> 480 정도로 낮추면 tracking 연속성은 유지한 채 추론 속도를 올릴 수
        # 있다. 낮추기 전/후로 반드시 evaluation/test_vision_api_validation.py로
        # 정확도 재확인할 것.
        self.yolo_imgsz = max(160, int(os.getenv("VISION_YOLO_IMGSZ", "640")))
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[vision_api] YOLO device={device_name}, imgsz={self.yolo_imgsz}, vid_stride={self.yolo_vid_stride}")

    def _tracking_features(self, video_path: Path) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        device: int | str = 0 if torch.cuda.is_available() else "cpu"
        results = self.yolo.track(
            source=str(video_path), tracker="botsort.yaml", persist=False, save=False,
            conf=0.3, classes=[0], stream=True, device=device, verbose=False,
            vid_stride=self.yolo_vid_stride, imgsz=self.yolo_imgsz,
        )
        for frame_idx, result in enumerate(results):
            if result.boxes is None:
                continue
            for box in result.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                rows.append({
                    "frame": frame_idx * self.yolo_vid_stride, "track_id": int(box.id[0]),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                })
        return pd.DataFrame(rows)

    def _pose_features(self, video_path: Path) -> pd.DataFrame:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Unable to open uploaded video.")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        previous: dict[int, tuple[int, float, float, float, float]] = {}
        rows: list[dict[str, Any]] = []
        base_options = mp.tasks.BaseOptions
        options = vision.PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=str(POSE_PATH)),
            running_mode=vision.RunningMode.VIDEO, num_poses=2,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        frame = -1
        try:
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    frame += 1
                    if frame % FRAME_INTERVAL:
                        continue
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                        int(frame / fps * 1000),
                    )
                    values = []
                    for pose_index, landmarks in enumerate(result.pose_landmarks):
                        left_shoulder, right_shoulder, right_wrist = landmarks[11], landmarks[12], landmarks[16]
                        center_x = (left_shoulder.x + right_shoulder.x) / 2
                        center_y = (left_shoulder.y + right_shoulder.y) / 2
                        hand_motion = body_motion = 0.0
                        if pose_index in previous:
                            prev_frame, wrist_x, wrist_y, body_x, body_y = previous[pose_index]
                            gap = frame - prev_frame
                            if gap:
                                hand_motion = math.dist((right_wrist.x, right_wrist.y), (wrist_x, wrist_y)) / gap
                                body_motion = math.dist((center_x, center_y), (body_x, body_y)) / gap
                        previous[pose_index] = (frame, right_wrist.x, right_wrist.y, center_x, center_y)
                        values.append((
                            hand_motion, body_motion,
                            math.dist((right_wrist.x, right_wrist.y), (right_shoulder.x, right_shoulder.y)),
                            math.degrees(math.atan2(right_shoulder.y - left_shoulder.y, right_shoulder.x - left_shoulder.x)),
                        ))
                    if values:
                        array = np.asarray(values, dtype=float)
                        rows.append({
                            "frame": frame, "hand_motion": array[:, 0].mean(), "body_motion": array[:, 1].mean(),
                            "arm_extension": array[:, 2].mean(), "upper_body_angle": array[:, 3].mean(),
                        })
        finally:
            capture.release()
        return pd.DataFrame(rows)

    def analyze(self, video_path: Path) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if fps <= 0 or frame_count <= 0:
            raise ValueError("Invalid MP4 metadata.")
        tracking, pose = self._tracking_features(video_path), self._pose_features(video_path)
        behavior = summarize_block(tracking) if {"frame", "track_id", "x1", "y1", "x2", "y2"}.issubset(tracking.columns) else {"has_tracking": 0, "tracked_person_count": 0}
        pose_values = pose_summary(pose, 0, frame_count - 1) if {"frame", *POSE_COLUMNS}.issubset(pose.columns) else {"has_pose": 0, "pose_frame_count": 0}
        features = pd.DataFrame([{**behavior, **pose_values}])
        for column in self.feature_columns:
            if column not in features:
                features[column] = 0.0
        probabilities = self.classifier.predict_proba(features[self.feature_columns])[0]
        prediction_id = int(np.argmax(probabilities))
        label = str(self.encoder.inverse_transform([prediction_id])[0])
        probability_map = {name: round(float(value), 6) for name, value in zip(self.encoder.classes_, probabilities)}
        duration = frame_count / fps
        warnings = []
        if not 5 <= duration <= 10.5:
            warnings.append("Expected a 5-10 second video; this result is outside the trained input duration.")
        return {
            "prediction": label, "event": LABEL_DETAILS[label],
            "confidence": probability_map[label], "classProbabilities": probability_map,
            "videoMetadata": {"durationSeconds": round(duration, 3), "fps": round(fps, 3), "frameCount": frame_count, "width": width, "height": height},
            "featureAvailability": {"hasTracking": bool(behavior["has_tracking"]), "hasPose": bool(pose_values["has_pose"])},
            "trackedPersonCount": int(behavior.get("tracked_person_count", 0)),
            "inferenceConfiguration": {"yoloVidStride": self.yolo_vid_stride, "yoloImageSize": self.yolo_imgsz, "poseFrameInterval": FRAME_INTERVAL},
            "warnings": warnings,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keypad integration must work even when this desktop has no camera and no
    # one has requested heavyweight video inference yet. Load models on demand.
    app.state.service = None
    yield


app = FastAPI(title="Vision Inference API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": "xgboost_fixed_10s",
        "videoInferenceLoaded": str(app.state.service is not None).lower(),
    }


@app.post("/triggers/keypad", status_code=202)
def receive_keypad_trigger(payload: KeypadTriggerRequest) -> dict:
    """Accept a keypad trigger without requiring a webcam or Raspberry Pi."""

    trigger = payload.model_dump(mode="json")
    _recent_keypad_triggers.append(trigger)
    return {
        "status": "accepted",
        "trigger": trigger,
        "nextAction": "RECORD_AND_UPLOAD_VIDEO",
        "message": "Keypad trigger received. Do not send the keypad PIN.",
    }


@app.get("/triggers/keypad/latest")
def latest_keypad_trigger() -> dict:
    """Return the latest in-memory trigger for LAN integration testing."""

    if not _recent_keypad_triggers:
        raise HTTPException(status_code=404, detail="No keypad trigger has been received.")
    return {"status": "ok", "trigger": _recent_keypad_triggers[-1]}


def _video_metadata(video_path: Path, file_name: str) -> dict[str, Any]:
    """팀 스키마의 video 필드(fileName/durationSeconds/fps/frameCount)를 채운다."""
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError("Invalid MP4 metadata.")
    return {
        "fileName": file_name,
        "durationSeconds": round(frame_count / fps, 3),
        "fps": round(fps, 3),
        "frameCount": frame_count,
        "width": width,
        "height": height,
    }


@app.post("/analyze/vision")
async def analyze_vision(
    file: UploadFile = File(...),
    triggerId: str = Form(...),
    triggerType: Literal["AUDIO", "KEYPAD"] = Form(...),
    securityEventId: str | None = Form(default=None),
    triggeredAt: datetime = Form(...),
) -> dict[str, Any]:
    """팀이 확정한 공통 응답 스키마(trigger/video/behavior/visionEvents/observations)로
    응답한다. AUDIO면 XGBoost 행동분류, KEYPAD면 Silent Signal 규칙 판정을 태운다."""
    if not (file.filename or "").lower().endswith(".mp4"):
        raise HTTPException(status_code=415, detail="Only .mp4 files are supported.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temporary_file:
            temporary_path = Path(temporary_file.name)
            shutil.copyfileobj(file.file, temporary_file)
        if temporary_path.stat().st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Video exceeds 100 MB limit.")

        video = _video_metadata(temporary_path, file.filename)

        behavior: dict[str, Any] | None = None
        vision_events: list[dict[str, Any]] = []
        observations = {"trackedPersonCount": 0, "hasTracking": False, "hasPose": False}

        if triggerType == "AUDIO":
            if app.state.service is None:
                app.state.service = InferenceService()
            analysis = app.state.service.analyze(temporary_path)
            behavior = {
                "prediction": analysis["prediction"],
                "label": analysis["event"],
                "confidence": analysis["confidence"],
                "classProbabilities": analysis["classProbabilities"],
            }
            observations = {
                "trackedPersonCount": analysis["trackedPersonCount"],
                "hasTracking": analysis["featureAvailability"]["hasTracking"],
                "hasPose": analysis["featureAvailability"]["hasPose"],
            }
            # A17/A19는 위험도 정책 매핑 대상이 아니라서 Agent가 알아서 무시하지만,
            # 확인용으로 서버 로그에는 남겨둔다.
            if analysis["prediction"] in ("A17", "A19"):
                print(f"[vision_api] AUDIO prediction={analysis['prediction']} (시연 매핑 대상 아님, 로그만)")

        else:  # KEYPAD
            if app.state.service is None:
                app.state.service = InferenceService()
            service = app.state.service

            signal = analyze_video_for_silent_signal(temporary_path)
            # SILENT_SIGNAL(손동작)뿐 아니라 REAR_CLOSE_APPROACH_SUSPECTED(후방 접근)와
            # 실험적 흉기 근접 감지도 같은 person tracking 결과를 공유해서 쓴다.
            tracking = service._tracking_features(temporary_path)
            # 정규화 ROI를 실제 프레임 좌표로 바꾸기 위한 원본 크기다.
            tracking.attrs["frame_width"] = video["width"]
            tracking.attrs["frame_height"] = video["height"]

            observations["hasPose"] = signal.hand_present_frames > 0
            observations["hasTracking"] = not tracking.empty
            observations["trackedPersonCount"] = (
                int(tracking["track_id"].nunique()) if not tracking.empty else 0
            )

            if signal.detected:
                vision_events.append({
                    "eventType": "SILENT_SIGNAL",
                    "confidence": signal.confidence,
                    "detectedFrame": signal.detected_frame,
                    "details": {
                        "signalCode": signal.signal_code,
                        "signalName": signal.signal_name,
                    },
                })

            rear_approach_event = detect_rear_approach(tracking, video["fps"])
            if rear_approach_event is not None:
                vision_events.append(rear_approach_event)

            if ENABLE_WEAPON_DETECTION:
                device: int | str = 0 if torch.cuda.is_available() else "cpu"
                weapon_event = detect_weapon_proximity(
                    temporary_path, service.yolo, service.yolo_imgsz, tracking, device,
                )
                if weapon_event is not None:
                    vision_events.append(weapon_event)

        result = {
            "trigger": {
                "triggerId": triggerId,
                "triggerType": triggerType,
                "securityEventId": securityEventId,
                "triggeredAt": triggeredAt.isoformat(),
            },
            "analyzedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            "video": video,
            "behavior": behavior,
            "visionEvents": vision_events,
            "observations": observations,
        }
        return {"status": "success", "result": result}
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await file.close()
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
