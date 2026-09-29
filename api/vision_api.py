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
from fastapi import FastAPI, File, HTTPException, UploadFile
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
        # 기본값은 학습 때의 frame-level 결과와 같은 모든 프레임 처리다.
        # 운영 환경에서는 환경변수로 속도/정확도 trade-off를 실험할 수 있다.
        self.yolo_vid_stride = max(1, int(os.getenv("VISION_YOLO_VID_STRIDE", "1")))
        self.yolo_imgsz = max(160, int(os.getenv("VISION_YOLO_IMGSZ", "640")))

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


@app.post("/analyze/vision")
async def analyze_vision(file: UploadFile = File(...)) -> dict[str, Any]:
    if not (file.filename or "").lower().endswith(".mp4"):
        raise HTTPException(status_code=415, detail="Only .mp4 files are supported.")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temporary_file:
            temporary_path = Path(temporary_file.name)
            shutil.copyfileobj(file.file, temporary_file)
        if temporary_path.stat().st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Video exceeds 100 MB limit.")
        if app.state.service is None:
            app.state.service = InferenceService()
        result = app.state.service.analyze(temporary_path)
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
