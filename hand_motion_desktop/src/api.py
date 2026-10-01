"""Unified Vision API for AUDIO and KEYPAD triggers.

Run:
    python -m uvicorn api:app --host 0.0.0.0 --port 8001

The canonical integration endpoint is POST /analyze/vision. It accepts multipart/form-data so
the caller sends both the recorded MP4 and trigger metadata in one request.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import tempfile
from typing import Literal

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from detector import analyze_video_for_silent_signal

app = FastAPI(title="Vision API", version="1.0.0")


class KeypadTriggerRequest(BaseModel):
    triggerId: str = Field(min_length=1, max_length=128)
    triggeredAt: datetime
    triggerType: Literal["KEYPAD"] = "KEYPAD"
    securityEventId: str | None = Field(default=None, max_length=128)
    deviceId: str | None = Field(default=None, max_length=128)


_recent_keypad_triggers: deque[dict] = deque(maxlen=100)
_behavior_service = None

# The established behavior pipeline lives at the project level.  This desktop
# API reuses its trained 10-second XGBoost bundle instead of maintaining a
# second, inconsistent implementation.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR_API_PATH = PROJECT_ROOT / "api" / "vision_api.py"
TARGET_BEHAVIOR_CLASSES = ("N1", "A18", "A20", "A21")


def _get_behavior_service():
    """Load the existing YOLO + Pose + XGBoost inference service on demand."""
    global _behavior_service
    if _behavior_service is not None:
        return _behavior_service
    if not BEHAVIOR_API_PATH.is_file():
        raise FileNotFoundError(f"Behavior inference module is missing: {BEHAVIOR_API_PATH}")
    spec = importlib.util.spec_from_file_location("project_behavior_vision", BEHAVIOR_API_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load behavior inference module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _behavior_service = module.InferenceService()
    return _behavior_service


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "vision-api", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "vision-api"}


@app.post("/triggers/keypad", status_code=202)
def receive_keypad_trigger(payload: KeypadTriggerRequest) -> dict:
    """Optional early keypad notification. The actual analysis is /analyze/vision."""
    trigger = payload.model_dump(mode="json")
    _recent_keypad_triggers.append(trigger)
    return {"status": "accepted", "trigger": trigger, "nextAction": "UPLOAD_VIDEO",
            "uploadEndpoint": "/analyze/vision"}


@app.get("/triggers/keypad/latest")
def latest_keypad_trigger() -> dict:
    if not _recent_keypad_triggers:
        raise HTTPException(status_code=404, detail="No keypad trigger has been received.")
    return {"status": "ok", "trigger": _recent_keypad_triggers[-1]}


def _video_metadata(video_path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to read video metadata.")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return (round(frame_count / fps, 3) if fps else 0.0, round(fps, 3), frame_count)


async def _analyze(file: UploadFile, trigger_id: str, trigger_type: str,
                   security_event_id: str | None, triggered_at: str):
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        return JSONResponse(status_code=415, content={"status": "error", "result": None,
            "message": "Only MP4 files are supported.", "code": "UNSUPPORTED_VIDEO_FORMAT"})

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        duration, fps, frame_count = _video_metadata(tmp_path)
        if trigger_type == "AUDIO":
            behavior_result = _get_behavior_service().analyze(tmp_path)
            prediction = behavior_result["prediction"]
            probabilities = behavior_result["classProbabilities"]
            # A17/A19 remain visible in logs but have no automatic risk mapping
            # for the current demonstration policy.
            if prediction not in TARGET_BEHAVIOR_CLASSES:
                print(f"[VISION] Out-of-scope behavior prediction: {prediction}")
            return {"status": "success", "result": {
                "trigger": {"triggerId": trigger_id, "triggerType": trigger_type,
                            "securityEventId": security_event_id, "triggeredAt": triggered_at},
                "analyzedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "video": {"fileName": file.filename, "durationSeconds": duration, "fps": fps,
                          "frameCount": frame_count},
                "behavior": {
                    "prediction": prediction,
                    "label": behavior_result["event"],
                    "confidence": behavior_result["confidence"],
                    "classProbabilities": {name: probabilities.get(name, 0.0)
                                           for name in TARGET_BEHAVIOR_CLASSES},
                },
                "visionEvents": [],
                "observations": {
                    "trackedPersonCount": 0,
                    "hasTracking": behavior_result["featureAvailability"]["hasTracking"],
                    "hasPose": behavior_result["featureAvailability"]["hasPose"],
                },
            }}
        signal_result = analyze_video_for_silent_signal(tmp_path)
    except ValueError:
        return JSONResponse(status_code=422, content={"status": "error", "result": None,
            "message": "Unable to read video metadata.", "code": "INVALID_VIDEO_FILE"})
    finally:
        tmp_path.unlink(missing_ok=True)
        await file.close()

    vision_events = []
    if signal_result.detected:
        vision_events.append({"eventType": "SILENT_SIGNAL", "confidence": signal_result.confidence,
            "detectedFrame": signal_result.detected_frame,
            "details": {"signalCode": signal_result.signal_code, "signalName": signal_result.signal_name}})

    return {"status": "success", "result": {
        "trigger": {"triggerId": trigger_id, "triggerType": trigger_type,
                    "securityEventId": security_event_id, "triggeredAt": triggered_at},
        "analyzedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": {"fileName": file.filename, "durationSeconds": duration, "fps": fps,
                  "frameCount": frame_count},
        "behavior": None, "visionEvents": vision_events,
        "observations": {"trackedPersonCount": 0, "hasTracking": False,
                         "hasPose": signal_result.hand_present_frames > 0},
    }}


@app.post("/analyze/vision")
async def analyze_vision(file: UploadFile = File(...), triggerId: str = Form(...),
                         triggerType: Literal["AUDIO", "KEYPAD"] = Form(...),
                         securityEventId: str | None = Form(None), triggeredAt: str = Form(...)):
    """Canonical multipart endpoint defined by the Vision API contract."""
    return await _analyze(file, triggerId, triggerType, securityEventId, triggeredAt)


@app.post("/analyze", include_in_schema=False)
async def analyze_compat(file: UploadFile = File(...), triggerId: str = Form(...),
                         triggerType: Literal["AUDIO", "KEYPAD"] = Form(...),
                         securityEventId: str | None = Form(None), triggeredAt: str = Form(...)):
    """Temporary compatibility alias for callers already switched to /analyze."""
    return await _analyze(file, triggerId, triggerType, securityEventId, triggeredAt)


@app.post("/analyze/silent-signal", include_in_schema=False)
async def legacy_analyze_silent_signal(file: UploadFile = File(...), triggerId: str = Form(...),
                                        triggerType: Literal["KEYPAD"] = Form("KEYPAD"),
                                        securityEventId: str | None = Form(None),
                                        triggeredAt: str = Form(...)):
    """Backward-compatible alias; new integrations must use /analyze."""
    return await _analyze(file, triggerId, triggerType, securityEventId, triggeredAt)
