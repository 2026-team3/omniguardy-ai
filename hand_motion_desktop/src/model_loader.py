"""MediaPipe Hand Landmarker 모델 파일을 준비한다.
파일이 없으면 공식 배포 URL에서 자동으로 내려받는다 (Pose Landmarker와 같은 방식)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


def ensure_model() -> Path:
    """모델 파일이 없으면 다운로드하고, 최종 경로를 반환한다."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"hand_landmarker.task 다운로드 중... -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("다운로드 완료")
    return MODEL_PATH
