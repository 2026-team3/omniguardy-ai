"""Vision 분석 결과를 Agent 연동용 이벤트로 만들고 전송한다."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from camera.classifier import WindowAnalysis, risk_level


def build_event(analysis: WindowAnalysis, window_seconds: float, inference_ms: float) -> dict[str, Any]:
    """Vision과 Agent 사이에서 공유할 JSON 이벤트를 생성한다."""
    if analysis.status == "no_person":
        event_name = "vision.no_person"
    elif analysis.status == "insufficient_tracking":
        event_name = "vision.insufficient_data"
    elif analysis.label == "N1":
        event_name = "vision.window.analyzed"
    else:
        event_name = "vision.suspicious_behavior_detected"

    return {
        "schema_version": "1.0",
        "event_id": str(uuid.uuid4()),
        "source": "vision",
        "event": event_name,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "window_seconds": window_seconds,
        "prediction": {
            "label": analysis.label,
            "behavior_confidence": round(analysis.confidence, 4),
            "risk_level": risk_level(analysis.label, analysis.confidence),
            "status": analysis.status,
        },
        "detection": {
            "person_count": analysis.person_count,
            "detection_confidence": round(analysis.detection_confidence, 4),
        },
        "runtime": {"device": "cpu", "inference_ms": round(inference_ms, 2)},
    }


def append_event(path: Path, event: dict[str, Any]) -> None:
    """모든 분석 결과를 JSON Lines 형식으로 로컬에 남긴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def publish_event(url: str | None, event: dict[str, Any]) -> None:
    """Agent API 주소가 주어졌을 때만 HTTP POST로 이벤트를 전달한다."""
    if not url:
        return
    try:
        response = requests.post(url, json=event, timeout=3)
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"[경고] Agent 이벤트 전송 실패: {error}", file=sys.stderr)
