"""Door Event 런타임 설정을 읽고 검증합니다."""

import json
import os
from pathlib import Path

from audio_guard.domain.pipeline.base import SAMPLE_RATE
from audio_guard.labels import EVENT_CLASSES, LABELS

DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "audio_config.json"


def load_config(path=None):
    config_path = Path(path or os.getenv("AUDIO_CONFIG_PATH", DEFAULT_CONFIG))
    with config_path.open(encoding="utf-8") as source:
        config = json.load(source)
    if config.get("labels") != LABELS:
        raise ValueError("Audio label mapping does not match the runtime")
    if config.get("sample_rate") != SAMPLE_RATE:
        raise ValueError("Audio sample rate does not match the runtime")
    if config.get("pipeline") != "door_event":
        raise ValueError("Unknown audio preprocessing pipeline")
    thresholds = config.get("thresholds", {})
    if set(thresholds) != set(EVENT_CLASSES):
        raise ValueError("Thresholds must be configured for knock and handle")
    if any(not 0 < float(value) < 1 for value in thresholds.values()):
        raise ValueError("Audio thresholds must be between 0 and 1")
    for field in ("window_seconds", "hop_seconds"):
        if float(config.get(field, 0)) <= 0:
            raise ValueError(f"{field} must be positive")
    if float(config["hop_seconds"]) > float(config["window_seconds"]):
        raise ValueError("hop_seconds must not exceed window_seconds")
    if float(config.get("cooldown_seconds", -1)) < 0:
        raise ValueError("cooldown_seconds must be non-negative")
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    config["model_path"] = str(model_path.resolve())
    config["thresholds"] = {
        name: float(thresholds[name]) for name in EVENT_CLASSES
    }
    for field in ("window_seconds", "hop_seconds", "cooldown_seconds"):
        config[field] = float(config[field])
    return config
