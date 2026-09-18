"""오디오 런타임 설정을 읽고 검증합니다."""

import json
import os
from pathlib import Path

from audio_guard.domain.pipeline.base import SAMPLE_RATE
from audio_guard.labels import LABELS, TARGET_CLASSES

DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "audio_config.json"


def load_config(path=None):
    config_path = Path(path or os.getenv("AUDIO_CONFIG_PATH", DEFAULT_CONFIG))
    with config_path.open(encoding="utf-8") as source:
        config = json.load(source)
    if config.get("labels") != LABELS:
        raise ValueError("Audio label mapping does not match the runtime")
    if set(config.get("esc_abnormal_categories", [])) != TARGET_CLASSES:
        raise ValueError("ESC abnormal category mapping does not match the runtime")
    if config.get("sample_rate") != SAMPLE_RATE:
        raise ValueError("Audio sample rate does not match the runtime")
    if config.get("pipeline") not in {"v3", "v4", "v5"}:
        raise ValueError("Unknown audio preprocessing pipeline")
    if not 0 < float(config.get("threshold", -1)) < 1:
        raise ValueError("Audio threshold must be between 0 and 1")
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    config["model_path"] = str(model_path.resolve())
    config["threshold"] = float(config["threshold"])
    config["max_windows"] = int(config.get("max_windows", 8))
    if config["max_windows"] < 1:
        raise ValueError("max_windows must be positive")
    return config
