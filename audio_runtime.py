"""평가·API·CLI에서 공통으로 사용하는 모델·전처리·임계치 설정을 관리합니다."""

import json
import os
from pathlib import Path

import numpy as np

from feature import SAMPLE_RATE, audio_to_mel, audio_to_mel_bag, audio_to_mel_windows
from audio_labels import LABELS, TARGET_CLASSES

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "audio_config.json"


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
    config["model_path"] = str(model_path)
    config["threshold"] = float(config["threshold"])
    config["max_windows"] = int(config.get("max_windows", 8))
    if config["max_windows"] < 1:
        raise ValueError("max_windows must be positive")
    return config


def model_input(audio, sr, config):
    if sr != config["sample_rate"]:
        raise ValueError("Audio must be resampled before inference")
    if config["pipeline"] == "v5":
        return audio_to_mel_bag(audio, sr, config["max_windows"])[None, ..., None]
    if config["pipeline"] == "v4":
        return audio_to_mel_windows(audio, sr)[..., None]
    return audio_to_mel(audio, sr)[None, ..., None]


def predict_score(model, audio, sr, config):
    scores = np.asarray(model.predict(model_input(audio, sr, config), verbose=0))
    return float(scores.reshape(-1).max())


def predict_label(score, config):
    return "abnormal" if score >= config["threshold"] else "normal"
