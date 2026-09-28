"""FastAPI 애플리케이션 의존성을 조립합니다."""

import logging
from pathlib import Path

from fastapi import FastAPI

from audio_guard.application.analyze_clip import AnalyzeClip
from audio_guard.config import load_config
from audio_guard.infrastructure.audio.ffmpeg_converter import FfmpegConverter
from audio_guard.infrastructure.ml.keras_model_repository import KerasModelRepository
from audio_guard.interfaces.api.router import create_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

config = load_config()
logger.info(
    "Door Event runtime: model=%s sample_rate=%s thresholds=%s window=%s hop=%s cooldown=%s",
    Path(config["model_path"]).name,
    config["sample_rate"],
    config["thresholds"],
    config["window_seconds"],
    config["hop_seconds"],
    config["cooldown_seconds"],
)
model_repository = KerasModelRepository(config["model_path"])
converter = FfmpegConverter(config["sample_rate"])
analyze_clip = AnalyzeClip(
    model_repository,
    config["pipeline"],
    config["thresholds"],
    config["window_seconds"],
    config["hop_seconds"],
    config["cooldown_seconds"],
)

app = FastAPI()
app.include_router(create_router(analyze_clip, converter, config))
