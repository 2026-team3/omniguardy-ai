"""FastAPI 애플리케이션 의존성을 조립합니다."""

import logging

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

config = load_config()
model_repository = KerasModelRepository(config["model_path"])
converter = FfmpegConverter(config["sample_rate"])
analyze_clip = AnalyzeClip(
    model_repository,
    config["pipeline"],
    config["threshold"],
    config["max_windows"],
)

app = FastAPI()
app.include_router(create_router(analyze_clip, converter, config))
