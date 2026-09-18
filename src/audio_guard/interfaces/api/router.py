"""오디오 예측과 상태 확인 FastAPI 라우터입니다."""

import logging
import os
import tempfile

from fastapi import APIRouter, File, UploadFile

from audio_guard.domain.audio_clip import AudioClip
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from audio_guard.interfaces.api.schemas import PredictResponse

logger = logging.getLogger(__name__)


def create_router(analyze_clip, converter, config):
    router = APIRouter()

    @router.post("/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...)):
        logger.info("REQUEST RECEIVED : %s", file.filename)
        tmp_path = None
        wav_path = None
        try:
            contents = await file.read()
            if not contents:
                return {"status": "error_empty_file", "probability": 0.0}
            suffix = os.path.splitext(file.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                temporary.write(contents)
                tmp_path = temporary.name
            wav_path = os.path.splitext(tmp_path)[0] + "_converted.wav"
            if not converter.convert_to_wav(tmp_path, wav_path):
                return {"status": "error_ffmpeg", "probability": 0.0}
            audio, sample_rate = load_audio(wav_path, config["sample_rate"])
            clip = AudioClip(audio, sample_rate)
            if clip.duration < 1.0:
                return {"status": "error_short_audio", "probability": 0.0}
            status, probability = analyze_clip.execute(clip)
            return {"status": status, "probability": probability}
        except Exception as error:
            logger.exception("API ERROR : %s", error)
            return {"status": "error", "probability": 0.0}
        finally:
            for path in (tmp_path, wav_path):
                if path and os.path.exists(path):
                    os.remove(path)

    @router.get("/health")
    def health():
        return {
            "status": "ok",
            "model": os.path.basename(config["model_path"]),
            "pipeline": config["pipeline"],
            "threshold": config["threshold"],
            "sample_rate": config["sample_rate"],
            "ffmpeg": converter.executable,
        }

    return router
