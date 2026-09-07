from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import numpy as np
import librosa
import subprocess
import tempfile
import os
import tensorflow as tf
import logging

from feature import audio_to_mel


# =============================
# Logging
# =============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =============================
# FastAPI
# =============================

keras = tf.keras

app = FastAPI()


# =============================
# 설정
# =============================

MODEL_PATH = os.path.join(
    "models",
    "audio_model_v3_weight_1_1.keras"
)

SAMPLE_RATE = 22050

# 현재 V3 기준 임시 Threshold
# 이후 0.7 ~ 0.9 세부 실험 후 변경 가능
THRESHOLD = 0.7


# =============================
# Response Schema
# =============================

class PredictResponse(BaseModel):
    status: str
    probability: float


# =============================
# Model Load
# =============================

logger.info("Loading model...")

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

logger.info(
    f"MODEL LOADED : {MODEL_PATH}"
)


# =============================
# Preprocess
# =============================

def preprocess(audio, sr):

    """
    Audio waveform
        ↓
    Mel Spectrogram
        ↓
    (128, 128)
    """

    mel = audio_to_mel(
        audio,
        sr
    )

    return mel.astype(
        np.float32
    )


# =============================
# Prediction
# =============================

def predict_risk(audio, sr):

    """
    하나의 3초 Audio Chunk를
    Normal / Abnormal로 분석한다.
    """

    # -----------------------------
    # Mel Spectrogram
    # -----------------------------

    mel = preprocess(
        audio,
        sr
    )


    logger.info(
        "========== AUDIO INFO =========="
    )

    logger.info(
        f"audio length : {len(audio)}"
    )

    logger.info(
        f"sample rate  : {sr}"
    )


    logger.info(
        "========== MEL INFO =========="
    )

    logger.info(
        f"mel shape : {mel.shape}"
    )

    logger.info(
        f"mel min   : {mel.min()}"
    )

    logger.info(
        f"mel max   : {mel.max()}"
    )

    logger.info(
        f"mel mean  : {mel.mean()}"
    )


    # -----------------------------
    # CNN Input Shape
    #
    # (128,128)
    # ↓
    # (1,128,128,1)
    # -----------------------------

    mel_input = np.expand_dims(
        mel,
        axis=(0, -1)
    )


    # -----------------------------
    # Model Prediction
    # -----------------------------

    pred = model.predict(
        mel_input,
        verbose=0
    )


    score = float(
        pred[0][0]
    )


    # -----------------------------
    # Threshold
    # -----------------------------

    status = (
        "abnormal"
        if score >= THRESHOLD
        else "normal"
    )


    logger.info(
        "========== RESULT =========="
    )

    logger.info(
        f"raw score : {score}"
    )

    logger.info(
        f"threshold : {THRESHOLD}"
    )

    logger.info(
        f"prediction : {status.upper()}"
    )

    logger.info(
        "============================"
    )


    return (
        status,
        score
    )


# =============================
# API
# =============================

@app.post(
    "/predict",
    response_model=PredictResponse
)
async def predict(
    file: UploadFile = File(...)
):

    """
    Spring에서 전달받은
    약 3초 길이의 오디오 파일을 분석한다.

    Flow

    Raspberry Pi
        ↓
    Spring
        ↓
    POST /predict
        ↓
    WAV 변환
        ↓
    Mel Spectrogram
        ↓
    Audio CNN V3
        ↓
    Normal / Abnormal
    """

    logger.info(
        f"REQUEST RECEIVED : {file.filename}"
    )


    tmp_path = None
    wav_path = None


    try:

        # =============================
        # Upload File Read
        # =============================

        contents = await file.read()


        if not contents:

            logger.warning(
                "Empty audio file received"
            )

            return {
                "status": "error_empty_file",
                "probability": 0.0
            }


        # =============================
        # 임시 원본 파일 저장
        # =============================

        suffix = (
            os.path.splitext(
                file.filename
            )[-1]
            or ".m4a"
        )


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(
                contents
            )

            tmp_path = tmp.name


        # =============================
        # WAV 변환 경로
        # =============================

        wav_path = (
            os.path.splitext(
                tmp_path
            )[0]
            + ".wav"
        )


        # =============================
        # FFmpeg
        # =============================

        FFMPEG_PATH = (
            r"C:\Users\DS\Downloads"
            r"\ffmpeg-2026-06-01-git-bf608f16fd-essentials_build"
            r"\ffmpeg-2026-06-01-git-bf608f16fd-essentials_build"
            r"\bin\ffmpeg.exe"
        )


        result = subprocess.run(
            [
                FFMPEG_PATH,
                "-y",
                "-i",
                tmp_path,

                # 학습 조건과 동일
                "-ar",
                str(SAMPLE_RATE),

                # Mono
                "-ac",
                "1",

                wav_path
            ],
            capture_output=True,
            text=True
        )


        # =============================
        # FFmpeg 실패
        # =============================

        if result.returncode != 0:

            logger.error(
                "FFmpeg conversion failed"
            )

            logger.error(
                result.stderr
            )

            return {
                "status": "error_ffmpeg",
                "probability": 0.0
            }


        # =============================
        # WAV Load
        # =============================

        audio, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE,
            mono=True
        )


        # =============================
        # Audio 길이 확인
        # =============================

        duration = (
            len(audio)
            / sr
        )


        logger.info(
            f"Audio duration : {duration:.3f}s"
        )


        # 너무 짧은 오디오는 분석하지 않음
        if duration < 1.0:

            logger.warning(
                "Audio is too short"
            )

            return {
                "status": "error_short_audio",
                "probability": 0.0
            }


        # =============================
        # Prediction
        #
        # Spring이 이미 약 3초 단위로
        # 전달하므로 Sliding Window 없음
        # =============================

        status, probability = predict_risk(
            audio,
            sr
        )


        logger.info(
            "========== FINAL RESULT =========="
        )

        logger.info(
            f"status      : {status}"
        )

        logger.info(
            f"probability : {probability}"
        )

        logger.info(
            "=================================="
        )


        # =============================
        # Response
        # =============================

        return {

            "status":
                status,

            "probability":
                probability
        }


    # =============================
    # Exception
    # =============================

    except Exception as e:

        logger.exception(
            f"API ERROR : {e}"
        )


        return {

            "status":
                "error",

            "probability":
                0.0
        }


    # =============================
    # 임시 파일 삭제
    # =============================

    finally:

        if (
            tmp_path
            and os.path.exists(
                tmp_path
            )
        ):

            os.remove(
                tmp_path
            )


        if (
            wav_path
            and os.path.exists(
                wav_path
            )
        ):

            os.remove(
                wav_path
            )


# =============================
# Health Check
# =============================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": os.path.basename(
            MODEL_PATH
        ),
        "threshold": THRESHOLD
    }