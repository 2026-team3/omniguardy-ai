from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import numpy as np
import librosa
import subprocess
import tempfile
import os
import tensorflow as tf
import logging
import shutil

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

app = FastAPI()


# =============================
# 설정
# =============================

MODEL_PATH = os.path.join(
    "models",
    "audio_model_v3_weight_1_1.keras"
)

SAMPLE_RATE = 22050

# 현재 V3 기준 Threshold
THRESHOLD = 0.7

# FFmpeg
FFMPEG_PATH = shutil.which("ffmpeg")


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

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

logger.info(
    f"MODEL LOADED : {MODEL_PATH}"
)

logger.info(
    f"THRESHOLD : {THRESHOLD}"
)

logger.info(
    f"FFMPEG PATH : {FFMPEG_PATH}"
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
    하나의 약 3초 Audio Chunk를
    Normal / Abnormal로 분석
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
    # CNN Input
    #
    # (128, 128)
    # ↓
    # (1, 128, 128, 1)
    # -----------------------------

    mel_input = np.expand_dims(
        mel,
        axis=(0, -1)
    )


    # -----------------------------
    # Prediction
    # -----------------------------

    pred = model.predict(
        mel_input,
        verbose=0
    )

    score = float(
        pred.reshape(-1)[0]
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
        f"raw score  : {score}"
    )

    logger.info(
        f"threshold  : {THRESHOLD}"
    )

    logger.info(
        f"prediction : {status.upper()}"
    )

    logger.info(
        "============================"
    )


    return status, score


# =============================
# Audio Conversion
# =============================

def convert_to_wav(
    input_path,
    output_path
):

    """
    입력 오디오를
    22050Hz / Mono WAV로 변환
    """

    if not FFMPEG_PATH:

        raise RuntimeError(
            "FFmpeg를 찾을 수 없습니다."
        )


    result = subprocess.run(
        [
            FFMPEG_PATH,
            "-y",
            "-i",
            input_path,

            "-ar",
            str(SAMPLE_RATE),

            "-ac",
            "1",

            "-c:a",
            "pcm_s16le",

            output_path
        ],
        capture_output=True,
        text=True
    )


    if result.returncode != 0:

        logger.error(
            "FFmpeg conversion failed"
        )

        logger.error(
            result.stderr
        )

        return False


    return True


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
    Raspberry Pi
        ↓
    약 3초 오디오 수집
        ↓
    Spring
        ↓
    POST /predict
        ↓
    FastAPI
        ↓
    22050Hz / Mono 변환
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
        # 파일 읽기
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
        # 확장자 확인
        # =============================

        suffix = (
            os.path.splitext(
                file.filename
            )[-1]
            or ".wav"
        )


        # =============================
        # 임시 파일 저장
        # =============================

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(
                contents
            )

            tmp_path = tmp.name


        # =============================
        # 변환된 WAV 경로
        # =============================

        wav_path = (
            os.path.splitext(
                tmp_path
            )[0]
            + "_converted.wav"
        )


        # =============================
        # FFmpeg 변환
        # =============================

        conversion_success = convert_to_wav(
            tmp_path,
            wav_path
        )


        if not conversion_success:

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
        # Audio Length
        # =============================

        duration = (
            len(audio)
            / sr
        )


        logger.info(
            f"Audio duration : {duration:.3f}s"
        )


        # 너무 짧으면 분석 X

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
        # Spring에서 이미 약 3초씩
        # 전달하므로 Sliding Window 없음
        # =============================

        status, probability = predict_risk(
            audio,
            sr
        )


        # =============================
        # Result
        # =============================

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


        return {
            "status": status,
            "probability": probability
        }


    # =============================
    # Exception
    # =============================

    except Exception as e:

        logger.exception(
            f"API ERROR : {e}"
        )

        return {
            "status": "error",
            "probability": 0.0
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
        "threshold": THRESHOLD,
        "sample_rate": SAMPLE_RATE,
        "ffmpeg": FFMPEG_PATH
    }