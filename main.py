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

from audio_runtime import load_config, model_input, predict_label


# =============================
# 로깅
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

CONFIG = load_config()
MODEL_PATH = CONFIG["model_path"]
SAMPLE_RATE = CONFIG["sample_rate"]
THRESHOLD = CONFIG["threshold"]

# FFmpeg
FFMPEG_PATH = shutil.which("ffmpeg")


# =============================
# 응답 형식
# =============================

class PredictResponse(BaseModel):
    status: str
    probability: float


# =============================
# 모델 로드
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
# 전처리
# =============================

# =============================
# 예측
# =============================

def predict_risk(audio, sr):

    """
    설정된 모델과 전처리로 Audio를 Normal / Abnormal로 분석
    """

    # -----------------------------
    # Mel 스펙트로그램
    # -----------------------------

    mel_input = model_input(audio, sr, CONFIG)
    mel = mel_input.reshape(-1, 128, 128)[0]

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
    # CNN 입력
    #
    # (128, 128)
    # ↓
    # (1, 128, 128, 1)
    # -----------------------------

    # -----------------------------
    # 예측
    # -----------------------------

    pred = model.predict(
        mel_input,
        verbose=0
    )

    score = float(pred.reshape(-1).max())


    # -----------------------------
    # 임계치
    # -----------------------------

    status = predict_label(score, CONFIG)


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
# 오디오 변환
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
        Configured Audio CNN
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
        # WAV 로드
        # =============================

        audio, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE,
            mono=True
        )


        # =============================
        # 오디오 길이
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
        # 예측: V5는 학습 때와 같은 제한된 윈도우 묶음을 사용합니다.
        # =============================

        status, probability = predict_risk(
            audio,
            sr
        )


        # =============================
        # 결과
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
    # 예외 처리
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
# 상태 확인
# =============================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": os.path.basename(
            MODEL_PATH
        ),
        "pipeline": CONFIG["pipeline"],
        "threshold": THRESHOLD,
        "sample_rate": SAMPLE_RATE,
        "ffmpeg": FFMPEG_PATH
    }
