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


# -----------------------------
# Logging 설정
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)



keras = tf.keras

app = FastAPI()



# -----------------------------
# Response Schema
# -----------------------------
class PredictResponse(BaseModel):
    status: str
    probability: float



# -----------------------------
# Model Load
# -----------------------------
MODEL_PATH = os.path.join(
    "models",
    "audio_model.keras"
)

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

logger.info("MODEL LOADED")



# -----------------------------
# Preprocess
# train.py와 동일
# -----------------------------
def preprocess(audio, sr):

    mel = audio_to_mel(
        audio,
        sr
    )

    return mel.astype(
        np.float32
    )



# -----------------------------
# Prediction
# -----------------------------
def predict_risk(audio, sr):

    mel = preprocess(
        audio,
        sr
    )


    logger.info("========== AUDIO INFO ==========")
    logger.info(f"audio length : {len(audio)}")
    logger.info(f"sample rate  : {sr}")


    logger.info("========== MEL INFO ==========")
    logger.info(f"mel shape : {mel.shape}")
    logger.info(f"mel min   : {mel.min()}")
    logger.info(f"mel max   : {mel.max()}")
    logger.info(f"mel mean  : {mel.mean()}")



    mel_input = np.expand_dims(
        mel,
        axis=(0, -1)
    )


    pred = model.predict(
        mel_input,
        verbose=0
    )


    score = float(
        pred[0][0]
    )


    logger.info("========== RESULT ==========")
    logger.info(f"raw score : {score}")


    if score >= 0.5:

        logger.info("prediction : ABNORMAL")

    else:

        logger.info("prediction : NORMAL")


    logger.info("============================")


    return score




# -----------------------------
# Sliding Window
# -----------------------------
def sliding_predict(audio, sr):

    block_size = sr * 5

    probs = []


    for start in range(
        0,
        len(audio),
        block_size
    ):


        chunk = audio[
            start:start + block_size
        ]


        if len(chunk) < sr:

            continue



        prob = predict_risk(
            chunk,
            sr
        )


        logger.info(
            f"chunk start={start}, probability={prob}"
        )


        probs.append(
            prob
        )



    if not probs:

        logger.info(
            "No valid audio chunk"
        )

        return {

            "status": "normal",

            "final_prob": 0.0

        }



    final_prob = max(
        probs
    )


    logger.info(
        f"FINAL PROBABILITY : {final_prob}"
    )


    return {


        "status":

            "abnormal"

            if final_prob >= 0.5

            else "normal",



        "final_prob":

            float(final_prob)

    }




# -----------------------------
# API
# -----------------------------
@app.post(
    "/predict",
    response_model=PredictResponse
)
async def predict(
    file: UploadFile = File(...)
):


    logger.info(
        f"REQUEST RECEIVED : {file.filename}"
    )


    tmp_path = None
    wav_path = None


    try:


        contents = await file.read()



        suffix = (

            os.path.splitext(
                file.filename
            )[-1]

            or ".mp4"

        )



        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:


            tmp.write(
                contents
            )

            tmp_path = tmp.name




        wav_path = (

            os.path.splitext(
                tmp_path
            )[0]

            + ".wav"

        )



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

                "-ar",
                "22050",

                "-ac",
                "1",

                wav_path
            ],

            capture_output=True,

            text=True

        )



        if result.returncode != 0:


            logger.error(
                result.stderr
            )


            return {

                "status":
                    "error_ffmpeg",

                "probability":
                    0.0

            }




        audio, sr = librosa.load(

            wav_path,

            sr=22050,

            mono=True

        )



        result = sliding_predict(
            audio,
            sr
        )



        logger.info(
            f"FINAL RESULT : {result}"
        )



        return {


            "status":

                result["status"],



            "probability":

                result["final_prob"]

        }




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




    finally:


        if tmp_path and os.path.exists(tmp_path):

            os.remove(
                tmp_path
            )



        if wav_path and os.path.exists(wav_path):

            os.remove(
                wav_path
            )