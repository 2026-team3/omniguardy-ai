import time
import requests
from pathlib import Path


# =====================================
# 설정
# =====================================

API_URL = "http://127.0.0.1:8000/predict"

# 테스트할 오디오가 들어있는 폴더
TEST_DIR = Path("dataset/abnormal")

# Spring이 전송할 주기와 동일하게 테스트
INTERVAL_SECONDS = 3


# =====================================
# API 호출
# =====================================

def send_audio(file_path):

    print()
    print("==============================")
    print("Sending:", file_path.name)
    print("==============================")

    try:

        with open(file_path, "rb") as audio_file:

            files = {
                "file": (
                    file_path.name,
                    audio_file,
                    "audio/wav"
                )
            }

            response = requests.post(
                API_URL,
                files=files,
                timeout=30
            )

        response.raise_for_status()

        result = response.json()

        status = result.get("status")
        probability = result.get("probability")

        print(
            f"Result      : {status}"
        )

        print(
            f"Probability : {probability}"
        )

        return result

    except Exception as e:

        print(
            f"Request ERROR: {e}"
        )

        return None


# =====================================
# Main
# =====================================

def main():

    audio_files = sorted(
        TEST_DIR.glob("*.wav")
    )

    if not audio_files:

        print(
            "No WAV files found:",
            TEST_DIR
        )

        return


    print("==============================")
    print("Realtime API Simulation")
    print("==============================")

    print(
        "API:",
        API_URL
    )

    print(
        "Files:",
        len(audio_files)
    )

    print(
        "Interval:",
        INTERVAL_SECONDS,
        "seconds"
    )

    print()


    for index, file_path in enumerate(
        audio_files,
        start=1
    ):

        print(
            f"[{index}/{len(audio_files)}]"
        )

        send_audio(
            file_path
        )


        # 마지막 파일 뒤에는 기다릴 필요 없음
        if index < len(audio_files):

            print(
                f"Waiting {INTERVAL_SECONDS} seconds..."
            )

            time.sleep(
                INTERVAL_SECONDS
            )


    print()
    print("==============================")
    print("Simulation Finished")
    print("==============================")


if __name__ == "__main__":
    main()