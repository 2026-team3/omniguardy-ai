import os
import zipfile
from collections import Counter

# =========================
# TL 라벨 zip 폴더
# =========================
label_zip_dir = "./videos/labels/TL"

# =========================
# TS 영상 zip 폴더
# =========================
video_zip_dir = "./videos/labels/TS"

# =========================
# 압축 해제 폴더
# =========================
extract_base = "./videos/unzipped"

os.makedirs(extract_base, exist_ok=True)

# =========================
# TL unzip + 통계
# =========================
action_counter = Counter()
sy_counter = Counter()

for zip_name in os.listdir(label_zip_dir):

    if not zip_name.endswith(".zip"):
        continue

    zip_path = os.path.join(
        label_zip_dir,
        zip_name
    )

    extract_path = os.path.join(
        extract_base,
        zip_name.replace(".zip", "")
    )

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print()
    print("=" * 50)
    print("압축 해제 완료:", zip_name)

    local_action_counter = Counter()

    for root, dirs, files in os.walk(extract_path):

        for file in files:

            if not file.endswith(".json"):
                continue

            parts = file.split("_")

            if len(parts) < 3:
                continue

            action_code = parts[1]
            sy_code = parts[2]

            action_counter[action_code] += 1
            sy_counter[sy_code] += 1

            local_action_counter[action_code] += 1

    print("이 zip의 ACTION:")
    print(local_action_counter)

# =========================
# TS unzip
# =========================

target_ts = {
    "TS11.zip",
    "TS12.zip",
    "TS13.zip",
    "TS14.zip",
    "TS15.zip"
}

for zip_name in os.listdir(video_zip_dir):

    if zip_name not in target_ts:
        continue

    zip_path = os.path.join(
        video_zip_dir,
        zip_name
    )

    extract_path = os.path.join(
        extract_base,
        zip_name.replace(".zip", "")
    )

    print()
    print("=" * 50)
    print("영상 압축 해제:", zip_name)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    mp4_count = 0

    for root, dirs, files in os.walk(extract_path):

        for file in files:

            if file.endswith(".mp4"):

                mp4_count += 1

    print("mp4 개수:", mp4_count)