import os
import json

label_base = "./videos/unzipped"
video_base = "./videos/unzipped"

matched_pairs = []

# TL11 ~ TL15
for i in range(11, 16):

    label_root = os.path.join(
        label_base,
        f"TL{i}"
    )

    video_root = os.path.join(
        video_base,
        f"TS{i}"
    )

    print()
    print("=" * 50)
    print(f"TL{i} <-> TS{i}")

    # label 이름 저장
    label_names = set()

    for root, dirs, files in os.walk(label_root):

        for file in files:

            if file.endswith(".json"):

                name = file.replace(".json", "")

                label_names.add(name)

    local_match = 0

    # 영상 매칭
    for root, dirs, files in os.walk(video_root):

        for file in files:

            if file.endswith(".mp4"):

                name = file.replace(".mp4", "")

                if name in label_names:

                    matched_pairs.append({

                        "name": name,

                        "json_path":
                            os.path.join(
                                label_root,
                                name + ".json"
                            ),

                        "video_path":
                            os.path.join(
                                root,
                                file
                            )
                    })

                    local_match += 1

    print("매칭 개수:", local_match)

print()
print("=" * 50)
print("전체 매칭:", len(matched_pairs))

print()
print("예시:")
print(matched_pairs[:3])

# 저장
with open(
    "./videos/matched_pairs.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        matched_pairs,
        f,
        ensure_ascii=False,
        indent=2
    )

print()
print("matched_pairs.json 저장 완료")