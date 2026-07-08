import json
import os

def load_video_infos():
    video_infos = []

    # AIHub
    with open(
        "./splits/train_pairs.json",
        "r",
        encoding="utf-8"
    ) as f:
        matched_pairs = json.load(f)

    for pair in matched_pairs:
        video_name = os.path.basename(
            pair["video_path"]
        )
        video_infos.append({
            "video": pair["name"],
            "label": pair["name"].split("_")[1],
            "video_path": os.path.join(
                "./videos/AIHub",
                video_name
            )
        })

    # mydata
    custom_root = "./videos/mydata"

    if os.path.exists(custom_root):
        for label in os.listdir(custom_root):
            label_path = os.path.join(
                custom_root,
                label
            )

            if not os.path.isdir(label_path):
                continue
            for video in os.listdir(label_path):
                if not video.endswith(".mp4"):
                    continue

                video_infos.append({
                    "video": video.replace(".mp4",""),
                    "label": label,
                    "video_path": os.path.join(
                        label_path,
                        video
                    )
                })
    return video_infos