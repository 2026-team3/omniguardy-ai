import os
from pathlib import Path

import torch
import pandas as pd
from ultralytics import YOLO

from utils.video_loader import load_video_infos


print("=" * 50)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 50)


# YOLO 모델
model = YOLO("./models/yolov8n.pt")

# 결과 저장 폴더
output_dir = Path("./results/tracking")
output_dir.mkdir(parents=True, exist_ok=True)


# train / valid / test 모두 수행
for split in ["train", "valid", "test"]:

    print(f"\n{'=' * 60}")
    print(f"Processing {split.upper()}")
    print(f"{'=' * 60}")

    video_infos = load_video_infos(split)

    tracking_data = []

    for idx, item in enumerate(video_infos):

        try:
            print("-" * 50)
            print(f"[{idx + 1}/{len(video_infos)}]")
            print("video :", item["video"])
            print("label :", item["label"])
            print("source:", item["source"])

            if not os.path.exists(item["video_path"]):
                continue

            results = model.track(
                source=item["video_path"],
                tracker="botsort.yaml",
                persist=False,
                save=False,
                conf=0.3,
                classes=[0],      # person
                stream=True,
                device=0
            )

            # frame별 tracking 결과 저장
            for frame_idx, r in enumerate(results):

                if r.boxes is None:
                    continue

                for box in r.boxes:

                    if box.id is None:
                        continue

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    tracking_data.append({

                        "video": item["video"],
                        "label": item["label"],
                        "source": item["source"],
                        "split": split,

                        "frame": frame_idx,
                        "track_id": int(box.id[0]),

                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),

                        "confidence": float(box.conf[0]),
                        "class": int(box.cls[0])
                    })

            print("Tracking 완료")

        except Exception as e:

            print(f"\nError : {item['video']}")
            print(e)
            continue

    # CSV 저장
    df = pd.DataFrame(tracking_data)

    save_path = output_dir / f"{split}_tracking.csv"

    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved : {save_path}")
    print(df.head())
    if len(df) > 0:
        print("Processed videos :", df["video"].nunique())

    print("Total rows :", len(df))
    print("총 row :", len(df))