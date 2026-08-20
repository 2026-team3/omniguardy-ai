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


# YOLO 사람 추적 모델을 불러온다.
model = YOLO("./models/yolov8n.pt")

# tracking 결과를 저장할 루트 폴더
output_dir = Path("./results/tracking")
output_dir.mkdir(parents=True, exist_ok=True)


# train과 validation split을 차례대로 처리한다.
for split in ["train", "valid"]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Processing {split.upper()}")
    print(f"{'=' * 60}")

    if split == "train":
        video_infos = load_video_infos(
            split="train",
            include_mydata=True,
            include_augmented=True
        )
    else:
        video_infos = load_video_infos(
            split=split,
            include_mydata=False,
            include_augmented=False
        )
    
    print(f"Total videos : {len(video_infos)}")
  
    missing = 0
    failed = 0

    for idx, item in enumerate(video_infos):
        tracking_data = []

        print(
            f"[{idx+1}/{len(video_infos)}] "
            f"{item['source']} | "  
            f"{item['label']} | "
            f"{item['video']}"
        )
        video_name = Path(item["video"]).stem
        save_path = split_dir / f"{video_name}.csv"

        if save_path.exists():
            print("이미 존재하여 건너뜁니다.")
            continue

        try:
            if not os.path.exists(item["video_path"]):
                print("영상 파일이 없습니다.")
                missing += 1
                continue

            results = model.track(
                source=item["video_path"],
                tracker="botsort.yaml",
                persist=False,
                save=False,
                conf=0.3,
                classes=[0],      # 사람 클래스만 추적
                stream=True,
                device=0,
                verbose=False
            )

            # 프레임별 tracking 결과를 행으로 저장한다.
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
                        "video_path": item["video_path"],

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

            print("추적 완료")
            # 영상별 tracking CSV를 저장한다.
            df = pd.DataFrame(tracking_data)
            video_name = Path(item["video"]).stem
            save_path = split_dir / f"{video_name}.csv"

            df.to_csv(
                save_path,
                index=False,
                encoding="utf-8-sig"
            )

        except Exception as e:
            failed += 1

            print(f"\nError : {item['video']}")
            print(e)
       
        finally:

            if "results" in locals():
                del results

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    print("\n" + "=" * 50)
    print(f"{split.upper()} ")
    print("=" * 50)

    print(f"Saved              : {save_path}")
    saved_files = len(list(split_dir.glob("*.csv")))
    print(f"Saved CSV          : {saved_files}")
    print(f"Total rows         : {len(df)}")
    print(f"Missing videos     : {missing}")
    print(f"Failed videos      : {failed}")


print("\n전체 추적 완료")

