import os
import pandas as pd

from ultralytics import YOLO
from utils.video_loader import load_video_infos

model = YOLO("./models/yolov8n.pt")
video_infos = load_video_infos()

# Tracking
tracking_data = []

for item in video_infos:
    try:
        print("="*50)
        print(item["video"])

        if not os.path.exists(item["video_path"]):
            print("파일 없음")
            continue
          
        results = model.track(
            source=item["video_path"],
            tracker="botsort.yaml",
            persist=True,
            save=False,
            conf=0.3
        )

        # 결과 추출
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
                    "frame": frame_idx,
                    "track_id": int(box.id[0]),

                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),

                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0])
                })
        print("tracking 완료")

    except Exception as e:
        print()
        print(item["video"])
        print(e)

        continue

# CSV 저장
df = pd.DataFrame(tracking_data)

df.to_csv(
    "./results/tracking_results(07.07).csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print(df.head())
print("총 row:", len(df))