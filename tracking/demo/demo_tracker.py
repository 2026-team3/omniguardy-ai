import os
import pandas as pd

from ultralytics import YOLO

model = YOLO("./models/yolov8n.pt")

# =========================
# 테스트 영상 폴더
# =========================
video_dir = "./videos/test/시연용"

# =========================
# 결과 저장
# =========================
tracking_data = []

# =========================
# 영상 반복
# =========================
for video_name in os.listdir(video_dir):

    if not video_name.endswith(".mp4"):
        continue

    try:

        video_path = os.path.join(
            video_dir,
            video_name
        )

        print()
        print("=" * 50)
        print("현재 영상:")
        print(video_path)

        # =========================
        # tracking
        # =========================
        results = model.track(
            source=video_path,
            tracker="botsort.yaml",
            persist=True,
            save=False,
            conf=0.3
        )

        # =========================
        # 결과 추출
        # =========================
        for frame_idx, r in enumerate(results):

            boxes = r.boxes

            if boxes is None:
                continue

            for box in boxes:

                if box.id is None:
                    continue

                track_id = int(box.id[0])

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )
                conf = float(box.conf[0])

                tracking_data.append({

                    "video":
                        video_name.replace(
                            ".mp4",
                            ""
                        ),

                    "frame": frame_idx,
                    "track_id": track_id,
                    "confidence": conf,

                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                })

        print("tracking 완료")

    except Exception as e:
        print()
        print("에러 발생")
        print(video_name)
        print(e)

# =========================
# CSV 저장
# =========================
df = pd.DataFrame(tracking_data)

df.to_csv(
    "./results/시연용/tracking_results(test).csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 50)
print("tracking_results(test).csv 저장 완료")
print("총 row:", len(df))