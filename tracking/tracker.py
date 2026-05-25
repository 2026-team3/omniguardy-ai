# 루트에서 실행

import json
import os
import pandas as pd
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# =========================
# 매칭 정보 로드
# =========================
with open(
    "./videos/matched_pairs.json",
    "r",
    encoding="utf-8"
) as f:

    matched_pairs = json.load(f)

print("총 pair:", len(matched_pairs))

# =========================
# tracking 결과 저장용
# =========================
tracking_data = []

# =========================
# 50개 테스트
# =========================
for pair in matched_pairs[:50]:

    try:

        video_name = os.path.basename(
            pair["video_path"]
        )

        video_path = os.path.join(
            "./videos/trimmed",
            video_name
        )

        print()
        print("=" * 50)
        print("현재 영상:")
        print(video_path)

        # 파일 없으면 skip
        if not os.path.exists(video_path):

            print("파일 없음 -> skip")
            continue

        # =========================
        # Tracking 실행
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

                # track id 없는 경우 skip
                if box.id is None:
                    continue

                track_id = int(box.id[0])

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                conf = float(box.conf[0])

                cls = int(box.cls[0])

                tracking_data.append({

                    "video": pair["name"],

                    "frame": frame_idx,

                    "track_id": track_id,

                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),

                    "confidence": conf,

                    "class": cls
                })

        print("tracking 완료")

    except Exception as e:

        print()
        print("에러 발생 -> skip")
        print(video_path)
        print(e)

        continue

# =========================
# CSV 저장
# =========================
df = pd.DataFrame(tracking_data)

df.to_csv(
    "./tracking_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 50)
print("tracking_results.csv 저장 완료")
print("총 row:", len(df))