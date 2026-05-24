import json
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# 매칭 정보 로드
with open(
    "./videos/matched_pairs.json",
    "r",
    encoding="utf-8"
) as f:

    matched_pairs = json.load(f)

print("총 pair:", len(matched_pairs))

# 테스트용 3개만
for pair in matched_pairs[:3]:

    video_path = pair["video_path"]

    print()
    print("=" * 50)
    print("현재 영상:")
    print(video_path)

    results = model.track(
        source=video_path,
        tracker="botsort.yaml",
        persist=True,
        save=True,
        conf=0.3
    )

    print("tracking 완료")