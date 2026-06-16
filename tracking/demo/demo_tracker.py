import os
import cv2
import pandas as pd
import numpy as np

from ultralytics import YOLO

model = YOLO("./models/yolov8n.pt")

# =========================
# 테스트 영상 폴더
# =========================
video_dir = "./videos/test/시연용"

pose_landmarks = pd.read_csv(
    "./results/시연용/pose_landmarks(test).csv"
)

# =========================
# 결과 저장
# =========================
tracking_data = []

# =========================
# ROI 영역 (사다리꼴)
# =========================
roi_points = np.array([
    [30, 50],
    [1890, 50],
    [1910, 1070],
    [10, 1070]
], dtype=np.int32)

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (23, 24)
]

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
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        output_path = os.path.join(
            "./runs/detect/track-2",
            video_name.replace(".mp4", "_pose.mp4")
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
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
        for frame_idx, r in enumerate(results, start=1):

            boxes = r.boxes

            # --------------------------
            # YOLO Detection Box 그리기
            # --------------------------
            annotated_frame = r.plot()

            # --------------------------
            # ROI Overlay
            # --------------------------
            overlay = annotated_frame.copy()

            cv2.fillPoly(
                overlay,
                [roi_points],
                (0, 0, 255)
            )

            cv2.addWeighted(
                overlay,
                0.10,
                annotated_frame,
                0.90,
                0,
                annotated_frame
            )

            cv2.polylines(
                annotated_frame,
                [roi_points],
                True,
                (0, 0, 255),
                3
            )

            cv2.putText(
                annotated_frame,
                "AI MONITORING AREA",
                (350, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # --------------------------
            # MediaPipe Skeleton Overlay
            # --------------------------
            current_pose = pose_landmarks[
                (pose_landmarks["video"] == video_name.replace(".mp4", ""))
                &
                (pose_landmarks["frame"] == frame_idx +1)
            ]

            points = {}

            for _, row in current_pose.iterrows():

                px = int(np.clip(row["x"], 0, 1) * width)
                py = int(np.clip(row["y"], 0, 1) * height)
                print(frame_idx, len(current_pose))

                landmark_id = int(row["landmark_id"])

                points[landmark_id] = (px, py)

                cv2.circle(
                    annotated_frame,
                    (px, py),
                    4,
                    (0, 255, 255),
                    -1
                )

            # skeleton 연결선
            for a, b in POSE_CONNECTIONS:
                if a in points and b in points:
                    cv2.line(
                        annotated_frame,
                        points[a],
                        points[b],
                        (0, 255, 255),
                        2
                    )

            # --------------------------
            # 최종 영상 저장
            # --------------------------
            writer.write(annotated_frame)
        
            # --------------------------
            # tracking csv 저장용
            # --------------------------
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
                    "video": video_name.replace(".mp4", ""),
                    "frame": frame_idx,
                    "track_id": track_id,
                    "confidence": conf,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                })

        writer.release()
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