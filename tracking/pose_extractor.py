"""영상별 pose 프레임 특징을 추출해 블록 단위 병합에 사용한다."""

from pathlib import Path
import math

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision

from utils.video_loader import load_video_infos


MODEL_PATH = Path("models/pose_landmarker_lite.task")
OUTPUT_ROOT = Path("results/pose")
SPLITS = ("train", "valid")
FRAME_INTERVAL = 10


def extract_video_features(item: dict, landmarker) -> pd.DataFrame:
    """영상 하나에서 일정 간격의 pose 프레임 특징을 만든다."""
    capture = cv2.VideoCapture(item["video_path"])
    if not capture.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {item['video_path']}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    previous: dict[int, tuple[int, float, float, float, float]] = {}
    rows: list[dict] = []
    frame = -1

    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame += 1
        if frame % FRAME_INTERVAL != 0:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(frame / fps * 1000))
        values = []

        for pose_index, landmarks in enumerate(result.pose_landmarks):
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            right_wrist = landmarks[16]
            center_x = (left_shoulder.x + right_shoulder.x) / 2
            center_y = (left_shoulder.y + right_shoulder.y) / 2
            arm_extension = math.dist(
                (right_wrist.x, right_wrist.y),
                (right_shoulder.x, right_shoulder.y),
            )
            upper_body_angle = math.degrees(math.atan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x,
            ))
            hand_motion = 0.0
            body_motion = 0.0
            if pose_index in previous:
                prev_frame, prev_wrist_x, prev_wrist_y, prev_center_x, prev_center_y = previous[pose_index]
                gap = frame - prev_frame
                if gap > 0:
                    hand_motion = math.dist(
                        (right_wrist.x, right_wrist.y), (prev_wrist_x, prev_wrist_y)
                    ) / gap
                    body_motion = math.dist(
                        (center_x, center_y), (prev_center_x, prev_center_y)
                    ) / gap
            previous[pose_index] = (frame, right_wrist.x, right_wrist.y, center_x, center_y)
            values.append((hand_motion, body_motion, arm_extension, upper_body_angle))

        if values:
            array = np.asarray(values, dtype=float)
            rows.append({
                "video": item["video"],
                "video_path": item["video_path"],
                "source": item["source"],
                "split": item["split"],
                "frame": frame,
                "pose_count": len(values),
                "hand_motion": array[:, 0].mean(),
                "body_motion": array[:, 1].mean(),
                "arm_extension": array[:, 2].mean(),
                "upper_body_angle": array[:, 3].mean(),
            })

    capture.release()
    return pd.DataFrame(rows)


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pose 모델이 없습니다: {MODEL_PATH}")

    base_options = mp.tasks.BaseOptions
    options = vision.PoseLandmarkerOptions(
        base_options=base_options(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    for split in SPLITS:
        output_dir = OUTPUT_ROOT / split
        output_dir.mkdir(parents=True, exist_ok=True)
        videos = load_video_infos(
            split=split,
            include_mydata=(split == "train"),
            include_augmented=(split == "train"),
        )
        print(f"[{split}] pose 추출 대상: {len(videos)}개")

        for index, item in enumerate(videos, start=1):
            output_path = output_dir / f"{Path(item['video']).stem}.csv"
            if output_path.exists():
                continue
            print(f"{index}/{len(videos)}: {item['video']}")
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                features = extract_video_features(item, landmarker)
            features.to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
