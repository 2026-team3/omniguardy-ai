# MediaPipe Tasks API 기반 Pose Extractor
import cv2
import mediapipe as mp
import pandas as pd
import math
import os
import numpy as np

# =========================
# 모델 경로
# =========================
model_path = (
    "./models/pose_landmarker_lite.task"
)

# =========================
# Pose Landmarker 생성
# =========================
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=2,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = (
    PoseLandmarker.create_from_options(
        options
    )
)

# =========================
# 영상 경로
# =========================
video_dir = "./videos/test/시연용"

video_list = [

    v for v in os.listdir(video_dir)
    if v.endswith(".mp4")
]

# =========================
# 결과 저장
# =========================
pose_features = []
pose_landmark_data = []

# =========================
# 영상 반복
# =========================
for video_name in video_list:

    try:
        # =========================
        # 영상마다 새 landmarker 생성
        # =========================
        landmarker = (
            PoseLandmarker.create_from_options(
                options
            )
        )
        video_path = os.path.join(video_dir, video_name)
        print()
        print("=" * 50)
        print("현재 영상:")
        print(video_name)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # fps 이상하면 기본값
        if fps <= 0:
            fps = 30
        

        frame_idx = 0
        right_hand_positions = []
        body_positions = []
        arm_lengths = []
        upper_body_angles = []

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # =========================
            # 10프레임마다만 처리
            # =========================
            # if frame_idx % 10 != 0:
            #     continue

            # =========================
            # timestamp(ms)
            # =========================
            timestamp_ms = int(frame_idx * 100)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, 
                                data=rgb_frame)

            # =========================
            # Pose 추론
            # =========================
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            

            if len(result.pose_landmarks) == 0:
                continue
            landmarks = (result.pose_landmarks[0])
            # =========================
            # landmark 저장
            # =========================
            for landmark_id, lm in enumerate(landmarks):
                pose_landmark_data.append({
                    "video": video_name.replace(".mp4", ""),
                    "frame": frame_idx,
                    "landmark_id": landmark_id,
                    "x": lm.x,
                    "y": lm.y
                })

            # =========================
            # 주요 landmark
            # =========================
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            right_wrist = landmarks[16]

            # =========================
            # 손 위치
            # =========================
            hand_x = right_wrist.x
            hand_y = right_wrist.y
            right_hand_positions.append(
                (hand_x, hand_y)
            )

            # =========================
            # body center
            # =========================
            body_center_x = (
                left_shoulder.x +
                right_shoulder.x
            ) / 2

            body_center_y = (
                left_shoulder.y +
                right_shoulder.y
            ) / 2

            body_positions.append(
                (
                    body_center_x,
                    body_center_y
                )
            )

            # =========================
            # 상체 각도
            # =========================
            dx = (right_shoulder.x - left_shoulder.x )
            dy = (right_shoulder.y - left_shoulder.y )

            angle = math.degrees(math.atan2(dy, dx))
            upper_body_angles.append(angle)

            # =========================
            # 팔 뻗음
            # =========================
            arm_length = math.sqrt(
                (right_wrist.x - right_shoulder.x) ** 2 
                +
                (right_wrist.y - right_shoulder.y) ** 2
            )

            arm_lengths.append(arm_length)

        cap.release()
        
        landmarker.close()

        # =========================
        # hand motion
        # =========================
        hand_motion = 0

        for i in range(1, len(right_hand_positions)):

            x1, y1 = (right_hand_positions[i - 1])
            x2, y2 = (right_hand_positions[i])

            dist = math.sqrt(
                (x2 - x1) ** 2 + (y2 - y1) ** 2
            )

            hand_motion += dist

        # =========================
        # body motion
        # =========================
        body_motion = 0

        for i in range( 1, len(body_positions)):
            x1, y1 = (body_positions[i - 1])
            x2, y2 = (body_positions[i])
            
            dist = math.sqrt(
                (x2 - x1) ** 2 + (y2 - y1) ** 2
            )

            body_motion += dist

        # =========================
        # 평균값
        # =========================
        avg_arm_extension = 0
        avg_upper_body_angle = 0

        if len(arm_lengths) > 0:
            avg_arm_extension = np.mean(
                arm_lengths
            )

        if len(upper_body_angles) > 0:
            avg_upper_body_angle = np.mean(
                upper_body_angles
            )

        # =========================
        # 저장
        # =========================
        pose_features.append({
            "video": video_name,
            "frame_count": frame_idx,
            "hand_motion": hand_motion,
            "body_motion": body_motion,
            "upper_body_angle": avg_upper_body_angle,
            "arm_extension": avg_arm_extension
        })

        print("pose feature 추출 완료")

    except Exception as e:

        print()
        print("에러 발생 -> skip")
        print(video_name)
        print(e)

        continue

# =========================
# CSV 저장
# =========================
pose_df = pd.DataFrame( pose_features )
pose_df.to_csv(
    "./results/시연용/pose_features(test).csv",
    index=False,
    encoding="utf-8-sig"
)
pose_landmark_df = pd.DataFrame(pose_landmark_data)
pose_landmark_df.to_csv(
    "./results/시연용/pose_landmarks(test).csv",
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 50)
print("pose_features(test).csv 저장 완료")
print("총 row:", len(pose_df))

