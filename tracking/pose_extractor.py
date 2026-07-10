import cv2
import mediapipe as mp
import pandas as pd
import math
import numpy as np
from utils.video_loader import load_video_infos
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

model_path = (
    "./models/pose_landmarker_lite.task"
)

video_infos = load_video_infos()
visualize_video = video_infos[0]["video"]

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image

# Pose Landmarker 생성
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


# 결과 저장
pose_features = []

# 영상 반복
for item in video_infos:
    video_name = item["video"]
    video_path = item["video_path"]
    label = item["label"]
    try:
        # 영상마다 새 landmarker 생성
        landmarker = (
            PoseLandmarker.create_from_options(
                options
            )
        )
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

            # 10프레임마다만 처리
            if frame_idx % 10 != 0:
                continue

            # timestamp(ms)
            timestamp_ms = int((frame_idx / fps) * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, 
                                data=rgb_frame)

            # Pose 추론
            result = (
                landmarker.detect_for_video(mp_image, timestamp_ms)
            )

            # Visualization
            if video_name == visualize_video:
                annotated_image = (
                    draw_landmarks_on_image(
                        rgb_frame,
                        result
                    )
                )
                cv2.imshow(
                    "Pose Visualization",
                    cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                )
                if result.segmentation_masks:
                    segmentation_mask = (
                        result.segmentation_masks[0]
                        .numpy_view()
                    )
                    segmentation_mask = np.squeeze( segmentation_mask )
                    visualized_mask = (segmentation_mask * 255).astype(np.uint8)
                    visualized_mask = np.stack([visualized_mask] * 3, axis=-1)
                    cv2.imshow(
                        "Segmentation Mask",
                        visualized_mask
                    )
                key = cv2.waitKey(1)

                    # ESC 누르면 종료
                if key == 27:
                    break

            if len(result.pose_landmarks) == 0:
                continue
            landmarks = (result.pose_landmarks[0])

            # 주요 landmark
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            right_wrist = landmarks[16]

            # 손 위치
            hand_x = right_wrist.x
            hand_y = right_wrist.y
            right_hand_positions.append(
                (hand_x, hand_y)
            )

            # body center
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

            # 상체 각도
            dx = (right_shoulder.x - left_shoulder.x )
            dy = (right_shoulder.y - left_shoulder.y )

            angle = math.degrees(math.atan2(dy, dx))
            upper_body_angles.append(angle)

            # 팔 뻗음
            arm_length = math.sqrt(
                (right_wrist.x - right_shoulder.x) ** 2 
                +
                (right_wrist.y - right_shoulder.y) ** 2
            )

            arm_lengths.append(arm_length)

        cap.release()
        landmarker.close()

        # hand motion
        hand_motion_list = []

        for i in range(1, len(right_hand_positions)):
            x1, y1 = (right_hand_positions[i - 1])
            x2, y2 = (right_hand_positions[i])

            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            hand_motion_list.append(dist)

        if len(hand_motion_list):
            hand_motion_sum = np.sum(hand_motion_list)
            hand_motion_mean = np.mean(hand_motion_list)
            hand_motion_std = np.std(hand_motion_list)
            hand_motion_max = np.max(hand_motion_list)
        else:
            hand_motion_sum = 0
            hand_motion_mean = 0
            hand_motion_std = 0
            hand_motion_max = 0

        # body motion
        body_motion_list = []

        for i in range( 1, len(body_positions)):
            x1, y1 = (body_positions[i - 1])
            x2, y2 = (body_positions[i])
            
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            body_motion_list.append(dist)

        if len(body_motion_list):
            body_motion_sum = np.sum(body_motion_list)
            body_motion_mean = np.mean(body_motion_list)
            body_motion_std = np.std(body_motion_list)
            body_motion_max = np.max(body_motion_list)
        else:
            body_motion_sum = 0
            body_motion_mean = 0
            body_motion_std = 0
            body_motion_max = 0

        # arm extension
        if len(arm_lengths):
            arm_extension_mean = np.mean(arm_lengths)
            arm_extension_std = np.std(arm_lengths)
            arm_extension_max = np.max(arm_lengths)
            arm_extension_min = np.min(arm_lengths)
        else:
            arm_extension_mean = 0
            arm_extension_std = 0
            arm_extension_max = 0
            arm_extension_min = 0

        # upper body angle
        if len(upper_body_angles):
            upper_body_angle_mean = np.mean(upper_body_angles)
            upper_body_angle_std = np.std(upper_body_angles)
            upper_body_angle_max = np.max(upper_body_angles)
            upper_body_angle_min = np.min(upper_body_angles)
        else:
            upper_body_angle_mean = 0
            upper_body_angle_std = 0
            upper_body_angle_max = 0
            upper_body_angle_min = 0

        # 저장
        pose_features.append({
            "video": video_name,
            "label": label,
            "frame_count": frame_idx,
            # Hand Motion
            "hand_motion_sum": hand_motion_sum,
            "hand_motion_mean": hand_motion_mean,
            "hand_motion_std": hand_motion_std,
            "hand_motion_max": hand_motion_max,

            # Body Motion
            "body_motion_sum": body_motion_sum,
            "body_motion_mean": body_motion_mean,
            "body_motion_std": body_motion_std,
            "body_motion_max": body_motion_max,

            # Upper Body Angle
            "upper_body_angle_mean": upper_body_angle_mean,
            "upper_body_angle_std": upper_body_angle_std,
            "upper_body_angle_max": upper_body_angle_max,
            "upper_body_angle_min": upper_body_angle_min,

            # Arm Extension
            "arm_extension_mean": arm_extension_mean,
            "arm_extension_std": arm_extension_std,
            "arm_extension_max": arm_extension_max,
            "arm_extension_min": arm_extension_min
        })

        print("pose feature 추출 완료")

    except Exception as e:

        print()
        print("에러 발생 -> skip")
        print(video_name)
        print(e)

        continue

# CSV 저장
pose_df = pd.DataFrame( pose_features )
pose_df.to_csv(
    "./results/pose_features(07.07).csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 50)
print("pose_features(07.07).csv 저장 완료")
print("총 row:", len(pose_df))

cv2.destroyAllWindows()