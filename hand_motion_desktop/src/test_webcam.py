"""노트북 웹캠으로 손동작 인식을 바로 눈으로 확인하는 스크립트.
FastAPI 연동 전에 gesture_rules.py의 판정 기준이 맞는지 빠르게 테스트할 때 사용한다.

실행:
    python test_webcam.py
종료:
    q 또는 Esc
"""

from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gesture_rules import (
    RepeatedFistToOpenTracker,
    classify_gesture,
    frame_gesture_from_hands,
)
from model_loader import ensure_model

# 트리거 감지 후 "SILENT SIGNAL" 문구를 화면에 띄워둘 프레임 수 (약 1.5초 @30fps)
SIGNAL_DISPLAY_FRAMES = 45

# 몇 번 연속으로 주먹 쥐었다 펴야 확정할지
REQUIRED_REPEATS = 2


def main() -> None:
    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 카메라 인덱스를 확인하세요.")

    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval_ms = 1000.0 / fps

    tracker = RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS)
    signal_countdown = 0  # 0보다 크면 "SILENT SIGNAL" 문구를 화면에 표시 중인 상태

    print(
        f"웹캠 테스트 시작. 주먹을 쥐었다 펴는 동작을 {REQUIRED_REPEATS}번 연속으로 하면 "
        "SILENT SIGNAL 표시. q 또는 Esc로 종료."
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_idx * frame_interval_ms)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            label_text = "no hand"
            gestures_this_frame: list[str] = []
            if result.hand_landmarks:
                for landmarks, handedness in zip(
                    result.hand_landmarks, result.handedness
                ):
                    hand_label = handedness[0].category_name
                    gestures_this_frame.append(classify_gesture(landmarks, hand_label))
                    # 손 landmark 점 그려서 눈으로 확인
                    h, w, _ = frame.shape
                    for point in landmarks:
                        cx, cy = int(point.x * w), int(point.y * h)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                label_text = ", ".join(gestures_this_frame)

            frame_gesture = frame_gesture_from_hands(gestures_this_frame)
            triggered = tracker.update(frame_gesture)
            if triggered:
                signal_countdown = SIGNAL_DISPLAY_FRAMES
                print(f"[frame {frame_idx}] SILENT SIGNAL 감지 (fist -> open_palm)")

            cv2.putText(
                frame,
                label_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"reps: {tracker.rep_count}/{REQUIRED_REPEATS}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            if signal_countdown > 0:
                cv2.putText(
                    frame,
                    "SILENT SIGNAL",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 255),
                    3,
                )
                signal_countdown -= 1

            cv2.imshow("hand gesture test", frame)

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()