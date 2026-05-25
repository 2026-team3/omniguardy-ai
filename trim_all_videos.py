import os
import cv2

input_root = "./videos/unzipped"
output_root = "./videos/trimmed"

os.makedirs(output_root, exist_ok=True)

# 몇 초짜리 clip 만들지
clip_seconds = 20

for root, dirs, files in os.walk(input_root):

    for file in files:

        if not file.endswith(".mp4"):
            continue

        input_path = os.path.join(root, file)

        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        duration = total_frames / fps

        # 중앙 기준
        start_sec = max(
            0,
            duration / 2 - clip_seconds / 2
        )

        end_sec = start_sec + clip_seconds

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        output_path = os.path.join(
            output_root,
            file
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame
        )

        current = start_frame

        while current <= end_frame:

            ret, frame = cap.read()

            if not ret:
                break

            out.write(frame)

            current += 1

        cap.release()
        out.release()

        print("완료:", file)

print()
print("전체 trimming 완료")