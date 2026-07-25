import os
import cv2
import albumentations as at

from utils.video_loader import load_video_infos

OUTPUT_ROOT = "./videos/augmented"
AUG_COUNT = {
    "A17": 2,
    "A18": 0,
    "A19": 1,
    "A20": 0,          # AIHub용
    "A21": 1,
    "delivery": 8,
    "normal": 4,
}

# Augmentation
transform = at.ReplayCompose([  # 각 프레임에 서로 다른 augmentation을 적용하지 않도록!! (시간적으로 일관된 augmentation 만들기)
    at.Affine(
        scale=(0.9, 1.1),   # 영상 크기를 랜덤하게 90% ~ 110% 범위로 변경
        translate_percent=(-0.05, 0.05),    # 영상을 최대 +/-5% 이동
        rotate=(-5, 5), # 영상을 +/- 5도 회전
        p=0.8   # 80% 확률로 Affine augmentation 적용
    ),

    at.RandomBrightnessContrast(    # 밝기와 대비
        brightness_limit=0.15,  # 밝기 변화 +/- 15%에서 변화
        contrast_limit=0.15,    # 대비도 마찬가지
        p=0.5
    ),

    at.GaussNoise(  # 30% 확률로 영상에 Gaussian noise 추가
        std_range=(0.01, 0.03),
        p=0.3
    ),
])

# 영상 증강
def augment_video(
    input_path,
    output_path
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("영상 열기 실패:", input_path)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    replay = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 첫 frame에서 augmentation parameter 생성
        if replay is None:
            result = transform(
                image=frame
            )

            aug_frame = result["image"]
            replay = result["replay"]

        # 이후 frame에는 동일 transform 적용
        else:
            result = at.ReplayCompose.replay(
                replay,
                image=frame
            )

            aug_frame = result["image"]

        writer.write(aug_frame)

    cap.release()
    writer.release()

    return True


# Main
video_infos = load_video_infos(
    split="train",
    include_mydata=True,
    include_augmented=False
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)

print("전체 원본 영상:", len(video_infos))


for item in video_infos:

    video_name = item["video"]
    video_path = item["video_path"]
    label = item["label"]

    source = item["source"]

    is_mydata = source == "mydata"
    is_aihub = source == "AIHub"

    # mydata 또는 AIHub 원본만 증강
    if not (is_mydata or is_aihub):
        continue

    # ------------------------
    # 증강 횟수 결정
    # ------------------------

    if is_mydata and label == "A20":
        aug_count = 15
    else:
        aug_count = AUG_COUNT.get(label, 0)

    if aug_count == 0:
        continue

    if not os.path.exists(video_path):
        print("파일 없음:", video_path)
        continue

    label_dir = os.path.join(OUTPUT_ROOT, label)

    os.makedirs(label_dir, exist_ok=True)

    for aug_idx in range(aug_count):
        base, ext = os.path.splitext(video_name)
        output_name = f"{base}_aug{aug_idx+1}{ext}"

        output_path = os.path.join(label_dir, output_name)
        # 이미 생성된 영상이면 skip
        if os.path.exists(output_path):
            print(
                "이미 존재 -> skip:",
                output_name
            )
            continue

        success = augment_video(video_path, output_path)

        if success:
            print("증강 완료")
        else:
            print("증강 실패")


print()
print("=" * 50)
print("전체 augmentation 완료")