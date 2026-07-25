import os
import pandas as pd

CLASS_MAP = {
    "lookingInside": "A20",
    "delivery": "delivery",
    "normal": "normal",
}


def load_video_infos(
        split="train",
        include_mydata=True,
        include_augmented=False
):

    if split == "train":
        if include_augmented:
            ann_path = "./splits/annotations/train_annotations_all.csv"
        else:
            ann_path = "./splits/annotations/train_annotations.csv"
    else:
        ann_path = f"./splits/annotations/{split}_annotations.csv"

    ann = pd.read_csv(ann_path)

     # 필요한 source만 선택
    valid_sources = ["AIHub"]

    if include_mydata:
        valid_sources.append("mydata")

    if include_augmented:
        valid_sources.append("augmented")

    ann = ann[
        ann["source"].isin(valid_sources)
    ]

    # annotation은 block 단위이므로
    # video 단위로 중복 제거
    videos = (
        ann[
            ["video", "label", "source"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ==========================
    # mydata 영상 추가
    # ==========================
    if include_mydata:

        for folder, label in CLASS_MAP.items():

            folder_path = os.path.join(
                "./videos/mydata",
                folder
            )

            if not os.path.exists(folder_path):
                continue

            for file in os.listdir(folder_path):

                if not file.endswith(".mp4"):
                    continue

                videos.loc[len(videos)] = {
                    "video": file,
                    "label": label,
                    "source": "mydata"
                }

    video_infos = []

    for _, row in videos.iterrows():

        video = row["video"]
        label = row["label"]
        source = row["source"]

        ##########################
        # video path 생성
        ##########################

        if source == "AIHub":

            video_path = None

            for folder in os.listdir("./videos/unzipped"):

                candidate = os.path.join(
                    "./videos/unzipped",
                    folder,
                    video
                )

                if os.path.exists(candidate):
                    video_path = candidate
                    break

        elif source == "mydata":

            folder = next(
                (
                    k
                    for k, v in CLASS_MAP.items()
                    if v == label
                ),
                label
            )

            video_path = os.path.join(
                "./videos/mydata",
                folder,
                video
            )

        elif source == "augmented":

            video_path = os.path.join(
                "./videos/augmented",
                label,
                video
            )

        else:
            continue

        ##########################
        # 존재하는 영상만 추가
        ##########################

        if video_path is None:
            continue

        if not os.path.exists(video_path):
            continue

        video_infos.append({

            "video": video,
            "label": label,
            "video_path": video_path,
            "source": source,
            "split": split

        })

    return video_infos


if __name__ == "__main__":

    from collections import Counter

    for split in ["train", "valid", "test"]:

        infos = load_video_infos(split)

        print()
        print("=" * 50)
        print(split.upper())
        print("=" * 50)

        print("Videos :", len(infos))

        print("\nSource")
        print(Counter(x["source"] for x in infos))

        print("\nLabel")
        print(Counter(x["label"] for x in infos))