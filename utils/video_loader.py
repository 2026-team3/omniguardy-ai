import json
import os

CLASS_MAP = {
    "lookingInside": "A20",
    "delivery": "delivery",
    "normal": "normal",
}

def load_video_infos(
    split="train",
    include_mydata=True,
    include_augmented=True
):
    video_infos = []

    # AIHub
    split_path = (
        f"./splits/{split}_pairs.json"
    )

    with open(
        split_path,
        "r",
        encoding="utf-8"
    ) as f:
        matched_pairs = json.load(f)

    for pair in matched_pairs:
        video_name = os.path.basename(
            pair["video_path"]
        )

        video_infos.append({
            "video": pair["name"],
            "label": pair["name"].split("_")[1],
            "video_path": os.path.join(
                "./videos/AIHub",
                video_name
            ),
            "source": "AIHub",
            "split": split
        })

    # mydata (train에만 포함)
    if (
        split == "train"
        and include_mydata
    ):
        custom_root = "./videos/mydata"

        if os.path.exists(custom_root):
            for folder_name in os.listdir(
                custom_root
            ):
                label_path = os.path.join(
                    custom_root,
                    folder_name
                )

                if not os.path.isdir(label_path):
                    continue

                mapped_label = CLASS_MAP.get(
                    folder_name,
                    folder_name
                )

                for video in os.listdir(label_path):
                    if not video.lower().endswith(".mp4"):
                        continue

                    video_infos.append({
                        "video": os.path.splitext(video)[0],
                        "label": mapped_label,
                        "video_path": os.path.join(
                            label_path,
                            video
                        ),
                        "source": "mydata",
                        "split": "train"
                    })

    # Augmented (train에만 포함)
    if (
        split == "train"
        and include_augmented
    ):
        augmented_root = "./videos/augmented"

        if os.path.exists(augmented_root):
            for label in os.listdir(augmented_root):
                label_path = os.path.join(augmented_root, label)

                if not os.path.isdir(label_path):
                    continue

                for video in os.listdir(label_path):
                    if not video.lower().endswith(".mp4"):
                        continue

                    video_infos.append({
                        "video": os.path.splitext(video)[0],
                        "label": label,
                        "video_path": os.path.join(
                            label_path,
                            video
                        ),
                        "source": "augmented",
                        "split": "train"
                    })

    return video_infos

if __name__ == "__main__":
    train_infos = load_video_infos(split="train")
    valid_infos = load_video_infos(split="valid")
    test_infos = load_video_infos(split="test")

    print()
    print("=" * 50)
    print("Train:", len(train_infos))
    print("Valid:", len(valid_infos))
    print("Test:", len(test_infos))

    print()
    print("Train source")
    from collections import Counter

    print(
        Counter(
            item["source"]
            for item in train_infos
        )
    )

    print()
    print("Train label")
    print(
        Counter(
            item["label"]
            for item in train_infos
        )
    )