import cv2
import pandas as pd
from pathlib import Path

from utils.video_loader import load_video_infos


AIHUB_ANNOTATION = "./splits/annotations/train_annotations.csv"
OUTPUT_PATH = "./splits/annotations/train_annotations_all.csv"


def read_aihub_annotations():
    return pd.read_csv(AIHUB_ANNOTATION)


def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"영상 열기 실패 : {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count


def generate_mydata_annotations(video_infos):

    rows = []

    for item in video_infos:

        if item["source"] != "mydata":
            continue

        frame_count = get_frame_count(item["video_path"])

        if frame_count is None:
            continue

        if item["label"] == "normal":
            label = "N1"
            block_type = "normal"

        else:
            label = item["label"]
            block_type = "action"

        rows.append({
            "video": item["video"],
            "label": label,
            "block_type": block_type,
            "start_frame": 0,
            "end_frame": frame_count - 1,
            "source": "mydata"
        })

    return pd.DataFrame(rows)


def generate_augmented_annotations(base_annotation_df):

    rows = []
    aug_root = Path("./videos/augmented")

    for label_dir in aug_root.iterdir():

        if not label_dir.is_dir():
            continue

        for video_path in label_dir.glob("*.mp4"):

            aug_video = video_path.name

            if "_aug" not in aug_video:
                continue

            base_video = aug_video.split("_aug")[0] + ".mp4"

            base_rows = base_annotation_df[
                base_annotation_df["video"] == base_video
            ]

            if len(base_rows) == 0:
                print(f"원본 annotation 없음 : {base_video}")
                continue

            for _, row in base_rows.iterrows():

                new_row = row.copy()

                new_row["video"] = aug_video
                new_row["source"] = "augmented"

                rows.append(new_row)

    return pd.DataFrame(rows)

def merge_annotations(
    aihub_df,
    mydata_df,
    augmented_df
):

    return pd.concat(
        [
            aihub_df,
            mydata_df,
            augmented_df
        ],
        ignore_index=True
    )


def main():

    print("=" * 60)
    print("Loading video infos...")

    video_infos = load_video_infos(
        split="train",
        include_mydata=True,
        include_augmented=False
    )

    print("Reading AIHub annotations...")
    aihub_df = read_aihub_annotations()

    print("Generating mydata annotations...")
    mydata_df = generate_mydata_annotations(video_infos)

    print("Generating augmented annotations...")
    base_annotation_df = pd.concat(
        [
            aihub_df,
            mydata_df
        ],
        ignore_index=True
    )

    augmented_df = generate_augmented_annotations(
        base_annotation_df
    )

    print("Merging annotations...")

    final_df = merge_annotations(
        aihub_df,
        mydata_df,
        augmented_df
    )

    final_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)
    print("완료")

    print(f"AIHub      : {len(aihub_df)}")
    print(f"MyData     : {len(mydata_df)}")
    print(f"Augmented  : {len(augmented_df)}")
    print("--------------------------")
    print(f"Total      : {len(final_df)}")

    print()
    print(final_df["source"].value_counts())

    print()
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()