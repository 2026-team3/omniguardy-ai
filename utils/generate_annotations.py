"""AIHub, mydata, augmentation을 합쳐 train 전용 manifest를 만든다."""

from pathlib import Path

import cv2
import pandas as pd


AIHUB_ANNOTATION = Path("splits/annotations/train_annotations.csv")
OUTPUT_PATH = Path("splits/annotations/train_annotations_all.csv")
MYDATA_ROOT = Path("videos/mydata")
AUGMENTED_ROOT = Path("videos/augmented")
TARGET_LABELS = {"N1", "A17", "A18", "A19", "A20", "A21"}

MYDATA_CONFIG = {
    "normal": {"label": "N1", "block_type": "normal", "normal_subtype": "resident"},
    "delivery": {"label": "N1", "block_type": "normal", "normal_subtype": "delivery"},
    "lookingInside": {"label": "A20", "block_type": "action", "normal_subtype": pd.NA},
}


def get_frame_count(video_path: Path) -> int | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"CANNOT OPEN: {video_path}")
        return None
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frame_count if frame_count > 0 else None


def generate_mydata_annotations() -> pd.DataFrame:
    rows = []
    for folder, config in MYDATA_CONFIG.items():
        for video_path in sorted((MYDATA_ROOT / folder).glob("*.mp4")):
            frame_count = get_frame_count(video_path)
            if frame_count is None:
                continue
            rows.append({
                "split": "train",
                "video": video_path.name,
                "annotation_video": pd.NA,
                "video_path": str(video_path.resolve()),
                "json_path": pd.NA,
                "label": config["label"],
                "block_type": config["block_type"],
                "start_frame": 0,
                "end_frame": frame_count - 1,
                "source": "mydata",
                "normal_subtype": config["normal_subtype"],
            })
    return pd.DataFrame(rows)


def generate_augmented_annotations(base_annotations: pd.DataFrame) -> pd.DataFrame:
    """train 원본에서 파생된 augmentation 영상에만 부모 블록을 복사한다."""
    rows = []
    base_by_video = {
        video: group
        for video, group in base_annotations.groupby("video", dropna=False)
    }

    for augmented_path in AUGMENTED_ROOT.rglob("*.mp4"):
        augmented_name = augmented_path.name
        if "_aug" not in augmented_name:
            continue

        base_name = augmented_name.split("_aug", 1)[0] + augmented_path.suffix
        parent_blocks = base_by_video.get(base_name)
        if parent_blocks is None:
            print(f"NO PARENT ANNOTATION: {augmented_path}")
            continue

        frame_count = get_frame_count(augmented_path)
        if frame_count is None:
            continue

        last_frame = frame_count - 1
        max_parent_end = int(parent_blocks["end_frame"].max())
        if max_parent_end > last_frame:
            print(
                f"TRUNCATED BY {max_parent_end - last_frame} FRAME(S): "
                f"{augmented_path}"
            )

        for _, parent in parent_blocks.iterrows():
            if int(parent["start_frame"]) > last_frame:
                print(f"BLOCK OUTSIDE AUGMENTED VIDEO: {augmented_path}")
                continue
            row = parent.to_dict()
            row.update({
                "video": augmented_name,
                "video_path": str(augmented_path.resolve()),
                "source": "augmented",
                "split": "train",
                "end_frame": min(int(parent["end_frame"]), last_frame),
            })
            rows.append(row)

    return pd.DataFrame(rows, columns=base_annotations.columns)


def main() -> None:
    aihub = pd.read_csv(AIHUB_ANNOTATION)
    aihub = aihub[aihub["label"].isin(TARGET_LABELS)].copy()
    aihub["normal_subtype"] = pd.NA

    mydata = generate_mydata_annotations()
    base = pd.concat([aihub, mydata], ignore_index=True)
    augmented = generate_augmented_annotations(base)
    final = pd.concat([base, augmented], ignore_index=True)

    if not set(final["label"]).issubset(TARGET_LABELS):
        raise ValueError("Unexpected label in train manifest")

    final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved -> {OUTPUT_PATH}")
    print("\nRows by source")
    print(final["source"].value_counts().to_string())
    print("\nRows by label")
    print(final["label"].value_counts().sort_index().to_string())
    print("\nMydata normal subtypes")
    print(mydata["normal_subtype"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

