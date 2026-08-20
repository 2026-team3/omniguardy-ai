"""블록 단위 annotation manifest에서 중복 없는 실제 영상 목록을 반환한다."""

from pathlib import Path

import pandas as pd


ANNOTATION_ROOT = Path("splits/annotations")
MYDATA_ROOT = Path("videos/mydata")


MYDATA_CONFIG = {
    "normal": "N1",
    "delivery": "N1",
    "lookingInside": "A20",
}


def _mydata_video_rows() -> list[dict]:
    rows = []
    for folder, label in MYDATA_CONFIG.items():
        for path in sorted((MYDATA_ROOT / folder).glob("*.mp4")):
            rows.append({
                "video": path.name,
                "label": label,
                "video_path": str(path.resolve()),
                "source": "mydata",
            })
    return rows


def load_video_infos(
    split: str = "train",
    include_mydata: bool = True,
    include_augmented: bool = False,
) -> list[dict]:
    """실제 영상은 한 번만 반환하고 블록 정보는 annotation CSV에 유지한다."""
    if split not in {"train", "valid"}:
        raise ValueError(f"Unknown split: {split}")
    if split != "train" and (include_mydata or include_augmented):
        # 평가 데이터는 split/source 누수를 막기 위해 AIHub 원본만 유지한다.
        include_mydata = False
        include_augmented = False

    annotation_name = (
        "train_annotations_all.csv"
        if split == "train" and include_augmented
        else f"{split}_annotations.csv"
    )
    annotations = pd.read_csv(ANNOTATION_ROOT / annotation_name)
    valid_sources = {"AIHub"}
    if split == "train" and include_mydata:
        valid_sources.add("mydata")
    if split == "train" and include_augmented:
        valid_sources.add("augmented")

    videos = annotations[annotations["source"].isin(valid_sources)].copy()

    if split == "train" and include_mydata and "mydata" not in set(videos["source"]):
        videos = pd.concat([videos, pd.DataFrame(_mydata_video_rows())], ignore_index=True)

    required = ["video", "label", "video_path", "source"]
    missing = set(required) - set(videos.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    videos = videos.dropna(subset=["video_path"])
    videos = videos.drop_duplicates(subset=["video_path", "source"])

    video_infos = []
    for row in videos.itertuples(index=False):
        video_path = Path(row.video_path)
        if not video_path.is_file():
            print(f"MISSING VIDEO: {video_path}")
            continue
        video_infos.append({
            "video": row.video,
            "label": row.label,
            "video_path": str(video_path),
            "source": row.source,
            "split": split,
        })
    return video_infos


if __name__ == "__main__":
    from collections import Counter

    for split in ["train", "valid"]:
        infos = load_video_infos(
            split,
            include_mydata=True,
            include_augmented=(split == "train"),
        )
        print(f"\n{split.upper()} videos: {len(infos)}")
        print("Source:", Counter(item["source"] for item in infos))
        print("Label:", Counter(item["label"] for item in infos))


