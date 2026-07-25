import json
from pathlib import Path
import pandas as pd

print("cwd :", Path.cwd())
# 추출할 block_type
VALID_BLOCK_TYPES = {"action", "normal"}


def load_annotation(json_path):
    """
    하나의 AIHub annotation JSON을 읽어서
    action/normal block을 DataFrame으로 반환
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_name = data["metadata"]["filename"]
    blocks = data["file"][0]["videos"]["block_information"]

    rows = []

    for block in blocks:

        # action, normal만 사용
        if block["block_type"] not in VALID_BLOCK_TYPES:
            continue

        rows.append({
            "video": video_name,
            "label": block["block_detail"],
            "block_type": block["block_type"],
            "start_frame": int(block["start_frame_index"]),
            "end_frame": int(block["end_frame_index"]),
            "source": "AIHub"
        })

    return pd.DataFrame(rows)


def load_annotations(pair_json):
    """
    train_pairs.json / valid_pairs.json / test_pairs.json을 읽어서
    해당 split의 annotation만 DataFrame으로 반환
    """

    with open(pair_json, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    root = Path("videos/unzipped")
    dfs = []

    for pair in pairs:

        # 파일 이름 (확장자 제외)
        name = pair["name"]

        # videos/unzipped 아래에서 재귀적으로 검색
        matches = list(root.rglob(f"{name}.json"))

        if len(matches) == 0:
            print(f"NOT FOUND : {name}.json")
            continue

        json_path = matches[0]

        try:
            df = load_annotation(json_path)

            if not df.empty:
                dfs.append(df)

        except Exception as e:
            print(f"Error : {json_path}")
            print(e)

    if not dfs:
        return pd.DataFrame(columns=[
            "video",
            "label",
            "block_type",
            "start_frame",
            "end_frame",
            "source"
        ])

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":

    split_files = {
        "train": "splits/train_pairs.json",
        "valid": "splits/valid_pairs.json",
        "test": "splits/test_pairs.json"
    }
    output_dir = Path("splits/annotations")
    output_dir.mkdir(parents=True, exist_ok=True)
    

    for split_name, pair_json in split_files.items():

        annotations = load_annotations(pair_json)

        print(f"\n[{split_name}]")
        print(annotations.head())
        print(f"Total blocks : {len(annotations)}")

        if not annotations.empty:
            print(annotations["block_type"].value_counts())

        annotations.to_csv(
            output_dir / f"{split_name}_annotations.csv",
            index=False
        )

        print(f"Saved {split_name}_annotations.csv")