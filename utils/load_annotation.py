"""TL/TS와 VL/VS 구조에서 AIHub 블록 annotation을 생성한다."""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd


VALID_BLOCK_TYPES = {"action", "normal"}
TARGET_LABELS = {"N1", "A17", "A18", "A19", "A20", "A21"}
MANIFEST_COLUMNS = [
    "split", "video", "annotation_video", "video_path", "json_path",
    "label", "block_type", "start_frame", "end_frame", "source",
]
SPLIT_ROOTS = {
    "train": (Path("videos/unzipped/TL"), Path("videos/unzipped/TS")),
    "valid": (Path("videos/unzipped/VL"), Path("videos/unzipped/VS")),
}


def build_name_index(root: Path, suffix: str) -> dict[str, list[Path]]:
    """폴더를 한 번만 스캔하여 파일명별 실제 경로를 만든다."""
    index: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob(f"*{suffix}"):
        index[path.name].append(path)
    return index


def frame_count(video_path: Path) -> int:
    """MP4의 실제 프레임 수를 반환한다."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count <= 0:
        raise ValueError(f"유효한 프레임 수가 없습니다: {video_path}")
    return count


def read_blocks(json_path: Path) -> tuple[str, list[dict]]:
    """JSON의 annotation 영상명과 대상 블록 목록을 읽는다."""
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    annotation_video = data["metadata"]["filename"]
    blocks = data["file"][0]["videos"]["block_information"]
    return annotation_video, blocks


def make_rows(
    json_path: Path,
    video_path: Path,
    split: str,
) -> tuple[list[dict], list[dict]]:
    """JSON 하나를 블록 행으로 변환하고 프레임 보정 내역을 반환한다."""
    annotation_video, blocks = read_blocks(json_path)
    last_frame = frame_count(video_path) - 1
    rows: list[dict] = []
    notices: list[dict] = []

    for block in blocks:
        block_type = block.get("block_type")
        label = block.get("block_detail")
        if block_type not in VALID_BLOCK_TYPES or label not in TARGET_LABELS:
            continue

        start_frame = int(block["start_frame_index"])
        end_frame = int(block["end_frame_index"])
        if start_frame < 0 or end_frame < start_frame:
            notices.append({
                "split": split, "issue": "invalid_frame_range",
                "json_path": str(json_path.resolve()), "video_path": str(video_path.resolve()),
                "detail": f"{start_frame}~{end_frame}",
            })
            continue
        if start_frame > last_frame:
            notices.append({
                "split": split, "issue": "block_outside_video",
                "json_path": str(json_path.resolve()), "video_path": str(video_path.resolve()),
                "detail": f"{start_frame}~{end_frame}, last={last_frame}",
            })
            continue
        if end_frame > last_frame:
            notices.append({
                "split": split, "issue": "end_frame_clipped",
                "json_path": str(json_path.resolve()), "video_path": str(video_path.resolve()),
                "detail": f"{end_frame}->{last_frame}",
            })
            end_frame = last_frame

        rows.append({
            "split": split,
            "video": video_path.name,
            "annotation_video": annotation_video,
            "video_path": str(video_path.resolve()),
            "json_path": str(json_path.resolve()),
            "label": label,
            "block_type": block_type,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "source": "AIHub",
        })
    return rows, notices


def create_manifest(split: str, json_root: Path, video_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """동일 파일명 JSON과 MP4를 직접 연결해 split manifest를 만든다."""
    json_index = build_name_index(json_root, ".json")
    video_index = build_name_index(video_root, ".mp4")
    rows: list[dict] = []
    notices: list[dict] = []

    for filename, json_paths in sorted(json_index.items()):
        video_name = f"{Path(filename).stem}.mp4"
        video_paths = video_index.get(video_name, [])
        if len(json_paths) != 1 or len(video_paths) != 1:
            # 다운로드 대상 행동 코드가 파일명에 있는 경우만 누락으로 알린다.
            if not any(f"_{label}_" in filename for label in TARGET_LABELS - {"N1"}):
                continue
            notices.append({
                "split": split,
                "issue": "ambiguous_json" if len(json_paths) > 1 else "missing_video" if not video_paths else "ambiguous_video",
                "json_path": " | ".join(str(path.resolve()) for path in json_paths),
                "video_path": " | ".join(str(path.resolve()) for path in video_paths),
                "detail": filename,
            })
            continue
        try:
            block_rows, block_notices = make_rows(json_paths[0], video_paths[0], split)
            rows.extend(block_rows)
            notices.extend(block_notices)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            notices.append({
                "split": split, "issue": "invalid_annotation",
                "json_path": str(json_paths[0].resolve()),
                "video_path": str(video_paths[0].resolve()),
                "detail": str(error),
            })

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    issue_columns = ["split", "issue", "json_path", "video_path", "detail"]
    return manifest, pd.DataFrame(notices, columns=issue_columns)


def main() -> None:
    output_root = Path("splits/annotations")
    output_root.mkdir(parents=True, exist_ok=True)

    for split, (json_root, video_root) in SPLIT_ROOTS.items():
        manifest, notices = create_manifest(split, json_root, video_root)
        output_path = output_root / f"{split}_annotations.csv"
        manifest.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"[{split}]")
        print(f"블록: {len(manifest)}")
        print(f"영상: {manifest['video_path'].nunique()}")
        print(f"검증 알림: {len(notices)}")
        print(manifest["label"].value_counts().sort_index().to_string())
        print(f"저장: {output_path}\n")


if __name__ == "__main__":
    main()


