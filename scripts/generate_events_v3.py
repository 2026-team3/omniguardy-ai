from pathlib import Path
import csv

from generate_events import detect_events


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT.parent / "dataset"

V3_MANIFEST_PATH = DATASET_DIR / "dataset_manifest_v3.csv"

OLD_EVENTS_PATH = DATASET_DIR / "events.csv"
V3_EVENTS_PATH = DATASET_DIR / "events_v3.csv"


# v3에서 새로 추가된 positive recording
NEW_RECORDING_IDS = {
    "K031",
    "K032",
    "K033",
    "K034",
    "K035",
    "H031",
    "H032",
    "H033",
    "H034",
    "H035",
    "H036",
    "H037",
    "H038",
    "H039",
    "H040",
}


def main():

    # --------------------------------------------------
    # 1. 기존 v1/v2 event annotation 그대로 읽기
    # --------------------------------------------------

    with OLD_EVENTS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        old_rows = list(csv.DictReader(f))

    print(f"Existing events: {len(old_rows)}")


    # --------------------------------------------------
    # 2. v3 manifest 읽기
    # --------------------------------------------------

    with V3_MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        manifest = list(csv.DictReader(f))


    # --------------------------------------------------
    # 3. 신규 positive 데이터만 event detection
    # --------------------------------------------------

    new_rows = []

    for item in manifest:

        recording_id = item["recording_id"]

        if recording_id not in NEW_RECORDING_IDS:
            continue

        class_name = item["class_name"]

        if class_name == "background":
            continue

        audio_path = DATASET_DIR / item["path"]

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        print()
        print(
            f"Processing: {recording_id} "
            f"({class_name}) -> {audio_path.name}"
        )

        events = detect_events(
            audio_path,
            class_name,
        )

        print(
            f"  detected events: {len(events)}"
        )

        for start, end in events:

            new_rows.append(
                {
                    "recording_id": recording_id,
                    "class_name": class_name,
                    "event_start_seconds": start,
                    "event_end_seconds": end,
                }
            )


    # --------------------------------------------------
    # 4. 기존 annotation + 신규 annotation
    # --------------------------------------------------

    all_rows = old_rows + new_rows


    # --------------------------------------------------
    # 5. events_v3.csv 생성
    # --------------------------------------------------

    with V3_EVENTS_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "recording_id",
                "class_name",
                "event_start_seconds",
                "event_end_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(all_rows)


    # --------------------------------------------------
    # 6. 결과 출력
    # --------------------------------------------------

    knock_events = sum(
        row["class_name"] == "knock"
        for row in new_rows
    )

    handle_events = sum(
        row["class_name"] == "handle"
        for row in new_rows
    )

    print()
    print("=" * 60)
    print("V3 EVENT ANNOTATION RESULT")
    print("=" * 60)

    print(f"Existing events : {len(old_rows)}")
    print(f"New knock       : {knock_events}")
    print(f"New handle      : {handle_events}")
    print(f"New total       : {len(new_rows)}")
    print(f"All events      : {len(all_rows)}")

    print()
    print(f"Created: {V3_EVENTS_PATH}")


if __name__ == "__main__":
    main()