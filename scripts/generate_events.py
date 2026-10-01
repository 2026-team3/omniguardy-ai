from pathlib import Path
import csv

import librosa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT.parent / "dataset"

MANIFEST_PATH = DATASET_DIR / "dataset_manifest.csv"
EVENTS_PATH = DATASET_DIR / "events.csv"

SAMPLE_RATE = 22050


def detect_events(audio_path: Path, class_name: str):
    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    # 급격한 소리 변화(onset) 탐지
    hop_length = 512

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        backtrack=True,
        delta=0.30,
        wait=8,
    )

    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=sr,
    )

    if len(onset_times) == 0:
        return []

    # 하나의 실제 동작에서 여러 onset이 발생한 경우 합치기
    merge_gap = 1.0 if class_name == "knock" else 1.5

    groups = []
    current = [float(onset_times[0])]

    for time in onset_times[1:]:
        time = float(time)

        if time - current[-1] <= merge_gap:
            current.append(time)
        else:
            groups.append(current)
            current = [time]

    groups.append(current)

    duration = librosa.get_duration(
        y=y,
        sr=sr,
    )

    events = []

    for group in groups:
        first = group[0]
        last = group[-1]

        if class_name == "knock":
            start = max(0.0, first - 0.15)
            end = min(duration, last + 0.45)

        else:  # handle
            start = max(0.0, first - 0.20)
            end = min(duration, last + 0.80)

        events.append(
            (
                round(start, 3),
                round(end, 3),
            )
        )

    return events


def main():
    rows = []

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        manifest = list(csv.DictReader(f))

    for item in manifest:
        recording_id = item["recording_id"]
        class_name = item["class_name"]

        # background에는 이벤트 annotation을 만들지 않음
        if class_name == "background":
            continue

        audio_path = DATASET_DIR / item["path"]

        print(
            f"Processing: {recording_id} -> {audio_path.name}"
        )

        events = detect_events(
            audio_path,
            class_name,
        )

        print(
            f"  detected events: {len(events)}"
        )

        for start, end in events:
            rows.append(
                {
                    "recording_id": recording_id,
                    "class_name": class_name,
                    "event_start_seconds": start,
                    "event_end_seconds": end,
                }
            )

    with EVENTS_PATH.open(
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
        writer.writerows(rows)

    print()
    print(f"Created: {EVENTS_PATH}")
    print(f"Total events: {len(rows)}")


if __name__ == "__main__":
    main()