import csv
import unittest

from audio_guard.application.prepare_dataset import (
    assign_group_splits,
    prepare_manifest,
)


class PrepareDatasetTest(unittest.TestCase):

    def test_assigns_connected_rows_to_one_of_all_three_splits(self):
        rows = []
        classes = ("background", "knock", "handle")

        for index in range(30):
            rows.append({
                "recording_id": f"R{index:03d}",
                "path": f"{index}.wav",
                "class_name": classes[index % 3],
                "dataset": "direct",
                "source_id": f"source-{index}",
                "session_id": f"session-{index}",
                "group_id": f"group-{index}",
            })

        splits = assign_group_splits(rows)

        self.assertEqual(
            set(splits),
            {"train", "validation", "test"},
        )

        # 같은 seed에서는 동일한 split 결과
        self.assertEqual(
            splits,
            assign_group_splits(rows),
        )

    def test_keeps_same_session_together(self):
        rows = []

        for index in range(12):
            rows.append({
                "recording_id": f"R{index:03d}",
                "path": f"{index}.wav",
                "class_name": (
                    "background",
                    "knock",
                    "handle",
                )[index % 3],
                "dataset": "direct",
                "source_id": f"source-{index}",
                "session_id": (
                    "shared"
                    if index < 2
                    else f"session-{index}"
                ),
                "group_id": f"group-{index}",
            })

        splits = assign_group_splits(rows)

        # 같은 session은 서로 다른 split으로 갈 수 없음
        self.assertEqual(
            splits[0],
            splits[1],
        )

    def test_prepare_manifest_keeps_recording_id_and_adds_split(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as directory:
            directory = Path(directory)

            input_path = directory / "recordings.csv"
            output_path = directory / "dataset_manifest.csv"

            fieldnames = [
                "recording_id",
                "path",
                "class_name",
                "dataset",
                "source_id",
                "session_id",
                "group_id",
            ]

            classes = (
                "background",
                "knock",
                "handle",
            )

            with input_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )

                writer.writeheader()

                for index in range(30):
                    writer.writerow({
                        "recording_id": f"R{index:03d}",
                        "path": f"{index}.wav",
                        "class_name": classes[index % 3],
                        "dataset": "direct",
                        "source_id": f"source-{index}",
                        "session_id": f"session-{index}",
                        "group_id": f"group-{index}",
                    })

            prepare_manifest(
                input_path,
                output_path,
            )

            with output_path.open(
                newline="",
                encoding="utf-8-sig",
            ) as file:
                rows = list(
                    csv.DictReader(file)
                )

            self.assertEqual(
                len(rows),
                30,
            )

            self.assertEqual(
                rows[0]["recording_id"],
                "R000",
            )

            self.assertIn(
                "split",
                rows[0],
            )

            self.assertEqual(
                {row["split"] for row in rows},
                {"train", "validation", "test"},
            )


if __name__ == "__main__":
    unittest.main()
