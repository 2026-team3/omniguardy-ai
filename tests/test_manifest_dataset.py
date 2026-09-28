import csv
import tempfile
import unittest
from pathlib import Path

from audio_guard.infrastructure.dataset.manifest_dataset import load_manifest


FIELDS = [
    "path", "class_name", "dataset", "source_id", "session_id", "group_id", "split"
]


class ManifestDatasetTest(unittest.TestCase):
    def _write(self, root, rows):
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_loads_group_safe_three_way_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, split in enumerate(("train", "validation", "test")):
                audio = root / f"{split}.wav"
                audio.touch()
                rows.append({
                    "path": audio.name,
                    "class_name": ("background", "knock", "handle")[index],
                    "dataset": "direct",
                    "source_id": f"source-{index}",
                    "session_id": f"session-{index}",
                    "group_id": f"group-{index}",
                    "split": split,
                })
            self.assertEqual(len(load_manifest(self._write(root, rows))), 3)

    def test_rejects_session_crossing_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, split in enumerate(("train", "validation", "test")):
                audio = root / f"{split}.wav"
                audio.touch()
                rows.append({
                    "path": audio.name,
                    "class_name": "background",
                    "dataset": "direct",
                    "source_id": f"source-{index}",
                    "session_id": "same-session" if index < 2 else "other-session",
                    "group_id": f"group-{index}",
                    "split": split,
                })
            with self.assertRaisesRegex(ValueError, "session_id crosses splits"):
                load_manifest(self._write(root, rows))


if __name__ == "__main__":
    unittest.main()
