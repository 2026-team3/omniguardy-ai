import tempfile
import unittest
from pathlib import Path

from audio_guard.infrastructure.dataset.field_dataset import split_field_three_way


class FieldDatasetTest(unittest.TestCase):
    def _create_manifest(self, root, rows):
        manifest = root / "field_splits.csv"
        manifest.write_text(
            "filename,split,group_id\n" + "\n".join(rows), encoding="utf-8")
        return manifest

    def test_manifest_assigns_each_file_once_without_group_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for label in ("normal", "abnormal"):
                (root / label).mkdir()
                for index, split in enumerate(("train", "validation", "test")):
                    path = root / label / f"take_{index}.wav"
                    path.touch()
                    rows.append(f"{label}/{path.name},{split},{label}_session_{index}")
            splits = split_field_three_way(root, self._create_manifest(root, rows))
            self.assertEqual({len(items) for items in splits.values()}, {2})
            self.assertTrue(all({label for _, label in items} == {0, 1}
                                for items in splits.values()))

    def test_manifest_rejects_a_recording_group_in_multiple_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for label in ("normal", "abnormal"):
                (root / label).mkdir()
                for index, split in enumerate(("train", "validation", "test")):
                    path = root / label / f"take_{index}.wav"
                    path.touch()
                    group = "shared" if index < 2 else f"{label}_test"
                    rows.append(f"{label}/{path.name},{split},{group}")
            with self.assertRaises(ValueError):
                split_field_three_way(root, self._create_manifest(root, rows))
