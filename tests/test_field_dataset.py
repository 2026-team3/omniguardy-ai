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

    def test_abnormal_only_manifest_assigns_each_file_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "abnormal").mkdir()
            rows = []
            for index, split in enumerate(("train", "validation", "test")):
                path = root / "abnormal" / f"take_{index}.wav"
                path.touch()
                rows.append(f"abnormal/{path.name},{split},session_{index}")
            splits = split_field_three_way(root, self._create_manifest(root, rows))
            self.assertEqual({len(items) for items in splits.values()}, {1})
            self.assertTrue(all({label for _, label in items} == {1}
                                for items in splits.values()))

    def test_manifest_rejects_a_recording_group_in_multiple_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "abnormal").mkdir()
            rows = []
            for index, split in enumerate(("train", "validation", "test")):
                path = root / "abnormal" / f"take_{index}.wav"
                path.touch()
                group = "shared" if index < 2 else "test_session"
                rows.append(f"abnormal/{path.name},{split},{group}")
            with self.assertRaises(ValueError):
                split_field_three_way(root, self._create_manifest(root, rows))

    def test_each_split_requires_at_least_one_abnormal_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "abnormal").mkdir()
            rows = []
            for index, split in enumerate(("train", "validation")):
                path = root / "abnormal" / f"take_{index}.wav"
                path.touch()
                rows.append(f"abnormal/{path.name},{split},session_{index}")
            with self.assertRaisesRegex(ValueError, "test split"):
                split_field_three_way(root, self._create_manifest(root, rows))
