import tempfile
import unittest
from pathlib import Path

from audio_guard.infrastructure.dataset.field_dataset import split_field_three_way


class FieldDatasetTest(unittest.TestCase):
    def _create_manifest(self, root, rows, include_event_type=False):
        manifest = root / "field_splits.csv"
        header = "filename,split,group_id,event_type" if include_event_type else (
            "filename,split,group_id")
        manifest.write_text(
            header + "\n" + "\n".join(rows), encoding="utf-8")
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
            self.assertTrue(all({example.label for example in items} == {1}
                                for items in splits.values()))
            self.assertTrue(all({example.event_type for example in items} == {"other"}
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

    def test_doorbell_is_ignored_and_event_types_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "abnormal").mkdir()
            rows = []
            for index, split in enumerate(("train", "validation", "test")):
                path = root / "abnormal" / f"knock_{index}.wav"
                path.touch()
                rows.append(
                    f"abnormal/{path.name},{split},knock_{index},knock")
            doorbell = root / "abnormal" / "doorbell.wav"
            doorbell.touch()
            rows.append("abnormal/doorbell.wav,ignored,doorbell_1,doorbell")

            splits = split_field_three_way(
                root, self._create_manifest(root, rows, include_event_type=True))

            self.assertEqual(sum(map(len, splits.values())), 3)
            self.assertTrue(all(
                example.event_type == "knock"
                for examples in splits.values() for example in examples))

    def test_doorbell_must_use_ignored_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "abnormal").mkdir()
            rows = []
            for index, split in enumerate(("train", "validation", "test")):
                path = root / "abnormal" / f"take_{index}.wav"
                path.touch()
                event_type = "doorbell" if split == "train" else "knock"
                rows.append(
                    f"abnormal/{path.name},{split},session_{index},{event_type}")
            with self.assertRaisesRegex(ValueError, "doorbell"):
                split_field_three_way(
                    root, self._create_manifest(
                        root, rows, include_event_type=True))

    def test_event_type_groups_must_be_distributed_across_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "abnormal").mkdir()
            rows = []
            for index in range(3):
                path = root / "abnormal" / f"knock_{index}.wav"
                path.touch()
                rows.append(f"abnormal/{path.name},train,knock_{index},knock")
            for index, split in enumerate(("train", "validation", "test")):
                path = root / "abnormal" / f"impact_{index}.wav"
                path.touch()
                rows.append(f"abnormal/{path.name},{split},impact_{index},impact")

            with self.assertRaisesRegex(ValueError, "event_type must span"):
                split_field_three_way(
                    root, self._create_manifest(
                        root, rows, include_event_type=True))
