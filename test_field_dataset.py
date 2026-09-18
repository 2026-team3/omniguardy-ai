import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_guard.infrastructure.dataset.field_dataset import (
    split_field_files,
    split_field_three_way,
)


class FieldDatasetTest(unittest.TestCase):
    def test_parent_directory_defines_label_and_files_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ("normal", "abnormal"):
                (root / folder).mkdir()
                for index in range(5):
                    (root / folder / f"take_{index}.wav").touch()
            train, val, train_y, val_y = split_field_files(root)
            self.assertFalse(set(train) & set(val))
            self.assertEqual(len(train) + len(val), 10)
            self.assertEqual(set(np.unique(train_y)), {0, 1})
            self.assertEqual(set(np.unique(val_y)), {0, 1})
            for path, label in list(zip(train, train_y)) + list(zip(val, val_y)):
                self.assertEqual(label, int(path.parent.name == "abnormal"))

    def test_manifest_keeps_recording_groups_in_one_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = ["filename,split,group_id"]
            for folder in ("normal", "abnormal"):
                (root / folder).mkdir()
                for index, split in enumerate(("train", "train", "validation", "test")):
                    name = f"take_{index}.wav"
                    (root / folder / name).touch()
                    rows.append(f"{folder}/{name},{split},session_{index}")
            manifest = root / "splits.csv"
            manifest.write_text("\n".join(rows), encoding="utf-8")
            splits = split_field_three_way(root, manifest)
            self.assertEqual(len(splits["train"]), 4)
            self.assertEqual(len(splits["validation"]), 2)
            self.assertEqual(len(splits["test"]), 2)
            rows[-1] = "abnormal/take_3.wav,train,session_3"
            manifest.write_text("\n".join(rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                split_field_three_way(root, manifest)


if __name__ == "__main__":
    unittest.main()
