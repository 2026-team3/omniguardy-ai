import tempfile
import unittest
from pathlib import Path

import numpy as np

from field_dataset import split_field_files


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


if __name__ == "__main__":
    unittest.main()
