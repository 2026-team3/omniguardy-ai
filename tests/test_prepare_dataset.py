import unittest

from audio_guard.application.prepare_dataset import assign_group_splits


class PrepareDatasetTest(unittest.TestCase):
    def test_assigns_connected_rows_to_one_of_all_three_splits(self):
        rows = []
        classes = ("background", "knock", "handle")
        for index in range(30):
            rows.append({
                "path": f"{index}.wav",
                "class_name": classes[index % 3],
                "dataset": "direct",
                "source_id": f"source-{index}",
                "session_id": f"session-{index}",
                "group_id": f"group-{index}",
            })
        splits = assign_group_splits(rows)
        self.assertEqual(set(splits), {"train", "validation", "test"})
        self.assertEqual(splits, assign_group_splits(rows))

    def test_keeps_same_session_together(self):
        rows = []
        for index in range(12):
            rows.append({
                "path": f"{index}.wav",
                "class_name": ("background", "knock", "handle")[index % 3],
                "dataset": "direct",
                "source_id": f"source-{index}",
                "session_id": "shared" if index < 2 else f"session-{index}",
                "group_id": f"group-{index}",
            })
        splits = assign_group_splits(rows)
        self.assertEqual(splits[0], splits[1])


if __name__ == "__main__":
    unittest.main()
