import json
import tempfile
import unittest
from pathlib import Path

from audio_guard.config import load_config


class ConfigTest(unittest.TestCase):
    def test_loads_three_class_threshold_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps({
                "model_path": "model.keras",
                "pipeline": "door_event",
                "sample_rate": 22050,
                "labels": {"background": 0, "knock": 1, "handle": 2},
                "thresholds": {"knock": 0.71, "handle": 0.64},
                "window_seconds": 1.0,
                "hop_seconds": 0.2,
                "cooldown_seconds": 1.0,
            }), encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config["thresholds"]["knock"], 0.71)
            self.assertEqual(config["pipeline"], "door_event")


if __name__ == "__main__":
    unittest.main()
