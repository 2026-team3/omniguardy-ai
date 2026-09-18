import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_guard.application.evaluate_model import choose_threshold
from audio_guard.config import load_config
from audio_guard.domain.pipeline import create_pipeline
from audio_guard.domain.risk_policy import predict_label
from audio_guard.domain.pipeline.v5_bag import audio_to_mel_bag


class AudioV5Test(unittest.TestCase):
    def test_bag_is_bounded_to_one_recording(self):
        audio = np.zeros(22050 * 12, dtype=np.float32)
        bag = audio_to_mel_bag(audio, 22050, max_windows=4)
        self.assertEqual(bag.shape, (4, 128, 128))

    def test_runtime_manifest_controls_labels_and_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_file = root / "model.config.json"
            config_file.write_text(json.dumps({
                "model_path": "model.keras", "pipeline": "v5",
                "sample_rate": 22050, "labels": {"normal": 0, "abnormal": 1},
                "esc_abnormal_categories": [
                    "door_wood_knock", "door_wood_creaks", "glass_breaking",
                    "siren", "chainsaw", "footsteps"],
                "threshold": 0.42, "max_windows": 4,
            }), encoding="utf-8")
            config = load_config(config_file)
            self.assertEqual(config["model_path"], str(root / "model.keras"))
            self.assertEqual(predict_label(0.41, config["threshold"]), "normal")
            self.assertEqual(predict_label(0.42, config["threshold"]), "abnormal")
            pipeline = create_pipeline(config["pipeline"], config["max_windows"])
            features = pipeline.transform(np.zeros(22050 * 3), 22050)
            self.assertEqual(features.shape, (1, 4, 128, 128, 1))

    def test_threshold_uses_validation_scores_and_field_recall_goal(self):
        esc_y = np.array([0, 0, 1, 1])
        esc_scores = np.array([0.1, 0.2, 0.7, 0.9])
        field_y = np.array([0, 0, 1, 1])
        field_scores = np.array([0.2, 0.3, 0.4, 0.8])
        threshold, _, field, _ = choose_threshold(
            esc_y, esc_scores, field_y, field_scores, min_recall=1.0)
        self.assertLessEqual(threshold, 0.4)
        self.assertEqual(field["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
