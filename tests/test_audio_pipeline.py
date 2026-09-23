import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_guard.application.evaluate_model import choose_threshold, field_recall_metrics
from audio_guard.config import load_config
from audio_guard.domain.pipeline import create_pipeline
from audio_guard.domain.risk_policy import predict_label
from audio_guard.labels import TARGET_CLASSES


class AudioPipelineTest(unittest.TestCase):
    def test_v4_runtime_manifest_controls_labels_and_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_file = root / "model.config.json"
            config_file.write_text(json.dumps({
                "model_path": "model.keras",
                "pipeline": "v4",
                "sample_rate": 22050,
                "labels": {"normal": 0, "abnormal": 1},
                "esc_abnormal_categories": [
                    "door_wood_knock", "door_wood_creaks", "glass_breaking",
                    "siren", "chainsaw", "footsteps",
                ],
                "threshold": 0.42,
            }), encoding="utf-8")

            config = load_config(config_file)
            self.assertEqual(config["model_path"], str(root / "model.keras"))
            self.assertEqual(predict_label(0.41, config["threshold"]), "normal")
            self.assertEqual(predict_label(0.42, config["threshold"]), "abnormal")
            features = create_pipeline("v4").transform(
                np.zeros(22050 * 3), 22050)
            self.assertEqual(features.shape, (1, 128, 128, 1))

    def test_unknown_pipeline_is_rejected(self):
        with self.assertRaises(ValueError):
            create_pipeline("unsupported")

    def test_threshold_uses_esc_validation_scores_and_recall_goal(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.3, 0.4, 0.8])
        selected = choose_threshold(
            labels, scores, min_recall=1.0, candidates=(0.3, 0.4, 0.8))
        self.assertEqual(selected["threshold"], 0.4)
        self.assertEqual(selected["recall"], 1.0)

    def test_field_abnormal_reports_recall_and_false_negatives_only(self):
        result = field_recall_metrics(
            np.ones(3, dtype=np.int32),
            np.array([0.9, 0.7, 0.2]),
            threshold=0.5,
        )
        self.assertEqual(result, {
            "threshold": 0.5,
            "recall": 2 / 3,
            "tp": 2,
            "fn": 1,
            "total": 3,
        })
        self.assertNotIn("accuracy", result)
        self.assertNotIn("precision", result)
        self.assertNotIn("f1", result)

    def test_all_existing_esc_target_classes_are_preserved(self):
        self.assertEqual(TARGET_CLASSES, frozenset({
            "door_wood_knock",
            "door_wood_creaks",
            "glass_breaking",
            "siren",
            "chainsaw",
            "footsteps",
        }))


if __name__ == "__main__":
    unittest.main()
