import unittest
from pathlib import Path

from audio_guard.config import load_config
from audio_guard.domain.risk_policy import predict_label


class DataAllConfigTest(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = (
            self.project_root / "models" / "audio_model_data_all.config.json"
        )

    def test_data_all_config_uses_v4_model_and_threshold(self):
        config = load_config(self.config_path)

        self.assertEqual(
            Path(config["model_path"]),
            self.project_root / "models" / "audio_model_data_all.keras",
        )
        self.assertEqual(config["pipeline"], "v4")
        self.assertEqual(config["sample_rate"], 22050)
        self.assertEqual(config["threshold"], 0.6)

    def test_threshold_includes_boundary_as_abnormal(self):
        self.assertEqual(predict_label(0.6, 0.6), "abnormal")
        self.assertEqual(predict_label(0.5999, 0.6), "normal")


if __name__ == "__main__":
    unittest.main()
