import unittest

import numpy as np

from audio_guard.application.calibrate_thresholds import choose_class_threshold
from audio_guard.application.evaluate_model import classification_metrics


class ThresholdCalibrationTest(unittest.TestCase):
    def test_selects_class_specific_threshold_with_background_fp(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        knock_scores = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.3])
        selected = choose_class_threshold(
            labels, knock_scores, 1, candidates=(0.4, 0.6, 0.8)
        )
        self.assertEqual(selected["threshold"], 0.6)
        self.assertEqual(selected["background_false_positives"], 0)

    def test_reports_background_false_positives_separately(self):
        result = classification_metrics(
            np.array([0, 0, 1, 2]), np.array([1, 2, 1, 2])
        )
        self.assertEqual(result["background_to_knock_false_positives"], 1)
        self.assertEqual(result["background_to_handle_false_positives"], 1)


if __name__ == "__main__":
    unittest.main()
