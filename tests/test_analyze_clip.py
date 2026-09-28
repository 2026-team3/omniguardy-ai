import unittest

import numpy as np

from audio_guard.application.analyze_clip import AnalyzeClip
from audio_guard.domain.audio_clip import AudioClip


class FakePipeline:
    def iter_features(self, audio, sample_rate):
        for offset in (0.0, 0.2, 0.4):
            yield np.zeros((128, 128, 1)), offset


class FakeRepository:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)
        self.call_count = 0

    def predict_probabilities(self, model_input):
        result = self.probabilities[self.call_count:self.call_count + 1]
        self.call_count += 1
        return result


class AnalyzeClipTest(unittest.TestCase):
    def _analyzer(self, probabilities):
        analyzer = AnalyzeClip(
            FakeRepository(probabilities),
            "door_event",
            {"knock": 0.7, "handle": 0.7},
            cooldown_seconds=1.0,
        )
        analyzer.pipeline = FakePipeline()
        return analyzer

    def test_returns_first_threshold_crossing_window(self):
        analyzer = self._analyzer([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        result = analyzer.execute(AudioClip(np.zeros(22050), 22050))
        self.assertEqual(result.status, "KNOCK_EVENT")
        self.assertAlmostEqual(result.window_start_seconds, 0.2)
        self.assertEqual(analyzer.model_repository.call_count, 2)

    def test_second_call_is_suppressed_by_cooldown(self):
        probabilities = [
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
        ]
        analyzer = self._analyzer(probabilities + probabilities)
        first = analyzer.execute(AudioClip(np.zeros(22050), 22050))
        second = analyzer.execute(AudioClip(np.zeros(22050), 22050))
        self.assertEqual(first.status, "KNOCK_EVENT")
        self.assertEqual(second.status, "NO_EVENT")
        self.assertTrue(second.cooldown_suppressed)


if __name__ == "__main__":
    unittest.main()
