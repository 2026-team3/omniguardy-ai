import unittest

import numpy as np

from audio_guard.domain.pipeline.door_event_pipeline import audio_windows


class AudioWindowsTest(unittest.TestCase):
    def test_short_audio_is_padded_to_one_second(self):
        windows = list(audio_windows(np.ones(100), 100, 1.0, 0.2))
        self.assertEqual(len(windows), 1)
        self.assertEqual(len(windows[0].samples), 100)
        self.assertEqual(windows[0].start_seconds, 0.0)

    def test_uses_point_two_second_hop_and_covers_end(self):
        windows = list(audio_windows(np.ones(230), 100, 1.0, 0.2))
        self.assertEqual(
            [round(window.start_seconds, 1) for window in windows],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3],
        )
        self.assertEqual(len(windows[-1].samples), 100)

    def test_rejects_hop_larger_than_window(self):
        with self.assertRaises(ValueError):
            list(audio_windows(np.ones(100), 100, 1.0, 1.1))


if __name__ == "__main__":
    unittest.main()
