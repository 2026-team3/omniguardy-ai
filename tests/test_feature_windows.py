import unittest

import numpy as np

from audio_guard.domain.pipeline.v4_windowed import audio_windows


class AudioWindowsTest(unittest.TestCase):
    def test_short_audio_is_padded_to_one_window(self):
        windows = list(audio_windows(np.ones(10), sr=10,
                                     window_seconds=3, hop_seconds=1))
        self.assertEqual(len(windows), 1)
        self.assertEqual(len(windows[0]), 30)
        np.testing.assert_array_equal(windows[0][:10], np.ones(10))

    def test_last_window_covers_end_of_clip(self):
        audio = np.arange(57)
        windows = list(audio_windows(audio, sr=10,
                                     window_seconds=3, hop_seconds=1))
        self.assertEqual(len(windows[0]), 30)
        self.assertEqual(windows[-1][-1], 56)

    def test_invalid_hop_is_rejected(self):
        with self.assertRaises(ValueError):
            list(audio_windows(np.ones(10), sr=10,
                               window_seconds=3, hop_seconds=0))


if __name__ == "__main__":
    unittest.main()
