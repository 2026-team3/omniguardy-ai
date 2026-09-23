import unittest

import numpy as np

from audio_guard.application.audio_augmentation import augment_audio


class AudioAugmentationTest(unittest.TestCase):
    def test_returns_original_and_one_weak_variant(self):
        audio = np.full(8, 0.5, dtype=np.float32)
        variants = augment_audio(
            audio,
            sample_rate=22050,
            noise_std=0,
            gain_range=(0.8, 0.8),
            max_time_shift_ms=0,
            rng=np.random.default_rng(42),
        )
        self.assertEqual(len(variants), 2)
        np.testing.assert_array_equal(variants[0], audio)
        np.testing.assert_allclose(variants[1], audio * 0.8)

    def test_rejects_invalid_noise_strength(self):
        with self.assertRaises(ValueError):
            augment_audio(np.zeros(4), sample_rate=22050, noise_std=-0.1)
