import unittest

import numpy as np

from audio_guard.application.audio_augmentation import augment_audio


class AudioAugmentationTest(unittest.TestCase):
    def test_returns_original_noise_and_volume_variants(self):
        audio = np.full(8, 0.5, dtype=np.float32)
        variants = augment_audio(audio, noise_std=0.001, rng=np.random.default_rng(42))
        self.assertEqual(len(variants), 4)
        np.testing.assert_array_equal(variants[0], audio)
        np.testing.assert_allclose(variants[2], audio * 0.8)
        np.testing.assert_allclose(variants[3], audio * 1.2)

    def test_rejects_invalid_noise_strength(self):
        with self.assertRaises(ValueError):
            augment_audio(np.zeros(4), noise_std=-0.1)
