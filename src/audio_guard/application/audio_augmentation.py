"""현장 오디오 fine-tuning에 사용하는 제한적 augmentation입니다."""

import numpy as np


def augment_audio(audio, noise_std=0.002, gains=(0.8, 1.2), rng=None):
    """원본 특징을 보존하는 약한 noise와 volume 변형을 생성합니다."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if any(gain <= 0 for gain in gains):
        raise ValueError("augmentation gains must be positive")
    rng = rng or np.random.default_rng()
    audio = np.asarray(audio, dtype=np.float32)
    variants = [audio]
    if noise_std:
        noise = rng.normal(0, noise_std, len(audio)).astype(np.float32)
        variants.append(np.clip(audio + noise, -1.0, 1.0))
    variants.extend(np.clip(audio * gain, -1.0, 1.0) for gain in gains)
    return variants
