"""현장 오디오 fine-tuning에 사용하는 제한적 augmentation입니다."""

import numpy as np


def augment_audio(
    audio,
    sample_rate,
    noise_std=0.001,
    gain_range=(0.9, 1.1),
    max_time_shift_ms=50,
    rng=None,
):
    """원본과 약한 noise/volume/time-shift 변형 하나를 생성합니다."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if len(gain_range) != 2 or gain_range[0] <= 0 or gain_range[0] > gain_range[1]:
        raise ValueError("gain_range must contain two ordered positive values")
    if max_time_shift_ms < 0:
        raise ValueError("max_time_shift_ms must be non-negative")
    rng = rng or np.random.default_rng()
    audio = np.asarray(audio, dtype=np.float32)

    gain = rng.uniform(gain_range[0], gain_range[1])
    augmented = audio * np.float32(gain)
    if noise_std:
        noise = rng.normal(0, noise_std, len(audio)).astype(np.float32)
        augmented = augmented + noise

    max_shift = min(
        max(0, len(audio) - 1),
        int(round(sample_rate * max_time_shift_ms / 1000)),
    )
    if max_shift:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        augmented = np.roll(augmented, shift)
        if shift > 0:
            augmented[:shift] = 0
        elif shift < 0:
            augmented[shift:] = 0

    return [audio, np.clip(augmented, -1.0, 1.0).astype(np.float32)]
