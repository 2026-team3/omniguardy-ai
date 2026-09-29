#현장 오디오 파인튜닝에 사용하는 제한적인 증강 코드

import numpy as np


def augment_audio(
    audio,
    sample_rate,
    noise_std=0.001,
    gain_range=(0.9, 1.1),
    rng=None,
):
    #원본과 약한 noise/volume 변형 하나를 생성

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    if (
        len(gain_range) != 2
        or gain_range[0] <= 0
        or gain_range[0] > gain_range[1]
    ):
        raise ValueError(
            "gain_range must contain two ordered positive values"
        )

    rng = rng or np.random.default_rng()
    audio = np.asarray(audio, dtype=np.float32)

    gain = rng.uniform(
        gain_range[0],
        gain_range[1],
    )

    augmented = audio * np.float32(gain)

    if noise_std:
        noise = rng.normal(
            0,
            noise_std,
            len(audio),
        ).astype(np.float32)

        augmented = augmented + noise

    return [
        audio,
        np.clip(
            augmented,
            -1.0,
            1.0,
        ).astype(np.float32),
    ]