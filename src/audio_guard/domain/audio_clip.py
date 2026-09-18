"""오디오 분석에 사용하는 값 객체입니다."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioClip:
    """모노 오디오 파형과 샘플레이트입니다."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self):
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True)
class MelSpectrogram:
    """모델에 전달할 Mel 스펙트로그램 배열입니다."""

    values: np.ndarray
