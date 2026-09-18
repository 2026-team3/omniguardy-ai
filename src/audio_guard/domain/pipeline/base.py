"""오디오 전처리 파이프라인의 공통 계약입니다."""

from typing import Protocol

import numpy as np

SAMPLE_RATE = 22050
WINDOW_SECONDS = 3.0
WINDOW_HOP_SECONDS = 1.0


class PreprocessingPipeline(Protocol):
    """오디오 파형을 모델 입력 특징으로 변환합니다."""

    name: str

    def transform(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """오디오 파형을 파이프라인별 Mel 특징으로 변환합니다."""
