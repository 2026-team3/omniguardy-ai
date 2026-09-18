"""librosa 기반 오디오 로더입니다."""

import librosa
import numpy as np

from audio_guard.domain.pipeline.base import SAMPLE_RATE


def load_audio(path, sample_rate=SAMPLE_RATE):
    """오디오를 지정한 샘플레이트의 모노 파형으로 읽습니다."""
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32), sr
