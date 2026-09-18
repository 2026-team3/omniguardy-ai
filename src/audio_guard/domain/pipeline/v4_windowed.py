"""v4 슬라이딩 윈도우 전처리입니다."""

import numpy as np

from audio_guard.domain.pipeline.base import WINDOW_HOP_SECONDS, WINDOW_SECONDS
from audio_guard.domain.pipeline.v3_pipeline import audio_to_mel


def audio_windows(
    audio,
    sr,
    window_seconds=WINDOW_SECONDS,
    hop_seconds=WINDOW_HOP_SECONDS,
):
    """끝부분 이벤트까지 포함하도록 클립 전체를 윈도우로 나눕니다."""
    window_samples = int(round(window_seconds * sr))
    hop_samples = int(round(hop_seconds * sr))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("Window and hop must be positive")

    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) <= window_samples:
        yield np.pad(audio, (0, max(0, window_samples - len(audio))))
        return

    last_start = len(audio) - window_samples
    starts = list(range(0, last_start + 1, hop_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        yield audio[start:start + window_samples]


def audio_to_mel_windows(audio, sr):
    return np.stack([audio_to_mel(window, sr) for window in audio_windows(audio, sr)])


class V4WindowedPipeline:
    """v4 윈도우 단위 파이프라인입니다."""

    name = "v4"

    def transform(self, audio, sample_rate):
        return audio_to_mel_windows(audio, sample_rate)[..., None]
