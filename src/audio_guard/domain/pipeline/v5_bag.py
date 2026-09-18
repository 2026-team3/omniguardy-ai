"""v5 녹음 단위 다중 인스턴스 전처리입니다."""

import librosa
import numpy as np

from audio_guard.domain.pipeline.v4_windowed import audio_windows


def audio_to_mel_v5(audio, sr):
    """3초 구간 전체를 128열 Mel 특징으로 유지합니다."""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
    )
    mel = librosa.power_to_db(mel, ref=np.max, top_db=80)
    if mel.shape[1] == 1:
        return np.repeat(mel, 128, axis=1).astype(np.float32)
    positions = np.linspace(0, mel.shape[1] - 1, 128)
    source = np.arange(mel.shape[1])
    resized = np.stack([np.interp(positions, source, row) for row in mel])
    return resized.astype(np.float32)


def audio_to_mel_bag(audio, sr, max_windows=8):
    """녹음 하나를 길이가 고정된 Mel 특징 묶음으로 변환합니다."""
    if max_windows < 1:
        raise ValueError("max_windows must be positive")
    windows = list(audio_windows(audio, sr))
    if len(windows) > max_windows:
        indexes = np.linspace(0, len(windows) - 1, max_windows)
        windows = [windows[int(round(index))] for index in indexes]
    features = [audio_to_mel_v5(window, sr) for window in windows]
    features.extend([features[-1]] * (max_windows - len(features)))
    return np.asarray(features, dtype=np.float32)


class V5BagPipeline:
    """v5 녹음 단위 다중 인스턴스 파이프라인입니다."""

    name = "v5"

    def __init__(self, max_windows=8):
        self.max_windows = max_windows

    def transform(self, audio, sample_rate):
        return audio_to_mel_bag(
            audio,
            sample_rate,
            self.max_windows,
        )[None, ..., None]
