"""v3 단일 Mel 스펙트로그램 전처리입니다."""

import librosa
import numpy as np


def audio_to_mel(audio, sr):
    """오디오 파형을 크기가 ``(128, 128)``인 Mel 특징으로 변환합니다."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    target_width = 128
    if mel.shape[1] < target_width:
        mel = np.pad(
            mel,
            ((0, 0), (0, target_width - mel.shape[1])),
            mode="constant",
        )
    else:
        mel = mel[:, :target_width]
    return mel.astype(np.float32)


class V3Pipeline:
    """v3 단일 구간 파이프라인입니다."""

    name = "v3"

    def transform(self, audio, sample_rate):
        return audio_to_mel(audio, sample_rate)[None, ..., None]
