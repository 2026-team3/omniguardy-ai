"""Door Event CNN 입력용 Mel spectrogram 변환입니다."""

import librosa
import numpy as np

N_MELS = 128
N_FFT = 2048
MEL_HOP_LENGTH = 512
TARGET_WIDTH = 128


def audio_to_mel(audio, sample_rate):
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=MEL_HOP_LENGTH,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    if mel.shape[1] < TARGET_WIDTH:
        mel = np.pad(mel, ((0, 0), (0, TARGET_WIDTH - mel.shape[1])))
    else:
        mel = mel[:, :TARGET_WIDTH]
    return mel.astype(np.float32)
