import librosa
import numpy as np


def audio_to_mel(audio, sr):

    """
    Audio waveform
    ->
    Mel Spectrogram
    ->
    (128,128)
    """

    # Mel Spectrogram 생성
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )


    # dB 변환
    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )


    # -------------------------
    # 크기 고정
    # 모델 입력:
    # (128,128,1)
    # -------------------------

    target_width = 128


    if mel.shape[1] < target_width:

        padding = target_width - mel.shape[1]

        mel = np.pad(
            mel,
            (
                (0,0),
                (0,padding)
            ),
            mode="constant"
        )


    else:

        mel = mel[:, :target_width]


    return mel.astype(np.float32)