import librosa
import numpy as np

SAMPLE_RATE = 22050
WINDOW_SECONDS = 3.0
WINDOW_HOP_SECONDS = 1.0


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


def load_audio(path):
    """Use the same sample rate and mono conversion for training and evaluation."""
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), sr


def audio_windows(audio, sr, window_seconds=WINDOW_SECONDS,
                  hop_seconds=WINDOW_HOP_SECONDS):
    """Cover the whole clip, including an event near its end."""
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
    return np.stack([audio_to_mel(window, sr)
                     for window in audio_windows(audio, sr)])
