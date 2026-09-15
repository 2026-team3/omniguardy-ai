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


def audio_to_mel_v5(audio, sr):
    """New-model feature: retain the full 3-second interval in 128 columns."""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32), sr=sr,
        n_mels=128, n_fft=2048, hop_length=512,
    )
    mel = librosa.power_to_db(mel, ref=np.max, top_db=80)
    if mel.shape[1] == 1:
        return np.repeat(mel, 128, axis=1).astype(np.float32)
    positions = np.linspace(0, mel.shape[1] - 1, 128)
    source = np.arange(mel.shape[1])
    resized = np.stack([np.interp(positions, source, row) for row in mel])
    return resized.astype(np.float32)


def audio_to_mel_bag(audio, sr, max_windows=8):
    """One recording is one bag; cap its windows to limit length bias."""
    if max_windows < 1:
        raise ValueError("max_windows must be positive")
    windows = list(audio_windows(audio, sr))
    if len(windows) > max_windows:
        indexes = np.linspace(0, len(windows) - 1, max_windows)
        windows = [windows[int(round(index))] for index in indexes]
    features = [audio_to_mel_v5(window, sr) for window in windows]
    # Repeating a real window avoids inventing a high-energy padded window.
    features.extend([features[-1]] * (max_windows - len(features)))
    return np.asarray(features, dtype=np.float32)
