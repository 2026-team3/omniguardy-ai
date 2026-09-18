"""FFmpeg를 이용해 입력 오디오를 모델용 WAV로 변환합니다."""

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


class FfmpegConverter:
    """오디오를 지정 샘플레이트의 모노 PCM WAV로 변환합니다."""

    def __init__(self, sample_rate, executable=None):
        self.sample_rate = sample_rate
        self.executable = executable or shutil.which("ffmpeg")

    def convert_to_wav(self, input_path, output_path):
        if not self.executable:
            raise RuntimeError("FFmpeg를 찾을 수 없습니다.")
        result = subprocess.run(
            [self.executable, "-y", "-i", str(input_path),
             "-ar", str(self.sample_rate), "-ac", "1",
             "-c:a", "pcm_s16le", str(output_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("FFmpeg conversion failed\n%s", result.stderr)
            return False
        return True
