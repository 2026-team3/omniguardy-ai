"""PC와 라즈베리파이 카메라 입력을 공통 인터페이스로 제공한다."""

from __future__ import annotations

import cv2
import numpy as np


class OpenCVFrameSource:
    """PC 웹캠, USB 웹캠, 동영상 파일을 OpenCV로 읽는 입력 어댑터다."""

    def __init__(self, source: int | str, width: int, height: int) -> None:
        self.capture = cv2.VideoCapture(source)
        if isinstance(source, int):
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.capture.isOpened():
            raise RuntimeError(f"카메라 또는 영상을 열 수 없습니다: {source}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        """다음 BGR 프레임을 읽는다."""
        return self.capture.read()

    def release(self) -> None:
        """카메라 또는 영상 파일 핸들을 해제한다."""
        self.capture.release()


class PiCameraFrameSource:
    """라즈베리파이 CSI 카메라용 Picamera2 입력 어댑터다."""

    def __init__(self, width: int, height: int) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError(
                "Pi 카메라는 라즈베리파이에서 picamera2 설치 후 사용할 수 있습니다."
            ) from error

        self.camera = Picamera2()
        self.camera.preview_configuration.main.size = (width, height)
        self.camera.preview_configuration.main.format = "RGB888"
        self.camera.preview_configuration.align()
        self.camera.configure("preview")
        self.camera.start()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Picamera2 RGB 프레임을 OpenCV용 BGR 형식으로 반환한다."""
        rgb_frame = self.camera.capture_array()
        return True, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        """Pi 카메라를 정지한다."""
        self.camera.stop()


def parse_source(value: str) -> int | str:
    """숫자 입력은 웹캠 번호로, 나머지는 동영상 경로로 해석한다."""
    return int(value) if value.isdigit() else value


def create_frame_source(source: str, width: int, height: int) -> OpenCVFrameSource | PiCameraFrameSource:
    """입력 종류에 맞는 프레임 공급자를 선택한다."""
    if source == "picamera":
        return PiCameraFrameSource(width, height)
    return OpenCVFrameSource(parse_source(source), width, height)
