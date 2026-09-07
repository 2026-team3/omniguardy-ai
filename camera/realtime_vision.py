"""실시간 Vision 파이프라인의 명령행 진입점이다."""

from __future__ import annotations

import sys
from pathlib import Path


# 직접 실행해도 camera 패키지를 불러올 수 있도록 프로젝트 루트를 경로에 넣는다.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.runner import parse_arguments, run


if __name__ == "__main__":
    run(parse_arguments())
