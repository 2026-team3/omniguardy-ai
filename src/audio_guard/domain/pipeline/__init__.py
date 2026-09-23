"""설정값에 맞는 오디오 전처리 파이프라인을 제공합니다."""

from audio_guard.domain.pipeline.v3_pipeline import V3Pipeline
from audio_guard.domain.pipeline.v4_windowed import V4WindowedPipeline


def create_pipeline(name):
    """파이프라인 이름으로 전처리 전략을 생성합니다."""
    pipelines = {
        "v3": V3Pipeline,
        "v4": V4WindowedPipeline,
    }
    try:
        return pipelines[name]()
    except KeyError as error:
        raise ValueError(f"Unknown audio preprocessing pipeline: {name}") from error
