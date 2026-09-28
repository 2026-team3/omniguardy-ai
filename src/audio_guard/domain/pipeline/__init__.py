"""Door Event 오디오 전처리 파이프라인을 제공합니다."""

from audio_guard.domain.pipeline.door_event_pipeline import DoorEventPipeline


def create_pipeline(name="door_event", window_seconds=1.0, hop_seconds=0.2):
    if name != "door_event":
        raise ValueError(f"Unknown audio preprocessing pipeline: {name}")
    return DoorEventPipeline(window_seconds, hop_seconds)
