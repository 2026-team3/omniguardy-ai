"""Door Event 추론 결과 값 객체입니다."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DoorEventResult:
    status: str
    predicted_class: str
    probabilities: dict[str, float] = field(default_factory=dict)
    window_start_seconds: float | None = None
    cooldown_suppressed: bool = False
