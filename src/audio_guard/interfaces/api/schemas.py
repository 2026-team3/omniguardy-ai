"""Door Event HTTP API 입출력 스키마입니다."""

from pydantic import BaseModel


class PredictResponse(BaseModel):
    status: str
    predicted_class: str | None = None
    probabilities: dict[str, float] | None = None
    window_start_seconds: float | None = None
    cooldown_suppressed: bool = False
