"""HTTP API 입출력 스키마입니다."""

from pydantic import BaseModel


class PredictResponse(BaseModel):
    status: str
    probability: float
