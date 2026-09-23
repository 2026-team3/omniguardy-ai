"""오디오 클립 분석 유스케이스입니다."""

import logging

from audio_guard.domain.pipeline import create_pipeline
from audio_guard.domain.risk_policy import predict_label

logger = logging.getLogger(__name__)


class AnalyzeClip:
    """전처리, 모델 추론, 위험도 판정을 오케스트레이션합니다."""

    def __init__(self, model_repository, pipeline_name, threshold):
        self.model_repository = model_repository
        self.pipeline = create_pipeline(pipeline_name)
        self.threshold = threshold

    def execute(self, clip):
        model_input = self.pipeline.transform(clip.samples, clip.sample_rate)
        score = self.model_repository.predict_max_score(model_input)
        status = predict_label(score, self.threshold)
        logger.info("분석 결과: score=%s threshold=%s status=%s",
                    score, self.threshold, status)
        return status, score
