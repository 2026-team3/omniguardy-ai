"""Keras 모델 로딩과 추론을 캡슐화합니다."""

import numpy as np
import tensorflow as tf


class KerasModelRepository:
    """설정 경로에서 Keras 모델을 한 번 로드해 재사용합니다."""

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def predict_max_score(self, model_input):
        scores = np.asarray(self.model.predict(model_input, verbose=0))
        return float(scores.reshape(-1).max())
