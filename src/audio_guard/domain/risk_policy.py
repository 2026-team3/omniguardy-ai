"""이상 위험도 점수를 라벨로 판정하는 도메인 정책입니다."""


def predict_label(score, threshold):
    """임계치 이상이면 abnormal, 미만이면 normal을 반환합니다."""
    return "abnormal" if score >= threshold else "normal"
