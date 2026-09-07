"""실시간 Vision 실행에 공통으로 쓰는 경로와 기본값을 정의한다."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "models" / "yolov8n.pt"
DEFAULT_POSE_MODEL = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
DEFAULT_CLASSIFIER = PROJECT_ROOT / "results" / "models" / "xgboost_validation" / "xgboost_validation.pkl"
DEFAULT_EVENT_LOG = PROJECT_ROOT / "outputs" / "realtime_events.jsonl"
POSE_COLUMNS = ("hand_motion", "body_motion", "arm_extension", "upper_body_angle")
