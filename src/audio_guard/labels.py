"""공통 이진 라벨과 비전 트리거로 분류할 ESC-50 클래스를 정의합니다."""

LABELS = {"normal": 0, "abnormal": 1}
TARGET_CLASSES = frozenset({
    "door_wood_knock", "door_wood_creaks", "glass_breaking",
    "siren", "chainsaw", "footsteps",
})
