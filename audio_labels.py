"""Shared binary labels and ESC-50 categories considered abnormal."""

LABELS = {"normal": 0, "abnormal": 1}
TARGET_CLASSES = frozenset({
    "door_wood_knock", "door_wood_creaks", "glass_breaking",
    "siren", "chainsaw", "footsteps",
})
