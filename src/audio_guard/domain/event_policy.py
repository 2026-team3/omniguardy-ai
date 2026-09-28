"""Class별 threshold 판정과 중복 이벤트 cooldown 정책입니다."""

import threading
import time

import numpy as np

from audio_guard.labels import EVENTS, EVENT_CLASSES, LABELS


class EventPolicy:
    def __init__(self, thresholds):
        if set(thresholds) != set(EVENT_CLASSES):
            raise ValueError("Thresholds must contain knock and handle")
        self.thresholds = {name: float(value) for name, value in thresholds.items()}
        if any(not 0 < value < 1 for value in self.thresholds.values()):
            raise ValueError("Thresholds must be between 0 and 1")

    def decide(self, probabilities):
        values = np.asarray(probabilities, dtype=float).reshape(-1)
        if values.shape != (len(LABELS),):
            raise ValueError("Expected three class probabilities")
        candidates = [
            name for name in EVENT_CLASSES
            if values[LABELS[name]] >= self.thresholds[name]
        ]
        if not candidates:
            return "background", EVENTS["background"]
        selected = max(candidates, key=lambda name: values[LABELS[name]])
        return selected, EVENTS[selected]


class CooldownGate:
    """한 detector에서 직전에 발생한 실제 이벤트의 중복 window를 억제합니다."""

    def __init__(self, cooldown_seconds, clock=time.monotonic):
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        self.cooldown_seconds = float(cooldown_seconds)
        self.clock = clock
        self._last_event_at = None
        self._lock = threading.Lock()

    def allow(self):
        now = self.clock()
        with self._lock:
            if (
                self._last_event_at is not None
                and now - self._last_event_at < self.cooldown_seconds
            ):
                return False
            self._last_event_at = now
            return True

    def reset(self):
        with self._lock:
            self._last_event_at = None
