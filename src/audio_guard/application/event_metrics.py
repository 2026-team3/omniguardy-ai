"""실제 환경 event matching, false trigger와 detection latency 평가입니다."""

import numpy as np


def evaluate_events(reference_events, detected_events, recording_duration_seconds):
    """같은 class이며 reference 구간 안에서 검출된 첫 이벤트를 1:1 매칭합니다."""
    if recording_duration_seconds <= 0:
        raise ValueError("recording_duration_seconds must be positive")
    used = set()
    latencies = []
    true_positives = 0
    for reference in reference_events:
        candidates = [
            (index, detected)
            for index, detected in enumerate(detected_events)
            if index not in used
            and detected["class_name"] == reference["class_name"]
            and reference["start_seconds"] <= detected["time_seconds"] <= reference["end_seconds"]
        ]
        if not candidates:
            continue
        index, detected = min(candidates, key=lambda item: item[1]["time_seconds"])
        used.add(index)
        true_positives += 1
        latencies.append(detected["time_seconds"] - reference["start_seconds"])
    false_triggers = len(detected_events) - len(used)
    false_negatives = len(reference_events) - true_positives
    precision = true_positives / len(detected_events) if detected_events else 0.0
    recall = true_positives / len(reference_events) if reference_events else 0.0
    return {
        "event_precision": float(precision),
        "event_recall": float(recall),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_triggers": false_triggers,
        "false_triggers_per_hour": float(
            false_triggers * 3600 / recording_duration_seconds
        ),
        "mean_detection_latency_seconds": (
            float(np.mean(latencies)) if latencies else None
        ),
        "p95_detection_latency_seconds": (
            float(np.percentile(latencies, 95)) if latencies else None
        ),
    }
