import unittest

from audio_guard.application.event_metrics import evaluate_events


class EventMetricsTest(unittest.TestCase):
    def test_matches_once_and_reports_false_trigger_and_latency(self):
        references = [
            {"class_name": "knock", "start_seconds": 1.0, "end_seconds": 2.0},
            {"class_name": "handle", "start_seconds": 5.0, "end_seconds": 6.0},
        ]
        detections = [
            {"class_name": "knock", "time_seconds": 1.2},
            {"class_name": "knock", "time_seconds": 1.4},
            {"class_name": "handle", "time_seconds": 5.3},
        ]
        result = evaluate_events(references, detections, 60)
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_triggers"], 1)
        self.assertAlmostEqual(result["event_recall"], 1.0)
        self.assertAlmostEqual(result["mean_detection_latency_seconds"], 0.25)


if __name__ == "__main__":
    unittest.main()
