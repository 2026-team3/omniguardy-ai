import unittest

import numpy as np

from audio_guard.domain.event_policy import CooldownGate, EventPolicy


class EventPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = EventPolicy({"knock": 0.7, "handle": 0.6})

    def test_threshold_boundary_triggers_immediately(self):
        self.assertEqual(
            self.policy.decide(np.array([0.2, 0.7, 0.1])),
            ("knock", "KNOCK_EVENT"),
        )

    def test_background_argmax_does_not_block_crossed_event_threshold(self):
        policy = EventPolicy({"knock": 0.4, "handle": 0.6})
        self.assertEqual(
            policy.decide(np.array([0.5, 0.4, 0.1])),
            ("knock", "KNOCK_EVENT"),
        )

    def test_higher_probability_wins_when_both_cross(self):
        self.assertEqual(
            self.policy.decide(np.array([0.01, 0.71, 0.72])),
            ("handle", "HANDLE_EVENT"),
        )

    def test_cooldown_starts_only_after_allowed_event(self):
        now = [10.0]
        gate = CooldownGate(1.0, clock=lambda: now[0])
        self.assertTrue(gate.allow())
        now[0] = 10.2
        self.assertFalse(gate.allow())
        now[0] = 11.0
        self.assertTrue(gate.allow())


if __name__ == "__main__":
    unittest.main()
