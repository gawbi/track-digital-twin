import unittest

import numpy as np

from dtwin.core import load_inputs
from dtwin.road_vision import GroundMark
from dtwin.vision_missions import VisionMissionDriver


class ScriptedDetector:
    def __init__(self):
        self.classes = []

    def detect(self, rgb):
        return [
            {"class": c, "xyxy": [300, 100, 340, 160], "confidence": 0.95, "source": "vision"}
            for c in self.classes
        ]


class VisionMissionTests(unittest.TestCase):
    def setUp(self):
        self.cfg, _ = load_inputs()
        self.detector = ScriptedDetector()
        self.driver = VisionMissionDriver(self.cfg, self.detector)
        self.marks = []
        self.driver.road.detect = lambda rgb, detections: list(self.marks)
        self.t = 0.0
        self.rng = np.random.default_rng(0)

    def step(self, moving=True):
        self.t += 1 / 30
        rgb = (
            self.rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
            if moving
            else np.full((480, 640, 3), 90, np.uint8)
        )
        return self.driver.step(rgb, self.t, self.t)

    def pink_at(self, distance_m):
        self.marks = [GroundMark("pink", (distance_m + self.driver.front, 0.0), 0.9)]

    def test_horn_once_until_the_next_pink_line_then_rearmed(self):
        self.detector.classes = ["horn"]
        horns = sum(self.step().horn for _ in range(20))
        self.assertEqual(horns, 1)
        # Pass the sign, then drive over the pink line: close, then out of view.
        self.detector.classes = []
        self.pink_at(0.10)
        self.step()
        self.marks = []
        self.assertEqual(sum(self.step().horn for _ in range(25)), 0)
        self.assertEqual(self.driver.pink.crossings, 1)
        # The next horn sign (next lap / next pink line) is a new mission.
        self.detector.classes = ["horn"]
        self.assertEqual(sum(self.step().horn for _ in range(10)), 1)

    def test_stop_waits_for_image_standstill_then_dwells(self):
        self.detector.classes = ["stop"]
        outputs = [self.step(moving=True) for _ in range(10)]
        self.assertTrue(all(o.stop for o in outputs[3:]))  # stopping, not yet still
        still = [self.step(moving=False) for _ in range(60)]
        released = next(i for i, o in enumerate(still) if not o.stop)
        # 4+ still frames to confirm, then 1 s dwell at 30 fps.
        self.assertGreaterEqual(released, 4 + 29)
        self.assertTrue(any(e["event"] == "standstill_confirmed_by_image" for e in self.driver.events))
        self.assertTrue(any(e["event"] == "mission_completed" and e["kind"] == "stop" for e in self.driver.events))

    def test_sign_seen_at_its_pink_line_is_too_late_and_crossing_while_stopping_faults(self):
        self.pink_at(0.0)
        self.detector.classes = ["speed20"]
        for _ in range(6):
            self.step()
        self.assertTrue(any(e["event"] == "too_close_to_pink_line" for e in self.driver.events))
        self.assertEqual(self.driver.speed_scale, 1.0)
        driver = VisionMissionDriver(self.cfg, self.detector)
        self.driver = driver
        driver.road.detect = lambda rgb, detections: list(self.marks)
        self.detector.classes = ["stop"]
        self.pink_at(0.2)
        for _ in range(5):
            self.step()
        self.pink_at(0.05)
        self.step()
        self.marks = []
        for _ in range(25):
            out = self.step()
        self.assertEqual(driver.fault, "pink_deadline_missed")
        self.assertTrue(out.stop)

    def test_red_signal_stops_before_crosswalk_and_waits_after_standstill(self):
        self.detector.classes = ["signal_red"]
        self.marks = [GroundMark("crosswalk", (self.driver.front + 0.4, 0.0), 0.9)]
        approach = [self.step() for _ in range(6)]
        self.assertFalse(approach[-1].stop)
        self.assertLessEqual(approach[-1].speed_scale, 0.5)
        self.marks = [GroundMark("crosswalk", (self.driver.front + 0.02, 0.0), 0.9)]
        self.assertTrue(self.step().stop)
        outputs = [self.step(moving=False) for _ in range(60)]
        self.assertFalse(outputs[-1].stop)
        self.assertTrue(any(e["event"] == "red_wait_done" for e in self.driver.events))


if __name__ == "__main__":
    unittest.main()
