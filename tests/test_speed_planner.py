import unittest

from dtwin.actions import SpeedPlanner


class SpeedPlannerTest(unittest.TestCase):
    def test_speeds_up_on_straight_brakes_in_curve_keeps_curvature(self):
        p = SpeedPlanner(straight=1.4, curve=1.0)
        for _ in range(200):  # 4 s straight at 50 Hz
            v, w = p.step(0.1, 0.05, 0.02)
        self.assertAlmostEqual(p.scale, 1.4)
        v, w = p.step(0.1, 1.0, 0.02)  # curvature 10 rad/m
        self.assertAlmostEqual(w / v, 10.0)
        self.assertAlmostEqual(p.scale, 1.4 - 0.04)  # brake slew 2/s
        for _ in range(20):
            p.step(0.1, 1.0, 0.02)
        self.assertAlmostEqual(p.scale, 1.0)

    def test_normalized_units(self):
        p = SpeedPlanner(straight=1.2, curve=1.0, yaw_per_speed=2.0 / 0.18)
        for _ in range(50):
            p.step(0.5, 0.45, 0.1)  # 0.45*2 / (0.5*0.18) = 10 rad/m -> curve
        self.assertAlmostEqual(p.scale, 1.0)

    def test_rejects_bad_plan(self):
        with self.assertRaises(ValueError):
            SpeedPlanner(straight=1.0, curve=1.2)


if __name__ == "__main__":
    unittest.main()
