import unittest

from dtwin.core import PathFollower, Progress, load_inputs, sample_episode, wheel_speeds
from dtwin.legacy import gpio_module, gpio_wheels
from dtwin.missions import MissionController, Observation


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.layout = load_inputs()

    def test_texture_units_and_narrow_road_clearance(self):
        self.assertEqual(self.layout["meta"]["board_size"], [2.355, 2.4])
        width = self.cfg["vehicle"]["track_width_m"] + self.cfg["vehicle"]["wheel_width_m"]
        self.assertAlmostEqual((self.layout["dims"]["center_road_ew_width"] - width) / 2, 0.0125)

    def test_wheel_units_sign_and_ratio_preserving_saturation(self):
        v = self.cfg["vehicle"]
        left, right = wheel_speeds(0.1, 0, v)
        self.assertAlmostEqual(left, 0.1 / v["wheel_radius_m"])
        self.assertEqual(left, right)
        left, right = wheel_speeds(0, 1, v)
        self.assertLess(left, 0)
        self.assertGreater(right, 0)
        left, right = wheel_speeds(10, 5, v)
        self.assertLessEqual(max(abs(left), abs(right)), v["max_wheel_rad_s"])
        self.assertAlmostEqual(
            left / right, (10 - 5 * v["track_width_m"] / 2) / (10 + 5 * v["track_width_m"] / 2)
        )

    def test_teacher_recovers_correct_direction(self):
        path = PathFollower([[0, 0], [2, 0], [2, 2], [0, 2]])
        _, omega, _ = path.command(1, 0.05, 0)
        self.assertLess(omega, 0)
        _, omega, _ = path.command(1, -0.05, 0)
        self.assertGreater(omega, 0)

    def test_lap_wrap_and_reverse_do_not_count_false_laps(self):
        p = Progress(10)
        for s in [9.9, 0.1, 2.6, 5.1, 7.6, 0.1]:
            p.update(s)
        self.assertAlmostEqual(p.distance, 10.2)
        self.assertEqual(p.update(0.2), 1)
        p = Progress(10)
        for s in [0.1, 9.9, 9.7]:
            p.update(s)
        self.assertLess(p.distance, 0)

    def test_randomization_is_reproducible_without_mutating_config(self):
        height = self.cfg["camera"]["z_m"]
        a = sample_episode(self.cfg, 123, True)
        self.assertEqual(a, sample_episode(self.cfg, 123, True))
        self.assertNotEqual(a, sample_episode(self.cfg, 124, True))
        self.assertEqual(self.cfg["camera"]["z_m"], height)

    def test_stress_parts_ablation_applies_only_the_named_change(self):
        cfg_all, all_parts = sample_episode(self.cfg, 10000, False, True)
        self.assertEqual((all_parts["motor_gain_left"], all_parts["motor_gain_right"]), (0.75, 1.2))
        self.assertEqual(all_parts["latency_frames"], 3)
        cfg_motor, motor = sample_episode(self.cfg, 10000, False, True, stress_parts=["motor"])
        self.assertEqual((motor["motor_gain_left"], motor["motor_gain_right"]), (0.75, 1.2))
        self.assertEqual(motor["latency_frames"], 0)
        cfg_none, _ = sample_episode(self.cfg, 10000, False, True, stress_parts=[])
        self.assertEqual(cfg_motor["camera"]["pitch_down_deg"], cfg_none["camera"]["pitch_down_deg"])
        self.assertNotEqual(cfg_none["camera"]["pitch_down_deg"], cfg_all["camera"]["pitch_down_deg"])
        cfg_cam, _ = sample_episode(self.cfg, 10000, False, True, stress_parts=["camera"])
        self.assertEqual(cfg_cam["camera"]["pitch_down_deg"], cfg_all["camera"]["pitch_down_deg"])
        with self.assertRaises(ValueError):
            sample_episode(self.cfg, 10000, False, True, stress_parts=["weather"])

    def test_harsh_training_draws_keep_ordinary_seeds_and_add_glare_lights(self):
        plain_cfg, plain = sample_episode(self.cfg, 123, True)
        harsh_cfg, harsh = sample_episode(self.cfg, 123, True, harsh=True)
        self.assertNotIn("glare_lights", plain)
        # Everything drawn before the harsh block is identical for the same seed.
        for name in ["dome_intensity", "key_intensity", "friction", "skid_yaw_efficiency"]:
            self.assertEqual(plain[name], harsh[name])
        h = self.cfg["randomization"]["harsh"]
        self.assertTrue(h["glare_lights"][0] <= len(harsh["glare_lights"]) <= h["glare_lights"][1])
        self.assertTrue(h["image_gain"][0] <= harsh["image_gain"] <= h["image_gain"][1])
        extra = harsh_cfg["camera"]["pitch_down_deg"] - plain_cfg["camera"]["pitch_down_deg"]
        self.assertTrue(h["extra_pitch_deg"][0] <= extra <= h["extra_pitch_deg"][1])
        glossy = [sample_episode(self.cfg, s, True, harsh=True)[1]["roughness"] for s in range(200, 260)]
        self.assertTrue(any(r <= h["glossy_roughness"][1] for r in glossy))

    def test_gpio_bridge_left_and_right_match_course_pins(self):
        devices = {}
        gpio = gpio_module(devices)
        for pin in [18, 22, 27, 23, 25, 24]:
            gpio.PWMOutputDevice(pin)
        for pin in [27, 24]:
            devices[pin].value = 1
        devices[18].value, devices[23].value = 0.4, 0.0
        self.assertEqual(gpio_wheels(devices, 20), (8.0, 0.0))


class MissionTests(unittest.TestCase):
    def test_horn_once_per_encounter_and_before_pink(self):
        ctrl = MissionController()
        sign = Observation("horn-lap1", "horn", before_pink_line=True)
        self.assertTrue(ctrl.update(0, 0.1, [sign]).horn)
        self.assertFalse(ctrl.update(0.1, 0.1, [sign]).horn)
        self.assertFalse(
            ctrl.update(0.2, 0.1, [Observation("late", "horn", before_pink_line=False)]).horn
        )
        self.assertTrue(
            ctrl.update(10, 0.1, [Observation("horn-lap2", "horn", before_pink_line=True)]).horn
        )

    def test_red_dwell_measures_stopped_time_then_right_turn(self):
        ctrl = MissionController(red_dwell_s=1)
        red = Observation("signal1", "signal", signal_color="red", stop_distance_m=0.03)
        self.assertTrue(ctrl.update(0, 0.1, [red]).stop)
        self.assertTrue(ctrl.update(1, 0.08, [red]).stop)
        self.assertTrue(ctrl.update(2, 0, [red]).stop)
        self.assertTrue(ctrl.update(2.9, 0, [red]).stop)
        result = ctrl.update(3.01, 0, [red])
        self.assertFalse(result.stop)
        self.assertEqual(result.direction, "right")
        self.assertFalse(ctrl.update(3.1, 0, [red]).stop)

    def test_green_turn_yellow_unknown_and_low_confidence_ignored(self):
        ctrl = MissionController()
        result = ctrl.update(0, 0.1, [Observation("g", "signal", signal_color="green")])
        self.assertEqual(result.direction, "right")
        result = ctrl.update(1, 0.1, [Observation("y", "signal", signal_color="yellow")])
        self.assertEqual(result.events[0]["event"], "unconfirmed_yellow_rule")
        self.assertFalse(
            ctrl.update(
                2, 0.1, [Observation("h", "horn", confidence=0.1, before_pink_line=True)]
            ).horn
        )

    def test_oracle_cannot_be_passed_off_as_vision(self):
        with self.assertRaises(ValueError):
            MissionController().update(0, 0, [Observation("s", "stop", source="oracle")])
        controller = MissionController()
        controller.update(0, 0, [Observation("v", "stop", before_pink_line=True)])
        with self.assertRaises(ValueError):
            controller.update(0.1, 0, [Observation("s", "stop", source="oracle")])


if __name__ == "__main__":
    unittest.main()
