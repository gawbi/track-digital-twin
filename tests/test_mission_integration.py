import math
import unittest

import numpy as np

from dtwin.autonomy import MapContext, MissionDriver
from dtwin.core import load_inputs, wheel_speeds
from dtwin.evaluation import GateCounter
from dtwin.geometry import CameraGeometry, to_vehicle
from dtwin.missions import MissionController, Observation
from dtwin.navigation import JunctionNavigator
from dtwin.road_vision import RoadMarkDetector


class StaticDetector:
    def __init__(self, geometry, kind, local_position):
        self.geometry, self.kind, self.position = geometry, kind, local_position

    def detect(self, rgb):
        z = 0.075 if self.kind.startswith("signal_") else 0.103
        u, v = self.geometry.project(*self.position, z)
        return [
            {
                "class": self.kind,
                "xyxy": [u - 12, v - 32, u + 12, v],
                "confidence": 0.98,
                "source": "vision",
            }
        ]


class MissionIntegrationTests(unittest.TestCase):
    def test_mission_laps_require_all_ordered_gates_after_route_switch(self):
        gates = GateCounter([(0, 0), (1, 0), (1, 1), (0, 0)], radius_m=0.1)
        for point in [(0, 0), (1, 1), (0, 0)]:
            self.assertEqual(gates.update(*point), 0)
        for point in [(1, 0), (1, 1), (0, 0)]:
            gates.update(*point)
        self.assertEqual(gates.laps, 1)

    def setUp(self):
        self.cfg, self.layout = load_inputs()
        self.rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_horn_through_detector_tracking_distance_and_command_once(self):
        initial = (-0.95, 0.8725, 0.0)
        detector = StaticDetector(CameraGeometry(self.cfg["camera"]), "horn", (0.4, -0.08))
        driver = MissionDriver(self.cfg, self.layout, initial, detector)
        horns = []
        for i in range(8):
            command, horn, telemetry = driver.step(self.rgb, i / 30, i / 30, 0.0, 0.0, (0.1, 0.0))
            horns.append(horn)
        self.assertEqual(sum(horns), 1)
        self.assertEqual(command, (0.1, 0.0))
        self.assertIsNone(telemetry["fault"])
        self.assertEqual(
            sum(e["event"] == "mission_completed" and e["kind"] == "horn" for e in driver.events), 1
        )
        self.assertTrue(
            any(e.get("range_source") == "configured_map+wheel_odometry" for e in driver.events)
        )

    def test_stale_camera_stops_then_recovers_without_counting_duplicate_frames(self):
        detector = StaticDetector(CameraGeometry(self.cfg["camera"]), "horn", (0.4, -0.08))
        driver = MissionDriver(self.cfg, self.layout, (-0.95, 0.8725, 0), detector)
        driver.step(self.rgb, 0, 0, 0, 0, (0.1, 0))
        for i in range(1, 4):
            driver.step(self.rgb, i * 0.1, 0, 0, 0, (0.1, 0))
        command, horn, info = driver.step(self.rgb, 0.4, 0, 0, 0, (0.1, 0))
        self.assertEqual(command, (0, 0))
        self.assertFalse(horn)
        self.assertEqual(info["fault"], "stale_or_future_camera_frame")
        command, _, info = driver.step(self.rgb, 0.5, 0.5, 0, 0, (0.1, 0))
        self.assertEqual(command, (0.1, 0))
        self.assertIsNone(info["fault"])

    def test_stop_and_slowdown_are_applied_and_completed_once(self):
        for kind in ["stop", "speed20"]:
            with self.subTest(kind=kind):
                sensor = StaticDetector(CameraGeometry(self.cfg["camera"]), kind, (0.4, -0.08))
                driver = MissionDriver(self.cfg, self.layout, (-0.95, 0.8725, 0), sensor)
                previous = (0.1, 0.0)
                commands = []
                for i in range(50):
                    left, right = wheel_speeds(*previous, self.cfg["vehicle"])
                    command, _, info = driver.step(
                        self.rgb, i / 30, i / 30, left, right, (0.1, 0.0)
                    )
                    self.assertIsNone(info["fault"])
                    commands.append(command)
                    previous = command
                completions = [
                    e
                    for e in driver.events
                    if e["event"] == "mission_completed" and e["kind"] == kind
                ]
                self.assertEqual(len(completions), 1)
                if kind == "stop":
                    self.assertGreaterEqual(sum(c == (0.0, 0.0) for c in commands), 30)
                    self.assertEqual(commands[-1], (0.1, 0.0))
                else:
                    self.assertEqual(commands[-1], (0.05, 0.0))
                    self.assertAlmostEqual(info["odometry_speed_m_s"], 0.05)

    def test_only_inner_crosswalk_can_trigger_signal_rule(self):
        context = MapContext(self.layout, 0.125, self.cfg["missions"])
        self.assertIsNone(context.select("crosswalk", (-0.8, 0.8725, 0), inner=True))
        self.assertEqual(context.select("crosswalk", (-0.8, 0, 0), inner=True), "center-W")

    def test_late_sign_cannot_switch_its_deadline_to_the_next_pink_line(self):
        context = MapContext(self.layout, 0.125, self.cfg["missions"])
        pose = (-0.5, -0.8725, 0.0)
        name = context.select("pink", pose, anchor=(-0.4, -0.80))
        self.assertEqual(name, "pink-SW-S")
        self.assertLess(context.distance(name, pose, 0.0)[0], 0.0)

    def test_actual_turning_is_not_counted_as_stopped_time(self):
        ctrl = MissionController()
        obs = Observation("s", "signal", signal_color="red", stop_distance_m=0.03)
        ctrl.update(0, 0, [obs])
        self.assertTrue(ctrl.update(0.1, 0, [obs], yaw_rate_rad_s=0.5).stop)
        self.assertTrue(ctrl.update(1.5, 0, [obs], yaw_rate_rad_s=0.5).stop)
        self.assertTrue(ctrl.update(2, 0, [obs], yaw_rate_rad_s=0).stop)
        self.assertFalse(ctrl.update(3.1, 0, [obs], yaw_rate_rad_s=0).stop)

    def test_red_approach_slows_and_preserves_curve_ratio(self):
        ctrl = MissionController(braking_m_s2=0.3)
        result = ctrl.update(
            0, 0.18, [Observation("r", "signal", signal_color="red", stop_distance_m=0.05)]
        )
        self.assertLess(result.speed_limit_m_s, 0.18)
        self.assertFalse(result.stop)

    def test_red_detection_to_stop_dwell_right_turn_and_exit_pipeline(self):
        geometry = CameraGeometry(self.cfg["camera"])

        class ProjectedSensor:
            pose = (-0.65, 0.0, 0.0)

            def detect(self, rgb):
                local = to_vehicle((0.15, 0.08), self.pose)
                if local[0] <= 0.12:
                    return []
                pixel = geometry.project(*local, 0.075)
                if pixel is None:
                    return []
                u, v = pixel
                if not 15 < u < 625 or not 35 < v < 475:
                    return []
                return [
                    {
                        "class": "signal_red",
                        "xyxy": [u - 12, v - 32, u + 12, v],
                        "confidence": 0.98,
                        "source": "vision",
                    }
                ]

        sensor = ProjectedSensor()
        driver = MissionDriver(self.cfg, self.layout, sensor.pose, sensor)
        pose, previous = list(sensor.pose), (0.0, 0.0)
        stopped_start = None
        for i in range(900):
            sensor.pose = tuple(pose)
            left, right = wheel_speeds(*previous, self.cfg["vehicle"])
            command, _, info = driver.step(self.rgb, i / 30, i / 30, left, right, (0.18, 0.0))
            self.assertIsNone(info["fault"], info)
            if (
                info["mission"]["stop"]
                and abs(info["odometry_speed_m_s"]) < 0.015
                and stopped_start is None
            ):
                stopped_start = i / 30
            pose[0] += command[0] / 30 * math.cos(pose[2])
            pose[1] += command[0] / 30 * math.sin(pose[2])
            pose[2] += command[1] / 30
            previous = command
            if any(
                e["event"] == "mission_completed" and e["kind"] == "signal" for e in driver.events
            ):
                break
        complete = [
            e for e in driver.events if e["event"] == "mission_completed" and e["kind"] == "signal"
        ]
        self.assertEqual(len(complete), 1)
        stop_event = next(e for e in driver.events if e["event"] == "stop_completed")
        self.assertGreaterEqual(stop_event["time_s"] - stopped_start, 1.0)
        self.assertLess(pose[1], -0.29)

    def test_stationary_right_turn_and_exit_with_ideal_encoder_replay(self):
        cfg = dict(self.cfg["missions"], front_bumper_m=0.125)
        nav = JunctionNavigator(self.layout, cfg, (-0.6, 0, 0))
        nav.request("center_4way", "right")
        pose, velocity, omega = [-0.4, 0.0, 0.0], 0.0, 0.0
        events = []
        for i in range(900):
            command, new = nav.command(i / 30, tuple(pose), velocity, (0.18, 0), omega)
            events += new
            velocity, omega = command
            pose[0] += velocity / 30 * math.cos(pose[2])
            pose[1] += velocity / 30 * math.sin(pose[2])
            pose[2] += omega / 30
            if any(e["event"] == "junction_completed" for e in events):
                break
        self.assertIsNone(nav.fault)
        self.assertTrue(any(e["event"] == "junction_completed" for e in events))
        self.assertAlmostEqual(pose[2], -math.pi / 2, delta=0.06)
        self.assertLess(pose[1], -0.29)

    def test_missing_direction_waits_and_impossible_turn_is_rejected(self):
        cfg = dict(self.cfg["missions"], front_bumper_m=0.125)
        nav = JunctionNavigator(self.layout, cfg, (-0.6, 0, 0))
        command, events = nav.command(0, (-0.19, 0, 0), 0.0, (0.18, 0))
        self.assertEqual(command, (0, 0))
        self.assertEqual(events[-1]["event"], "waiting_for_direction")
        nav = JunctionNavigator(self.layout, cfg, (0.8625, -0.6, math.pi / 2))
        nav.request("T_east", "right")  # would lead east off the track
        command, events = nav.command(0, (0.8625, -0.01, math.pi / 2), 0.0, (0.18, 0))
        self.assertEqual(command, (0, 0))
        self.assertEqual(nav.fault, "direction_has_no_road")

    def test_projected_pink_and_crosswalk_pixels_generate_metric_observations(self):
        import cv2

        geometry = CameraGeometry(self.cfg["camera"])
        rgb = np.full((480, 640, 3), 25, dtype=np.uint8)

        def rectangle(x0, y0, x1, y1, color):
            pixels = [geometry.project(x, y) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
            cv2.fillConvexPoly(rgb, np.array(pixels, dtype=np.int32), color)

        for y in np.arange(-0.09, 0.09, 0.03):
            rectangle(0.38, y, 0.384, y + 0.018, (228, 140, 210))
            rectangle(0.55, y, 0.67, y + 0.018, (240, 240, 238))
        marks = RoadMarkDetector(geometry).detect(rgb)
        kinds = {m.kind for m in marks}
        self.assertEqual(kinds, {"pink", "crosswalk"})
        self.assertAlmostEqual(
            next(m.point[0] for m in marks if m.kind == "pink"), 0.382, delta=0.025
        )
        self.assertAlmostEqual(
            next(m.point[0] for m in marks if m.kind == "crosswalk"), 0.55, delta=0.035
        )
