import threading
import time
import unittest

import numpy as np

from dtwin.actions import DutyMixer
from dtwin.realtime import Frame, LatestSlot, Result, VehicleRuntime, monotonic_ns
from dtwin.vehicle_io import FakeMotors

MS = 1_000_000


class FakeCamera:
    """Frames at ``period_s``; ``delay_s`` is capture->receive pipeline delay."""

    def __init__(self, period_s=0.02, delay_s=0.0, stall_after=None, fail_after=None):
        self.period, self.delay = period_s, round(delay_s * 1e9)
        self.stall_after, self.fail_after = stall_after, fail_after
        self.count, self.closed = 0, False

    def read(self):
        if self.fail_after is not None and self.count >= self.fail_after:
            raise RuntimeError("camera unplugged")
        if self.stall_after is not None and self.count >= self.stall_after:
            time.sleep(self.period)
            return None
        time.sleep(self.period)
        now = monotonic_ns()
        frame = Frame(self.count, now - self.delay, "sensor", now, np.zeros((4, 4, 3), np.uint8))
        self.count += 1
        return frame

    def close(self):
        self.closed = True


def perceiver(delay_s=0.0, fail_after=None, hang_after=None, action=(0.5, 0.0)):
    calls = {"n": 0}
    hang = threading.Event()

    def perceive(frame):
        calls["n"] += 1
        if fail_after is not None and calls["n"] > fail_after:
            raise ValueError("model crashed")
        if hang_after is not None and calls["n"] > hang_after:
            hang.wait(5)
        time.sleep(delay_s)
        return {"action": action, "stages": {"lane_inference_ns": round(delay_s * 1e9)}}

    perceive.calls = calls
    perceive.release = hang.set
    return perceive


def runtime(camera, perceive, **kwargs):
    motors = FakeMotors()
    rt = VehicleRuntime(camera, perceive, motors, DutyMixer(max_duty=0.5), **kwargs)
    return rt, motors


class LatestSlotTests(unittest.TestCase):
    def test_reader_gets_newest_and_overwritten_frames_are_counted(self):
        slot = LatestSlot()
        for i in range(3):
            slot.put(i)
        version, item = slot.take_newer(0, 0.01)
        self.assertEqual(item, 2)
        self.assertEqual(slot.dropped, 2)
        self.assertEqual(slot.take_newer(version, 0.01), (version, None))


class DecisionTests(unittest.TestCase):
    """Deterministic checks of the stop rules with an injected clock."""

    def setUp(self):
        self.now = 10_000 * MS
        self.rt, self.motors = runtime(None, None, clock=lambda: self.now, max_result_age_s=0.2)
        self.rt.last_frame_ns = self.now

    def put(self, capture_ns, finished_ns):
        self.rt.results.put(
            Result(1, capture_ns, "sensor", capture_ns, finished_ns - 1, finished_ns, (0.4, 0.1))
        )

    def test_fresh_result_is_applied(self):
        self.put(self.now - 50 * MS, self.now - 1 * MS)
        self.assertEqual(self.rt.decide(self.now)[1], "applied")

    def test_new_result_from_old_capture_is_rejected(self):
        # Inference finished just now, but the image is 300 ms old.
        self.put(self.now - 300 * MS, self.now)
        action, reason, _ = self.rt.decide(self.now)
        self.assertEqual((action, reason), ((0.0, 0.0), "stale_result"))

    def test_camera_gap_stalled_inference_fault_and_future_time_stop(self):
        self.put(self.now - 10 * MS, self.now)
        self.rt.last_frame_ns = self.now - 400 * MS
        self.assertEqual(self.rt.decide(self.now)[1], "camera_stalled")
        self.rt.last_frame_ns = self.now
        self.rt.inference_started_ns = self.now - 600 * MS
        self.assertEqual(self.rt.decide(self.now)[1], "inference_stalled")
        self.rt.inference_started_ns = None
        self.put(self.now + 5 * MS, self.now)
        self.assertEqual(self.rt.decide(self.now)[1], "future_capture_timestamp")
        self.rt.fault = "perception: crash"
        self.assertEqual(self.rt.decide(self.now)[1], "fault")

    def test_tick_drives_motors_through_mixer_and_stops_otherwise(self):
        self.put(self.now - 10 * MS, self.now)
        self.rt.tick(self.now)
        self.assertEqual(self.motors.state, DutyMixer(max_duty=0.5).duty((0.4, 0.1)))
        self.now += 300 * MS
        self.rt.last_frame_ns = self.now
        self.assertEqual(self.rt.tick(self.now), "stale_result")
        self.assertEqual(self.motors.state, (0.0, 0.0))


class ThreadedFaultInjectionTests(unittest.TestCase):
    def test_slow_inference_does_not_block_motor_updates(self):
        camera = FakeCamera(0.01)
        rt, motors = runtime(camera, perceiver(delay_s=0.15), control_hz=50, max_result_age_s=0.4)
        summary = rt.run(duration_s=0.8)
        # ~40 ticks at 50 Hz while only ~5 inferences finished.
        self.assertGreater(summary["counts"]["ticks"], 30)
        self.assertLess(summary["counts"]["results"], 8)
        self.assertGreater(rt.frames.dropped, 10)
        self.assertGreater(summary["counts"]["applied_ticks"], 0)
        self.assertEqual(motors.state, (0.0, 0.0))
        self.assertTrue(camera.closed)

    def test_inference_slower_than_age_limit_never_drives(self):
        rt, motors = runtime(FakeCamera(0.01), perceiver(delay_s=0.3), max_result_age_s=0.2)
        summary = rt.run(duration_s=0.9)
        self.assertEqual(summary["counts"]["applied_ticks"], 0)
        self.assertTrue(all(call[1:] == (0.0, 0.0) for call in motors.calls))

    def test_hung_inference_stops_motors_within_limit(self):
        perceive = perceiver(hang_after=3)
        rt, motors = runtime(FakeCamera(0.02), perceive, max_inference_s=0.2, max_result_age_s=0.15)
        stopper = threading.Timer(1.0, perceive.release)
        stopper.start()
        summary = rt.run(duration_s=0.8)
        stopper.cancel()
        perceive.release()
        self.assertGreater(summary["counts"]["applied_ticks"], 0)
        self.assertTrue(
            {"stale_result", "inference_stalled"} & set(summary["stop_reasons"]), summary
        )
        ticks = [r for r in rt.telemetry.records if r["kind"] == "tick"]
        self.assertEqual(ticks[-1]["duty"], [0.0, 0.0])

    def test_perception_exception_latches_fault_and_stops(self):
        rt, motors = runtime(FakeCamera(0.02), perceiver(fail_after=3))
        summary = rt.run(duration_s=0.6)
        self.assertIn("perception", summary["fault"])
        self.assertGreater(summary["stop_reasons"]["fault"], 5)
        self.assertEqual(motors.state, (0.0, 0.0))

    def test_camera_failure_and_stall_stop(self):
        rt, _ = runtime(FakeCamera(0.02, fail_after=5), perceiver())
        self.assertIn("camera", rt.run(duration_s=0.5)["fault"])
        rt, motors = runtime(FakeCamera(0.02, stall_after=5), perceiver(), max_camera_gap_s=0.1)
        summary = rt.run(duration_s=0.6)
        self.assertIn("camera_stalled", summary["stop_reasons"])
        self.assertEqual(motors.state, (0.0, 0.0))

    def test_motors_stop_when_control_loop_raises(self):
        rt, motors = runtime(FakeCamera(0.02), perceiver())
        rt.tick = lambda scheduled: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            rt.run(duration_s=1)
        self.assertEqual(motors.state, (0.0, 0.0))
        self.assertGreaterEqual(motors.stops, 2)


if __name__ == "__main__":
    unittest.main()
