import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dtwin.actions import ACTION_SPACE, DutyMixer, denormalize_action, normalize_command
from dtwin.core import load_inputs
from dtwin.latency import summarize, summarize_trials
from dtwin.real_data import (
    SessionWriter,
    assign_groups,
    build_manifest,
    load_manifest,
    record_test_access,
    split_frames,
    write_manifest,
)

INFO = dict(
    location="310호",
    lighting="test",
    camera={"width": 32, "height": 24},
    calibration_version="unmeasured",
    clock={"reference": "test"},
    encoder=False,
)


def make_session(root, name, frames=4, source=None, seed=0, labels=None):
    writer = SessionWriter(root, name, dict(INFO, source_recording=source or name))
    rng = np.random.default_rng(seed)
    for i in range(frames):
        rgb = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
        writer.write(i, rgb, 1000 + i, "sensor", 2000 + i,
                     dict(action=[0.5, -0.1], duty=[0.2, 0.3], applied_ns=1999 + i, source="keyboard"))
    writer.close(status="completed")
    if labels is not None:
        (writer.path / "labels").mkdir()
        for i, text in labels.items():
            (writer.path / "labels" / f"{i:06d}.txt").write_text(text)
    return writer.path


class ActionContractTests(unittest.TestCase):
    def test_normalized_round_trip_and_clipping(self):
        self.assertEqual(normalize_command(0.09, -1.0, 0.18, 2.0), (0.5, -0.5))
        self.assertEqual(normalize_command(1.0, 0, 0.18, 2.0), (1.0, 0.0))
        self.assertEqual(denormalize_action((0.5, -0.5), 0.18, 2.0), (0.09, -1.0))

    def test_duty_mixer_signs_saturation_and_inverse(self):
        mixer = DutyMixer(max_duty=0.5, turn_mix=0.72)
        left, right = mixer.duty((0.0, 1.0))
        self.assertLess(left, 0)
        self.assertGreater(right, 0)
        self.assertEqual(mixer.duty((1.0, 0.0)), (0.5, 0.5))
        left, right = mixer.duty((1.0, 1.0))
        self.assertLessEqual(max(abs(left), abs(right)), 0.5)
        for action in [(0.3, 0.2), (-0.2, 0.4), (0.0, -0.5)]:
            np.testing.assert_allclose(mixer.action(*mixer.duty(action)), action, atol=1e-9)

    def test_simulator_ratio_matches_wheel_kinematics(self):
        from dtwin.core import wheel_speeds

        cfg, _ = load_inputs()
        mixer = DutyMixer.from_simulator(cfg)
        sim = cfg["simulation"]
        left, right = wheel_speeds(0.5 * sim["max_speed_m_s"], 0.3 * sim["max_yaw_rad_s"], cfg["vehicle"])
        duty_left, duty_right = mixer.duty((0.5, 0.3))
        self.assertAlmostEqual(left / right, duty_left / duty_right, places=9)

    def test_calibration_file_is_tied_to_action_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "car.json"
            path.write_text(json.dumps(DutyMixer(max_duty=0.4).to_json()))
            self.assertEqual(DutyMixer.load(path).max_duty, 0.4)
            path.write_text(json.dumps({"action_space": "angle_deg", "duty_mixer": {}}))
            with self.assertRaises(ValueError):
                DutyMixer.load(path)


class RealDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_session_records_commands_not_measurements(self):
        path = make_session(self.root, "s1")
        info = json.loads((path / "session.json").read_text())
        self.assertEqual(info["action_space"], ACTION_SPACE)
        row = json.loads((path / "frames.jsonl").read_text().splitlines()[0])
        self.assertIsNone(row["encoder"])
        self.assertEqual(row["command"]["duty"], [0.2, 0.3])
        writer = SessionWriter(self.root, "s2", dict(INFO))
        with self.assertRaises(ValueError):
            writer.write(0, np.zeros((2, 2, 3), np.uint8), 0, "sensor", 0, None,
                         encoder={"left_rad_s": 1.0})

    def test_split_by_source_recording_and_fixed_manifest(self):
        for i, name in enumerate(["a", "b", "c", "d"]):
            make_session(self.root, name, seed=i)
        # A clip of recording "a" must follow "a" into the same split.
        make_session(self.root, "a-clip", source="a", seed=10)
        manifest = build_manifest([self.root], seed=1, explicit={"d": "test"})
        splits = {entry["split"] for name, entry in manifest["sessions"].items() if name.startswith("a")}
        self.assertEqual(len(splits), 1)
        self.assertIn("d", manifest["assignment"]["test"])
        path = write_manifest(manifest, self.root / "m.json")
        with self.assertRaises(FileExistsError):
            write_manifest(manifest, path)
        loaded = load_manifest(path)
        rows = split_frames(loaded, "train")
        self.assertTrue(all(r["session_id"] in {"a", "a-clip", "b", "c"} for r in rows))
        # Editing a session after fixing the split is detected.
        with (self.root / "b" / "frames.jsonl").open("a") as file:
            file.write("\n")
        with self.assertRaises(ValueError):
            load_manifest(path)

    def test_copied_frames_across_recordings_are_rejected(self):
        make_session(self.root, "a", seed=1)
        make_session(self.root, "b", seed=2)
        make_session(self.root, "copy", seed=1)  # same pixels, different source id
        with self.assertRaises(ValueError):
            build_manifest([self.root], seed=0)

    def test_too_few_recordings_and_class_statistics(self):
        make_session(self.root, "a", seed=1, labels={0: "1 0.5 0.5 0.2 0.2\n", 1: ""})
        make_session(self.root, "b", seed=2)
        with self.assertRaises(ValueError):
            build_manifest([self.root])
        make_session(self.root, "c", seed=3)
        manifest = build_manifest([self.root], explicit={"a": "train"})
        report = manifest["report"]
        self.assertEqual(report["class_counts"]["train"]["stop"], 1)
        self.assertEqual(report["background_frames"]["train"], 1)
        self.assertIn("horn", report["empty_classes"]["train"])
        self.assertTrue(manifest["limitations"])
        self.assertEqual(assign_groups(["x", "y", "z"], 0)["train"].__len__(), 1)

    def test_test_access_ledger_counts_previous_uses(self):
        path = self.root / "m.json"
        path.write_text("{}")
        record_test_access(path, "final", "model.onnx", {"mae": 1})
        _, previous = record_test_access(path, "again", "model.onnx", {"mae": 1})
        self.assertEqual(previous, 1)


class LatencyTests(unittest.TestCase):
    def test_end_to_end_uses_capture_time_and_excludes_warmup(self):
        ms = 1_000_000
        records = []
        for k in range(200):
            t = k * 20 * ms
            records.append(dict(kind="tick", scheduled_ns=t, start_ns=t + ms, applied_ns=t + 2 * ms,
                                lateness_ns=ms, reason="applied", frame_id=k, first_use=True,
                                capture_ns=t - 50 * ms, capture_clock="sensor", duty=[0.1, 0.1]))
            records.append(dict(kind="frame", frame_id=k, capture_ns=t - 60 * ms, received_ns=t - 40 * ms,
                                capture_clock="sensor", dropped_total=k // 2))
            records.append(dict(kind="result", frame_id=k, capture_ns=t - 50 * ms, capture_clock="sensor",
                                received_ns=t - 40 * ms, started_ns=t - 30 * ms, finished_ns=t - 5 * ms,
                                stages={"lane_inference_ns": 20 * ms}))
        report = summarize(records, warmup_s=1.0, control_hz=50)
        self.assertEqual(report["end_to_end_capture_to_motor_ms"]["p50"], 52.0)
        self.assertEqual(report["control"]["ticks"], 150)
        self.assertEqual(report["perception"]["stages_ms"]["lane_inference_ns"]["p95"], 20.0)
        self.assertEqual(report["end_to_end_reference"], "camera sensor timestamp")
        self.assertEqual(report["control"]["deadline_misses"], 0)

    def test_trials_require_fields_and_flag_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.jsonl"
            rows = [
                dict(trial=1, route="outer", attempted_at="t", result="completed", evidence="v1.mp4",
                     missions={"stop": {"completed": 1, "duplicate": 1}}),
                dict(trial=2, route="outer", attempted_at="t", result="offroad", evidence="",
                     stop_reason="lost_lane"),
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows))
            report = summarize_trials(path)
            self.assertEqual(report["completed"], 1)
            self.assertEqual(report["missions"]["stop"]["duplicate"], 1)
            self.assertEqual(report["without_evidence"], [2])
            path.write_text(json.dumps(dict(trial=3, route="outer", result="completed")))
            with self.assertRaises(ValueError):
                summarize_trials(path)


class KinematicModelTests(unittest.TestCase):
    def test_skid_model_lag_slip_and_measured_wheel_rates(self):
        from dtwin.core import SkidSteerModel, wheel_speeds

        cfg, _ = load_inputs()
        model = SkidSteerModel(cfg["vehicle"], (0, 0, 0), 0.5, 0.05)
        left, right = wheel_speeds(0, 1.0, cfg["vehicle"])
        model.step(left, right, 0.01)
        self.assertLess(abs(model.wheel_rates[1]), abs(right))  # first-order lag
        for _ in range(300):
            model.step(left, right, 0.01)
        np.testing.assert_allclose(model.wheel_rates, (left, right), rtol=1e-6)
        yaw_rate = cfg["vehicle"]["wheel_radius_m"] * (right - left) / cfg["vehicle"]["track_width_m"]
        before = model.pose[2]
        model.step(left, right, 0.01)
        self.assertAlmostEqual((model.pose[2] - before) / 0.01, 0.5 * yaw_rate, places=6)


if __name__ == "__main__":
    unittest.main()
