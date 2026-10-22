import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dtwin.core import load_inputs
from dtwin.geometry import CameraGeometry, WheelOdometry
from dtwin.perception import CLASSES, OnnxDetector, decode_yolo, letterbox
from dtwin.tracking import EncounterMemory, ObjectTracker, Track


def detection(kind="horn", box=(100, 120, 140, 170)):
    return {"class": kind, "xyxy": list(box), "confidence": 0.95, "source": "vision"}


class DetectorTests(unittest.TestCase):
    def test_letterbox_inverse_class_nms_and_signal_exclusivity(self):
        rgb = np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8)
        tensor, ratio, pad = letterbox(rgb, (640, 640))
        self.assertEqual(pad, (0, 80))
        self.assertEqual(float(tensor[0, 0, 100, 0]), 1)
        self.assertEqual(float(tensor[0, 2, 100, 0]), 0)
        raw = np.zeros((1, 4 + len(CLASSES), 4), dtype=np.float32)
        raw[0, :4, :] = np.array([[320, 320, 200, 100]] * 4).T
        raw[0, 4, 0], raw[0, 4, 1] = 0.95, 0.8
        raw[0, 4 + 3, 2], raw[0, 4 + 5, 3] = 0.9, 0.8
        boxes = decode_yolo(raw, CLASSES, rgb.shape, ratio, pad)
        self.assertEqual([b["class"] for b in boxes], ["horn", "signal_red"])
        np.testing.assert_allclose(boxes[0]["xyxy"], [220, 190, 420, 290])

    def test_real_onnx_session_contract_without_weights_download(self):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector.onnx"
            raw = np.zeros((1, 4 + len(CLASSES), 1), dtype=np.float32)
            raw[0, :4, 0] = [320, 320, 80, 60]
            raw[0, 4 + 1, 0] = 0.96
            graph = helper.make_graph(
                [helper.make_node("Constant", [], ["boxes"], value=numpy_helper.from_array(raw))],
                "detector_fixture",
                [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 640, 640])],
                [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, list(raw.shape))],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            model.ir_version = 10
            onnx.save(model, path)
            path.with_suffix(".json").write_text(
                json.dumps({"format": "yolo11_raw_xywh_classes", "classes": CLASSES})
            )
            result = OnnxDetector(path).detect(np.zeros((480, 640, 3), dtype=np.uint8))
            self.assertEqual(result[0]["class"], "stop")
            self.assertEqual(result[0]["source"], "vision")


class TrackingTests(unittest.TestCase):
    def test_signal_color_change_retains_track_and_requires_green_confirmation(self):
        tracker = ObjectTracker(min_hits=3)
        for timestamp in [0.0, 0.1, 0.2]:
            tracks = tracker.update([detection("signal_red")], timestamp)
        identifier = tracks[0].id
        self.assertEqual(tracker.update([detection("signal_green")], 0.3)[0].kind, "signal_red")
        green = tracker.update([detection("signal_green")], 0.4)[0]
        self.assertEqual((green.id, green.kind), (identifier, "signal_green"))
        self.assertEqual(tracker.update([detection("signal_green")], 0.4), [])

    def test_occlusion_reassociation_and_revisit(self):
        tracker = ObjectTracker(min_hits=2, max_age_s=0.5)
        tracker.update([detection()], 0.0)
        first = tracker.update([detection()], 0.1)[0]
        tracker.update([], 0.3)
        self.assertEqual(tracker.update([detection()], 0.4)[0].id, first.id)
        tracker.update([], 1.0)
        self.assertEqual(tracker.update([detection()], 1.1), [])
        second = tracker.update([detection()], 1.2)[0]
        self.assertNotEqual(first.id, second.id)
        memory = EncounterMemory()
        a = memory.associate(first, (0.5, 0.1), 0.1, 0.0)
        b = memory.associate(second, (0.52, 0.1), 5.0, 1.0)
        self.assertEqual(a.id, b.id)  # path length alone cannot rearm a sign
        memory.mark_passed((1.0, 0.0, 0.0), 1.0)
        later = Track(99, "horn", [100, 120, 140, 170], 0.9, 10.0)
        c = memory.associate(later, (0.5, 0.1), 10.0, 2.0)
        self.assertNotEqual(a.id, c.id)

    def test_renderer_labels_are_rejected(self):
        box = dict(detection(), source="renderer_ground_truth")
        with self.assertRaises(ValueError):
            ObjectTracker().update([box], 0.0)


class DistanceTests(unittest.TestCase):
    def test_ground_and_raised_sign_projection_roundtrip(self):
        cfg, _ = load_inputs()
        camera = CameraGeometry(dict(cfg["camera"], yaw_deg=7, pitch_down_deg=22))
        for z in [0.0, 0.075, 0.103]:
            pixel = camera.project(0.6, 0.08, z)
            point = camera.intersect(*pixel, plane_height_m=z)
            np.testing.assert_allclose(point, [0.6, 0.08], atol=1e-8)
        self.assertIsNone(camera.intersect(320, -1000))

    def test_encoder_units_turn_in_place_and_delayed_pose_interpolation(self):
        cfg, _ = load_inputs()
        odom = WheelOdometry(cfg["vehicle"], (0.0, 0.0, 0.0))
        odom.update(0.0, 2.0, 2.0)
        odom.update(0.1, 2.0, 2.0)
        self.assertAlmostEqual(odom.pose[0], 0.0065)
        self.assertAlmostEqual(odom.at(0.05)[0], 0.00325)
        before = odom.pose[:2]
        odom.update(0.2, -2.0, 2.0)
        np.testing.assert_allclose(odom.pose[:2], before)
        self.assertAlmostEqual(odom.pose[2], 0.1)
        self.assertGreater(odom.yaw_rate, 0)
        with self.assertRaises(ValueError):
            odom.update(0.15, 0.0, 0.0)
        with self.assertRaises(ValueError):
            odom.at(-1)

    def test_measured_intrinsics_distortion_and_camera_below_sign(self):
        cfg, _ = load_inputs()
        camera = CameraGeometry(
            dict(
                cfg["camera"],
                z_m=0.08,
                intrinsics={"fx": 500, "fy": 510, "cx": 315, "cy": 235},
                distortion=[0.05, -0.01, 0.001, 0.001, 0],
            )
        )
        pixel = camera.project(0.5, 0.07, 0.103)
        np.testing.assert_allclose(
            camera.intersect(*pixel, plane_height_m=0.103), [0.5, 0.07], atol=1e-5
        )
