"""Shared RGB preprocessing and ONNX camera-only inference."""

import json
from pathlib import Path

import numpy as np


def preprocess(rgb):
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image, not BGR or a normalized float array")
    image = rgb[rgb.shape[0] // 2 :]
    # Matches the book's BGR->YUV after converting the simulator RGB correctly.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0


class OnnxPolicy:
    def __init__(self, path, threads=None):
        import onnxruntime as ort

        from .actions import ACTION_SPACE

        self.metadata = json.loads(Path(path).with_suffix(".json").read_text())
        if self.metadata["preprocess"] != "rgb_bottom_half_yuv_blur3_200x66_chw_div255":
            raise ValueError("Unsupported preprocessing contract")
        # Exports before the action contract existed used the same normalized output.
        if self.metadata.get("action_space", ACTION_SPACE) != ACTION_SPACE:
            raise ValueError(f"Policy action space must be {ACTION_SPACE}")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name

    def action(self, rgb, timings=None):
        """Normalized [linear, angular] in [-1, 1]; optional stage timings in ns."""
        import time

        start = time.monotonic_ns()
        tensor = preprocess(rgb)[None]
        middle = time.monotonic_ns()
        values = self.session.run(None, {self.input: tensor})[0][0]
        if timings is not None:
            timings["lane_preprocess_ns"] = middle - start
            timings["lane_inference_ns"] = time.monotonic_ns() - middle
        if len(values) != 2 or not np.isfinite(values).all():
            raise ValueError("Policy must return two finite values")
        return float(np.clip(values[0], -1, 1)), float(np.clip(values[1], -1, 1))

    def command(self, rgb):
        linear, angular = self.action(rgb)
        return linear * self.metadata["max_speed_m_s"], angular * self.metadata["max_yaw_rad_s"]
