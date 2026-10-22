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
    def __init__(self, path):
        import onnxruntime as ort

        self.metadata = json.loads(Path(path).with_suffix(".json").read_text())
        if self.metadata["preprocess"] != "rgb_bottom_half_yuv_blur3_200x66_chw_div255":
            raise ValueError("Unsupported preprocessing contract")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def command(self, rgb):
        values = self.session.run(None, {self.input: preprocess(rgb)[None]})[0][0]
        if len(values) != 2 or not np.isfinite(values).all():
            raise ValueError("Policy must return two finite values")
        linear = float(np.clip(values[0], -1, 1)) * self.metadata["max_speed_m_s"]
        angular = float(np.clip(values[1], -1, 1)) * self.metadata["max_yaw_rad_s"]
        return linear, angular
