"""YOLO11 raw ONNX decoding. No Ultralytics/torch import in the simulator."""

import json
from pathlib import Path

import numpy as np

CLASSES = [
    "horn",
    "stop",
    "speed20",
    "signal_red",
    "signal_yellow",
    "signal_green",
    "direction_left",
    "direction_right",
    "direction_straight",
]


def letterbox(rgb, size):
    import cv2

    height, width = rgb.shape[:2]
    ratio = min(size[0] / width, size[1] / height)
    resized = round(width * ratio), round(height * ratio)
    left = round((size[0] - resized[0]) / 2 - 0.1)
    top = round((size[1] - resized[1]) / 2 - 0.1)
    image = np.full((size[1], size[0], 3), 114, dtype=np.uint8)
    image[top : top + resized[1], left : left + resized[0]] = cv2.resize(rgb, resized)
    return (
        np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0,
        ratio,
        (left, top),
    )


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (
        max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        - intersection
    )
    return intersection / max(union, 1e-9)


def decode_yolo(raw, classes, image_shape, ratio, padding, confidence=0.5, nms_iou=0.45):
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError("Expected YOLO11 raw [1,4+classes,anchors] output")
    rows = raw[0].T if raw.shape[1] == len(classes) + 4 else raw[0]
    if rows.shape[1] != len(classes) + 4:
        raise ValueError(
            "Unsupported detector output: use YOLO11 nms=False export and matching class metadata"
        )
    if not np.isfinite(rows).all():
        raise ValueError("Detector returned nonfinite values")
    height, width = image_shape[:2]
    candidates = []
    for row in rows:
        idx = int(np.argmax(row[4:]))
        score = float(row[4 + idx])
        if score < confidence:
            continue
        cx, cy, w, h = row[:4]
        x0, x1 = (cx - w / 2 - padding[0]) / ratio, (cx + w / 2 - padding[0]) / ratio
        y0, y1 = (cy - h / 2 - padding[1]) / ratio, (cy + h / 2 - padding[1]) / ratio
        box = [float(max(0, x0)), float(max(0, y0)), float(min(width, x1)), float(min(height, y1))]
        if box[2] - box[0] >= 4 and box[3] - box[1] >= 4:
            candidates.append(
                {"class": classes[idx], "confidence": score, "xyxy": box, "source": "vision"}
            )
    kept = []
    for candidate in sorted(candidates, key=lambda d: d["confidence"], reverse=True)[:300]:
        if not any(
            (
                c["class"] == candidate["class"]
                or (c["class"].startswith("signal_") and candidate["class"].startswith("signal_"))
            )
            and iou(c["xyxy"], candidate["xyxy"]) > nms_iou
            for c in kept
        ):
            kept.append(candidate)
    return kept


class OnnxDetector:
    def __init__(self, path, confidence=None, threads=None):
        import onnxruntime as ort

        self.metadata = json.loads(Path(path).with_suffix(".json").read_text())
        if self.metadata.get("format") != "yolo11_raw_xywh_classes":
            raise ValueError(
                "Detector export metadata missing/unsupported; export using train_detector.py"
            )
        self.classes = self.metadata["classes"]
        if (
            not self.classes
            or len(set(self.classes)) != len(self.classes)
            or not set(self.classes) <= set(CLASSES)
        ):
            raise ValueError("Unknown or duplicate detector classes")
        # Explicit value, else the validation-selected threshold, else 0.5.
        self.confidence = confidence or self.metadata.get("confidence_threshold") or 0.5
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        if (
            len(inp.shape) != 4
            or inp.shape[:2] != [1, 3]
            or not all(isinstance(v, int) for v in inp.shape[2:])
        ):
            raise ValueError("Export a fixed-size RGB batch-1 detector")
        self.input, self.size = inp.name, (inp.shape[3], inp.shape[2])

    def detect(self, rgb):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Detector expects uint8 RGB")
        tensor, ratio, padding = letterbox(rgb, self.size)
        raw = self.session.run(None, {self.input: tensor})[0]
        return decode_yolo(raw, self.classes, rgb.shape, ratio, padding, self.confidence)


def yolo_label(detection, width, height):
    x0, y0, x1, y1 = detection["xyxy"]
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid bounding box")
    return [
        CLASSES.index(detection["class"]),
        (x0 + x1) / 2 / width,
        (y0 + y1) / 2 / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    ]
