"""Detector ONNX adapter. Mission triggering/tracking remains a separate layer."""

CLASSES = ["horn", "stop", "speed20", "signal_red", "signal_yellow", "signal_green"]


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
