"""Ground-plane pink-line and zebra-crossing measurements from RGB pixels."""

from dataclasses import dataclass

import numpy as np


@dataclass
class GroundMark:
    kind: str
    point: tuple
    confidence: float


class RoadMarkDetector:
    def __init__(self, geometry):
        self.geometry = geometry

    def detect(self, rgb, object_boxes=()):
        import cv2

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        pink = cv2.inRange(hsv, (130, 45, 75), (175, 255, 255))
        white = cv2.inRange(hsv, (0, 0, 140), (179, 65, 255))
        # Elevated signs are not ground paint even when their projected colors match.
        for box in object_boxes:
            x0, y0, x1, y1 = (int(v) for v in box["xyxy"])
            pink[max(0, y0 - 3) : y1 + 4, max(0, x0 - 3) : x1 + 4] = 0
            white[max(0, y0 - 3) : y1 + 4, max(0, x0 - 3) : x1 + 4] = 0

        def components(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = []
            for contour in contours:
                if cv2.contourArea(contour) < 3:
                    continue
                samples = contour.reshape(-1, 2)
                points = [self.geometry.intersect(float(u), float(v)) for u, v in samples]
                points = np.asarray([p for p in points if p is not None and 0.08 < p[0] < 1.2])
                if len(points) < 3:
                    continue
                lo, hi = points.min(axis=0), points.max(axis=0)
                center = (lo + hi) / 2
                if abs(center[1]) < 0.22:
                    result.append((center, hi - lo))
            return result

        result = []
        pink_parts = components(pink)
        # Dashed marks have several small components spanning the driving corridor.
        for center, size in pink_parts:
            group = [(c, s) for c, s in pink_parts if abs(c[0] - center[0]) < 0.045]
            if len(group) >= 3:
                points = np.asarray([c for c, _ in group])
                if np.ptp(points[:, 1]) >= 0.055:
                    result.append(GroundMark("pink", tuple(np.median(points, axis=0)), 0.9))
                    break
            elif size[1] >= 0.07 and size[0] < 0.07:
                result.append(GroundMark("pink", tuple(center), 0.8))
        stripes = [
            (c, s) for c, s in components(white) if 0.045 <= s[0] <= 0.23 and 0.006 <= s[1] <= 0.055
        ]
        for center, _ in stripes:
            group = [(c, s) for c, s in stripes if abs(c[0] - center[0]) < 0.06]
            if len(group) >= 3:
                points = np.asarray([c for c, _ in group])
                if np.ptp(points[:, 1]) >= 0.055:
                    # Stop before the first stripe edge along travel, not its center.
                    front = min(c[0] - s[0] / 2 for c, s in group)
                    result.append(
                        GroundMark("crosswalk", (float(front), float(np.median(points[:, 1]))), 0.9)
                    )
                    break
        return result
