"""OpenCV lens distortion for simulated camera frames and their labels.

The RTX camera renders an ideal pinhole image with a wider FOV; ``apply`` resamples
it into the distorted output camera described by ``hfov_deg`` (or ``intrinsics``)
and ``distortion`` -- the same model CameraGeometry uses for range estimates.
"""

from __future__ import annotations

import math

import numpy as np


def output_matrix(camera):
    width, height = camera["width"], camera["height"]
    intrinsics = camera.get("intrinsics")
    if intrinsics:
        fx, fy, cx, cy = (float(intrinsics[k]) for k in ["fx", "fy", "cx", "cy"])
    else:
        fx = fy = width / (2 * math.tan(math.radians(camera["hfov_deg"] / 2)))
        cx, cy = width / 2, height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


class Lens:
    def __init__(self, camera, margin=1.03):
        import cv2

        self.cv2 = cv2
        self.width, self.height = camera["width"], camera["height"]
        self.matrix = output_matrix(camera)
        self.distortion = np.array(camera.get("distortion", []), dtype=np.float64)
        if self.distortion.size not in {0, 4, 5, 8}:
            raise ValueError("Expected OpenCV distortion coefficients with length 4, 5 or 8")
        if not self.distortion.size or not np.any(self.distortion):
            self.render_matrix, self.maps = self.matrix, None
            self.render_size = (self.width, self.height)
            return
        # Wide pinhole renders are rendered larger so the magnified center stays sharp.
        scale = camera.get("render_scale", 2)
        self.render_size = (self.width * scale, self.height * scale)
        # Normalized pinhole coordinates seen by every output pixel.
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        pixels = np.stack([u.ravel(), v.ravel()], -1)[:, None]
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
        rays = cv2.undistortPointsIter(pixels, self.matrix, self.distortion, None, None, criteria)[:, 0]
        # Pinhole render FOV covering all of them, centered like the output camera.
        span_x = max(abs(rays[:, 0]).max(), 1e-6) * margin
        span_y = max(abs(rays[:, 1]).max(), 1e-6) * margin
        f = min(self.width / 2 / span_x, self.height / 2 / span_y) * scale
        center = [self.render_size[0] / 2, self.render_size[1] / 2]
        self.render_matrix = np.array(
            [[f, 0, center[0]], [0, f, center[1]], [0, 0, 1]], dtype=np.float64
        )
        source = rays * f + center
        self.maps = (
            source[:, 0].reshape(self.height, self.width).astype(np.float32),
            source[:, 1].reshape(self.height, self.width).astype(np.float32),
        )

    @property
    def render_hfov_deg(self):
        return math.degrees(2 * math.atan(self.render_size[0] / 2 / self.render_matrix[0, 0]))

    def apply(self, rgb):
        if self.maps is None:
            return rgb
        if rgb.shape[1::-1] != self.render_size:
            raise ValueError(f"Render size {rgb.shape[1::-1]} != {self.render_size}")
        return self.cv2.remap(rgb, *self.maps, self.cv2.INTER_LINEAR,
                              borderMode=self.cv2.BORDER_REPLICATE)

    def box(self, xyxy, samples=9):
        """Tight output-image box of a render-image box (edges sampled)."""
        if self.maps is None:
            return list(xyxy)
        x0, y0, x1, y1 = xyxy
        xs, ys = np.linspace(x0, x1, samples), np.linspace(y0, y1, samples)
        edge = np.concatenate(
            [np.stack([xs, np.full(samples, y)], -1) for y in (y0, y1)]
            + [np.stack([np.full(samples, x), ys], -1) for x in (x0, x1)]
        )
        f, c = self.render_matrix[0, 0], self.render_matrix[:2, 2]
        rays = (edge - c) / f
        points = np.concatenate([rays, np.ones((len(rays), 1))], 1)[:, None]
        projected = self.cv2.projectPoints(points, np.zeros(3), np.zeros(3), self.matrix, self.distortion)[0][:, 0]
        return [
            float(projected[:, 0].min()),
            float(projected[:, 1].min()),
            float(projected[:, 0].max()),
            float(projected[:, 1].max()),
        ]
