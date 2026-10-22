"""Calibrated monocular range and wheel odometry, without simulator pose access."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .core import wrap_angle


class CameraGeometry:
    def __init__(self, camera, max_range_m=1.5, sign_bottom_m=0.103, signal_bottom_m=0.075):
        self.width, self.height = camera["width"], camera["height"]
        self.fx = self.width / (2 * math.tan(math.radians(camera["hfov_deg"] / 2)))
        self.fy = self.fx
        self.cx, self.cy = self.width / 2, self.height / 2
        intrinsics = camera.get("intrinsics")
        if intrinsics:
            self.fx, self.fy, self.cx, self.cy = (
                float(intrinsics[k]) for k in ["fx", "fy", "cx", "cy"]
            )
            if (
                not all(math.isfinite(v) for v in [self.fx, self.fy, self.cx, self.cy])
                or min(self.fx, self.fy) <= 0
            ):
                raise ValueError("Invalid measured camera intrinsics")
        self.distortion = np.array(camera.get("distortion", []), dtype=np.float64)
        if self.distortion.size not in {0, 4, 5, 8} or not np.isfinite(self.distortion).all():
            raise ValueError("Expected OpenCV distortion coefficients with length 4, 5 or 8")
        self.matrix = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64
        )
        self.sign_bottom, self.signal_bottom = sign_bottom_m, signal_bottom_m
        self.origin = np.array([camera["x_m"], 0.0, camera["z_m"]])
        p, y = math.radians(camera["pitch_down_deg"]), math.radians(camera["yaw_deg"])
        pitch = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        yaw = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        self.rotation = yaw @ pitch
        self.max_range = max_range_m

    def intersect(self, u, v, plane_height_m=0.0):
        if self.distortion.size:
            import cv2

            xy = cv2.undistortPoints(
                np.array([[[u, v]]], dtype=np.float64), self.matrix, self.distortion
            )[0, 0]
            u, v = self.cx + self.fx * xy[0], self.cy + self.fy * xy[1]
        ray = self.rotation @ np.array([1.0, -(u - self.cx) / self.fx, -(v - self.cy) / self.fy])
        if abs(ray[2]) < 1e-5:
            return None
        scale = (plane_height_m - self.origin[2]) / ray[2]
        point = self.origin + scale * ray
        if scale <= 0 or point[0] <= 0 or np.linalg.norm(point[:2]) > self.max_range:
            return None
        return tuple(float(value) for value in point[:2])

    def project(self, x, y, z=0.0):
        ray = self.rotation.T @ (np.array([x, y, z]) - self.origin)
        if ray[0] <= 0:
            return None
        if self.distortion.size:
            import cv2

            pixel = cv2.projectPoints(
                np.array([[[-ray[1], -ray[2], ray[0]]]], dtype=np.float64),
                np.zeros(3),
                np.zeros(3),
                self.matrix,
                self.distortion,
            )[0][0, 0]
            return tuple(float(v) for v in pixel)
        return self.cx - self.fx * ray[1] / ray[0], self.cy - self.fy * ray[2] / ray[0]

    def object_position(self, detection):
        x0, y0, x1, y1 = detection["xyxy"]
        if x0 <= 1 or x1 >= self.width - 1 or y1 >= self.height - 1:
            return None  # cropped/occluded boxes cannot establish a known bottom height
        height = (
            self.signal_bottom if detection["class"].startswith("signal_") else self.sign_bottom
        )
        return self.intersect((x0 + x1) / 2, y1, height)


def to_world(point, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * point[0] - s * point[1], y + s * point[0] + c * point[1]


def to_vehicle(point, pose):
    x, y, yaw = pose
    dx, dy = point[0] - x, point[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


class WheelOdometry:
    def __init__(self, vehicle, initial_pose):
        self.radius, self.width = vehicle["wheel_radius_m"], vehicle["track_width_m"]
        self.pose = tuple(initial_pose)
        self.distance = 0.0
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.time = None
        self.history = deque(maxlen=300)

    def update(self, now, left_rad_s, right_rad_s):
        if not all(math.isfinite(value) for value in [now, left_rad_s, right_rad_s]):
            raise ValueError("Odometry inputs must be finite")
        dt = 0.0 if self.time is None else now - self.time
        if dt < 0 or dt > 0.5:
            raise ValueError("Nonmonotonic or interrupted wheel-odometry timestamps")
        v = self.radius * (left_rad_s + right_rad_s) / 2
        omega = self.radius * (right_rad_s - left_rad_s) / self.width
        x, y, yaw = self.pose
        angle = omega * dt
        if abs(omega) < 1e-8:
            x, y = x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw)
        else:
            x += v / omega * (math.sin(yaw + angle) - math.sin(yaw))
            y += v / omega * (-math.cos(yaw + angle) + math.cos(yaw))
        self.pose = x, y, wrap_angle(yaw + angle)
        self.speed, self.time = v, now
        self.yaw_rate = omega
        self.distance += abs(v) * dt
        self.history.append((now, self.pose))
        return self.pose

    def at(self, timestamp):
        if (
            not self.history
            or timestamp < self.history[0][0] - 1e-6
            or timestamp > self.history[-1][0] + 1e-6
        ):
            raise ValueError("Camera timestamp is outside odometry history")
        for (ta, a), (tb, b) in zip(self.history, list(self.history)[1:]):
            if ta <= timestamp <= tb and tb > ta:
                ratio = (timestamp - ta) / (tb - ta)
                return (
                    a[0] + ratio * (b[0] - a[0]),
                    a[1] + ratio * (b[1] - a[1]),
                    wrap_angle(a[2] + ratio * wrap_angle(b[2] - a[2])),
                )
        return self.history[-1][1]


def nearest_on_segment(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(
        0.0,
        min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / max(dx * dx + dy * dy, 1e-12)),
    )
    return a[0] + t * dx, a[1] + t * dy
