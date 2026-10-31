"""Geometry, parameter sampling and controls independent of Isaac Sim."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_inputs(config_path=None):
    cfg = json.loads(Path(config_path or ROOT / "configs/rtx.json").read_text())
    layout = json.loads((ROOT / "track_layout.json").read_text())
    from PIL import Image

    with Image.open(ROOT / "track_texture.png") as image:
        expected = tuple(round(s * 1000) for s in layout["meta"]["board_size"])
        if image.size != expected:
            raise ValueError(f"Texture/JSON mismatch: {image.size} != {expected}")
    v, c, s = cfg["vehicle"], cfg["camera"], cfg["simulation"]
    if not (s["physics_hz"] > 0 and s["camera_hz"] > 0 and s["physics_hz"] % s["camera_hz"] == 0):
        raise ValueError("physics_hz must be an integer multiple of camera_hz")
    if (
        min(
            v["wheel_radius_m"],
            v["track_width_m"],
            v["wheelbase_m"],
            v["mass_kg"],
            v["max_wheel_rad_s"],
            s["max_speed_m_s"],
            s["max_yaw_rad_s"],
        )
        <= 0
    ):
        raise ValueError("Vehicle and speed parameters must be positive")
    if s.get("drive_model", "physx") not in {"physx", "kinematic_skid"}:
        raise ValueError("simulation.drive_model must be physx or kinematic_skid")
    if s.get("drive_model") == "kinematic_skid" and not (
        0 < v["skid_yaw_efficiency"] <= 1 and v["motor_time_constant_s"] > 0
    ):
        raise ValueError("kinematic_skid needs skid_yaw_efficiency and motor_time_constant_s")
    if not 1 < c["hfov_deg"] < 175 or min(c["width"], c["height"], c["z_m"]) <= 0:
        raise ValueError("Invalid camera parameters")
    spots = {p["id"] for p in layout["prop_spots"]}
    for p in cfg["props"]:
        if p["spot"] not in spots or p["kind"] not in {
            "speed20",
            "stop",
            "horn",
            "signal",
            "direction_left",
            "direction_right",
            "direction_straight",
        }:
            raise ValueError(f"Invalid prop: {p}")
    m = cfg.get("missions")
    if m is not None:
        if not 0 < m["confidence"] <= 1 or m["min_track_hits"] < 2:
            raise ValueError("Invalid mission confidence/track confirmation settings")
        if any(
            m[k] <= 0
            for k in [
                "max_frame_age_s",
                "braking_m_s2",
                "junction_timeout_s",
                "junction_speed_m_s",
                "stop_dwell_s",
                "red_dwell_s",
            ]
        ):
            raise ValueError("Mission times, speeds and braking must be positive")
        if m["yellow_behavior"] not in {"stop", "follow_lane"} or not 0 < m["slowdown_scale"] <= 1:
            raise ValueError("Invalid yellow behavior or slowdown scale")
    return cfg, layout


def sample_episode(base, seed, randomize=False, stress=False):
    """A seed completely identifies scene parameters. Stress uses held-out ranges."""
    cfg = copy.deepcopy(base)
    rng = random.Random(seed)
    r = cfg["randomization"]

    def draw(name):
        lo, hi = r[name]
        return rng.uniform(lo, hi)

    params = dict(
        seed=seed,
        randomize=randomize,
        stress=stress,
        dome_intensity=650.0,
        key_intensity=1500.0,
        light_temperature_k=5500.0,
        roughness=0.45,
        friction=cfg["vehicle"]["friction"],
        motor_gain_left=1.0,
        motor_gain_right=1.0,
        spawn_lateral_m=0.0,
        spawn_yaw_deg=0.0,
        image_gain=1.0,
        image_noise_std=0.0,
        latency_frames=0,
        background_rgb=[0.35, 0.32, 0.30],
        skid_yaw_efficiency=cfg["vehicle"].get("skid_yaw_efficiency", 1.0),
        motor_time_constant_s=cfg["vehicle"].get("motor_time_constant_s", 0.08),
    )
    if randomize or stress:
        for name in [
            "dome_intensity",
            "key_intensity",
            "light_temperature_k",
            "roughness",
            "friction",
            "spawn_lateral_m",
            "spawn_yaw_deg",
            "image_gain",
            "image_noise_std",
        ]:
            params[name] = draw(name)
        params["motor_gain_left"], params["motor_gain_right"] = (
            draw("motor_gain"),
            draw("motor_gain"),
        )
        params["latency_frames"] = rng.randint(*r["latency_frames"])
        params["background_rgb"] = [rng.uniform(0.08, 0.65) for _ in range(3)]
        cfg["camera"]["z_m"] += draw("camera_height_delta_m")
        cfg["camera"]["pitch_down_deg"] += draw("camera_pitch_delta_deg")
        cfg["camera"]["yaw_deg"] += draw("camera_yaw_delta_deg")
        cfg["camera"]["hfov_deg"] += draw("hfov_delta_deg")
        # Drawn last so earlier parameters of existing seeds are unchanged.
        for name in ["skid_yaw_efficiency", "motor_time_constant_s", "image_blur_sigma"]:
            if name in r:
                params[name] = draw(name)
        if "image_channel_gain" in r:
            params["image_channel_gains"] = [draw("image_channel_gain") for _ in range(3)]
        if "distortion_k1_delta" in r and cfg["camera"].get("distortion"):
            cfg["camera"]["distortion"][0] += draw("distortion_k1_delta")
        if "focal_scale_delta" in r and cfg["camera"].get("intrinsics"):
            scale = 1 + draw("focal_scale_delta")
            cfg["camera"]["intrinsics"] = dict(
                cfg["camera"]["intrinsics"],
                fx=cfg["camera"]["intrinsics"]["fx"] * scale,
                fy=cfg["camera"]["intrinsics"]["fy"] * scale,
            )
    if stress:
        params["image_gain"] = rng.choice([0.55, 1.5])
        params["image_noise_std"] = 9.0
        params["latency_frames"] = 3
        params["roughness"] = 0.08
        params["motor_gain_left"] = 0.75
        params["motor_gain_right"] = 1.2
        cfg["camera"]["pitch_down_deg"] += rng.choice([-8, 8])
        if "skid_yaw_efficiency" in r:
            params["skid_yaw_efficiency"] = min(r["skid_yaw_efficiency"]) * 0.9
            params["motor_time_constant_s"] = max(r["motor_time_constant_s"]) * 1.3
    return cfg, params


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def randomize_props(cfg, layout, seed):
    """Place props on distinct blue circles, with fronts facing the nearest road."""
    rng = random.Random(seed ^ 0x5A91)
    spots = rng.sample(layout["prop_spots"], len(cfg["props"]))
    segments = []
    for name, points in layout["centerlines"].items():
        tail = points[1:] + (points[:1] if name.startswith("outer") else [])
        segments += list(zip(points, tail))
    for prop, spot in zip(cfg["props"], spots):
        x, y = spot["center"]
        closest = None
        for a, b in segments:
            dx, dy = b[0] - a[0], b[1] - a[1]
            t = max(
                0.0, min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / max(dx * dx + dy * dy, 1e-9))
            )
            px, py = a[0] + t * dx, a[1] + t * dy
            candidate = (math.hypot(px - x, py - y), px, py)
            if closest is None or candidate[0] < closest[0]:
                closest = candidate
        prop["spot"] = spot["id"]
        prop["yaw_deg"] = math.degrees(math.atan2(closest[1] - x, -(closest[2] - y))) + rng.uniform(
            -10, 10
        )


def wheel_speeds(linear, angular, vehicle):
    r, width = vehicle["wheel_radius_m"], vehicle["track_width_m"]
    left, right = (linear - angular * width / 2) / r, (linear + angular * width / 2) / r
    scale = max(
        1.0, abs(left) / vehicle["max_wheel_rad_s"], abs(right) / vehicle["max_wheel_rad_s"]
    )
    return left / scale, right / scale


class SkidSteerModel:
    """Kinematic skid-steer ground motion for the simulator's ``kinematic_skid`` mode.

    Wheel rates follow motor targets through gain, first-order lag and saturation.
    Ground yaw is the ideal differential rate scaled by ``yaw_efficiency`` (lateral
    tire scrub). ``wheel_rates`` is what an encoder measures, not the target.
    """

    def __init__(self, vehicle, pose, yaw_efficiency, time_constant_s, gains=(1.0, 1.0)):
        if not 0 < yaw_efficiency <= 1 or time_constant_s <= 0:
            raise ValueError("Skid model needs 0 < yaw_efficiency <= 1 and positive time constant")
        self.vehicle, self.pose = vehicle, tuple(pose)
        self.yaw_efficiency, self.time_constant = yaw_efficiency, time_constant_s
        self.gains = tuple(gains)
        self.wheel_rates = (0.0, 0.0)

    def step(self, left_target, right_target, dt):
        limit = self.vehicle["max_wheel_rad_s"]
        alpha = 1 - math.exp(-dt / self.time_constant)
        self.wheel_rates = tuple(
            rate + (max(-limit, min(limit, target * gain)) - rate) * alpha
            for rate, target, gain in zip(
                self.wheel_rates, (left_target, right_target), self.gains
            )
        )
        left, right = self.wheel_rates
        r, width = self.vehicle["wheel_radius_m"], self.vehicle["track_width_m"]
        linear = r * (left + right) / 2
        angular = r * (right - left) / width * self.yaw_efficiency
        x, y, yaw = self.pose
        mid = yaw + angular * dt / 2
        self.pose = (
            x + linear * math.cos(mid) * dt,
            y + linear * math.sin(mid) * dt,
            wrap_angle(yaw + angular * dt),
        )
        return self.pose


class PathFollower:
    """Privileged geometric teacher, never used as a camera-only evaluation policy."""

    def __init__(
        self,
        points,
        lookahead=0.16,
        speed=0.18,
        max_yaw=2.0,
        sharp_turn_rad=0.87,
        pivot_rad=0.6,
        pivot_exit_rad=0.06,
        corner_gain=0.09,
        dt=1 / 30,
    ):
        self.points = points
        self.lookahead, self.speed, self.max_yaw = lookahead, speed, max_yaw
        self.lengths = [math.dist(a, b) for a, b in zip(points, points[1:] + points[:1])]
        self.total_length = sum(self.lengths)
        self.pivot_rad, self.pivot_exit_rad, self.corner_gain = pivot_rad, pivot_exit_rad, corner_gain
        self.pivoting = False
        # Privileged estimate of realized/commanded yaw rate (skid slip), see command().
        # yaw_bias_per_speed: left/right motor mismatch turns the car on straights.
        self.dt, self.yaw_ratio, self.yaw_bias_per_speed, self._last = dt, 1.0, 0.0, None
        # Sharp polyline vertices (junction turns): drive to the vertex, then pivot.
        headings = [
            math.atan2(b[1] - a[1], b[0] - a[0])
            for a, b in zip(points, points[1:] + points[:1])
        ]
        self.sharp, along = [], 0.0
        for i, length in enumerate(self.lengths):
            if length > 1e-9 and abs(wrap_angle(headings[i] - headings[i - 1])) > sharp_turn_rad:
                self.sharp.append((along, tuple(points[i])))
            along += length

    def project(self, x, y):
        best = None
        along = 0.0
        for i, (a, b, length) in enumerate(
            zip(self.points, self.points[1:] + self.points[:1], self.lengths)
        ):
            if length < 1e-9:
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            t = max(0.0, min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / length**2))
            px, py = a[0] + t * dx, a[1] + t * dy
            d = math.hypot(x - px, y - py)
            if best is None or d < best[0]:
                best = (d, along + t * length, i, math.atan2(dy, dx))
            along += length
        if best is None:
            raise ValueError("Route must contain distinct points")
        return best

    def at(self, s):
        s %= self.total_length
        for a, b, length in zip(self.points, self.points[1:] + self.points[:1], self.lengths):
            if length > 0 and s <= length:
                t = s / length
                return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            s -= length
        return tuple(self.points[0])

    def command(self, x, y, yaw):
        """Teacher command for the NEXT step; assumes the previous command was applied."""
        if self._last is not None:
            last_yaw, last_speed, last_omega = self._last
            rate = wrap_angle(yaw - last_yaw) / self.dt
            if abs(last_omega) > 0.3:
                realized = (rate - self.yaw_bias_per_speed * last_speed) / last_omega
                self.yaw_ratio = 0.9 * self.yaw_ratio + 0.1 * min(1.5, max(0.2, realized))
            elif last_speed > 0.05:
                bias = (rate - self.yaw_ratio * last_omega) / last_speed
                self.yaw_bias_per_speed = 0.8 * self.yaw_bias_per_speed + 0.2 * bias
        speed, omega, info = self._command(x, y, yaw)
        self._last = (yaw, speed, omega)
        return speed, omega, dict(
            info, teacher_yaw_ratio=self.yaw_ratio, teacher_yaw_bias=self.yaw_bias_per_speed
        )

    def record_applied(self, speed, omega):
        """The command actually applied differs from ours (injected noise)."""
        if self._last is not None:
            self._last = (self._last[0], speed, omega)

    def _command(self, x, y, yaw):
        distance, s, _, tangent = self.project(x, y)
        target = self.at(s + self.lookahead)
        for vertex_s, point in self.sharp:
            ahead = (vertex_s - s) % self.total_length
            if ahead < self.lookahead and math.dist(point, (x, y)) > 0.02:
                target = point
                break
        dx, dy = target[0] - x, target[1] - y
        bearing = wrap_angle(math.atan2(dy, dx) - yaw)
        heading_error = wrap_angle(tangent - yaw)
        info = {"route_distance_m": distance, "route_s_m": s, "heading_error_rad": heading_error}
        # Narrow junctions: rotate in place until facing the next segment.
        if abs(bearing) > self.pivot_rad:
            self.pivoting = True
        elif abs(bearing) < self.pivot_exit_rad:
            self.pivoting = False
        if self.pivoting:
            omega = max(-self.max_yaw, min(self.max_yaw, 3.0 * bearing / self.yaw_ratio))
            return 0.0, omega, dict(info, teacher_phase="pivot")
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        curvature = 2 * lateral / max(dx * dx + dy * dy, 1e-6)
        speed = self.speed * math.cos(bearing) ** 2 / max(1.0, abs(curvature) * self.corner_gain)
        # Slow down when the slip-compensated yaw command would saturate.
        speed = min(speed, self.max_yaw * self.yaw_ratio / max(abs(curvature), 1e-6))
        omega = (speed * curvature - self.yaw_bias_per_speed * speed) / self.yaw_ratio
        omega = max(-self.max_yaw, min(self.max_yaw, omega))
        return speed, omega, dict(info, teacher_phase="pursuit")


class Progress:
    def __init__(self, route_length):
        self.length, self.previous, self.distance = route_length, None, 0.0

    def update(self, s):
        if self.previous is not None:
            delta = (s - self.previous + self.length / 2) % self.length - self.length / 2
            self.distance += delta
        self.previous = s
        return max(0, int(self.distance / self.length))


def body_corners(x, y, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx in [-length / 2, length / 2]
        for dy in [-width / 2, width / 2]
    ]
