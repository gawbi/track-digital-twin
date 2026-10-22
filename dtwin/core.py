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
    if stress:
        params["image_gain"] = rng.choice([0.55, 1.5])
        params["image_noise_std"] = 9.0
        params["latency_frames"] = 3
        params["roughness"] = 0.08
        params["motor_gain_left"] = 0.75
        params["motor_gain_right"] = 1.2
        cfg["camera"]["pitch_down_deg"] += rng.choice([-8, 8])
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


class PathFollower:
    """Privileged geometric teacher, never used as a camera-only evaluation policy."""

    def __init__(self, points, lookahead=0.16, speed=0.18, max_yaw=2.0):
        self.points = points
        self.lookahead, self.speed, self.max_yaw = lookahead, speed, max_yaw
        self.lengths = [math.dist(a, b) for a, b in zip(points, points[1:] + points[:1])]
        self.total_length = sum(self.lengths)

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
        distance, s, _, tangent = self.project(x, y)
        tx, ty = self.at(s + self.lookahead)
        dx, dy = tx - x, ty - y
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        curvature = 2 * lateral / max(dx * dx + dy * dy, 1e-6)
        heading_error = wrap_angle(tangent - yaw)
        speed = self.speed / max(1.0, abs(curvature) * 0.09)
        omega = max(-self.max_yaw, min(self.max_yaw, speed * curvature))
        return (
            speed,
            omega,
            {"route_distance_m": distance, "route_s_m": s, "heading_error_rad": heading_error},
        )


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
