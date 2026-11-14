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


STRESS_PARTS = ("image", "latency", "gloss", "motor", "camera", "dynamics")
# Evaluation-only parts for the --disturb conditions; never part of the default stress set,
# so earlier stress results stay comparable. Fixed harsh values (원철's bench "harsh" profile).
EXTRA_STRESS_PARTS = ("vehicle", "camerafx", "people")
STRESS_VEHICLE = dict(start_duty=[0.33, 0.33], breakaway_duty=[0.05, 0.05], max_effective_duty=[0.52, 0.52],
                      delay_s=[0.12, 0.12], speed_gain=[0.95, 1.05], turn_eff=[0.85, 0.85], drift_rad_s=[0.02, 0.02],
                      pivot_slide_m_per_rad=[0.01, 0.01], wander=[0.05, 0.05], trend_per_min=[-0.05, -0.05],
                      patches=[4, 4], patch_speed=[0.75, 0.85], patch_turn=[0.65, 0.8], patch_size_m=[0.3, 0.4],
                      push_per_min=[3.0, 3.0], push_lateral_m=[0.02, 0.04], push_yaw_rad=[0.05, 0.15],
                      push_duration_s=0.25, spin_per_min=[2.0, 2.0], spin_factor=[0.3, 0.5], spin_duration_s=[0.3, 0.8])
STRESS_CAMERA_FX = dict(exposure_s=[0.09, 0.09], focus_sigma=[2.0, 2.0], haze=[0.12, 0.12], target_rel=[0.7, 1.3],
                        ae_tau_s=[0.6, 0.6], ae_hunt=[0.25, 0.25], start_gain=[0.3, 0.3], awb_drift=[0.08, 0.08],
                        start_magenta=[0.6, 0.6], vignette=[0.3, 0.3], magenta_corner=[0.25, 0.25],
                        read_noise=[4.0, 4.0], shot_noise=[1.0, 1.0], warm_blur_probability=0.0,
                        glare_per_min=[1.5, 1.5], occlusion_per_min=[0.8, 0.8], dropout_per_min=[0.6, 0.6],
                        flare_per_min=[1.5, 1.5])
STRESS_PEOPLE = dict(count=[4, 4], edge_offset_m=[-0.05, 0.1], height_m=[1.6, 1.8], radius_m=[0.18, 0.22],
                     shade=[0.05, 0.2], walker_probability=0.5, walk_speed_m_s=[0.3, 0.5], walk_span_m=[1.0, 1.6],
                     light_elevation_deg=[50.0, 50.0], light_intensity=[3500.0, 3500.0])


def sample_episode(base, seed, randomize=False, stress=False, harsh=False, stress_parts=None, disturb=False):
    """A seed completely identifies scene parameters. Stress uses held-out ranges.

    ``harsh`` (training only) widens appearance/actuation draws toward the stress
    conditions and adds low distant lights whose specular glare the real mat shows.
    ``stress_parts`` applies only some stress changes (ablation); None = all of them.
    ``disturb`` (training) adds the real-car motor/body effects of :class:`Actuation`.
    """
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
        if harsh:
            # Drawn after everything above so non-harsh seeds keep their parameters.
            h = r["harsh"]
            pick = lambda name: rng.uniform(*h[name])
            if rng.random() < h["glossy_probability"]:
                params["roughness"] = pick("glossy_roughness")
            params["image_gain"] = pick("image_gain")
            params["image_noise_std"] = pick("image_noise_std")
            params["latency_frames"] = rng.randint(*h["latency_frames"])
            params["motor_gain_left"], params["motor_gain_right"] = pick("motor_gain"), pick("motor_gain")
            cfg["camera"]["pitch_down_deg"] += pick("extra_pitch_deg")
            # The camera sits 7cm high looking 23 deg down, so it only sees reflections of
            # light sources low and far away (windows, distant lamps), not of lamps overhead.
            params["glare_lights"] = []
            for _ in range(rng.randint(*h["glare_lights"])):
                angle, distance = rng.uniform(-math.pi, math.pi), pick("light_distance_m")
                params["glare_lights"].append(dict(
                    position=[distance * math.cos(angle), distance * math.sin(angle), pick("light_height_m")],
                    radius=pick("light_radius_m"),
                    intensity=pick("light_intensity"),
                ))
    if disturb:
        # Drawn after the harsh block so earlier parameters of every seed stay the same.
        params["actuation"] = sample_actuation(r["vehicle"], rng)
        from .camera_fx import sample_camera_fx

        params["camera_fx"] = sample_camera_fx(r["camera_fx"], rng)
        params["people"], params["shadow_light"] = sample_people(r["people"], rng)
    if stress:
        # Both choices are always drawn so a subset of parts keeps the same values.
        gain, pitch = rng.choice([0.55, 1.5]), rng.choice([-8, 8])
        parts = set(STRESS_PARTS if stress_parts is None else stress_parts)
        if not parts <= set(STRESS_PARTS + EXTRA_STRESS_PARTS):
            raise ValueError(f"Unknown stress parts {sorted(parts - set(STRESS_PARTS + EXTRA_STRESS_PARTS))}")
        params["stress_parts"] = sorted(parts)
        if "image" in parts:
            params["image_gain"], params["image_noise_std"] = gain, 9.0
        if "latency" in parts:
            params["latency_frames"] = 3
        if "gloss" in parts:
            params["roughness"] = 0.08
        if "motor" in parts:
            params["motor_gain_left"], params["motor_gain_right"] = 0.75, 1.2
        if "camera" in parts:
            cfg["camera"]["pitch_down_deg"] += pitch
        if "dynamics" in parts and "skid_yaw_efficiency" in r:
            params["skid_yaw_efficiency"] = min(r["skid_yaw_efficiency"]) * 0.9
            params["motor_time_constant_s"] = max(r["motor_time_constant_s"]) * 1.3
        # Drawn after the original parts (always, in a fixed order) so they never shift them.
        from .camera_fx import sample_camera_fx

        extra = dict(actuation=sample_actuation(STRESS_VEHICLE, rng), camera_fx=sample_camera_fx(STRESS_CAMERA_FX, rng))
        extra["people"], extra["shadow_light"] = sample_people(STRESS_PEOPLE, rng)
        if "vehicle" in parts:
            params["actuation"] = extra["actuation"]
        if "camerafx" in parts:
            params["camera_fx"] = extra["camera_fx"]
        if "people" in parts:
            params["people"], params["shadow_light"] = extra["people"], extra["shadow_light"]
    return cfg, params


def sample_actuation(v, rng):
    """One car personality for :class:`Actuation` from the ``randomization.vehicle`` ranges."""
    pick = lambda name: rng.uniform(*v[name])
    pair = lambda name: [pick(name), pick(name)]
    mixer = json.loads((ROOT / "configs/real_car.json").read_text())["duty_mixer"]
    patches = []
    for _ in range(rng.randint(*v["patches"])):
        # shortcut: outer-road centerline taken as a 0.86 m half-size square; fine for 20-40 cm patches.
        side, along, size = rng.randrange(4), rng.uniform(-0.8, 0.8), pick("patch_size_m") / 2
        cx, cy = [(0.86, along), (-0.86, along), (along, 0.86), (along, -0.86)][side]
        patches.append(dict(box=[cx - size, cx + size, cy - size, cy + size],
                            speed=pick("patch_speed"), turn=pick("patch_turn")))
    return dict(
        min_duty=mixer["min_duty"], max_duty=mixer["max_duty"],
        start_duty=pair("start_duty"), breakaway_duty=pick("breakaway_duty"),
        max_effective_duty=pick("max_effective_duty"), delay_s=pick("delay_s"),
        speed_gain=pair("speed_gain"), turn_eff=pair("turn_eff"), drift_rad_s=pick("drift_rad_s"),
        pivot_slide_m_per_rad=pick("pivot_slide_m_per_rad"), wander=pick("wander"), wander_tau_s=8.0,
        trend_per_min=pick("trend_per_min"), patches=patches,
        push_per_min=pick("push_per_min"), push_lateral_m=v["push_lateral_m"], push_yaw_rad=v["push_yaw_rad"],
        push_duration_s=v["push_duration_s"], spin_per_min=pick("spin_per_min"),
        spin_factor=v["spin_factor"], spin_duration_s=v["spin_duration_s"],
    )


def sample_people(v, rng):
    """People standing or walking just outside the mat edge, and a low light that throws their
    shadows across the road (the real classroom: students around the track)."""
    pick = lambda name: rng.uniform(*v[name])
    people = []
    for _ in range(rng.randint(*v["count"])):
        side, along = rng.randrange(4), rng.uniform(-0.9, 0.9)
        across = 1.15 + pick("edge_offset_m")  # mat half-size ~1.18 m; negative = on the green border
        start, direction = [((across, along), (0, 1)), ((-across, along), (0, 1)),
                            ((along, across), (1, 0)), ((along, -across), (1, 0))][side]
        walk = rng.random() < v["walker_probability"]
        people.append(dict(start=list(start), direction=list(direction), height=pick("height_m"),
                           radius=pick("radius_m"), shade=pick("shade"),
                           speed=pick("walk_speed_m_s") if walk else 0.0, span=pick("walk_span_m")))
    # The light stands behind one person (seen from the mat centre), so that person's shadow
    # falls inward across the road, as under the classroom ceiling lamps.
    anchor = people[rng.randrange(len(people))]["start"] if people else [rng.uniform(-1, 1), 1.0]
    light = dict(azimuth=math.atan2(anchor[1], anchor[0]) + rng.uniform(-0.5, 0.5),
                 elevation=math.radians(pick("light_elevation_deg")), intensity=pick("light_intensity"))
    return people, light


def person_position(person, t):
    """Walkers go back and forth along the mat edge; standing people stay at ``start``."""
    if not person["speed"]:
        return person["start"]
    period = 2 * person["span"] / person["speed"]
    phase = (t % period) / period
    along = person["span"] * (2 * phase if phase < 0.5 else 2 - 2 * phase) - person["span"] / 2
    return [person["start"][0] + person["direction"][0] * along, person["start"][1] + person["direction"][1] * along]


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


class Actuation:
    """Real-car motor effects the plain skid model lacks (after 원철's bench vehicle model).

    Wheel targets are mapped to the team duty (configs/real_car.json: min_duty offset,
    max_duty at the simulator top speed). A side below its start duty does not turn (plus
    breakaway friction from standstill); above the effective max duty it saturates.
    Commands are delayed; side speeds drift (wander, battery sag, floor patches, short
    spins); turns use separate left/right efficiencies; the body drifts, slides sideways
    while pivoting and is occasionally pushed. All draws come from ``seed``.
    """

    def __init__(self, params, vehicle, max_speed_m_s, seed):
        self.p, self.vehicle = params, vehicle
        self.rate_per_unit = max_speed_m_s / vehicle["wheel_radius_m"]  # wheel rad/s at duty max
        self.min_duty, self.max_duty = params["min_duty"], params["max_duty"]
        self.rng = random.Random(seed ^ 0xAC71)
        self.t, self.queue = 0.0, []
        self.wander = {k: 0.0 for k in ("speed_left", "speed_right", "turn_left", "turn_right", "start")}
        self.push = self.spin = None
        self.events = []

    def _duty(self, rate):
        unit = min(1.0, abs(rate) / self.rate_per_unit)
        return 0.0 if unit == 0 else self.min_duty + (self.max_duty - self.min_duty) * unit

    def _patch(self, pose):
        for patch in self.p.get("patches", []):
            x0, x1, y0, y1 = patch["box"]
            if x0 <= pose[0] <= x1 and y0 <= pose[1] <= y1:
                return patch["speed"], patch["turn"]
        return 1.0, 1.0

    def _events(self, dt):
        """Poisson pushes and wheel spins; each lasts a while once started."""
        p, rng = self.p, self.rng
        if self.push is None and rng.random() < p.get("push_per_min", 0) / 60 * dt:
            self.push = dict(left=p["push_duration_s"], lateral=rng.choice([-1, 1]) * rng.uniform(*p["push_lateral_m"]),
                             yaw=rng.choice([-1, 1]) * rng.uniform(*p["push_yaw_rad"]))
            self.push["total"] = self.push["left"]
            self.events.append(dict(t=round(self.t, 2), kind="push", lateral_m=self.push["lateral"], yaw_rad=self.push["yaw"]))
        if self.spin is None and rng.random() < p.get("spin_per_min", 0) / 60 * dt:
            self.spin = dict(left=rng.uniform(*p["spin_duration_s"]), side=rng.randrange(2),
                             factor=rng.uniform(*p["spin_factor"]))
            self.events.append(dict(t=round(self.t, 2), kind="spin", side="LR"[self.spin["side"]], factor=self.spin["factor"]))

    def targets(self, left, right, rates, pose, dt):
        """Delayed, duty-limited wheel targets (rad/s) and the left/right turn factors."""
        p = self.p
        self.t += dt
        tau, sigma = p.get("wander_tau_s", 8.0), p.get("wander", 0.0)
        for k in self.wander:  # Ornstein-Uhlenbeck around the base values
            self.wander[k] += -self.wander[k] * dt / tau + sigma * math.sqrt(2 * dt / tau) * self.rng.gauss(0, 1)
        self._events(dt)
        self.queue.append((left, right))
        delay_steps = round(p.get("delay_s", 0.0) / dt)
        left, right = self.queue[0] if len(self.queue) > delay_steps else (0.0, 0.0)
        del self.queue[: max(0, len(self.queue) - delay_steps)]
        patch_speed, patch_turn = self._patch(pose)
        sag = max(0.3, 1 + p.get("trend_per_min", 0.0) * self.t / 60)
        out = []
        for i, (target, side) in enumerate(((left, "left"), (right, "right"))):
            duty = self._duty(target)
            start = p["start_duty"][i] + self.wander["start"] * 0.1
            if abs(rates[i]) < 0.5:  # standstill: static friction needs extra duty
                start += p.get("breakaway_duty", 0.0)
            if duty < start:
                out.append(0.0)
                continue
            cap = p.get("max_effective_duty", self.max_duty)
            if duty > cap:
                target *= (cap - self.min_duty) / (duty - self.min_duty)
            gain = p["speed_gain"][i] * (1 + self.wander["speed_" + side]) * sag * patch_speed
            if self.spin and self.spin["side"] == i:
                gain *= self.spin["factor"]
            out.append(target * gain)
        if self.spin:
            self.spin["left"] -= dt
            self.spin = self.spin if self.spin["left"] > 0 else None
        turn = tuple(p["turn_eff"][i] * (1 + self.wander["turn_" + s]) * patch_turn for i, s in enumerate(("left", "right")))
        return out[0], out[1], turn

    def body(self, linear, angular, dt):
        """Extra (lateral m, yaw rad) motion this step: drift, pivot slide, push."""
        p = self.p
        dyaw = p.get("drift_rad_s", 0.0) * linear / 0.05 * dt
        lateral = 0.0
        if abs(linear) < 0.02 and abs(angular) > 0.3:  # pivoting: the body slides one way
            lateral += p.get("pivot_slide_m_per_rad", 0.0) * abs(angular) * dt
        if self.push:
            share = dt / self.push["total"]
            lateral += self.push["lateral"] * share
            dyaw += self.push["yaw"] * share
            self.push["left"] -= dt
            self.push = self.push if self.push["left"] > 0 else None
        return lateral, dyaw


class SkidSteerModel:
    """Kinematic skid-steer ground motion for the simulator's ``kinematic_skid`` mode.

    Wheel rates follow motor targets through gain, first-order lag and saturation.
    Ground yaw is the ideal differential rate scaled by ``yaw_efficiency`` (lateral
    tire scrub). ``wheel_rates`` is what an encoder measures, not the target.
    ``actuation`` (optional) adds the real-car motor effects of :class:`Actuation`.
    """

    def __init__(self, vehicle, pose, yaw_efficiency, time_constant_s, gains=(1.0, 1.0), actuation=None):
        if not 0 < yaw_efficiency <= 1 or time_constant_s <= 0:
            raise ValueError("Skid model needs 0 < yaw_efficiency <= 1 and positive time constant")
        self.vehicle, self.pose = vehicle, tuple(pose)
        self.yaw_efficiency, self.time_constant = yaw_efficiency, time_constant_s
        self.gains = tuple(gains)
        self.wheel_rates = (0.0, 0.0)
        self.actuation = actuation

    def step(self, left_target, right_target, dt):
        limit = self.vehicle["max_wheel_rad_s"]
        turn = (1.0, 1.0)
        if self.actuation:
            left_target, right_target, turn = self.actuation.targets(
                left_target, right_target, self.wheel_rates, self.pose, dt
            )
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
        angular *= turn[0] if angular > 0 else turn[1]  # left (ccw) and right turns slip differently
        x, y, yaw = self.pose
        mid = yaw + angular * dt / 2
        x, y, yaw = x + linear * math.cos(mid) * dt, y + linear * math.sin(mid) * dt, yaw + angular * dt
        if self.actuation:
            lateral, dyaw = self.actuation.body(linear, angular, dt)
            x, y, yaw = x - lateral * math.sin(yaw), y + lateral * math.cos(yaw), yaw + dyaw
        self.pose = (x, y, wrap_angle(yaw))
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
        straight_gain=1.0,
        brake_m=0.3,
        gain_rise_per_s=0.5,
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
        # Speed profile: faster on straights, brake over brake_m before a curve (visible to the
        # camera ~35 cm ahead), curve at the base speed, then accelerate at gain_rise_per_s.
        self.straight_gain, self.brake_m, self.gain_rise = straight_gain, brake_m, gain_rise_per_s
        self.speed_factor, self.corner_starts = 1.0, []
        if straight_gain > 1.0:
            step, window = 0.01, 0.05
            n = int(self.total_length / step)
            heading = lambda s: math.atan2(*reversed([b - a for a, b in zip(self.at(s), self.at(s + 0.005))]))
            self.turning = [abs(wrap_angle(heading(i * step + window) - heading(i * step))) / window > 1.5
                            for i in range(n)]  # > 1.5 rad/m: curve or junction turn
            self.corner_starts = [i * step for i in range(n) if self.turning[i] and not self.turning[i - 1]]
            self.profile_step = step

    def _profile(self, s):
        """Speed factor from the route ahead: 1 inside a curve, straight_gain far from one."""
        if not self.corner_starts:
            return 1.0
        inside = self.turning[int(s % self.total_length / self.profile_step) % len(self.turning)]
        ahead = min((a - s) % self.total_length for a in self.corner_starts)
        target = 1.0 if inside else 1.0 + (self.straight_gain - 1.0) * min(1.0, ahead / self.brake_m)
        # brake at once, accelerate gradually
        self.speed_factor = min(target, self.speed_factor + self.gain_rise * self.dt)
        return self.speed_factor

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
        factor = self._profile(s)
        info = {"route_distance_m": distance, "route_s_m": s, "heading_error_rad": heading_error,
                "teacher_speed_factor": factor}
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
        speed = self.speed * factor * math.cos(bearing) ** 2 / max(1.0, abs(curvature) * self.corner_gain)
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
