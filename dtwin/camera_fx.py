"""Real Pi-camera behaviour applied to rendered frames (training randomization, --disturb).

After 원철's bench (Edwin29/AutoMobile drive-v4-bench bench/realcam.py, bench/disturb.py),
whose ranges come from ~2,500 real Pi frames: motion blur from the car's own motion during
the exposure, focus blur, haze, lens shading and magenta corners, auto-exposure that lags
scene changes (and hunts), white-balance drift, a dark magenta start, gain-amplified sensor
noise, and random glare / lens-occlusion / white-out bursts. Rendered glare and shadows are
scene effects (dtwin/scene.py); this module only changes pixels.
"""

from __future__ import annotations

import math
import random

import numpy as np

FX = 365.0  # camera focal length in px (configs/rtx.json intrinsics), for yaw blur


def sample_camera_fx(ranges, rng):
    """One camera personality from ``randomization.camera_fx``; uniform within each range."""
    out = {k: rng.uniform(*v) for k, v in ranges.items() if isinstance(v, list)}
    out["warm_blur"] = rng.random() < ranges["warm_blur_probability"]
    out["seed"] = rng.randrange(1 << 30)
    return out


class CameraFx:
    def __init__(self, p):
        self.p, self.rng = p, random.Random(p["seed"])
        self.np = np.random.default_rng(p["seed"])
        self.gain, self.meter_ref, self.awb = max(0.2, p["start_gain"]), None, np.ones(3)
        self.awb_target, self.t, self.bursts, self.events = np.ones(3), 0.0, [], []
        self.shading = None

    def _lens(self, h, w):
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        vig = 1 - self.p["vignette"] * np.clip(r2 / 2, 0, 1)
        corner = self.p["magenta_corner"] * np.clip(r2 - 1, 0, 1)
        return np.stack([vig * (1 + corner), vig * (1 - corner), vig * (1 + corner)], -1)

    def _bursts(self, dt, h, w):
        """Poisson glare / occlusion / white-out events with random place and length."""
        for kind, base_s in (("glare", 3.0), ("occlusion", 2.5), ("dropout", 1.2), ("flare", 1.5)):
            if self.rng.random() < self.p[kind + "_per_min"] / 60 * dt:
                e = dict(kind=kind, left=base_s * self.rng.uniform(0.5, 1.5),
                         x=self.rng.uniform(0.15, 0.85) * w, y=self.rng.uniform(0.45, 0.8) * h,
                         size=self.rng.uniform(0.8, 1.4), side=self.rng.randrange(2))
                self.bursts.append(e)
                self.events.append(dict(t=round(self.t, 2), kind=kind))
        for e in self.bursts:
            e["left"] -= dt
        self.bursts = [e for e in self.bursts if e["left"] > 0]

    def apply(self, frame, dt, linear_m_s, angular_rad_s):
        import cv2

        p = self.p
        self.t += dt
        h, w = frame.shape[:2]
        x = frame.astype(np.float32)
        # motion blur: yaw sweeps the image sideways, forward motion smears it vertically
        exposure = p["exposure_s"] * float(np.clip(1 / max(self.gain, 1e-3), 0.8, 1.6))  # darker -> longer
        sweep = int(abs(angular_rad_s) * exposure * FX)
        smear = int(abs(linear_m_s) * exposure * 400)  # ~400 px/m on the near road
        if sweep > 1:
            x = cv2.filter2D(x, -1, np.full((1, sweep), 1 / sweep, np.float32))
        if smear > 1:
            x = cv2.filter2D(x, -1, np.full((smear, 1), 1 / smear, np.float32))
        if p["focus_sigma"] > 0.3:
            x = cv2.GaussianBlur(x, (0, 0), p["focus_sigma"])
        if p["warm_blur"]:
            x = cv2.GaussianBlur(x, (7, 7), 1.4) * np.array([1.12, 1.02, 0.85], np.float32)  # RGB order
        x = x * (1 - p["haze"]) + p["haze"] * x.mean()
        if self.shading is None or self.shading.shape[:2] != (h, w):
            self.shading = self._lens(h, w)
        x = x * self.shading
        self._bursts(dt, h, w)
        yy, xx = np.mgrid[:h, :w]
        for e in self.bursts:
            if e["kind"] in ("glare", "flare"):
                sx, sy = (57.6, 110.4) if e["kind"] == "glare" else (35.0, 60.0)
                alpha = 0.99 * np.exp(-0.5 * (((xx - e["x"]) / (sx * e["size"])) ** 2 + ((yy - e["y"]) / (sy * e["size"])) ** 2))
                tint = np.ones(3, np.float32) if e["kind"] == "glare" else np.array([0.6, 1.0, 1.3], np.float32)  # bluish ghost
                x = x + (255 - x) * alpha[..., None] * tint
            elif e["kind"] == "occlusion":
                x0 = 0 if e["side"] == 0 else int(w * 0.6)
                x[int(h * 0.33): int(h * 0.94), x0: x0 + int(w * 0.4)] = 70
        # auto exposure (centre-weighted, relative to the first frame) and white balance, both lagging
        meas = float(x[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean())
        self.meter_ref = self.meter_ref or meas
        want = self.meter_ref * p["target_rel"] / max(meas, 1.0)
        a = 1 - math.exp(-dt / p["ae_tau_s"])
        # 원철's bench clamps at 0.15-4; our renders are already darker than the real mat
        # (road V ~50 vs 74), so a lower floor would make frames black rather than realistic.
        self.gain = float(np.clip(math.exp(math.log(self.gain) + a * (math.log(want) - math.log(self.gain))), 0.45, 2.5))
        gain = self.gain * (1 + p["ae_hunt"] * math.sin(self.t * 2 * math.pi / 2.7))
        self.awb_target = np.clip(self.awb_target + self.np.normal(0, p["awb_drift"] * math.sqrt(dt), 3), 0.85, 1.15)
        self.awb += (self.awb_target - self.awb) * (1 - math.exp(-dt / 0.6))
        magenta = p["start_magenta"] * math.exp(-self.t / 1.0)  # first second after start-up
        awb = self.awb * np.array([1 + magenta, 1 - magenta, 1 + magenta])
        x = x * gain * awb.astype(np.float32)
        # sensor noise: read noise amplified by gain, shot noise grows with signal
        sigma = np.sqrt(p["read_noise"] ** 2 * max(gain, 1) + p["shot_noise"] * np.clip(x, 0, None))
        x = x + sigma * self.np.standard_normal(x.shape, dtype=np.float32)
        if any(e["kind"] == "dropout" for e in self.bursts):
            x[:] = 245  # white-out
        return np.clip(x, 0, 255).astype(np.uint8)
