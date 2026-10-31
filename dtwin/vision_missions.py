"""Sign/signal missions for a car WITHOUT wheel encoders: camera evidence only.

Distances come from ground-projecting the pink dashed line and the zebra crossing
with the camera calibration; standstill comes from the image not changing. PWM
commands are never treated as measured motion. Junction turns are NOT handled
here (no yaw measurement): they are reported as unverified and the lane policy
keeps driving.

Rules (docs/mission_rules.md): horn / STOP / speed20 once per sign, before its pink
line; red signal: stop before the crosswalk, wait ~1 s after a confirmed stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import CameraGeometry
from .road_vision import RoadMarkDetector
from .tracking import ObjectTracker

SIGN_KINDS = {"horn", "stop", "speed20"}


@dataclass
class VisionMissionOutput:
    speed_scale: float = 1.0
    stop: bool = False
    horn: bool = False
    events: list = field(default_factory=list)
    telemetry: dict = field(default_factory=dict)


class StandstillDetector:
    """Image-based standstill: the ground area of consecutive frames stops changing."""

    def __init__(self, occluder_px=0, threshold=2.5, frames=4):
        self.occluder, self.threshold, self.frames = occluder_px, threshold, frames
        self.previous, self.still = None, 0
        self.last_difference = None

    def update(self, rgb):
        import cv2

        height = rgb.shape[0]
        bottom = height - self.occluder if self.occluder else height
        ground = cv2.cvtColor(rgb[int(height * 0.45) : bottom], cv2.COLOR_RGB2GRAY)
        small = cv2.resize(cv2.GaussianBlur(ground, (5, 5), 0), (80, 40)).astype(np.float32)
        if self.previous is not None:
            self.last_difference = float(np.abs(small - self.previous).mean())
            self.still = self.still + 1 if self.last_difference < self.threshold else 0
        self.previous = small
        return self.still >= self.frames


class PinkLineMonitor:
    """Distance to the next pink line (front bumper, m) and 'line passed' events."""

    def __init__(self, front_bumper_m, near_m=0.30, lost_s=0.6):
        self.front, self.near, self.lost = front_bumper_m, near_m, lost_s
        self.distance, self.seen_at, self.crossings = None, None, 0

    def update(self, marks, now_s):
        crossed = False
        pinks = sorted(m.point[0] - self.front for m in marks if m.kind == "pink")
        ahead = [d for d in pinks if d > -0.05]
        if ahead:
            self.distance, self.seen_at = ahead[0], now_s
        elif self.distance is not None and now_s - self.seen_at > self.lost:
            # A line that was close and then left the bottom of the image was driven over.
            if self.distance < self.near:
                crossed = True
                self.crossings += 1
            self.distance, self.seen_at = None, None
        return crossed


class VisionMissionDriver:
    def __init__(self, cfg, detector, calibration=None):
        self.cfg, self.settings, self.detector = cfg, cfg["missions"], detector
        camera = calibration or cfg["camera"]
        self.geometry = CameraGeometry(camera, 1.3, self.settings["sign_bottom_m"], self.settings["signal_bottom_m"])
        self.road = RoadMarkDetector(self.geometry)
        self.front = cfg["vehicle"]["length_m"] / 2
        self.tracker = ObjectTracker(self.settings["min_track_hits"], self.settings["track_max_age_s"])
        self.pink = PinkLineMonitor(self.front)
        self.standstill = StandstillDetector(camera.get("bottom_occluder_px", 0))
        self.done = {}  # kind -> pink-crossing count when executed (re-armed after next line)
        self.active = None  # (kind, phase, started_s)
        self.speed_scale = 1.0
        self.fault = None
        self.events = []
        self.last_capture = -1e9

    def _armed(self, kind):
        return kind not in self.done or self.pink.crossings > self.done[kind]

    def step(self, rgb, now_s, capture_s):
        out = VisionMissionOutput(speed_scale=self.speed_scale)
        if capture_s <= self.last_capture:
            out.telemetry = dict(skipped="repeated_frame")
            return out
        self.last_capture = capture_s
        if now_s - capture_s > self.settings["max_frame_age_s"]:
            out.stop = True
            out.events.append({"event": "stale_camera_frame"})
        detections = self.detector.detect(rgb)
        if any(d.get("source") != "vision" for d in detections):
            raise ValueError("Vision missions cannot consume renderer ground truth")
        marks = self.road.detect(rgb, detections)
        if self.pink.update(marks, capture_s):
            out.events.append({"event": "pink_line_passed", "count": self.pink.crossings})
            if self.active and self.active[0] in SIGN_KINDS and self.active[1] != "done":
                self.fault = self.fault or "pink_deadline_missed"
        stopped = self.standstill.update(rgb)
        crosswalk = min(
            (m.point[0] - self.front for m in marks if m.kind == "crosswalk" and m.point[0] - self.front > -0.05),
            default=None,
        )
        tracks = [
            t for t in self.tracker.update(detections, capture_s) if t.confidence >= self.settings["confidence"]
        ]
        kinds = {t.kind for t in tracks}

        # Start a mission for a confidently tracked, re-armed sign.
        if self.active is None and not self.fault:
            for kind in ["stop", "horn", "speed20"]:
                if kind in kinds and self._armed(kind):
                    if self.pink.distance is not None and self.pink.distance <= self.settings["pink_margin_m"]:
                        out.events.append({"event": "too_close_to_pink_line", "kind": kind})
                        self.done[kind] = self.pink.crossings
                        continue
                    self.active = (kind, "start", now_s)
                    out.events.append({"event": "mission_started", "kind": kind, "pink_distance_m": self.pink.distance})
                    break
            if self.active is None and "signal_red" in kinds and self._armed("signal"):
                self.active = ("signal", "approach", now_s)
                out.events.append({"event": "red_signal_seen", "crosswalk_distance_m": crosswalk})
            elif self.active is None and "signal_green" in kinds and self._armed("signal"):
                self.done["signal"] = self.pink.crossings
                out.events.append({"event": "green_signal_seen", "junction_turn": "not_implemented_vision_only"})

        if self.active:
            kind, phase, started = self.active
            if kind == "horn":
                out.horn = True
                self._finish(kind, out)
            elif kind == "speed20":
                self.speed_scale = self.settings["slowdown_scale"]
                self._finish(kind, out)
            elif kind == "stop":
                out.stop = True
                if phase == "start" and stopped:
                    self.active = (kind, "dwell", now_s)
                    out.events.append({"event": "standstill_confirmed_by_image"})
                elif phase == "dwell" and now_s - started >= self.settings["stop_dwell_s"]:
                    out.stop = False
                    self._finish(kind, out)
            elif kind == "signal":
                braking = self.settings["junction_speed_m_s"] ** 2 / (2 * self.settings["braking_m_s2"]) + 0.02
                if phase == "approach":
                    if crosswalk is not None and crosswalk <= braking:
                        out.stop = True
                        self.active = (kind, "stopping", now_s)
                    else:
                        out.speed_scale = min(out.speed_scale, 0.5)  # approach slowly until the crossing is seen
                elif phase == "stopping":
                    out.stop = True
                    if stopped:
                        self.active = (kind, "dwell", now_s)
                        out.events.append({"event": "standstill_confirmed_by_image"})
                elif phase == "dwell":
                    out.stop = True
                    if now_s - started >= self.settings["red_dwell_s"]:
                        out.stop = False
                        out.events.append({"event": "red_wait_done", "junction_turn": "not_implemented_vision_only"})
                        self._finish(kind, out)
                if now_s - started > self.settings["junction_timeout_s"]:
                    self.fault = self.fault or "signal_timeout"
        if self.fault:
            out.stop = True
        out.speed_scale = min(out.speed_scale, self.speed_scale)
        out.telemetry = dict(
            source="vision_only(no encoders)",
            detections=detections,
            tracks=[t.kind for t in tracks],
            pink_distance_m=self.pink.distance,
            pink_crossings=self.pink.crossings,
            crosswalk_distance_m=crosswalk,
            image_difference=self.standstill.last_difference,
            standstill=stopped,
            active=None if self.active is None else list(self.active[:2]),
            fault=self.fault,
        )
        self.events += [dict(e, time_s=now_s) for e in out.events]
        return out

    def _finish(self, kind, out):
        self.done[kind] = self.pink.crossings
        self.active = None
        out.events.append({"event": "mission_completed", "kind": kind})
