"""Camera detections -> tracks -> range -> missions -> wheel commands.

Public input contains RGB, camera timestamps and wheel angular speeds only.
The configured map/start pose is a prior; no simulator pose/labels are consumed.
"""

from __future__ import annotations

import math
from dataclasses import asdict

from .geometry import CameraGeometry, WheelOdometry, nearest_on_segment, to_vehicle, to_world
from .missions import MissionController, Observation
from .navigation import JunctionNavigator
from .road_vision import RoadMarkDetector
from .tracking import EncounterMemory, ObjectTracker


class MapContext:
    def __init__(self, layout, front_bumper, cfg):
        self.cfg, self.front = cfg, front_bumper
        self.marks = {}
        for mark in layout["road_marks"]:
            if mark["kind"] == "pink_dashed":
                center = tuple((a + b) / 2 for a, b in zip(mark["p0"], mark["p1"]))
                angle = math.atan2(mark["p1"][1] - mark["p0"][1], mark["p1"][0] - mark["p0"][0])
                self.marks[mark["id"]] = dict(
                    kind="pink",
                    center=center,
                    axis=angle + math.pi / 2,
                    half_length=0.0015,
                    inner=False,
                    segment=(mark["p0"], mark["p1"]),
                )
        for mark in layout["crosswalks"]:
            self.marks[mark["id"]] = dict(
                kind="crosswalk",
                center=tuple(mark["center"]),
                axis=0.0 if mark["road_direction"] == "east-west" else math.pi / 2,
                half_length=mark["length_along_road"] / 2,
                inner=mark["id"].startswith("center-"),
            )
        self.visual = {}

    def update(self, features, capture_pose, timestamp):
        for feature in features:
            candidates = []
            for name, mark in self.marks.items():
                if (
                    mark["kind"] != feature.kind
                    or abs(math.cos(mark["axis"] - capture_pose[2])) < 0.8
                ):
                    continue
                point = to_world(
                    (
                        feature.point[0]
                        + (mark["half_length"] if feature.kind == "crosswalk" else 0),
                        feature.point[1],
                    ),
                    capture_pose,
                )
                candidates.append((math.dist(point, mark["center"]), name))
            candidates.sort()
            if candidates and candidates[0][0] <= self.cfg["map_match_m"]:
                if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.025:
                    continue
                self.visual[candidates[0][1]] = (to_world(feature.point, capture_pose), timestamp)

    def select(self, kind, pose, inner=False, anchor=None):
        candidates = []
        for name, mark in self.marks.items():
            if mark["kind"] != kind or (inner and not mark["inner"]):
                continue
            if abs(math.cos(mark["axis"] - pose[2])) < 0.8:
                continue
            x, y = to_vehicle(mark["center"], pose)
            distance = x - mark["half_length"] - self.front
            if anchor is not None and kind == "pink":
                score = math.dist(anchor, nearest_on_segment(anchor, *mark["segment"]))
                if (
                    -0.4 < distance < self.cfg["max_mark_distance_m"]
                    and abs(y) < 0.13
                    and score < 0.35
                ):
                    candidates.append((score, name))
            elif 0.01 < distance < self.cfg["max_mark_distance_m"] and abs(y) < 0.13:
                candidates.append((distance, name))
        return min(candidates)[1] if candidates else None

    def distance(self, name, pose, now):
        if name is None:
            return None, "unavailable"
        if name in self.visual and now - self.visual[name][1] <= self.cfg["mark_memory_s"]:
            return to_vehicle(self.visual[name][0], pose)[
                0
            ] - self.front, "camera_ground_projection+wheel_odometry"
        mark = self.marks[name]
        return to_vehicle(mark["center"], pose)[0] - mark[
            "half_length"
        ] - self.front, "configured_map+wheel_odometry"


class MissionDriver:
    def __init__(self, cfg, layout, initial_pose, detector, calibration=None):
        self.cfg, self.settings, self.detector = cfg, cfg["missions"], detector
        self.geometry = CameraGeometry(
            calibration or cfg["camera"],
            self.settings["max_mark_distance_m"] + 0.6,
            self.settings["sign_bottom_m"],
            self.settings["signal_bottom_m"],
        )
        self.road = RoadMarkDetector(self.geometry)
        self.odometry = WheelOdometry(cfg["vehicle"], initial_pose)
        self.tracker = ObjectTracker(
            self.settings["min_track_hits"], self.settings["track_max_age_s"]
        )
        self.encounters = EncounterMemory(
            self.settings["encounter_merge_m"], self.settings["revisit_distance_m"]
        )
        self.map = MapContext(layout, cfg["vehicle"]["length_m"] / 2, self.settings)
        nav_cfg = dict(self.settings, front_bumper_m=cfg["vehicle"]["length_m"] / 2)
        self.navigator = JunctionNavigator(layout, nav_cfg, initial_pose)
        self.controller = MissionController(
            stop_dwell_s=self.settings["stop_dwell_s"],
            red_dwell_s=self.settings["red_dwell_s"],
            slowdown_scale=self.settings["slowdown_scale"],
            braking_m_s2=self.settings["braking_m_s2"],
            confidence=self.settings["confidence"],
        )
        self.context = {}
        self.retained = {}
        self.fault = None
        self.last_capture = -math.inf
        self.events = []
        self.completed = set()
        self.speed_pending = set()
        self.signal_pending = {}
        self.last_guard = None
        self.ignored_signs = set()

    def step(self, rgb, now_s, capture_s, left_rad_s, right_rad_s, lane_command):
        if rgb.shape[:2] != (self.geometry.height, self.geometry.width):
            raise ValueError("Camera image dimensions differ from calibration")
        if not all(math.isfinite(v) for v in [capture_s, *lane_command]):
            raise ValueError("Camera timestamp and lane command must be finite")
        pose = self.odometry.update(now_s, left_rad_s, right_rad_s)
        self.encounters.mark_passed(pose, self.odometry.distance)
        events, detections, observations, transient = [], [], [], []
        if now_s - capture_s > self.settings["max_frame_age_s"] or capture_s > now_s + 1e-6:
            transient.append("stale_or_future_camera_frame")
        if capture_s > self.last_capture and self.fault is None and not transient:
            capture_pose = self.odometry.at(capture_s)
            detections = self.detector.detect(rgb)
            if any(d.get("source") != "vision" for d in detections):
                raise ValueError("MissionDriver cannot consume renderer-ground-truth labels")
            features = self.road.detect(rgb, detections)
            self.map.update(features, capture_pose, capture_s)
            for track in self.tracker.update(detections, capture_s):
                local = self.geometry.object_position({"class": track.kind, "xyxy": track.xyxy})
                if local is None or local[0] > self.settings["max_sign_distance_m"]:
                    continue
                world = to_world(local, capture_pose)
                encounter = self.encounters.associate(track, world, now_s, self.odometry.distance)
                key = encounter.id
                if key not in self.context:
                    self.context[key] = dict(
                        pink=self.map.select("pink", capture_pose, anchor=encounter.world),
                        crosswalk=self.map.select("crosswalk", capture_pose, inner=True),
                        junction=self.navigator.active
                        or self.navigator.next_junction(capture_pose),
                    )
                else:
                    context = self.context[key]
                    if context["pink"] is None:
                        context["pink"] = self.map.select(
                            "pink", capture_pose, anchor=encounter.world
                        )
                    if context["crosswalk"] is None:
                        context["crosswalk"] = self.map.select(
                            "crosswalk", capture_pose, inner=True
                        )
                    if context["junction"] is None:
                        context["junction"] = self.navigator.active or self.navigator.next_junction(
                            capture_pose
                        )
                self.retained[key] = dict(
                    kind=track.kind,
                    confidence=track.confidence,
                    world=encounter.world,
                    last_seen=now_s,
                )
            self.last_capture = capture_s
        for key, obj in list(self.retained.items()):
            if key in self.controller.done:
                continue
            if now_s - obj["last_seen"] > self.settings["observation_memory_s"] and not obj[
                "kind"
            ].startswith("signal_"):
                continue
            context = self.context[key]
            confidence = obj["confidence"]
            if confidence < self.settings["confidence"]:
                continue
            local = to_vehicle(obj["world"], pose)
            if local[0] < -0.05:
                continue
            pink_distance, pink_source = self.map.distance(context["pink"], pose, now_s)
            stop_distance, stop_source = self.map.distance(context["crosswalk"], pose, now_s)
            common = dict(encounter_id=key, confidence=confidence, junction_id=context["junction"])
            kind = obj["kind"]
            if kind.startswith("signal_"):
                # Outer crossings are deliberately excluded. A signal has no effect
                # without a geometrically compatible INNER crossing ahead.
                if context["crosswalk"] is None or context["junction"] != "center_4way":
                    continue
                color = kind.removeprefix("signal_")
                if color == "yellow":
                    events.append({"event": "yellow_rule_unconfirmed", "id": key})
                    if self.settings.get("yellow_behavior", "stop") == "stop":
                        transient.append("yellow_requires_operator_rule")
                observations.append(
                    Observation(
                        kind="signal", signal_color=color, stop_distance_m=stop_distance, **common
                    )
                )
                events.append(
                    {
                        "event": "signal_range",
                        "id": key,
                        "distance_m": stop_distance,
                        "range_source": stop_source,
                    }
                )
            elif kind.startswith("direction_"):
                observations.append(
                    Observation(
                        kind="direction", direction=kind.removeprefix("direction_"), **common
                    )
                )
            else:
                if pink_distance is None:
                    # No pink line of OUR road ahead is tied to this sign (yet): it is
                    # still far away or belongs to another road. Not ours to act on.
                    if key not in self.ignored_signs:
                        self.ignored_signs.add(key)
                        events.append({"event": "sign_not_on_route_ignored", "id": key, "kind": kind})
                    continue
                observations.append(
                    Observation(
                        kind=kind,
                        before_pink_line=pink_distance is not None
                        and pink_distance > self.settings["pink_margin_m"],
                        **common,
                    )
                )
                events.append(
                    {
                        "event": "pink_range",
                        "id": key,
                        "distance_m": pink_distance,
                        "range_source": pink_source,
                    }
                )
        # Remember red approaches after a temporary detector dropout: the stop
        # distance continues to come from the established crossing and odometry.
        for key, obj in self.retained.items():
            if (
                obj["kind"] == "signal_red"
                and key not in self.controller.done
                and key not in {o.encounter_id for o in observations}
            ):
                context = self.context[key]
                if context["junction"] == "center_4way" and context["crosswalk"] is not None:
                    transient.append("signal_lost_before_stop")
        active_observations = observations if not self.fault and not transient else []
        if self.controller.holding:
            key, kind = self.controller.holding
            distance, _ = self.map.distance(
                self.context[key]["pink" if kind == "stop" else "crosswalk"], pose, now_s
            )
            if distance is not None and distance <= 0:
                self.fault = self.fault or "stop_deadline_missed"
        output = self.controller.update(
            now_s, self.odometry.speed, active_observations, self.odometry.yaw_rate
        )
        events += output.events
        if any(
            e["event"] in {"stop_line_already_passed", "pink_deadline_unknown_or_missed"}
            for e in events
        ):
            self.fault = self.fault or "mission_deadline_missed"
        if output.direction:
            context = self.context.get(output.direction_encounter_id)
            if context:
                self.navigator.request(
                    context["junction"], output.direction, output.direction_priority
                )
                if output.direction_priority == 2:
                    self.signal_pending[output.direction_encounter_id] = context["junction"]
        # Multiple ordinary direction signs for the same junction are a conflict.
        for observation in observations:
            if observation.kind == "direction":
                self.navigator.request(observation.junction_id, observation.direction, 1)
        if self.fault or transient or output.stop:
            command = (0.0, 0.0)
        else:
            command, nav_events = self.navigator.command(
                now_s, pose, self.odometry.speed, lane_command, self.odometry.yaw_rate
            )
            events += nav_events
            velocity, angular = command
            scale = output.speed_scale
            command = velocity * scale, angular * scale
            if output.speed_limit_m_s is not None and abs(command[0]) > output.speed_limit_m_s:
                ratio = output.speed_limit_m_s / max(abs(command[0]), 1e-9)
                command = command[0] * ratio, command[1] * ratio
        if self.navigator.fault:
            self.fault = self.navigator.fault
        for event in output.events:
            if event["event"] == "mission_started" and event.get("kind") == "speed20":
                self.speed_pending.add(event["id"])
            if (
                event["event"] == "stop_completed"
                and self.retained.get(event["id"], {}).get("kind") == "stop"
            ):
                self._complete(event["id"], "stop", events)
        for key in list(self.speed_pending):
            distance, _ = self.map.distance(self.context[key]["pink"], pose, now_s)
            if distance is not None and distance <= 0:
                self.fault = self.fault or "slowdown_deadline_missed"
            elif (
                not self.fault
                and not transient
                and abs(self.odometry.speed) <= abs(lane_command[0]) * output.speed_scale + 0.015
            ):
                self._complete(key, "speed20", events)
                self.speed_pending.remove(key)
        if self.controller.holding and self.controller.holding[1] == "stop":
            key = self.controller.holding[0]
            distance, _ = self.map.distance(self.context[key]["pink"], pose, now_s)
            if distance is not None and distance <= 0:
                self.fault = self.fault or "stop_deadline_missed"
        for event in list(events):
            if event["event"] == "junction_completed":
                for key, junction in self.signal_pending.items():
                    if event["junction"] == junction:
                        self._complete(key, "signal", events)
                for key, context in self.context.items():
                    if context["junction"] == event["junction"] and self.retained[key][
                        "kind"
                    ].startswith("direction_"):
                        self.controller.done.add(key)
        horn = output.horn and self.fault is None and not transient
        if horn:
            for event in output.events:
                if event["event"] == "mission_started" and event.get("kind") == "horn":
                    self._complete(event["id"], "horn", events)
        if self.fault:
            command = (0.0, 0.0)
        command = (
            max(
                -self.cfg["simulation"]["max_speed_m_s"],
                min(self.cfg["simulation"]["max_speed_m_s"], command[0]),
            ),
            max(
                -self.cfg["simulation"]["max_yaw_rad_s"],
                min(self.cfg["simulation"]["max_yaw_rad_s"], command[1]),
            ),
        )
        self.events += [dict(event, time_s=now_s) for event in events]
        telemetry = dict(
            source="vision+wheel_encoders+configured_map",
            odometry_pose=list(pose),
            odometry_speed_m_s=self.odometry.speed,
            wheel_rad_s=[left_rad_s, right_rad_s],
            detections=detections,
            observations=[asdict(o) for o in observations],
            mission=asdict(output),
            navigation_phase=self.navigator.phase,
            fault=self.fault or next(iter(transient), None),
            events=events,
            capture_time_s=capture_s,
            control_time_s=now_s,
        )
        self.last_guard = telemetry["fault"]
        return command, horn, telemetry

    def _complete(self, key, kind, events):
        if key not in self.completed and self.fault is None:
            self.completed.add(key)
            events.append({"event": "mission_completed", "kind": kind, "id": key})
