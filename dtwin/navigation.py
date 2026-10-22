"""Junction approach, stationary turning and exit driven by wheel odometry."""

from __future__ import annotations

import math

from .core import wrap_angle
from .geometry import nearest_on_segment, to_vehicle


class JunctionNavigator:
    def __init__(self, layout, cfg, initial_pose):
        self.layout, self.cfg = layout, cfg
        self.phase = "lane"
        self.active = None
        self.requests = {}
        self.cleared = {
            name
            for name, p in layout["intersections"].items()
            if math.dist(p, initial_pose[:2]) < 0.12
        }
        self.heading = None
        self.entered = None
        self.fault = None

    def next_junction(self, pose, max_distance=1.2):
        candidates = []
        for name, point in self.layout["intersections"].items():
            if name in self.cleared:
                continue
            x, y = to_vehicle(point, pose)
            if 0 <= x <= max_distance and abs(y) < 0.12:
                candidates.append((x, name))
        return min(candidates)[1] if candidates else None

    def request(self, junction, direction, priority=1):
        if junction is None or direction not in {"left", "right", "straight"}:
            return
        previous = self.requests.get(junction)
        if previous and previous[0] == priority and previous[1] != direction:
            self.fault = "conflicting_direction_requests"
        elif previous is None or priority >= previous[0]:
            self.requests[junction] = priority, direction

    def _allowed(self, point, heading):
        probe = (point[0] + 0.25 * math.cos(heading), point[1] + 0.25 * math.sin(heading))
        distances = []
        for name, points in self.layout["centerlines"].items():
            tail = points[1:] + (points[:1] if name.startswith("outer") else [])
            distances += [
                math.dist(probe, nearest_on_segment(probe, a, b)) for a, b in zip(points, tail)
            ]
        return min(distances) < 0.07

    def command(self, now, pose, speed, base_command, yaw_rate=0.0):
        events = []
        for name in list(self.cleared):
            if (
                math.dist(pose[:2], self.layout["intersections"][name])
                > self.cfg["junction_clear_m"]
            ):
                self.cleared.remove(name)
        if self.fault:
            return (0.0, 0.0), [{"event": self.fault}]
        if self.active is None:
            name = self.next_junction(pose, self.cfg["junction_approach_m"])
            if name is None:
                return base_command, events
            self.active, self.phase, self.heading, self.entered = (
                name,
                "approach",
                round(pose[2] / (math.pi / 2)) * (math.pi / 2),
                now,
            )
            events.append({"event": "junction_approach", "junction": name})
        point = self.layout["intersections"][self.active]
        if now - self.entered > self.cfg["junction_timeout_s"]:
            self.fault = "junction_timeout"
            return (0.0, 0.0), events + [{"event": self.fault, "junction": self.active}]
        direction = self.requests.get(self.active, (0, None))[1]
        longitudinal, lateral = to_vehicle(point, pose)
        if self.phase == "approach":
            if direction is None:
                distance = max(
                    0.0, longitudinal - self.cfg["front_bumper_m"] - self.cfg["junction_wait_m"]
                )
                velocity = min(
                    self.cfg["junction_speed_m_s"],
                    math.sqrt(2 * self.cfg["braking_m_s2"] * distance),
                )
                if distance <= 0.01:
                    return (0.0, 0.0), events + [
                        {"event": "waiting_for_direction", "junction": self.active}
                    ]
                return (velocity, max(-0.7, min(0.7, 3 * lateral))), events
            if math.dist(pose[:2], point) <= self.cfg["junction_center_tolerance_m"]:
                self.target = wrap_angle(
                    self.heading
                    + {"left": math.pi / 2, "right": -math.pi / 2, "straight": 0}[direction]
                )
                if not self._allowed(point, self.target):
                    self.fault = "direction_has_no_road"
                    return (0.0, 0.0), events + [{"event": self.fault, "junction": self.active}]
                self.phase = "settle"
            else:
                if longitudinal < -self.cfg["junction_center_tolerance_m"]:
                    self.fault = "junction_center_overshoot"
                    return (0.0, 0.0), events + [{"event": self.fault, "junction": self.active}]
                bearing = math.atan2(lateral, max(longitudinal, 0.02))
                return (
                    min(self.cfg["junction_speed_m_s"], max(0.025, longitudinal)),
                    max(-0.7, min(0.7, 2 * bearing)),
                ), events
        if self.phase == "settle":
            if abs(speed) > self.cfg["stopped_speed_m_s"] or abs(yaw_rate) > 0.08:
                return (0.0, 0.0), events
            self.phase = "turn"
            events.append(
                {"event": "junction_turn", "junction": self.active, "direction": direction}
            )
        if self.phase == "turn":
            error = wrap_angle(self.target - pose[2])
            if abs(error) > self.cfg["turn_tolerance_rad"]:
                return (
                    0.0,
                    max(-self.cfg["turn_yaw_rad_s"], min(self.cfg["turn_yaw_rad_s"], 3 * error)),
                ), events
            self.phase = "exit"
        if self.phase == "exit":
            dx, dy = pose[0] - point[0], pose[1] - point[1]
            along = dx * math.cos(self.target) + dy * math.sin(self.target)
            if along >= self.cfg["junction_exit_m"]:
                name = self.active
                self.cleared.add(name)
                self.requests.pop(name, None)
                self.active, self.phase = None, "lane"
                return base_command, events + [{"event": "junction_completed", "junction": name}]
            return (
                self.cfg["junction_speed_m_s"],
                max(-0.6, min(0.6, 3 * wrap_angle(self.target - pose[2]))),
            ), events
        return (0.0, 0.0), events
