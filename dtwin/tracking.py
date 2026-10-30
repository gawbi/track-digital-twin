"""Short-term image tracks plus odometry-based encounter deduplication."""

import math
from dataclasses import dataclass, field

from .perception import iou


def family(kind):
    # One physical signal changes color; one direction sign may be misread between
    # left/right/straight from frame to frame. Both stay one track.
    if kind.startswith("signal_"):
        return "signal"
    return "direction" if kind.startswith("direction_") else kind


@dataclass
class Track:
    id: int
    kind: str
    xyxy: list
    confidence: float
    last_seen: float
    hits: int = 1
    candidate: str = ""
    color_hits: int = 0
    votes: dict = field(default_factory=dict)


class ObjectTracker:
    def __init__(self, min_hits=3, max_age_s=0.7):
        self.min_hits, self.max_age = min_hits, max_age_s
        self.tracks = {}
        self.next_id = 1
        self.time = -math.inf

    def update(self, detections, timestamp):
        if timestamp < self.time:
            raise ValueError("Nonmonotonic detector timestamp")
        if timestamp == self.time:
            return []  # a delayed/repeated camera frame is not additional evidence
        self.time = timestamp
        if any(d.get("source") != "vision" for d in detections):
            raise ValueError("Tracker accepts vision detections only, not renderer labels")
        self.tracks = {
            i: t for i, t in self.tracks.items() if timestamp - t.last_seen <= self.max_age
        }
        pairs = []
        for i, t in self.tracks.items():
            for j, d in enumerate(detections):
                if family(t.kind) != family(d["class"]):
                    continue
                overlap = iou(t.xyxy, d["xyxy"])
                ac = ((t.xyxy[0] + t.xyxy[2]) / 2, (t.xyxy[1] + t.xyxy[3]) / 2)
                bc = ((d["xyxy"][0] + d["xyxy"][2]) / 2, (d["xyxy"][1] + d["xyxy"][3]) / 2)
                span = max(t.xyxy[2] - t.xyxy[0], t.xyxy[3] - t.xyxy[1], 8.0)
                distance = math.dist(ac, bc) / span
                if overlap >= 0.1 or distance <= 0.8:
                    pairs.append((overlap - distance * 0.1, i, j))
        used_tracks, used_detections = set(), set()
        for _, i, j in sorted(pairs, reverse=True):
            if i in used_tracks or j in used_detections:
                continue
            used_tracks.add(i)
            used_detections.add(j)
            t, d = self.tracks[i], detections[j]
            t.hits += 1
            t.xyxy, t.confidence, t.last_seen = list(d["xyxy"]), d["confidence"], timestamp
            if d["class"] == t.candidate:
                t.color_hits += 1
            else:
                t.candidate, t.color_hits = d["class"], 1
            if family(d["class"]) == "direction":
                # The arrow does not change: confidence-weighted vote over the track.
                t.votes[d["class"]] = t.votes.get(d["class"], 0.0) + d["confidence"]
                t.kind = max(t.votes, key=t.votes.get)
            # Red evidence takes effect immediately; releasing it needs two frames.
            elif d["class"] == "signal_red" or t.color_hits >= 2:
                t.kind = d["class"]
        for j, d in enumerate(detections):
            if j not in used_detections:
                i = self.next_id
                self.next_id += 1
                self.tracks[i] = Track(i, d["class"], list(d["xyxy"]), d["confidence"], timestamp)
                self.tracks[i].votes = {d["class"]: d["confidence"]}
        return [
            t for t in self.tracks.values() if t.hits >= self.min_hits and t.last_seen == timestamp
        ]


@dataclass
class Encounter:
    id: str
    family: str
    world: tuple
    last_seen: float
    first_distance: float
    track_ids: set = field(default_factory=set)
    passed_distance: float | None = None


class EncounterMemory:
    def __init__(self, merge_radius_m=0.16, revisit_distance_m=0.8):
        self.merge_radius, self.revisit_distance = merge_radius_m, revisit_distance_m
        self.encounters = []
        self.next_id = 1

    def associate(self, track, world, now_s, traveled_m):
        candidates = [
            e
            for e in self.encounters
            if e.family == family(track.kind)
            and (track.id in e.track_ids or math.dist(e.world, world) <= self.merge_radius)
            and not (
                e.passed_distance is not None
                and traveled_m - e.passed_distance >= self.revisit_distance
            )
        ]
        if candidates:
            encounter = min(candidates, key=lambda e: math.dist(e.world, world))
            encounter.world = tuple(0.8 * a + 0.2 * b for a, b in zip(encounter.world, world))
            encounter.last_seen = now_s
        else:
            encounter = Encounter(
                f"encounter-{self.next_id}", family(track.kind), tuple(world), now_s, traveled_m
            )
            self.next_id += 1
            self.encounters.append(encounter)
        encounter.track_ids.add(track.id)
        return encounter

    def mark_passed(self, pose, traveled_m):
        from .geometry import to_vehicle

        for encounter in self.encounters:
            if encounter.passed_distance is None and to_vehicle(encounter.world, pose)[0] < -0.15:
                encounter.passed_distance = traveled_m
