"""Mission state machine fed by tracked visual observations, not map coordinates.

Distances refer to the front bumper, in meters. Observation IDs identify encounters
and must remain stable across frames; a later lap must use a new encounter ID.
"""

from dataclasses import dataclass, field


@dataclass
class Observation:
    encounter_id: str
    kind: str  # horn / stop / speed20 / signal / direction
    confidence: float = 1.0
    before_pink_line: bool | None = None
    signal_color: str | None = None
    stop_distance_m: float | None = None
    direction: str | None = None  # left / right / straight
    source: str = "vision"
    junction_id: str | None = None


@dataclass
class MissionOutput:
    speed_scale: float = 1.0
    stop: bool = False
    horn: bool = False
    direction: str | None = None
    events: list = field(default_factory=list)
    speed_limit_m_s: float | None = None
    direction_encounter_id: str | None = None
    direction_priority: int = 0


class MissionController:
    def __init__(
        self,
        stop_dwell_s=1.0,
        red_dwell_s=1.0,
        slowdown_scale=0.5,
        stop_tolerance_m=0.035,
        confidence=0.7,
        allow_oracle=False,
        braking_m_s2=0.3,
    ):
        # STOP dwell and reduction ratio are provisional, not confirmed course rules.
        self.stop_dwell_s, self.red_dwell_s = stop_dwell_s, red_dwell_s
        self.slowdown_scale, self.stop_tolerance_m = slowdown_scale, stop_tolerance_m
        self.confidence, self.allow_oracle = confidence, allow_oracle
        self.done = set()
        self.holding = None
        self.started = None
        self.speed_scale = 1.0
        self.braking = braking_m_s2

    def update(self, now_s, speed_m_s, observations, yaw_rate_rad_s=0.0):
        if not self.allow_oracle and any(obs.source != "vision" for obs in observations):
            raise ValueError("Ground-truth mission observations are forbidden in vision evaluation")
        result = MissionOutput(speed_scale=self.speed_scale)
        if self.holding:
            encounter, kind = self.holding
            # The dwell begins only once the vehicle has actually stopped.
            if abs(speed_m_s) > 0.015 or abs(yaw_rate_rad_s) > 0.08:
                self.started = None
            elif self.started is None:
                self.started = now_s
            dwell = self.red_dwell_s if kind == "signal" else self.stop_dwell_s
            if self.started is None or now_s - self.started < dwell:
                result.stop = True
                return result
            self.done.add(encounter)
            self.holding, self.started = None, None
            result.events.append({"event": "stop_completed", "id": encounter, "time_s": now_s})
            if kind == "signal":
                result.direction = "right"
                result.direction_encounter_id = encounter
                result.direction_priority = 2
                return result
        for obs in observations:
            if obs.confidence < self.confidence or obs.encounter_id in self.done:
                continue
            if obs.kind == "direction":
                if obs.direction in {"left", "right", "straight"} and result.direction_priority < 2:
                    result.direction = obs.direction
                    result.direction_encounter_id = obs.encounter_id
                    result.direction_priority = 1
                continue
            if obs.kind == "signal":
                if obs.signal_color == "green":
                    result.direction = "right"
                    result.direction_encounter_id = obs.encounter_id
                    result.direction_priority = 2
                    self.done.add(obs.encounter_id)
                elif (
                    obs.signal_color == "red"
                    and obs.stop_distance_m is not None
                    and 0 <= obs.stop_distance_m <= self.stop_tolerance_m
                ):
                    self.holding = obs.encounter_id, "signal"
                    result.stop = True
                elif (
                    obs.signal_color == "red"
                    and obs.stop_distance_m is not None
                    and obs.stop_distance_m < 0
                ):
                    result.events.append(
                        {"event": "stop_line_already_passed", "id": obs.encounter_id}
                    )
                    result.stop = True
                # Yellow behavior was not specified; report, do not invent a rule.
                elif obs.signal_color == "yellow":
                    result.events.append(
                        {"event": "unconfirmed_yellow_rule", "id": obs.encounter_id}
                    )
                if obs.signal_color == "red" and obs.stop_distance_m is not None:
                    remaining = max(0.0, obs.stop_distance_m - self.stop_tolerance_m)
                    limit = (2 * self.braking * remaining) ** 0.5
                    result.speed_limit_m_s = (
                        min(result.speed_limit_m_s, limit)
                        if result.speed_limit_m_s is not None
                        else limit
                    )
                continue
            if obs.kind not in {"horn", "stop", "speed20"}:
                continue
            if obs.before_pink_line is not True:
                result.events.append(
                    {"event": "pink_deadline_unknown_or_missed", "id": obs.encounter_id}
                )
                continue
            if obs.kind == "horn":
                result.horn = True
                self.done.add(obs.encounter_id)
            elif obs.kind == "speed20":
                # "20" means course slowdown, not 20m/s or 20km/h for this RC car.
                self.speed_scale = self.slowdown_scale
                result.speed_scale = self.speed_scale
                self.done.add(obs.encounter_id)
            else:
                self.holding = obs.encounter_id, "stop"
                result.stop = True
            result.events.append(
                {
                    "event": "mission_started",
                    "kind": obs.kind,
                    "id": obs.encounter_id,
                    "time_s": now_s,
                }
            )
        return result
