"""One action contract shared by simulator labels, training and the real car.

The lane policy outputs ``normalized_linear_angular_v1``: two values in [-1, 1],
(linear / max_speed, angular / max_yaw). The simulator scales them to m/s and
rad/s with its config. The real car mixes them into signed left/right PWM duty
with a calibration file. Duty is a COMMAND, never a measured wheel speed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

ACTION_SPACE = "normalized_linear_angular_v1"


def clip_unit(value):
    if not math.isfinite(value):
        raise ValueError("Action values must be finite")
    return max(-1.0, min(1.0, float(value)))


def normalize_command(linear_m_s, angular_rad_s, max_speed_m_s, max_yaw_rad_s):
    return clip_unit(linear_m_s / max_speed_m_s), clip_unit(angular_rad_s / max_yaw_rad_s)


def denormalize_action(action, max_speed_m_s, max_yaw_rad_s):
    return clip_unit(action[0]) * max_speed_m_s, clip_unit(action[1]) * max_yaw_rad_s


@dataclass(frozen=True)
class DutyMixer:
    """Normalized action -> signed motor duty in [-1, 1] for each side.

    left = gain * (linear - turn_mix * angular) * trim_left, right likewise with +.
    ``turn_mix`` makes the left/right ratio match the simulator kinematics;
    gain/trim/deadband must come from real measurements before they are trusted.
    """

    max_duty: float = 0.5
    turn_mix: float = 0.72
    trim_left: float = 1.0
    trim_right: float = 1.0
    deadband: float = 0.0
    # Smallest duty that actually moves the car on the mat (static friction). Nonzero
    # commands map to [min_duty, max_duty] so slow commands do not stall the motors.
    min_duty: float = 0.0
    provenance: str = "[추정] simulator ratio; measure on the real car"

    def __post_init__(self):
        if not 0 < self.max_duty <= 1 or self.turn_mix <= 0 or not 0 <= self.deadband < 1:
            raise ValueError("Invalid duty mixer calibration")
        if not 0 <= self.min_duty < self.max_duty:
            raise ValueError("min_duty must be in [0, max_duty)")
        if not (0 < self.trim_left <= 1.5 and 0 < self.trim_right <= 1.5):
            raise ValueError("Motor trims must be in (0, 1.5]")

    @classmethod
    def from_simulator(cls, cfg, max_duty=0.5):
        """Ratio that reproduces the simulated wheel speed ratio, still [추정]."""
        s, v = cfg["simulation"], cfg["vehicle"]
        mix = s["max_yaw_rad_s"] * v["track_width_m"] / 2 / s["max_speed_m_s"]
        return cls(max_duty=max_duty, turn_mix=mix)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("action_space") != ACTION_SPACE:
            raise ValueError("Calibration is not for " + ACTION_SPACE)
        return cls(**data["duty_mixer"])

    def to_json(self):
        return {"action_space": ACTION_SPACE, "duty_mixer": asdict(self)}

    def _side(self, value, trim):
        value = max(-1.0, min(1.0, value))
        if abs(value) <= self.deadband:
            return 0.0
        magnitude = self.min_duty + (self.max_duty - self.min_duty) * abs(value)
        return max(-1.0, min(1.0, math.copysign(magnitude, value) * trim))

    def _unside(self, duty, trim):
        duty = duty / trim
        if duty == 0:
            return 0.0
        return math.copysign((abs(duty) - self.min_duty) / (self.max_duty - self.min_duty), duty)

    def duty(self, action):
        linear, angular = clip_unit(action[0]), clip_unit(action[1])
        left = linear - self.turn_mix * angular
        right = linear + self.turn_mix * angular
        # Preserve the left/right ratio on saturation, like core.wheel_speeds.
        scale = max(1.0, abs(left), abs(right))
        return self._side(left / scale, self.trim_left), self._side(right / scale, self.trim_right)

    def action(self, left_duty, right_duty):
        """Inverse mixing for recorded duty commands (teleop or legacy GPIO).

        Exact inverse only without saturation/deadband; the result is a command
        label in the shared space, not a measurement of how the car moved.
        """
        left = self._unside(left_duty, self.trim_left)
        right = self._unside(right_duty, self.trim_right)
        return clip_unit((left + right) / 2), clip_unit((right - left) / (2 * self.turn_mix))
