#!/usr/bin/env python3
"""Fixed PWM steps for measuring the real car's speed and skid turning (Raspberry Pi).

Measure with a tape mark and tape measure (straight: start line to the front bumper;
spins: count whole turns, longer with --spin-seconds 6), or film from above with a phone. Every step is logged with its
monotonic start/stop time; the measured distance/angle from the video is entered
afterwards with --measured to fit duty->speed gain and skid yaw efficiency.
Motors stop between steps, on Ctrl+C and on any error. Start with --motors fake.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEPS = [
    # name, left duty, right duty, seconds
    ("straight_0.3", 0.3, 0.3, 3.0),
    ("straight_0.5", 0.5, 0.5, 2.0),
    ("spin_left_0.5", -0.5, 0.5, 2.0),
    ("spin_right_0.5", 0.5, -0.5, 2.0),
    ("arc_left_0.25_0.5", 0.25, 0.5, 3.0),
    ("arc_right_0.5_0.25", 0.5, 0.25, 3.0),
]


def fit(log, measured, vehicle):
    """measured: {step: {"distance_m": d} or {"turn_deg": a}} from the overhead video."""
    rows = []
    for step in log["steps"]:
        m = measured.get(step["name"])
        if not m:
            continue
        seconds = (step["stop_ns"] - step["start_ns"]) / 1e9
        mean_duty = (step["left"] + step["right"]) / 2
        diff_duty = (step["right"] - step["left"]) / 2
        row = dict(step=step["name"], seconds=round(seconds, 3))
        if "distance_m" in m and abs(mean_duty) > 1e-6:
            row["speed_m_s"] = m["distance_m"] / seconds
            row["speed_per_duty_m_s"] = row["speed_m_s"] / mean_duty
        if "turn_deg" in m and abs(diff_duty) > 1e-6:
            row["yaw_rate_rad_s"] = math.radians(m["turn_deg"]) / seconds
            row["yaw_per_diff_duty"] = row["yaw_rate_rad_s"] / diff_duty
        rows.append(row)
    gains = [r["speed_per_duty_m_s"] for r in rows if "speed_per_duty_m_s" in r and "straight" in r["step"]]
    result = dict(rows=rows, note="Includes motor start-up; use long steps and repeat 3 times")
    if gains:
        gain = sum(gains) / len(gains)
        result["speed_per_duty_m_s"] = gain
        yaw = [r["yaw_per_diff_duty"] for r in rows if "yaw_per_diff_duty" in r]
        if yaw:
            ideal = 2 * gain / vehicle["track_width_m"]  # yaw/diff duty if wheels did not scrub
            result["skid_yaw_efficiency"] = sum(yaw) / len(yaw) / ideal
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="Log JSON (not overwritten)")
    p.add_argument("--motors", choices=["gpio", "fake"], default="fake")
    p.add_argument("--pause", type=float, default=3.0, help="Stopped seconds between steps")
    p.add_argument("--steps", help="Comma separated subset of step names")
    p.add_argument(
        "--spin-seconds",
        type=float,
        help="Longer in-place spins so whole turns can be counted by eye (no overhead video)",
    )
    p.add_argument("--measured", type=Path, help="Fit an existing log with video measurements")
    a = p.parse_args()
    if a.measured:
        from dtwin.core import load_inputs

        log = json.loads(a.output.read_text())
        print(json.dumps(fit(log, json.loads(a.measured.read_text()), load_inputs()[0]["vehicle"]), indent=2))
        return
    from dtwin.realtime import monotonic_ns
    from dtwin.vehicle_io import FakeMotors, GpioMotors, system_snapshot

    motors = GpioMotors() if a.motors == "gpio" else FakeMotors()
    chosen = [s for s in STEPS if not a.steps or s[0] in a.steps.split(",")]
    if a.spin_seconds:
        chosen = [(n, lt, rt, a.spin_seconds if n.startswith("spin") else t) for n, lt, rt, t in chosen]
    log = dict(system=system_snapshot(), motors=motors.description, steps=[],
               warning="Duty is a command; distances/angles must come from video or tape")
    try:
        with a.output.open("x") as file:
            for name, left, right, seconds in chosen:
                input(f"[{name}] place the car on the start mark, start filming, press Enter...")
                start = monotonic_ns()
                motors.set(left, right)
                time.sleep(seconds)
                motors.stop()
                stop = monotonic_ns()
                log["steps"].append(dict(name=name, left=left, right=right, start_ns=start, stop_ns=stop))
                print(f"  ran {name} for {(stop - start) / 1e9:.3f}s; measure distance/angle in the video")
                time.sleep(a.pause)
            json.dump(log, file, indent=2)
    finally:
        motors.stop()
    print(f"Saved {a.output}. Then: --measured measured.json (e.g. "
          '{"straight_0.3": {"distance_m": 0.42}, "spin_left_0.5": {"turn_deg": 210}})')


if __name__ == "__main__":
    main()
