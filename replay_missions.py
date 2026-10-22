#!/usr/bin/env python3
"""Replay recorded RGB/encoders through mission control without Isaac or an RTX GPU.

Command changes do not change the recorded trajectory; this is not a closed-loop
performance evaluation. Use run_sim.py for closed-loop runs on the RTX notebook.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from dtwin.autonomy import MissionDriver
from dtwin.core import load_inputs
from dtwin.perception import OnnxDetector
from dtwin.policy import OnnxPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path, help="A --missions --record output episode")
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--detector", type=Path, required=True)
    p.add_argument(
        "--initial-pose", nargs=3, type=float, required=True, metavar=("X", "Y", "YAW_RAD")
    )
    p.add_argument("--config", type=Path)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg, layout = load_inputs(a.config)
    camera = json.loads(a.calibration.read_text()) if a.calibration else cfg["camera"]
    driver = MissionDriver(cfg, layout, a.initial_pose, OnnxDetector(a.detector), camera)
    policy = OnnxPolicy(a.policy)
    frames = [
        json.loads(line)
        for line in (a.episode / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not frames or any(
        "image" not in row or "autonomy" not in row or "wheel_rad_s" not in row["autonomy"]
        for row in frames
    ):
        p.error(
            "Replay requires recorded image, camera timestamp and measured wheel_rad_s in every frame"
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as file:
        for row in frames:
            rgb = np.asarray(Image.open(a.episode / row["image"]).convert("RGB"))
            sensors = row["autonomy"]
            command, horn, telemetry = driver.step(
                rgb,
                row["time_s"],
                sensors["capture_time_s"],
                *sensors["wheel_rad_s"],
                policy.command(rgb),
            )
            if not all(math.isfinite(v) for v in command):
                raise ValueError("Replay returned an invalid command")
            file.write(
                json.dumps(
                    dict(
                        time_s=row["time_s"],
                        command=list(command),
                        horn=horn,
                        telemetry=telemetry,
                        replay_only=True,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        f"Replayed {len(frames)} frames; commands were not applied to a simulator or real vehicle."
    )


if __name__ == "__main__":
    main()
