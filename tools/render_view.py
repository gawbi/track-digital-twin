#!/usr/bin/env python3
"""Render the car camera at given poses, e.g. to compare with real car photos.

Run with Isaac Sim's Python. Writes view-<i>.jpg (and -labels.jpg with renderer
boxes). Camera overrides use the same --set syntax as physics_probe.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtwin.core import ROOT, PathFollower, load_inputs, sample_episode  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/rtx.json")
    p.add_argument("--pose", nargs=3, type=float, action="append", required=True,
                   metavar=("X", "Y", "YAW_DEG"))
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=JSON")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--signal", default="red", choices=["red", "yellow", "green"])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    base, layout = load_inputs(a.config)
    for item in a.set:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        base[section][name] = json.loads(value)
    cfg, params = sample_episode(base, a.seed, a.randomize)
    cfg["simulation"]["drive_model"] = "kinematic_skid"
    a.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    sim = None
    try:
        from PIL import Image, ImageDraw

        from dtwin.runtime import Simulator

        x, y, yaw = a.pose[0]
        start = [x, y], [x + math.cos(math.radians(yaw)), y + math.sin(math.radians(yaw))]
        sim = Simulator(app, cfg, layout, params, PathFollower(list(start)), a.output)
        sim.set_signal(a.signal)
        for i, (x, y, yaw) in enumerate(a.pose):
            sim.skid.pose = (x, y, math.radians(yaw))
            for _ in range(3):  # let the renderer settle on the new pose
                sim.step(0, 0, count=False)
            rgb = sim.image()
            Image.fromarray(rgb).save(a.output / f"view-{i}.jpg", quality=95)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            for box in sim.last_boxes:
                draw.rectangle(box["xyxy"], outline=(255, 0, 255), width=2)
                draw.text((box["xyxy"][0], max(0, box["xyxy"][1] - 11)), box["class"], fill=(255, 0, 255))
            image.save(a.output / f"view-{i}-labels.jpg", quality=90)
        (a.output / "views.json").write_text(json.dumps(dict(poses=a.pose, camera=cfg["camera"]), indent=2))
    finally:
        if sim:
            sim.close()
        app.close()


if __name__ == "__main__":
    main()
