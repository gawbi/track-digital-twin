#!/usr/bin/env python3
"""Measure wheel contact, drive response and skid-steer turning in Isaac Sim.

Run with Isaac Sim 5.1's Python before collecting data. Simulator pose is used
for this DIAGNOSIS only; it is never a control input. Writes probe.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtwin.core import ROOT, PathFollower, load_inputs, sample_episode, wheel_speeds  # noqa: E402

PHASES = [
    ("settle", 0.0, 0.0, 1.0),
    ("forward", 0.15, 0.0, 3.0),
    ("brake", 0.0, 0.0, 1.0),
    ("spin_left", 0.0, 1.5, 2.0),
    ("spin_right", 0.0, -1.5, 2.0),
    ("arc_left", 0.12, 0.8, 3.0),
    ("reverse", -0.1, 0.0, 2.0),
]


def tilt_deg(stage, path):
    from pxr import UsdGeom

    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    up = matrix.TransformDir((0, 0, 1))
    return math.degrees(math.acos(max(-1.0, min(1.0, up[2] / max(up.GetLength(), 1e-9)))))


def height(stage, path):
    from pxr import UsdGeom

    return float(
        UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()[2]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/rtx.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=JSON",
        help="Diagnostic override, e.g. vehicle.wheel_collision='\"sphere\"'",
    )
    p.add_argument(
        "--usd",
        action="append",
        default=[],
        metavar="PRIM:ATTR=JSON",
        help="Diagnostic USD attribute, e.g. /World/Physics:physxScene:frictionType='\"patch\"'",
    )
    a = p.parse_args()
    base, layout = load_inputs(a.config)
    for item in a.set:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        base[section][name] = json.loads(value)
    cfg, params = sample_episode(base, a.seed)
    a.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    sim = None
    report = {
        "config": str(a.config),
        "overrides": a.set,
        "vehicle": cfg["vehicle"],
        "simulation": cfg["simulation"],
        "phases": [],
    }
    try:
        import dtwin.scene as scene_module
        from dtwin.runtime import Simulator

        report["usd_overrides"] = a.usd
        if a.usd:
            original = scene_module.build_scene

            def build_with_overrides(stage, *args, **kwargs):
                from pxr import PhysxSchema, Sdf

                result = original(stage, *args, **kwargs)
                types = {str: Sdf.ValueTypeNames.Token, bool: Sdf.ValueTypeNames.Bool}
                types |= {int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
                for item in a.usd:
                    target, value = item.split("=", 1)
                    path, attr = target.split(":", 1)
                    prim, value = stage.GetPrimAtPath(path), json.loads(value)
                    if attr.startswith("physxArticulation:"):
                        PhysxSchema.PhysxArticulationAPI.Apply(prim)
                    elif attr.startswith("physxScene:"):
                        PhysxSchema.PhysxSceneAPI.Apply(prim)
                    # Applied schema attributes already carry their declared type.
                    attribute = prim.GetAttribute(attr)
                    if not attribute:
                        attribute = prim.CreateAttribute(attr, types[type(value)])
                    attribute.Set(value)
                return result

            scene_module.build_scene = build_with_overrides
        # Spawn at the west end of the long straight north road, heading east.
        route = PathFollower([[-0.6, 0.8725], [0.6, 0.8725], [0.6, -0.8725], [-0.6, -0.8725]])
        sim = Simulator(app, cfg, layout, params, route, a.output)
        wheels = [f"/World/Car/{s}_{x}" for s in ["left", "right"] for x in ["front", "rear"]]
        r = cfg["vehicle"]["wheel_radius_m"]
        for name, linear, angular, duration in PHASES:
            target_left, target_right = wheel_speeds(linear, angular, cfg["vehicle"])
            rows = []
            for _ in range(round(duration / sim.dt)):
                x0, y0, yaw0 = sim.pose()
                sim.step(linear, angular)
                x1, y1, yaw1 = sim.pose()
                left, right = sim.wheel_measurement()
                forward = (x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0)
                rows.append(
                    dict(
                        v=forward / sim.dt,
                        w=math.remainder(yaw1 - yaw0, 2 * math.pi) / sim.dt,
                        left=left,
                        right=right,
                        z=height(sim.stage, sim.objects["body_path"]),
                        tilt=tilt_deg(sim.stage, sim.objects["body_path"]),
                        wheel_z=[height(sim.stage, w) for w in wheels],
                    )
                )
            # Ignore the first 0.5 s of each phase as the drive transient.
            steady = rows[round(0.5 / sim.dt) :] or rows

            def stats(key, values=None):
                values = values or [row[key] for row in steady]
                return dict(
                    mean=statistics.fmean(values),
                    std=statistics.pstdev(values),
                    min=min(values),
                    max=max(values),
                )

            report["phases"].append(
                dict(
                    phase=name,
                    command=dict(linear_m_s=linear, angular_rad_s=angular),
                    target_wheel_rad_s=dict(left=target_left, right=target_right),
                    measured_linear_m_s=stats("v"),
                    measured_yaw_rate_rad_s=stats("w"),
                    measured_left_rad_s=stats("left"),
                    measured_right_rad_s=stats("right"),
                    chassis_z_m=stats("z"),
                    chassis_tilt_deg=stats("tilt"),
                    max_wheel_lift_m=max(
                        max(abs(z - r) for z in row["wheel_z"]) for row in steady
                    ),
                )
            )
            print(json.dumps(report["phases"][-1]), flush=True)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        (a.output / "probe.json").write_text(json.dumps(report, indent=2))
        if sim:
            sim.close()
        app.close()


if __name__ == "__main__":
    main()
