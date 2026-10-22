#!/usr/bin/env python3
"""RTX entry point. --check works without Isaac Sim or an NVIDIA GPU."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from dtwin.core import ROOT, PathFollower, Progress, load_inputs, randomize_props, sample_episode


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/rtx.json")
    p.add_argument("--check", action="store_true")
    p.add_argument("--mode", choices=["manual", "expert", "onnx"], default="manual")
    p.add_argument("--route", choices=["outer", "inner"], default="outer")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--duration", type=float)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--record", action="store_true", help="Save camera images for training")
    p.add_argument("--output", type=Path, default=ROOT / "outputs")
    p.add_argument("--policy", type=Path, help="ONNX file with matching .json metadata")
    p.add_argument(
        "--legacy-script",
        type=Path,
        help="External course 8_4.py; never copied into this repository",
    )
    p.add_argument("--legacy-model", type=Path)
    return p


def route_points(layout, name):
    outer = layout["centerlines"]["outer_loop_clockwise"]
    if name == "outer":
        return outer
    # Follow northern centerline to central junction, turn right/east,
    # join east outer road then complete the outer loop back to north midpoint.
    east = max(p[0] for p in outer)
    idx = min(range(len(outer)), key=lambda i: math.dist(outer[i], (east, 0)))
    north_y = max(p[1] for p in outer)
    return [[0, north_y], [0, 0], [east, 0]] + outer[idx + 1 :]


def preflight(cfg, layout):
    import importlib.util

    report = dict(
        board_m=layout["meta"]["board_size"],
        camera=cfg["camera"],
        vehicle=cfg["vehicle"],
        texture_json_consistent=True,
        narrow_road_side_clearance_m=(
            layout["dims"]["center_road_ew_width"]
            - cfg["vehicle"]["track_width_m"]
            - cfg["vehicle"]["wheel_width_m"]
        )
        / 2,
        estimated_parameters=cfg["provenance"],
        isaacsim_installed=bool(importlib.util.find_spec("isaacsim")),
    )
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        report["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    except FileNotFoundError:
        report["nvidia_smi"] = "unavailable on this machine; run --check again on the RTX notebook"
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    args = parser().parse_args()
    base, layout = load_inputs(args.config)
    if args.check:
        preflight(base, layout)
        return
    if args.headless and args.mode == "manual" and not args.legacy_script:
        raise SystemExit("Headless manual mode has no input: choose --mode expert or --mode onnx")
    if args.mode == "onnx" and not args.policy:
        raise SystemExit("--mode onnx requires --policy")
    if args.legacy_script and (not args.legacy_model or args.headless):
        raise SystemExit("Legacy course script requires --legacy-model and GUI (OpenCV windows)")
    if args.legacy_script and not (args.legacy_script.is_file() and args.legacy_model.is_file()):
        raise SystemExit("Legacy script/model file does not exist")
    cfg, params = sample_episode(base, args.seed, args.randomize, args.stress)
    if args.randomize or args.stress:
        randomize_props(cfg, layout, args.seed)
    if args.duration is not None:
        if args.duration <= 0:
            raise SystemExit("--duration must be positive")
        cfg["simulation"]["duration_s"] = args.duration
    # An episode directory is never overwritten, including failed runs.
    output = args.output / f"{args.mode}-{args.route}-seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": "RaytracedLighting",
            "width": cfg["simulation"]["render_width"],
            "height": cfg["simulation"]["render_height"],
        }
    )
    sim = recorder = None
    summary = {"seed": args.seed, "mode": args.mode, "route": args.route, "status": "initializing"}
    try:
        import cv2

        from dtwin.runtime import Recorder, Simulator

        simcfg = cfg["simulation"]
        follower = PathFollower(
            route_points(layout, args.route),
            simcfg["lookahead_m"],
            simcfg["max_speed_m_s"],
            simcfg["max_yaw_rad_s"],
        )
        progress = Progress(follower.total_length)
        sim = Simulator(app, cfg, layout, params, follower, output)
        sim.save_stage()
        if args.legacy_script:
            from dtwin.legacy import run_book

            run_book(sim, args.legacy_script, args.legacy_model)
            summary["status"] = "legacy_script_finished"
            return
        if args.mode == "onnx":
            from dtwin.policy import OnnxPolicy

            policy = OnnxPolicy(args.policy)
        recorder = Recorder(output, cfg, params, args.mode, args.route, args.record)
        command = (0.0, 0.0)
        off_frames = 0
        maximum_frames = round(simcfg["duration_s"] / sim.dt)
        clearances, route_errors = [], []
        summary.update(status="running", laps=0, offroad_frames=0)
        print(
            "Controls in camera window: W forward, S reverse, A/D turn, SPACE stop, Q quit. Commands time out after 0.3s."
        )
        while app.is_running() and sim.frame < maximum_frames:
            rgb = sim.image()
            pose = sim.pose()
            expert_v, expert_w, route_info = follower.command(*pose)
            footprint = sim.footprint()
            clearances.append(footprint["min_clearance_m"])
            route_errors.append(route_info["route_distance_m"])
            laps = progress.update(route_info["route_s_m"])
            key = -1
            if not args.headless:
                cv2.imshow("Digital twin camera (RGB -> BGR)", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:
                    summary["status"] = "user_stopped"
                    break
            if args.mode == "expert":
                command = expert_v, expert_w
            elif args.mode == "onnx":
                command = policy.command(rgb)
            else:
                import time

                if key in [ord("w"), ord("s"), ord("a"), ord("d"), 32]:
                    commands = {
                        ord("w"): (simcfg["max_speed_m_s"], 0),
                        ord("s"): (-simcfg["max_speed_m_s"], 0),
                        ord("a"): (0, simcfg["max_yaw_rad_s"]),
                        ord("d"): (0, -simcfg["max_yaw_rad_s"]),
                        32: (0, 0),
                    }
                    command = commands[key]
                    last_key_time = time.monotonic()
                if "last_key_time" not in locals() or time.monotonic() - last_key_time > 0.3:
                    command = (0.0, 0.0)
            if not footprint["footprint_on_road"]:
                off_frames += 1
                summary["offroad_frames"] += 1
            else:
                off_frames = 0
            record = dict(
                frame=sim.frame,
                time_s=sim.frame * sim.dt,
                pose_xy_yaw=list(pose),
                linear_m_s=command[0],
                angular_rad_s=command[1],
                detections=sim.last_boxes,
                **footprint,
                **route_info,
            )
            recorder.write(record, rgb)
            summary["laps"] = laps
            if off_frames * sim.dt >= simcfg["off_track_grace_s"]:
                summary["status"] = "offroad"
                break
            sim.set_signal(["red", "green", "yellow"][int(sim.frame * sim.dt // 8) % 3])
            sim.step(*command)
        if summary["status"] == "running":
            summary["status"] = (
                "duration_completed" if sim.frame >= maximum_frames else "window_closed"
            )
        summary.update(
            frames=sim.frame,
            distance_m=progress.distance,
            min_clearance_m=min(clearances, default=0),
            mean_route_error_m=sum(route_errors) / max(1, len(route_errors)),
            policy_uses_ground_truth=(args.mode == "expert"),
            success=(summary["laps"] >= 1 and summary["offroad_frames"] == 0),
        )
    except Exception as error:
        summary.update(status="error", error=repr(error))
        raise
    finally:
        if recorder:
            recorder.close(summary)
        else:
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        if sim:
            sim.close()
        app.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
