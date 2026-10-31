#!/usr/bin/env python3
"""RTX entry point. --check works without Isaac Sim or an NVIDIA GPU."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path

from dtwin.core import ROOT, PathFollower, Progress, load_inputs, randomize_props, sample_episode


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/rtx.json")
    p.add_argument("--check", action="store_true")
    p.add_argument("--mode", choices=["manual", "expert", "onnx", "dagger"], default="manual")
    p.add_argument("--route", choices=["outer", "inner"], default="outer")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--harsh", action="store_true",
                   help="Training randomization: glare, wide brightness, motor mismatch (needs --randomize)")
    p.add_argument("--teacher-speed-scale", type=float, default=1.0,
                   help="Teacher top speed as a fraction of max_speed (real drivers use about half)")
    p.add_argument("--duration", type=float)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--record", action="store_true", help="Save camera images for training")
    p.add_argument(
        "--policy-speed-scale",
        type=float,
        default=1.0,
        help="ONNX mode: scale linear and angular commands together (same path curvature, "
        "more reaction time), like run_vehicle.py --speed-scale",
    )
    p.add_argument(
        "--chase-video",
        action="store_true",
        help="Also record a display-only overhead view (images_chase/) for videos",
    )
    p.add_argument(
        "--offroad-grace", type=float, help="Override simulation.off_track_grace_s (collection)"
    )
    p.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
        help="Expert mode: std (rad/s) of correlated noise added to the APPLIED yaw command; "
        "labels stay the clean teacher command (DART-style recovery data)",
    )
    p.add_argument("--output", type=Path, default=ROOT / "outputs")
    p.add_argument("--policy", type=Path, help="ONNX file with matching .json metadata")
    p.add_argument(
        "--legacy-script",
        type=Path,
        help="External course 8_4.py; never copied into this repository",
    )
    p.add_argument("--legacy-model", type=Path)
    p.add_argument(
        "--missions",
        action="store_true",
        help="Vision detector + wheel encoders + map mission control (ONNX mode)",
    )
    p.add_argument(
        "--vision-missions",
        action="store_true",
        help="Missions WITHOUT wheel encoders: camera-only pink/crosswalk distance and "
        "image standstill (dtwin.vision_missions); junction turns are not handled",
    )
    p.add_argument(
        "--teacher-lane-diagnostic",
        action="store_true",
        help="Missions with the privileged teacher as LANE input (--mode expert). Checks the "
        "sign/signal pipeline separately; never counts as camera autonomy",
    )
    p.add_argument(
        "--detector", type=Path, help="YOLO11 raw ONNX detector with matching .json metadata"
    )
    p.add_argument(
        "--calibration",
        type=Path,
        help="Measured camera JSON; defaults to nominal config, not randomized truth",
    )
    return p


def route_points(layout, name):
    outer = layout["centerlines"]["outer_loop_clockwise"]
    if name == "outer":
        return outer
    # Follow northern centerline to central junction, turn east,
    # join east outer road then complete the outer loop back to north midpoint.
    east = max(p[0] for p in outer)
    idx = min(range(len(outer)), key=lambda i: math.dist(outer[i], (east, 0)))
    north_y = max(p[1] for p in outer)
    # T-junction pivots sit 2.5cm toward the center road: the car's 0.148m turning
    # radius exceeds the 0.13m outer-road half width on the centerline.
    inset = 0.025
    tail = outer[idx + 1 :]
    # Rejoin/leave the outer centerline 0.16m from each T so both turns stay sharp.
    while tail and tail[0][1] > -0.16:
        tail = tail[1:]
    while tail and tail[-1][1] > north_y - 0.05 and tail[-1][0] > -0.16:
        tail = tail[:-1]
    return [[0, north_y - inset], [0, 0], [east - inset, 0]] + tail


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
    if args.vision_missions:
        args.missions = True
    base, layout = load_inputs(args.config)
    if args.check:
        preflight(base, layout)
        return
    if args.headless and args.mode == "manual" and not args.legacy_script:
        raise SystemExit("Headless manual mode has no input: choose --mode expert or --mode onnx")
    if args.mode in {"onnx", "dagger"} and not args.policy:
        raise SystemExit("--mode onnx requires --policy")
    lane_ok = args.mode == "onnx" or (args.mode == "expert" and args.teacher_lane_diagnostic)
    if args.missions and (not lane_ok or not args.detector or args.legacy_script):
        raise SystemExit(
            "--missions requires --mode onnx --policy ... --detector ... (or --mode expert "
            "--teacher-lane-diagnostic) and excludes the legacy bridge"
        )
    if args.missions and "missions" not in base:
        raise SystemExit("Missing missions settings; start with configs/rtx.json")
    if args.missions:
        for path in [args.detector] + ([args.policy] if args.mode == "onnx" else []):
            if not path.is_file() or not path.with_suffix(".json").is_file():
                raise SystemExit(f"Model or export metadata is missing: {path}")
    if args.legacy_script and (not args.legacy_model or args.headless):
        raise SystemExit("Legacy course script requires --legacy-model and GUI (OpenCV windows)")
    if args.legacy_script and not (args.legacy_script.is_file() and args.legacy_model.is_file()):
        raise SystemExit("Legacy script/model file does not exist")
    if args.harsh and (args.stress or not args.randomize):
        raise SystemExit("--harsh is a training randomization: use it with --randomize, not --stress")
    if not 0 < args.teacher_speed_scale <= 1:
        raise SystemExit("--teacher-speed-scale must be in (0, 1]")
    cfg, params = sample_episode(base, args.seed, args.randomize, args.stress, args.harsh)
    if args.randomize or args.stress:
        randomize_props(cfg, layout, args.seed)
    if args.chase_video:
        cfg["simulation"]["overhead_camera"] = True
    if args.offroad_grace is not None:
        cfg["simulation"]["off_track_grace_s"] = args.offroad_grace
    if args.duration is not None:
        if args.duration <= 0:
            raise SystemExit("--duration must be positive")
        cfg["simulation"]["duration_s"] = args.duration
    # An episode directory is never overwritten, including failed runs.
    suffix = "-missions" if args.missions else ""
    output = args.output / f"{args.mode}-{args.route}-seed{args.seed}{suffix}"
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
    sim = recorder = mission_driver = None
    summary = {
        "seed": args.seed,
        "mode": args.mode,
        "route": args.route,
        "status": "initializing",
        "mission_enabled": args.missions,
    }
    try:
        import cv2

        from dtwin.runtime import Recorder, Simulator

        simcfg = cfg["simulation"]
        follower = PathFollower(
            route_points(layout, args.route),
            simcfg["lookahead_m"],
            simcfg["max_speed_m_s"] * args.teacher_speed_scale,
            simcfg["max_yaw_rad_s"],
        )
        summary["teacher_speed_scale"] = args.teacher_speed_scale
        progress = Progress(follower.total_length)
        mission_gates = None
        if args.missions:
            from dtwin.evaluation import GateCounter

            mission_gates = GateCounter(base["missions"]["evaluation_gates"][args.route])
            summary["evaluation_gates"] = base["missions"]["evaluation_gates"][args.route]
        sim = Simulator(app, cfg, layout, params, follower, output)
        sim.save_stage()
        if args.legacy_script:
            from dtwin.legacy import run_book

            run_book(sim, args.legacy_script, args.legacy_model)
            summary["status"] = "legacy_script_finished"
            return
        if args.mode in {"onnx", "dagger"}:
            from dtwin.policy import OnnxPolicy

            policy = OnnxPolicy(args.policy)
        mission_driver = None
        if args.missions:
            from dtwin.autonomy import MissionDriver
            from dtwin.perception import OnnxDetector

            # The initial pose is a configured starting prior. Do not pass sim.pose()
            # or sampled camera randomization into the perception/control pipeline.
            nominal_route = route_points(layout, args.route)
            first, second = nominal_route[:2]
            initial_pose = (*first, math.atan2(second[1] - first[1], second[0] - first[0]))
            calibration = (
                json.loads(args.calibration.read_text()) if args.calibration else base["camera"]
            )
            if args.vision_missions:
                from dtwin.vision_missions import VisionMissionDriver

                mission_driver = VisionMissionDriver(base, OnnxDetector(args.detector), calibration)
                summary["control_sources"] = "RGB only (no encoders, no map pose); junction turns not handled"
            else:
                mission_driver = MissionDriver(
                    base, layout, initial_pose, OnnxDetector(args.detector), calibration
                )
                summary["control_sources"] = (
                    "RGB detector + measured wheel joints + configured map/start; no simulator pose or labels"
                )
            if args.teacher_lane_diagnostic:
                summary["lane_source"] = "privileged teacher (diagnostic only, not autonomy)"
        params["action_noise_rad_s"] = args.action_noise if args.mode == "expert" else 0.0
        # Expert noise and DAgger apply a different command than the recorded label.
        uses_applied = args.mode == "dagger" or (args.mode == "expert" and args.action_noise > 0)
        noise_rng, noise = random.Random(args.seed ^ 0xD4A7), 0.0
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
            reference_laps = progress.update(route_info["route_s_m"])
            laps = mission_gates.update(*pose[:2]) if mission_gates else reference_laps
            key = -1
            if not args.headless:
                cv2.imshow("Digital twin camera (RGB -> BGR)", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:
                    summary["status"] = "user_stopped"
                    break
            if args.mode == "expert":
                command = expert_v, expert_w
                if args.action_noise > 0:
                    # Ornstein-Uhlenbeck (0.5 s correlation) yaw disturbance; the
                    # teacher learns of it through its applied-command record.
                    decay = math.exp(-sim.dt / 0.5)
                    noise = noise * decay + noise_rng.gauss(0, args.action_noise) * math.sqrt(
                        1 - decay**2
                    )
                    applied = expert_v, max(
                        -simcfg["max_yaw_rad_s"], min(simcfg["max_yaw_rad_s"], expert_w + noise)
                    )
                    follower.record_applied(*applied)
            elif args.mode == "onnx":
                linear, angular = policy.command(rgb)
                command = linear * args.policy_speed_scale, angular * args.policy_speed_scale
            elif args.mode == "dagger":
                # The camera policy drives; the teacher labels every frame and takes
                # over only when the car drifts, so data covers the policy's own states.
                takeover = (
                    route_info["route_distance_m"] > 0.04
                    or footprint["min_clearance_m"] < 0.01
                    or abs(route_info["heading_error_rad"]) > 0.6
                )
                applied = (expert_v, expert_w) if takeover else policy.command(rgb)
                follower.record_applied(*applied)
                command = expert_v, expert_w
                route_info["teacher_takeover"] = takeover
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
            mission_info = None
            if mission_driver and args.vision_missions:
                result = mission_driver.step(rgb, sim.frame * sim.dt, sim.last_capture_time_s)
                scale = 0.0 if result.stop else result.speed_scale
                command, horn, mission_info = (command[0] * scale, command[1] * scale), result.horn, result.telemetry
                if horn:
                    sim.horn(base["missions"]["horn_duration_s"])
                if args.mode == "expert":
                    follower.record_applied(*command)
            elif mission_driver:
                left, right = sim.wheel_measurement()
                command, horn, mission_info = mission_driver.step(
                    rgb, sim.frame * sim.dt, sim.last_capture_time_s, left, right, command
                )
                if horn:
                    sim.horn(base["missions"]["horn_duration_s"])
                if args.mode == "expert":
                    follower.record_applied(*command)  # missions may stop/slow the teacher
            record = dict(
                frame=sim.frame,
                time_s=sim.frame * sim.dt,
                pose_xy_yaw=list(pose),
                linear_m_s=command[0],
                angular_rad_s=command[1],
                applied_command=list(applied if uses_applied else command),
                detections=sim.last_boxes,
                **footprint,
                **route_info,
            )
            if mission_info is not None:
                record["autonomy"] = mission_info
            recorder.write(record, rgb, sim.overhead_rgb() if args.chase_video else None)
            summary["laps"] = laps
            if off_frames * sim.dt >= simcfg["off_track_grace_s"]:
                summary["status"] = "offroad"
                break
            sim.set_signal(["red", "green", "yellow"][int(sim.frame * sim.dt // 8) % 3])
            sim.step(*(applied if uses_applied else command))
        if summary["status"] == "running":
            summary["status"] = (
                "duration_completed" if sim.frame >= maximum_frames else "window_closed"
            )
        summary.update(
            frames=sim.frame,
            distance_m=mission_gates.distance if mission_gates else progress.distance,
            min_clearance_m=min(clearances, default=0),
            mean_route_error_m=sum(route_errors) / max(1, len(route_errors)),
            policy_uses_ground_truth=(args.mode == "expert"),
            success=(summary["laps"] >= 1 and summary["offroad_frames"] == 0),
        )
        if mission_driver:
            completed = [e for e in mission_driver.events if e["event"] == "mission_completed"]
            counts = {
                kind: sum(e["kind"] == kind for e in completed)
                for kind in ["horn", "stop", "speed20", "signal"]
            }
            fault = mission_driver.fault if args.vision_missions else mission_driver.last_guard
            phase = "not_handled" if args.vision_missions else mission_driver.navigator.phase
            summary.update(
                lane_success=summary["success"],
                mission_counts=counts,
                mission_fault=fault,
                mission_navigation_phase=phase,
                required_missions=base["missions"]["required_missions"],
                mission_mode="vision_only" if args.vision_missions else "encoder_odometry",
            )
            summary["success"] = (
                summary["lane_success"]
                and not fault
                and phase in {"lane", "not_handled"}
                and all(counts[kind] > 0 for kind in base["missions"]["required_missions"])
            )
            (output / "mission_events.json").write_text(
                json.dumps(mission_driver.events, ensure_ascii=False, indent=2)
            )
    except Exception as error:
        summary.update(status="error", error=repr(error))
        raise
    finally:
        if mission_driver:
            (output / "mission_events.json").write_text(
                json.dumps(mission_driver.events, ensure_ascii=False, indent=2)
            )
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
