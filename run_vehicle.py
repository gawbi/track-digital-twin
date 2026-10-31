#!/usr/bin/env python3
"""Run the lane policy on the Raspberry Pi with separate perception and motor loops.

Writes <output>/env.json, telemetry.jsonl and summary.json (latency/period stats,
stop reasons). Start with --motors fake and the wheels off the ground.

Without wheel encoders the encoder-based MissionDriver (--missions) is refused because
PWM must not stand in for measured wheel speed. --vision-missions runs camera-only
missions (pink/crosswalk range from the calibrated camera, image standstill) in the
detector thread; the lane loop stops if that result is older than 0.6 s. Without it
the detector is observation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import threading
import time
from pathlib import Path

from collect_real import open_camera, open_motors
from dtwin.actions import DutyMixer
from dtwin.latency import summarize
from dtwin.policy import OnnxPolicy
from dtwin.realtime import Telemetry, VehicleRuntime
from dtwin.vehicle_io import system_snapshot


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DetectorWorker:
    """Sign detector on its own thread so a slow detector never delays lane control.

    It reads the newest camera frame (without consuming it) and keeps the latest
    detections with their capture time. Observation only: nothing here drives motors.
    """

    def __init__(self, detector, frames, telemetry, clock, missions=None, buzzer=None, horn_s=0.2):
        self.detector, self.frames, self.telemetry, self.clock = detector, frames, telemetry, clock
        self.missions, self.buzzer, self.horn_s = missions, buzzer, horn_s
        self.latest, self.mission, self.stop, self.error = None, None, threading.Event(), None
        self.thread = threading.Thread(target=self._loop, name="detector", daemon=True)

    def _loop(self):
        version = 0
        try:
            while not self.stop.is_set():
                current, frame = self.frames.peek()
                if frame is None or current == version:
                    time.sleep(0.003)
                    continue
                version = current
                started = self.clock()
                if self.missions is not None:
                    # Vision-only missions (no encoders): detection + pink/crosswalk range +
                    # image standstill decide stop/slow/horn for the lane loop.
                    result = self.missions.step(frame.rgb, started / 1e9, frame.capture_ns / 1e9)
                    detections = result.telemetry.get("detections", [])
                    self.mission = dict(
                        capture_ns=frame.capture_ns, stop=result.stop, speed_scale=result.speed_scale
                    )
                    if result.horn and self.buzzer is not None:
                        self.buzzer.on()
                        threading.Timer(self.horn_s, self.buzzer.off).start()
                    if result.events:
                        self.telemetry.write("mission_events", capture_ns=frame.capture_ns, events=result.events)
                else:
                    detections = self.detector.detect(frame.rgb)
                finished = self.clock()
                self.latest = dict(frame_id=frame.frame_id, capture_ns=frame.capture_ns, detections=detections)
                self.telemetry.write(
                    "detection",
                    frame_id=frame.frame_id,
                    capture_ns=frame.capture_ns,
                    started_ns=started,
                    finished_ns=finished,
                    detections=detections,
                )
        except Exception as error:  # noqa: BLE001 - lane loop sees the stale mission and stops
            self.error = repr(error)
            self.telemetry.write("detector_error", error=self.error)


def make_perceiver(policy, worker=None, mission_max_age_s=0.6):
    def perceive(frame):
        stages = {}
        action = policy.action(frame.rgb, stages)
        info = {}
        if worker is not None and worker.missions is not None:
            mission = worker.mission
            age = None if mission is None else (frame.capture_ns - mission["capture_ns"]) / 1e9
            if mission is None or age > mission_max_age_s or worker.error:
                action, info["mission_gate"] = (0.0, 0.0), "stale_or_missing_mission_result"
            elif mission["stop"]:
                action, info["mission_gate"] = (0.0, 0.0), "mission_stop"
            else:
                scale = mission["speed_scale"]
                action = (action[0] * scale, action[1] * scale)
                info["mission_gate"] = f"scale {scale}"
        if worker is not None and worker.latest is not None:
            latest = worker.latest
            info.update(
                detector_role="vision_missions" if worker.missions is not None else "observation_only",
                detection_frame_id=latest["frame_id"],
                detection_age_ms=round((frame.capture_ns - latest["capture_ns"]) / 1e6, 1),
                detections=[d["class"] for d in latest["detections"]],
            )
        return {"action": action, "stages": stages, "info": info}

    return perceive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--detector", type=Path, help="Logged only (no encoders)")
    p.add_argument("--missions", action="store_true", help="Encoder-based missions; refused (no encoders)")
    p.add_argument(
        "--vision-missions",
        action="store_true",
        help="Camera-only missions (needs --detector): horn/STOP/speed20/red signal; no junction turns",
    )
    p.add_argument("--buzzer-pin", type=int, default=12, help="Horn buzzer GPIO (board label GP12)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration", type=Path, help="Duty mixer JSON (default simulator ratio [추정])")
    p.add_argument("--camera", default="picamera2")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--flip180", action="store_true")
    p.add_argument("--motors", choices=["gpio", "fake"], default="fake")
    p.add_argument("--control-hz", type=float, default=50)
    p.add_argument("--max-result-age", type=float, default=0.25, help="Seconds since CAPTURE")
    p.add_argument("--max-camera-gap", type=float, default=0.3)
    p.add_argument("--max-inference", type=float, default=0.5)
    p.add_argument("--speed-scale", type=float, default=0.6)
    p.add_argument("--threads", type=int, default=2, help="ONNX Runtime intra-op threads")
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--warmup", type=float, default=3.0, help="Excluded from latency statistics")
    a = p.parse_args()
    if a.missions:
        p.error(
            "Encoder-based mission control needs measured wheel speed and this car has no "
            "encoders. Use --vision-missions (camera-only) with --detector."
        )
    if a.vision_missions and not a.detector:
        p.error("--vision-missions needs --detector")
    a.output.mkdir(parents=True, exist_ok=False)
    mixer = DutyMixer.load(a.calibration) if a.calibration else None
    if mixer is None:
        from dtwin.core import load_inputs

        mixer = DutyMixer.from_simulator(load_inputs()[0])
    policy = OnnxPolicy(a.policy, threads=a.threads)
    detector = None
    if a.detector:
        from dtwin.perception import OnnxDetector

        detector = OnnxDetector(a.detector, threads=a.threads)
    # A folder source loops so desk tests measure steady-state timing.
    camera = open_camera(a.camera, a.width, a.height, a.fps, a.flip180, loop=True)
    motors = open_motors(a.motors)
    env = dict(
        system_start=system_snapshot(),
        policy=dict(path=str(a.policy), sha256=sha256(a.policy), metadata=policy.metadata),
        detector=None
        if not a.detector
        else dict(path=str(a.detector), sha256=sha256(a.detector), thread="separate"),
        camera=getattr(camera, "description", {}),
        motors=getattr(motors, "description", {}),
        mixer=mixer.to_json(),
        settings={k: v for k, v in vars(a).items() if not isinstance(v, Path)},
        speed_note="Duty commands are not measured speed; real motion needs video/encoders",
    )
    (a.output / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2, default=str))
    telemetry = Telemetry(a.output / "telemetry.jsonl")
    runtime = VehicleRuntime(
        camera,
        None,
        motors,
        mixer,
        control_hz=a.control_hz,
        max_result_age_s=a.max_result_age,
        max_camera_gap_s=a.max_camera_gap,
        max_inference_s=a.max_inference,
        speed_scale=a.speed_scale,
        telemetry=telemetry,
    )
    worker = None
    if detector is not None:
        missions = buzzer = None
        horn_s = 0.2
        if a.vision_missions:
            from dtwin.core import load_inputs
            from dtwin.vision_missions import VisionMissionDriver

            cfg = load_inputs()[0]
            horn_s = cfg["missions"]["horn_duration_s"]
            missions = VisionMissionDriver(cfg, detector)
            if a.motors == "gpio":
                from gpiozero import DigitalOutputDevice

                buzzer = DigitalOutputDevice(a.buzzer_pin, initial_value=False)
        worker = DetectorWorker(detector, runtime.frames, telemetry, runtime.clock, missions, buzzer, horn_s)
        worker.thread.start()
    runtime.perceive = make_perceiver(policy, worker)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        result = runtime.run(duration_s=a.duration, stop_event=stop)
    except KeyboardInterrupt:
        result = dict(counts=runtime.counts, stop_reasons=runtime.stop_reasons, fault="user_interrupt")
    finally:
        motors.stop()
        if worker is not None:
            worker.stop.set()
            worker.thread.join(timeout=2)
            if worker.buzzer is not None:
                worker.buzzer.off()
                worker.buzzer.close()
        telemetry.close()
        if hasattr(motors, "close"):
            motors.close()
    env["system_end"] = system_snapshot()
    (a.output / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2, default=str))
    summary = dict(run=result, latency=summarize(a.output / "telemetry.jsonl", a.warmup, a.control_hz))
    (a.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
