#!/usr/bin/env python3
"""Record a real session on the Raspberry Pi: RGB frames + sent commands + times.

Keyboard teleop over SSH: arrow keys (up forward, left/right forward arc, down back,
space stop) or w/s, q/e arcs, a/d turn in place, x stop; p quits. A command
expires after --command-timeout and the motors stop. Labels are the COMMANDS that
were sent (normalized action and PWM duty), never measured speed; encoder data is
written only if a real encoder adapter exists (not implemented: --encoder none).

Example (Pi):  python3 collect_real.py --output real/sessions --session-id 2026-11-03-310-a \
    --location 310호 --lighting "형광등 오후" --calibration configs/real_car.json
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import threading
import time
from pathlib import Path

from dtwin.actions import DutyMixer
from dtwin.real_data import SessionWriter
from dtwin.realtime import monotonic_ns

# (linear, angular) multipliers; q/e arcs give smoother steering labels than spin-only keys.
KEYS = {
    "w": (1.0, 0.0),
    "s": (-1.0, 0.0),
    "a": (0.0, 1.0),
    "d": (0.0, -1.0),
    "q": (1.0, 0.6),
    "e": (1.0, -0.6),
    "x": (0.0, 0.0),
    " ": (0.0, 0.0),
}
# Arrow keys arrive as ESC [ A/B/C/D: up = forward, left/right = forward arcs, down = back.
ARROWS = {"A": "w", "B": "s", "C": "e", "D": "q"}


def open_camera(spec, width, height, fps, flip180, loop=False):
    from dtwin import vehicle_io

    if spec == "picamera2":
        return vehicle_io.PicameraSource(width, height, fps, flip180)
    if spec.startswith("opencv:"):
        return vehicle_io.OpenCVSource(int(spec.split(":", 1)[1]), width, height, flip180)
    if spec.startswith("folder:"):
        return vehicle_io.ImageFolderSource(spec.split(":", 1)[1], fps, loop=loop)
    raise ValueError("camera must be picamera2, opencv:N or folder:DIR")


def open_motors(spec):
    from dtwin import vehicle_io

    return vehicle_io.GpioMotors() if spec == "gpio" else vehicle_io.FakeMotors()


class Teleop:
    """Latest key command with its time; read from a raw terminal on its own thread."""

    def __init__(self, linear, angular):
        self.scale = (linear, angular)
        self.action, self.time_ns, self.quit = (0.0, 0.0), None, threading.Event()

    def press(self, key):
        if key == "p":
            self.quit.set()
        elif key in KEYS:
            base = KEYS[key]
            self.action = (base[0] * self.scale[0], base[1] * self.scale[1])
            self.time_ns = monotonic_ns()

    def run_terminal(self):
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.quit.is_set():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1)
                    if key == "\x1b" and select.select([sys.stdin], [], [], 0.02)[0]:
                        if sys.stdin.read(1) == "[" and select.select([sys.stdin], [], [], 0.02)[0]:
                            key = ARROWS.get(sys.stdin.read(1), "")
                    self.press(key.lower())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--source-recording", help="Shared by clips/copies of one recording")
    p.add_argument("--location", required=True)
    p.add_argument("--lighting", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--calibration", type=Path, help="Duty mixer JSON (default: simulator ratio [추정])")
    p.add_argument("--camera-calibration-version", default="unmeasured")
    p.add_argument("--camera", default="picamera2")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--flip180", action="store_true", help="Camera mounted upside down")
    p.add_argument("--motors", choices=["gpio", "fake"], default="gpio")
    p.add_argument("--driver", choices=["keyboard", "none"], default="keyboard")
    p.add_argument("--teleop-linear", type=float, default=0.6)
    p.add_argument("--teleop-angular", type=float, default=0.8)
    p.add_argument("--command-timeout", type=float, default=0.3)
    p.add_argument("--control-hz", type=float, default=50)
    p.add_argument("--every", type=int, default=1, help="Keep every Nth camera frame")
    p.add_argument("--duration", type=float, help="Seconds; default until q / Ctrl+C")
    p.add_argument("--encoder", choices=["none"], default="none")
    a = p.parse_args()
    if a.calibration:
        mixer, calibration = DutyMixer.load(a.calibration), json.loads(a.calibration.read_text())
    else:
        from dtwin.core import load_inputs

        mixer = DutyMixer.from_simulator(load_inputs()[0])
        calibration = mixer.to_json()
    camera = open_camera(a.camera, a.width, a.height, a.fps, a.flip180)
    motors = open_motors(a.motors)
    teleop = Teleop(a.teleop_linear, a.teleop_angular)
    applied = {"action": (0.0, 0.0), "duty": (0.0, 0.0), "applied_ns": None, "source": a.driver}
    lock, running = threading.Lock(), threading.Event()
    running.set()

    def control_loop():
        period = 1 / a.control_hz
        while running.is_set():
            now = monotonic_ns()
            fresh = teleop.time_ns is not None and now - teleop.time_ns <= a.command_timeout * 1e9
            action = teleop.action if fresh else (0.0, 0.0)
            duty = mixer.duty(action) if action != (0.0, 0.0) else (0.0, 0.0)
            if duty == (0.0, 0.0):
                motors.stop()
            else:
                motors.set(*duty)
            with lock:
                applied.update(action=action, duty=duty, applied_ns=monotonic_ns())
            time.sleep(period)

    info = dict(
        source_recording=a.source_recording or a.session_id,
        location=a.location,
        lighting=a.lighting,
        notes=a.notes,
        camera=dict(getattr(camera, "description", {}), stored="RGB JPEG q95"),
        calibration_version=a.camera_calibration_version,
        vehicle_calibration=calibration,
        clock=dict(
            reference="CLOCK_BOOTTIME ns (dtwin.realtime.monotonic_ns)",
            capture="libcamera SensorTimestamp when capture_clock=sensor, else receive time",
        ),
        encoder=False,
        driver=a.driver,
        label_rule="command = most recent motor command applied when the frame was received",
        motors=getattr(motors, "description", {}),
    )
    writer = SessionWriter(a.output, a.session_id, info)
    threads = [threading.Thread(target=control_loop, name="motors", daemon=True)]
    if a.driver == "keyboard":
        threads.append(threading.Thread(target=teleop.run_terminal, name="keys", daemon=True))
    for thread in threads:
        thread.start()
    print(
        "Recording. Arrows: up fwd, left/right forward arc, down back, space stop. "
        "Also w/s, q/e arcs, a/d spin, x stop. p quit.",
        file=sys.stderr,
    )
    start, kept, seen, status = monotonic_ns(), 0, 0, "completed"
    try:
        while not teleop.quit.is_set():
            if a.duration and monotonic_ns() - start > a.duration * 1e9:
                break
            frame = camera.read()
            if frame is None:
                if a.camera.startswith("folder:"):
                    break
                continue
            seen += 1
            if (seen - 1) % a.every:
                continue
            with lock:
                command = dict(
                    action=list(applied["action"]),
                    duty=list(applied["duty"]),
                    applied_ns=applied["applied_ns"],
                    source=applied["source"],
                )
            writer.write(kept, frame.rgb, frame.capture_ns, frame.capture_clock, frame.received_ns, command)
            kept += 1
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status = f"error: {error!r}"
        raise
    finally:
        running.clear()
        teleop.quit.set()
        motors.stop()
        for thread in threads:
            thread.join(timeout=1)
        motors.stop()
        camera.close()
        writer.close(status=status, camera_frames_seen=seen, frames_kept=kept)
        print(json.dumps(dict(session=str(writer.path), frames=kept, status=status)), file=sys.stderr)


if __name__ == "__main__":
    main()
