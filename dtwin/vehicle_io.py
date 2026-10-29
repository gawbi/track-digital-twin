"""Raspberry Pi camera/motor adapters and fakes for testing without hardware.

Pin map matches the course car used by dtwin.legacy (8_4.py):
left PWM 18 / IN 22, 27; right PWM 23 / IN 25, 24; (IN1, IN2) = (0, 1) is forward.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from .realtime import Frame, monotonic_ns

MOTOR_PINS = {"left": (18, 22, 27), "right": (23, 25, 24)}


class PicameraSource:
    """Picamera2 frames as RGB with the libcamera sensor timestamp.

    Picamera2 "RGB888" is BGR in memory (libcamera naming), so channels are
    reversed here once. SensorTimestamp uses CLOCK_BOOTTIME like realtime.monotonic_ns.
    """

    def __init__(self, width=640, height=480, fps=30, flip180=False):
        from picamera2 import Picamera2

        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
            buffer_count=2,
        )
        self.camera.configure(config)
        self.camera.start()
        self.flip180, self.next_id = flip180, 0
        self.description = dict(
            kind="picamera2", width=width, height=height, fps=fps, pixel_order_in="BGR",
            flip180=flip180,
        )

    def read(self):
        request = self.camera.capture_request()
        try:
            bgr = request.make_array("main")
            metadata = request.get_metadata()
        finally:
            request.release()
        received = monotonic_ns()
        sensor = metadata.get("SensorTimestamp")
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        if self.flip180:
            rgb = np.ascontiguousarray(rgb[::-1, ::-1])
        # Sanity: a sensor time in the future or >1 s old is not trusted.
        if sensor is not None and 0 <= received - sensor < 1_000_000_000:
            capture, clock = int(sensor), "sensor"
        else:
            capture, clock = received, "received"
        frame = Frame(self.next_id, capture, clock, received, rgb)
        self.next_id += 1
        return frame

    def close(self):
        self.camera.stop()
        self.camera.close()


class OpenCVSource:
    """USB/V4L2 camera; no sensor timestamp, so capture time = receive time."""

    def __init__(self, device=0, width=640, height=480, flip180=False):
        import cv2

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(device)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError(f"Camera {device} did not open")
        self.flip180, self.next_id = flip180, 0
        self.description = dict(kind="opencv", device=device, width=width, height=height,
                                pixel_order_in="BGR", flip180=flip180)

    def read(self):
        ok, bgr = self.capture.read()
        received = monotonic_ns()
        if not ok:
            raise RuntimeError("Camera read failed")
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        if self.flip180:
            rgb = np.ascontiguousarray(rgb[::-1, ::-1])
        frame = Frame(self.next_id, received, "received", received, rgb)
        self.next_id += 1
        return frame

    def close(self):
        self.capture.release()


class ImageFolderSource:
    """Replays saved RGB JPEG/PNG files at a fixed rate for desk tests (no motors moving).

    ``delay_s`` simulates a camera pipeline delay: capture is that much before receive.
    """

    def __init__(self, folder, fps=30.0, loop=True, delay_s=0.0, clock=monotonic_ns):
        self.paths = sorted(
            p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.paths:
            raise ValueError(f"No images in {folder}")
        self.period, self.loop, self.delay_ns, self.clock = 1 / fps, loop, round(delay_s * 1e9), clock
        self.next_id, self.next_time = 0, None
        self.description = dict(kind="image_folder", folder=str(folder), fps=fps, frames=len(self.paths),
                                simulated_delay_s=delay_s)

    def read(self):
        from PIL import Image

        if self.next_id >= len(self.paths) and not self.loop:
            time.sleep(self.period)
            return None
        now = time.monotonic()
        if self.next_time is not None and now < self.next_time:
            time.sleep(self.next_time - now)
        self.next_time = time.monotonic() + self.period
        path = self.paths[self.next_id % len(self.paths)]
        rgb = np.asarray(Image.open(path).convert("RGB"))
        received = self.clock()
        frame = Frame(self.next_id, received - self.delay_ns, "simulated", received, rgb)
        self.next_id += 1
        return frame

    def close(self):
        pass


class GpioMotors:
    """Signed duty in [-1, 1] -> gpiozero PWM + direction pins. ``stop`` is idempotent."""

    def __init__(self, pins=MOTOR_PINS, frequency=100):
        from gpiozero import DigitalOutputDevice, PWMOutputDevice

        self.devices = {}
        for side, (pwm, in1, in2) in pins.items():
            self.devices[side] = (
                PWMOutputDevice(pwm, frequency=frequency, initial_value=0),
                DigitalOutputDevice(in1, initial_value=False),
                DigitalOutputDevice(in2, initial_value=False),
            )
        self.lock = threading.Lock()
        self.description = dict(kind="gpiozero", pins=pins, pwm_frequency_hz=frequency)

    def _side(self, side, duty):
        pwm, in1, in2 = self.devices[side]
        duty = max(-1.0, min(1.0, float(duty)))
        if duty > 0:
            in1.off()
            in2.on()
        elif duty < 0:
            in1.on()
            in2.off()
        else:
            in1.off()
            in2.off()
        pwm.value = abs(duty)

    def set(self, left, right):
        with self.lock:
            self._side("left", left)
            self._side("right", right)

    def stop(self):
        with self.lock:
            for side in self.devices:
                self._side(side, 0.0)

    def close(self):
        self.stop()
        for devices in self.devices.values():
            for device in devices:
                device.close()


class FakeMotors:
    """Records every command with a timestamp; used for fault-injection tests."""

    def __init__(self, clock=monotonic_ns):
        self.clock, self.calls, self.stops = clock, [], 0
        self.state = (0.0, 0.0)
        self.description = dict(kind="fake")

    def set(self, left, right):
        self.state = (float(left), float(right))
        self.calls.append((self.clock(), *self.state))

    def stop(self):
        self.stops += 1
        self.state = (0.0, 0.0)
        self.calls.append((self.clock(), 0.0, 0.0))

    def close(self):
        self.stop()


def system_snapshot():
    """Pi model, OS, temperature and throttling flags when available (no hard failure)."""
    import platform
    import subprocess

    info = dict(platform=platform.platform(), python=platform.python_version(), machine=platform.machine())
    for key, path in [("model", "/proc/device-tree/model"), ("os_release", "/etc/os-release")]:
        try:
            text = Path(path).read_text(errors="ignore").replace("\x00", "").strip()
            info[key] = text if key == "model" else dict(
                line.split("=", 1) for line in text.splitlines() if "=" in line
            ).get("PRETTY_NAME", "").strip('"')
        except OSError:
            info[key] = None
    try:
        info["cpu_temp_c"] = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        info["cpu_temp_c"] = None
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2)
        info["throttled"] = out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["throttled"] = None
    try:
        import onnxruntime

        info["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        info["onnxruntime"] = None
    return info
