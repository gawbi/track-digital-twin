"""Real-car runtime: latest-frame perception and a separate motor control loop.

Camera thread  -> LatestSlot (older unread frames are dropped and counted)
Perception     -> newest frame only; result keeps the ORIGINAL capture time
Control loop   -> fixed period; applies the newest valid result or stops

Every age is measured on one monotonic clock (CLOCK_BOOTTIME on Linux, which
matches libcamera sensor timestamps). A result older than ``max_result_age_s``
since CAPTURE is never applied, even if inference just finished. Python threads
on Linux are not hard real time: the loop records its actual timing instead.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field

STOP = (0.0, 0.0)


def monotonic_ns():
    if hasattr(time, "CLOCK_BOOTTIME"):
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    return time.monotonic_ns()


@dataclass
class Frame:
    frame_id: int
    capture_ns: int
    capture_clock: str  # "sensor" or "received": what capture_ns really measures
    received_ns: int
    rgb: object = field(repr=False)


@dataclass
class Result:
    frame_id: int
    capture_ns: int
    capture_clock: str
    received_ns: int
    started_ns: int
    finished_ns: int
    action: tuple
    horn: bool = False
    stages: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


class LatestSlot:
    """Single-item mailbox: writers overwrite, readers get only newer items."""

    def __init__(self):
        self._cond = threading.Condition()
        self._item, self._version, self._taken = None, 0, True
        self.dropped = 0
        self.closed = False

    def put(self, item):
        with self._cond:
            if not self._taken:
                self.dropped += 1
            self._item, self._taken = item, False
            self._version += 1
            self._cond.notify_all()

    def take_newer(self, version, timeout_s):
        with self._cond:
            self._cond.wait_for(lambda: self._version > version or self.closed, timeout_s)
            if self._version <= version:
                return version, None
            self._taken = True
            return self._version, self._item

    def peek(self):
        with self._cond:
            return self._version, self._item

    def close(self):
        with self._cond:
            self.closed = True
            self._cond.notify_all()


class Telemetry:
    """JSONL writer on its own thread so file I/O never blocks motor updates."""

    def __init__(self, path=None):
        self.path, self.records = path, []
        self._queue = queue.SimpleQueue()
        self._thread = None
        if path:
            self._file = open(path, "x", encoding="utf-8")
            self._thread = threading.Thread(target=self._drain, name="telemetry", daemon=True)
            self._thread.start()

    def write(self, kind, **data):
        record = dict(kind=kind, **data)
        if self._thread:
            self._queue.put(record)
        else:
            self.records.append(record)

    def _drain(self):
        while True:
            record = self._queue.get()
            if record is None:
                break
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.close()

    def close(self):
        if self._thread:
            self._queue.put(None)
            self._thread.join(timeout=5)


class VehicleRuntime:
    """``camera.read()`` -> Frame (blocking), ``perceive(frame)`` -> Result fields,
    ``motors.set(left, right)`` / ``motors.stop()``, ``mixer.duty(action)``."""

    def __init__(
        self,
        camera,
        perceive,
        motors,
        mixer,
        control_hz=50.0,
        max_result_age_s=0.25,
        max_camera_gap_s=0.3,
        max_inference_s=0.5,
        speed_scale=1.0,
        clock=monotonic_ns,
        telemetry=None,
    ):
        if control_hz <= 0 or min(max_result_age_s, max_camera_gap_s, max_inference_s) <= 0:
            raise ValueError("Control rate and timeouts must be positive")
        if not 0 <= speed_scale <= 1:
            raise ValueError("speed_scale must be in [0, 1]")
        self.camera, self.perceive, self.motors, self.mixer = camera, perceive, motors, mixer
        self.period_ns = round(1e9 / control_hz)
        self.max_result_age_ns = round(max_result_age_s * 1e9)
        self.max_camera_gap_ns = round(max_camera_gap_s * 1e9)
        self.max_inference_ns = round(max_inference_s * 1e9)
        self.speed_scale, self.clock = speed_scale, clock
        self.telemetry = telemetry or Telemetry()
        self.frames, self.results = LatestSlot(), LatestSlot()
        self.running = threading.Event()
        self.fault = None
        self.last_frame_ns = None
        self.inference_started_ns = None
        self.applied = set()
        self.threads = []
        self.counts = dict(ticks=0, applied_ticks=0, stopped_ticks=0, results=0, frames=0)
        self.stop_reasons = {}

    # ---- worker threads -------------------------------------------------
    def _fail(self, where, error):
        if self.fault is None:
            self.fault = f"{where}: {error!r}"
            self.telemetry.write("fault", where=where, error=repr(error), t_ns=self.clock())

    def _camera_loop(self):
        try:
            while self.running.is_set():
                frame = self.camera.read()
                if frame is None:
                    continue
                self.last_frame_ns = frame.received_ns
                self.counts["frames"] += 1
                self.frames.put(frame)
                self.telemetry.write(
                    "frame",
                    frame_id=frame.frame_id,
                    capture_ns=frame.capture_ns,
                    capture_clock=frame.capture_clock,
                    received_ns=frame.received_ns,
                    dropped_total=self.frames.dropped,
                )
        except Exception as error:  # noqa: BLE001 - any camera error must stop the car
            self._fail("camera", error)

    def _inference_loop(self):
        version = 0
        try:
            while self.running.is_set():
                version, frame = self.frames.take_newer(version, 0.05)
                if frame is None:
                    continue
                started = self.clock()
                self.inference_started_ns = started
                output = self.perceive(frame)
                finished = self.clock()
                self.inference_started_ns = None
                action = tuple(float(v) for v in output["action"])
                if len(action) != 2 or not all(-1.0 <= v <= 1.0 for v in action):
                    raise ValueError(f"Perception returned invalid action {action}")
                result = Result(
                    frame_id=frame.frame_id,
                    capture_ns=frame.capture_ns,
                    capture_clock=frame.capture_clock,
                    received_ns=frame.received_ns,
                    started_ns=started,
                    finished_ns=finished,
                    action=action,
                    horn=bool(output.get("horn", False)),
                    stages=output.get("stages", {}),
                    info=output.get("info", {}),
                )
                self.counts["results"] += 1
                self.results.put(result)
                self.telemetry.write(
                    "result",
                    frame_id=result.frame_id,
                    capture_ns=result.capture_ns,
                    capture_clock=result.capture_clock,
                    received_ns=result.received_ns,
                    started_ns=started,
                    finished_ns=finished,
                    action=list(action),
                    horn=result.horn,
                    stages=result.stages,
                    info=result.info,
                )
        except Exception as error:  # noqa: BLE001 - perception errors must stop the car
            self.inference_started_ns = None
            self._fail("perception", error)

    # ---- control --------------------------------------------------------
    def decide(self, now):
        """Return (action, reason, result). Pure function of state and ``now``."""
        if self.fault:
            return STOP, "fault", None
        if self.last_frame_ns is None or now - self.last_frame_ns > self.max_camera_gap_ns:
            return STOP, "camera_stalled", None
        started = self.inference_started_ns
        if started is not None and now - started > self.max_inference_ns:
            return STOP, "inference_stalled", None
        _, result = self.results.peek()
        if result is None:
            return STOP, "no_result", None
        if now - result.capture_ns > self.max_result_age_ns:
            return STOP, "stale_result", result
        if result.capture_ns > now:
            return STOP, "future_capture_timestamp", result
        return result.action, "applied", result

    def tick(self, scheduled_ns):
        now = self.clock()
        action, reason, result = self.decide(now)
        if reason == "applied":
            scaled = (action[0] * self.speed_scale, action[1] * self.speed_scale)
            duty = self.mixer.duty(scaled)
            self.motors.set(*duty)
            self.counts["applied_ticks"] += 1
        else:
            duty = STOP
            self.motors.stop()
            self.counts["stopped_ticks"] += 1
            self.stop_reasons[reason] = self.stop_reasons.get(reason, 0) + 1
        applied_ns = self.clock()
        self.counts["ticks"] += 1
        first_use = result is not None and reason == "applied" and result.frame_id not in self.applied
        if first_use:
            self.applied.add(result.frame_id)
        self.telemetry.write(
            "tick",
            scheduled_ns=scheduled_ns,
            start_ns=now,
            applied_ns=applied_ns,
            lateness_ns=now - scheduled_ns,
            reason=reason,
            frame_id=None if result is None else result.frame_id,
            first_use=first_use,
            capture_ns=None if result is None else result.capture_ns,
            capture_clock=None if result is None else result.capture_clock,
            duty=list(duty),
        )
        return reason

    def start(self):
        self.running.set()
        for name, target in [("camera", self._camera_loop), ("perception", self._inference_loop)]:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)

    def run(self, duration_s=None, stop_event=None):
        """Control loop on the calling thread. Motors stop on every exit path."""
        self.motors.stop()
        self.start()
        start = self.clock()
        k = 0
        try:
            while not (stop_event and stop_event.is_set()):
                scheduled = start + k * self.period_ns
                if duration_s is not None and scheduled - start > duration_s * 1e9:
                    break
                delay = scheduled - self.clock()
                if delay > 0:
                    time.sleep(delay / 1e9)
                self.tick(scheduled)
                k += 1
                # Skip missed deadlines instead of bursting to catch up.
                behind = (self.clock() - start) // self.period_ns
                if behind > k:
                    self.telemetry.write("skipped_ticks", count=int(behind - k))
                    k = int(behind)
        finally:
            self.shutdown()
        return dict(counts=self.counts, stop_reasons=self.stop_reasons, fault=self.fault)

    def shutdown(self):
        self.running.clear()
        try:
            self.motors.stop()
        finally:
            self.frames.close()
            self.results.close()
            for thread in self.threads:
                thread.join(timeout=1.0)
            try:
                self.camera.close()
            except Exception as error:  # noqa: BLE001 - motors are already stopped
                self.telemetry.write("camera_close_error", error=repr(error))
            self.motors.stop()
            self.telemetry.write("shutdown", t_ns=self.clock(), dropped_frames=self.frames.dropped)
