"""Local bridge for the user's external 8_4.py. No course code is redistributed."""

import runpy
import sys
import types

import numpy as np


def gpio_module(devices):
    class Device:
        def __init__(self, pin, *args, initial_value=0, **kwargs):
            self.pin, self.value = pin, initial_value
            devices[pin] = self

        def on(self):
            self.value = 1

        def off(self):
            self.value = 0

        def close(self):
            self.value = 0

    module = types.ModuleType("gpiozero")
    module.DigitalOutputDevice = Device
    module.PWMOutputDevice = Device
    return module


def gpio_wheels(devices, max_rad_s):
    wheels = []
    for pwm, in1, in2 in [(18, 22, 27), (23, 25, 24)]:
        if not all(pin in devices for pin in (pwm, in1, in2)):
            wheels.append(0.0)
            continue
        direction = {(0, 1): 1.0, (1, 0): -1.0}.get((devices[in1].value, devices[in2].value), 0.0)
        wheels.append(direction * max(0.0, min(1.0, float(devices[pwm].value))) * max_rad_s)
    return tuple(wheels)


def run_book(sim, script_path, model_path):
    # TensorFlow is optional, only imported in legacy mode. Prefer CPU inference
    # so TensorFlow does not reserve the GPU needed by Isaac's renderer.
    import os

    import cv2

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import tensorflow.keras.models as models

    original_load_model = models.load_model
    original_wait_key = cv2.waitKey
    devices = {}
    camera_module = types.ModuleType("mycamera")

    class Camera:
        def __init__(self, width, height):
            self.width, self.height, self.open = width, height, True

        def isOpened(self):
            return (
                self.open
                and sim.app.is_running()
                and sim.frame * sim.dt < sim.cfg["simulation"]["duration_s"]
            )

        def read(self):
            if not self.isOpened():
                return False, np.zeros((self.height, self.width, 3), dtype=np.uint8)
            sim.step(
                0, 0, wheel_override=gpio_wheels(devices, sim.cfg["vehicle"]["max_wheel_rad_s"])
            )
            bgr = cv2.cvtColor(sim.image(), cv2.COLOR_RGB2BGR)
            # The course script flips by -1. Give it the same inverted raw image.
            return True, cv2.flip(cv2.resize(bgr, (self.width, self.height)), -1)

        def release(self):
            self.open = False

    camera_module.MyPiCamera = Camera
    replacements = {"mycamera": camera_module, "gpiozero": gpio_module(devices)}
    previous = {key: sys.modules.get(key) for key in replacements}

    def load_model(_path, *args, **kwargs):
        kwargs.setdefault("compile", False)
        return original_load_model(str(model_path), *args, **kwargs)

    def wait_key(delay=1):
        key = original_wait_key(delay)
        if key in [ord("w"), 82, 65362, 2490368]:
            return 82
        if key in [ord("s"), 84, 65364, 2621440]:
            return 84
        return key

    models.load_model = load_model
    cv2.waitKey = wait_key
    sys.modules.update(replacements)
    print("Legacy camera window: W/up starts driving, S/down stops, Q exits.")
    try:
        runpy.run_path(str(script_path.resolve()), run_name="__main__")
    finally:
        models.load_model = original_load_model
        cv2.waitKey = original_wait_key
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        sim.step(0, 0)
        cv2.destroyAllWindows()
