from collections import deque
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from dtwin.runtime import Simulator


class RuntimeContractTests(TestCase):
    def test_wheel_odometry_uses_measured_joints_instead_of_drive_targets(self):
        sim = Simulator.__new__(Simulator)
        sim.legacy = True
        sim._encoder = Mock()
        sim._encoder_indices = [0, 1, 2, 3]
        sim._encoder.get_joint_velocities.return_value = np.array([2.0, 4.0, 6.0, 8.0])
        self.assertEqual(sim.wheel_measurement(), (3.0, 7.0))

    def test_delayed_pixels_keep_their_original_detection_labels(self):
        sim = Simulator.__new__(Simulator)
        sim.params = {"image_gain": 1, "image_noise_std": 0}
        sim.rng = np.random.default_rng(1)
        sim.delay = deque(maxlen=2)
        sim.frame, sim.dt = 0, 1 / 30
        sim.raw_rgb = Mock(
            side_effect=[
                np.full((2, 2, 3), 20, dtype=np.uint8),
                np.full((2, 2, 3), 80, dtype=np.uint8),
            ]
        )
        sim.detections = Mock(side_effect=[[{"class": "stop"}], [{"class": "horn"}]])
        sim.image()
        delayed = sim.image()
        self.assertEqual(int(delayed[0, 0, 0]), 20)
        self.assertEqual(sim.last_boxes, [{"class": "stop"}])

    def test_four_physics_steps_and_one_render_per_camera_frame(self):
        from dtwin.core import load_inputs

        sim = Simulator.__new__(Simulator)
        sim.cfg, _ = load_inputs()
        sim.objects = {"wheel_drives": []}
        sim.params = {}
        sim.legacy = True
        sim.world = Mock()
        sim.frame = 0
        sim.step(0, 0)
        self.assertEqual(sim.world.step.call_count, 4)
        for call in sim.world.step.call_args_list:
            self.assertEqual(call.kwargs, {"render": False})
        sim.world.render.assert_called_once()
        self.assertEqual(sim.frame, 1)
