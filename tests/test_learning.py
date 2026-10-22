import unittest

import numpy as np

from dtwin.perception import yolo_label
from dtwin.policy import preprocess
from train_policy import LanePolicy, split_groups


class LearningTests(unittest.TestCase):
    def test_detection_box_normalization_and_class_identity(self):
        self.assertEqual(
            yolo_label({"class": "stop", "xyxy": [0, 0, 640, 480]}, 640, 480),
            [1, 0.5, 0.5, 1.0, 1.0],
        )
        with self.assertRaises(ValueError):
            yolo_label({"class": "stop", "xyxy": [10, 10, 5, 5]}, 640, 480)

    def test_scene_seeds_never_leak_between_splits(self):
        episodes = [{"group": str(seed)} for seed in range(10)] * 2
        splits = split_groups(episodes, 42)
        a, b, c = (set(splits[key]) for key in ["train", "validation", "test"])
        self.assertFalse(a & b or b & c or a & c)
        self.assertEqual(a | b | c, {str(i) for i in range(10)})

    def test_rgb_preprocessing_and_model_contract(self):
        import torch

        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[240:, :, 0] = 255
        image = preprocess(rgb)
        self.assertEqual(image.shape, (3, 66, 200))
        self.assertEqual(image.dtype, np.float32)
        # RGB red Y channel is approximately 76; BGR mixup would produce ~29.
        self.assertAlmostEqual(float(image[0].mean()), 76 / 255, places=2)
        result = LanePolicy().eval()(torch.from_numpy(image)[None]).detach().numpy()
        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(np.isfinite(result).all())
