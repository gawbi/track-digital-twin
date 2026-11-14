import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch

    from train_policy import Frames, LanePolicy
except ImportError:  # the Pi / Isaac envs have no torch
    torch = None


@unittest.skipIf(torch is None, "needs the training environment (torch)")
class TwoFramePolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        (self.root / "images").mkdir()
        from PIL import Image

        for i in range(5):
            Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(self.root / f"images/{i:06d}.jpg")

    def tearDown(self):
        self.tmp.cleanup()

    def export(self, model, contract):
        path = self.root / "policy.onnx"
        model.eval()
        channels = model.net[0].in_channels
        torch.onnx.export(model, torch.zeros(1, channels, 66, 200), str(path), input_names=["image"],
                          output_names=["command"], opset_version=17, dynamo=False)
        path.with_suffix(".json").write_text(json.dumps(dict(preprocess=contract, max_speed_m_s=0.18, max_yaw_rad_s=2.0)))
        return path

    def test_checkpoint_expansion_reproduces_the_single_frame_parent(self):
        from dtwin.policy import BASE_CONTRACT, OnnxPolicy

        torch.manual_seed(0)
        parent = LanePolicy(3)
        state = parent.state_dict()
        first = state["net.0.weight"]
        state["net.0.weight"] = torch.cat([first, torch.zeros_like(first)], dim=1)
        child = LanePolicy(6)
        child.load_state_dict(state)
        single = OnnxPolicy(self.export(parent, BASE_CONTRACT))
        frames = [np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(self.root / f"images/{i:06d}.jpg")) for i in range(3)]
        expected = single.action(frames[2])
        double = OnnxPolicy(self.export(child, BASE_CONTRACT + "_prev2"))
        self.assertEqual(double.gap, 2)
        np.testing.assert_allclose(double.action(frames[2], previous_rgb=frames[0]), expected, atol=1e-5)

    def test_history_uses_the_frame_two_calls_ago(self):
        from dtwin.policy import BASE_CONTRACT, OnnxPolicy

        torch.manual_seed(1)
        policy = OnnxPolicy(self.export(LanePolicy(6), BASE_CONTRACT + "_prev2"))
        from PIL import Image

        frames = [np.asarray(Image.open(self.root / f"images/{i:06d}.jpg")) for i in range(4)]
        first = policy.action(frames[0])  # no history yet: the frame pairs with itself
        np.testing.assert_allclose(first, policy.action(frames[0], previous_rgb=frames[0]), atol=1e-6)
        policy.action(frames[1])
        live = policy.action(frames[2])
        np.testing.assert_allclose(live, policy.action(frames[2], previous_rgb=frames[0]), atol=1e-6)

    def test_dataset_returns_six_channels_with_the_earlier_recorded_frame(self):
        from dtwin.policy import previous_image

        self.assertEqual(previous_image(self.root, "images/000004.jpg", 2), Path("images/000002.jpg"))
        self.assertEqual(previous_image(self.root, "images/000001.jpg", 2), Path("images/000000.jpg"))
        rows = [dict(path=self.root, image="images/000004.jpg", action=[0.5, 0.1])]
        image, target = Frames([], [], False, rows, gap=2)[0]
        self.assertEqual(tuple(image.shape), (6, 66, 200))
        self.assertFalse(torch.equal(image[:3], image[3:]))  # two different recorded frames
        flipped = Frames([], [], True, rows, gap=2)
        for _ in range(10):  # augmentation flips both frames together with the steering label
            image, target = flipped[0]
            self.assertEqual(tuple(image.shape), (6, 66, 200))
            self.assertIn(round(float(target[1]), 3), (0.1, -0.1))


if __name__ == "__main__":
    unittest.main()
