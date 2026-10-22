"""Tiny generated dataset checks train -> export -> ONNX inference -> detector conversion.

This fixture establishes software contracts only. It is NOT a driving model or
training evidence. All temporary data and weights are removed on exit.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtwin.core import ROOT, load_inputs
from dtwin.policy import OnnxPolicy


def main():
    cfg, _ = load_inputs()
    with tempfile.TemporaryDirectory(prefix="track-smoke-") as tmp:
        root = Path(tmp)
        data = root / "data"
        for seed in range(5):
            ep = data / f"episode-{seed}"
            (ep / "images").mkdir(parents=True)
            (ep / "episode.json").write_text(
                json.dumps({"mode": "expert", "domain": {"seed": seed}, "config": cfg})
            )
            frames = []
            for frame in range(3):
                rgb = np.full((480, 640, 3), 30 + seed * 15 + frame, dtype=np.uint8)
                rgb[240:, 150:170] = [240, 220, 40]
                relative = f"images/{frame:06d}.jpg"
                Image.fromarray(rgb).save(ep / relative)
                frames.append(
                    {
                        "frame": frame,
                        "image": relative,
                        "footprint_on_road": True,
                        "linear_m_s": 0.1,
                        "angular_rad_s": 0.1,
                        "detections": [{"class": "stop", "xyxy": [10, 10, 30, 30]}],
                    }
                )
            (ep / "frames.jsonl").write_text("\n".join(json.dumps(f) for f in frames) + "\n")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "train_policy.py"),
                str(data),
                "--epochs",
                "1",
                "--batch-size",
                "3",
                "--device",
                "cpu",
                "--output",
                str(root / "model"),
            ],
            check=True,
        )
        command = OnnxPolicy(root / "model/policy.onnx").command(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        assert all(np.isfinite(command)), command
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "make_detector_dataset.py"),
                str(data),
                "--output",
                str(root / "detector"),
            ],
            check=True,
        )
        assert len(list((root / "detector/labels").rglob("*.txt"))) == 15
        print(
            "PASS: tiny software fixture trained, exported, inferred, and converted 15 detection frames. No driving quality claim."
        )


if __name__ == "__main__":
    main()
