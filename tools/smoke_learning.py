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
        real_pipeline(root, cfg)
        print(
            "PASS: tiny software fixture trained, exported, inferred, converted 15 detection frames, "
            "ran A/B/C real-split fine-tuning, final test ledger and a fake-motor Pi run. "
            "No driving quality claim."
        )


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def real_pipeline(root, cfg):
    from dtwin.real_data import SessionWriter

    info = dict(location="fixture", lighting="fixture", camera={"width": 640, "height": 480},
                calibration_version="fixture", clock={"reference": "fixture"}, encoder=False)
    for index in range(4):
        writer = SessionWriter(root / "real", f"session-{index}", dict(info))
        for frame in range(3):
            rgb = np.full((480, 640, 3), 60 + index * 30 + frame * 3, dtype=np.uint8)
            rgb[240:, 200 + 20 * index : 230 + 20 * index] = [250, 250, 250]
            writer.write(frame, rgb, frame, "sensor", frame,
                         dict(action=[0.4, 0.1 * index - 0.1], duty=[0.2, 0.2], applied_ns=frame,
                              source="fixture"))
        writer.close(status="completed")
        labels = writer.path / "labels"
        labels.mkdir()
        (labels / "000000.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    run(ROOT / "split_real.py", root / "real", "--output", root / "manifest.json",
        "--assign", "session-3=test")
    common = ["--epochs", "1", "--batch-size", "3", "--device", "cpu"]
    run(ROOT / "train_policy.py", root / "data", "--real-manifest", root / "manifest.json",
        "--train-on", "synthetic", "--experiment", "A", "--output", root / "A", *common)
    run(ROOT / "train_policy.py", "--real-manifest", root / "manifest.json", "--experiment", "B",
        "--output", root / "B", *common)
    run(ROOT / "train_policy.py", "--real-manifest", root / "manifest.json", "--experiment", "C",
        "--init-checkpoint", root / "A/policy.pt", "--lr", "1e-5", "--output", root / "C", *common)
    meta = json.loads((root / "C/policy.json").read_text())
    assert meta["init_checkpoint"]["experiment"] == "A" and meta["synthetic_seed_history"]
    assert meta["real_data"]["used_for_training"] and not meta["test_evaluated"]
    run(ROOT / "evaluate_lane_offline.py", root / "C/policy.onnx", "--split", "validation",
        "--real-manifest", root / "manifest.json")
    denied = subprocess.run([sys.executable, str(ROOT / "evaluate_lane_offline.py"),
                             str(root / "C/policy.onnx"), "--split", "test",
                             "--real-manifest", str(root / "manifest.json")], capture_output=True)
    assert denied.returncode != 0, "test split must require --final"
    run(ROOT / "evaluate_lane_offline.py", root / "C/policy.onnx", "--split", "test", "--final",
        "--real-manifest", root / "manifest.json", "--purpose", "smoke")
    assert (root / "manifest.test_access.jsonl").exists()
    run(ROOT / "make_detector_dataset.py", "--real-manifest", root / "manifest.json",
        "--output", root / "real-detector")
    assert len(list((root / "real-detector/labels").rglob("*.txt"))) == 4
    run(ROOT / "run_vehicle.py", "--policy", root / "C/policy.onnx", "--camera",
        f"folder:{root / 'real/session-0/images'}", "--motors", "fake", "--duration", "2",
        "--warmup", "0.5", "--output", root / "vehicle-run", "--max-result-age", "0.5")
    summary = json.loads((root / "vehicle-run/summary.json").read_text())
    latency = summary["latency"]
    assert latency["control"]["ticks"] > 50 and latency["perception"]["results"] > 10, summary
    assert latency["control"]["applied_fraction"] > 0.5, summary
    assert latency["end_to_end_capture_to_motor_ms"]["n"] > 10, summary


if __name__ == "__main__":
    main()
