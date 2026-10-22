#!/usr/bin/env python3
"""Train the camera policy outside Isaac Sim; split by scene seed, never by frame."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dtwin.policy import preprocess


class LanePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 2),
            nn.Tanh(),
        )

    def forward(self, image):
        return self.net(image)


def split_groups(episodes, seed):
    groups = sorted({e["group"] for e in episodes})
    if len(groups) < 3:
        raise ValueError(
            "At least 3 distinct scene seeds required for train/validation/test separation"
        )
    random.Random(seed).shuffle(groups)
    n = max(1, round(len(groups) * 0.2))
    return {"test": groups[:n], "validation": groups[n : 2 * n], "train": groups[2 * n :]}


def read_episodes(root):
    episodes = []
    for metadata_path in sorted(Path(root).rglob("episode.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata["mode"] not in {"expert", "manual"}:
            continue
        frames_path = metadata_path.parent / "frames.jsonl"
        frames = [json.loads(line) for line in frames_path.read_text().splitlines() if line.strip()]
        frames = [f for f in frames if "image" in f and f["footprint_on_road"]]
        if frames:
            episodes.append(
                dict(
                    path=metadata_path.parent,
                    frames=frames,
                    group=str(metadata["domain"]["seed"]),
                    metadata=metadata,
                )
            )
    if not episodes:
        raise ValueError(
            "No recorded expert/manual camera frames found; run run_sim.py --record first"
        )
    return episodes


class Frames(Dataset):
    def __init__(self, episodes, groups, augment=False):
        self.rows = [(e, f) for e in episodes if e["group"] in groups for f in e["frames"]]
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        episode, frame = self.rows[index]
        rgb = np.asarray(Image.open(episode["path"] / frame["image"]).convert("RGB")).copy()
        cfg = episode["metadata"]["config"]["simulation"]
        target = np.array(
            [
                frame["linear_m_s"] / cfg["max_speed_m_s"],
                frame["angular_rad_s"] / cfg["max_yaw_rad_s"],
            ],
            dtype=np.float32,
        )
        if np.max(np.abs(target)) > 1.001:
            raise ValueError("Recorded commands exceed configured policy limits")
        if self.augment:
            if random.random() < 0.5:
                rgb = rgb[:, ::-1].copy()
                target[1] *= -1
            gain = random.uniform(0.7, 1.3)
            channels = np.array([random.uniform(0.9, 1.1), 1, random.uniform(0.9, 1.1)])
            rgb = np.clip(
                rgb * gain * channels + np.random.normal(0, random.uniform(0, 5), rgb.shape), 0, 255
            ).astype(np.uint8)
        return torch.from_numpy(preprocess(rgb)), torch.from_numpy(target)


def evaluate(model, loader, device):
    total = np.zeros(2)
    count = 0
    model.eval()
    with torch.no_grad():
        for image, target in loader:
            delta = (model(image.to(device)) - target.to(device)).abs().sum(0).cpu().numpy()
            total += delta
            count += len(image)
    return (total / max(1, count)).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("--output", type=Path, default=Path("models/lane"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.epochs <= 0 or a.batch_size <= 0:
        p.error("epochs and batch size must be positive")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    episodes = read_episodes(a.data)
    splits = split_groups(episodes, a.seed)
    # Every episode must use one action scale; ONNX metadata has one fixed contract.
    scales = {
        (
            e["metadata"]["config"]["simulation"]["max_speed_m_s"],
            e["metadata"]["config"]["simulation"]["max_yaw_rad_s"],
        )
        for e in episodes
    }
    if len(scales) != 1:
        raise ValueError("Mixed action scales: use separate data collections")
    loaders = {
        name: DataLoader(
            Frames(episodes, groups, name == "train"),
            batch_size=a.batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, groups in splits.items()
    }
    model = LanePolicy().to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_loss, best, history = float("inf"), None, []
    for epoch in range(a.epochs):
        model.train()
        for image, target in loaders["train"]:
            image, target = image.to(a.device), target.to(a.device)
            prediction = model(image)
            # Give curves more weight than overrepresented straight sections.
            weights = 1 + 2 * target[:, 1].abs()
            loss = (
                nn.functional.smooth_l1_loss(prediction, target, reduction="none").mean(1) * weights
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation = evaluate(model, loaders["validation"], a.device)
        history.append({"epoch": epoch + 1, "validation_normalized_mae": validation})
        print(json.dumps(history[-1]), flush=True)
        if sum(validation) < best_loss:
            best_loss, best = sum(validation), copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    test_mae = evaluate(model, loaders["test"], a.device)
    a.output.mkdir(parents=True, exist_ok=False)
    torch.save(model.cpu().state_dict(), a.output / "policy.pt")
    model.eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 66, 200),
        str(a.output / "policy.onnx"),
        input_names=["image"],
        output_names=["command"],
        opset_version=17,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(str(a.output / "policy.onnx")))
    max_speed, max_yaw = next(iter(scales))
    metadata = dict(
        preprocess="rgb_bottom_half_yuv_blur3_200x66_chw_div255",
        max_speed_m_s=max_speed,
        max_yaw_rad_s=max_yaw,
        seed=a.seed,
        splits=splits,
        history=history,
        test_normalized_mae=test_mae,
        test_physical_mae={
            "linear_m_s": test_mae[0] * max_speed,
            "angular_rad_s": test_mae[1] * max_yaw,
        },
        warning="Offline MAE does not establish closed-loop robustness or mission recognition",
        frames={name: len(loader.dataset) for name, loader in loaders.items()},
    )
    (a.output / "policy.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
