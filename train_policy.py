#!/usr/bin/env python3
"""Train/fine-tune the camera lane policy outside Isaac Sim.

Synthetic data splits by scene seed, real data by the fixed session manifest.
The test splits are never evaluated here (see evaluate_lane_offline.py).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dtwin.actions import ACTION_SPACE
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


def read_real(manifest_path, split):
    """Rows from a fixed real-data manifest; labels are recorded command actions."""
    from dtwin.real_data import load_manifest, split_frames

    manifest = load_manifest(manifest_path)
    rows = [
        dict(path=item["path"], image=item["row"]["image"], action=item["row"]["command"]["action"])
        for item in split_frames(manifest, split, need="lane")
    ]
    return manifest, rows


class Frames(Dataset):
    """Synthetic expert frames (by scene seed group) and/or real recorded frames."""

    def __init__(self, episodes, groups, augment=False, real_rows=()):
        self.rows = [("sim", e, f) for e in episodes if e["group"] in groups for f in e["frames"]]
        self.rows += [("real", None, row) for row in real_rows]
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        source, episode, frame = self.rows[index]
        if source == "real":
            rgb = np.asarray(Image.open(frame["path"] / frame["image"]).convert("RGB")).copy()
            target = np.array(frame["action"], dtype=np.float32)
        else:
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


def sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path, nargs="?", help="Synthetic expert episodes (run_sim --record)")
    p.add_argument("--output", type=Path, default=Path("models/lane"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        help="policy.pt to fine-tune from (its policy.json lineage is preserved)",
    )
    p.add_argument("--real-manifest", type=Path, help="Fixed real split from split_real.py")
    p.add_argument(
        "--train-on",
        choices=["synthetic", "real", "both"],
        help="Default: synthetic if only DATA, real if only --real-manifest, both otherwise",
    )
    p.add_argument(
        "--select-on",
        choices=["synthetic", "real"],
        help="Validation used to pick the epoch (default: real when a manifest is given)",
    )
    p.add_argument("--experiment", default="", help="Name recorded in metadata, e.g. A/B/C")
    p.add_argument(
        "--frame-stride", type=int, default=1, help="Use every Nth synthetic frame (near-duplicates)"
    )
    p.add_argument("--workers", type=int, default=8, help="DataLoader workers (JPEG decoding)")
    a = p.parse_args()
    if a.epochs <= 0 or a.batch_size <= 0 or a.lr <= 0:
        p.error("epochs, batch size and learning rate must be positive")
    if not a.data and not a.real_manifest:
        p.error("Give synthetic DATA, --real-manifest, or both")
    train_on = a.train_on or ("both" if a.data and a.real_manifest else "synthetic" if a.data else "real")
    select_on = a.select_on or ("real" if a.real_manifest else "synthetic")
    if (train_on in {"synthetic", "both"} or select_on == "synthetic") and not a.data:
        p.error("Synthetic training/selection needs DATA")
    if (train_on in {"real", "both"} or select_on == "real") and not a.real_manifest:
        p.error("Real training/selection needs --real-manifest")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    episodes, splits = [], {"train": [], "validation": [], "test": []}
    if a.data:
        episodes = read_episodes(a.data)
        for episode in episodes:
            episode["frames"] = episode["frames"][:: a.frame_stride]
        splits = split_groups(episodes, a.seed)
    # Every episode must use one action scale; ONNX metadata has one fixed contract.
    scales = {
        (
            e["metadata"]["config"]["simulation"]["max_speed_m_s"],
            e["metadata"]["config"]["simulation"]["max_yaw_rad_s"],
        )
        for e in episodes
    }
    if len(scales) > 1:
        raise ValueError("Mixed action scales: use separate data collections")
    parent = None
    if a.init_checkpoint:
        parent = json.loads(a.init_checkpoint.with_name("policy.json").read_text())
        scales.add((parent["max_speed_m_s"], parent["max_yaw_rad_s"]))
        if len(scales) > 1:
            raise ValueError("Checkpoint and data use different action scales")
    if not scales:
        # Real-only from scratch: real labels are already normalized; keep the
        # simulator's scale so the exported model runs in run_sim.py as well.
        from dtwin.core import load_inputs

        sim = load_inputs()[0]["simulation"]
        scales.add((sim["max_speed_m_s"], sim["max_yaw_rad_s"]))
    manifest, real = None, {}
    if a.real_manifest:
        for split in ["train", "validation"]:
            manifest, real[split] = read_real(a.real_manifest, split)
    real_train = real.get("train", []) if train_on in {"real", "both"} else []
    sim_train = splits["train"] if train_on in {"synthetic", "both"} else []
    loaders = {
        "train": DataLoader(
            Frames(episodes, sim_train, True, real_train),
            batch_size=a.batch_size,
            shuffle=True,
            num_workers=a.workers,
        ),
        "validation": DataLoader(
            Frames(episodes, splits["validation"] if select_on == "synthetic" else [], False,
                   real.get("validation", []) if select_on == "real" else []),
            batch_size=a.batch_size,
            num_workers=a.workers,
        ),
    }
    if not len(loaders["train"].dataset) or not len(loaders["validation"].dataset):
        raise ValueError("Empty training or validation set for the chosen sources")
    if a.real_manifest and select_on == "synthetic":
        loaders["real_validation"] = DataLoader(
            Frames([], [], False, real["validation"]), batch_size=a.batch_size, num_workers=a.workers
        )
    model = LanePolicy().to(a.device)
    if a.init_checkpoint:
        model.load_state_dict(torch.load(a.init_checkpoint, map_location=a.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    best_loss, best, history, steps = float("inf"), None, [], 0
    started = time.monotonic()
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
            steps += 1
        validation = evaluate(model, loaders["validation"], a.device)
        row = {"epoch": epoch + 1, "validation_normalized_mae": validation}
        if "real_validation" in loaders:
            row["real_validation_normalized_mae"] = evaluate(model, loaders["real_validation"], a.device)
        history.append(row)
        print(json.dumps(row), flush=True)
        if sum(validation) < best_loss:
            best_loss, best = sum(validation), copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
    model.load_state_dict(best)
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
    history_seeds = set(parent.get("synthetic_seed_history", []) if parent else [])
    if parent:
        history_seeds |= {str(x) for groups in parent["splits"].values() for x in groups}
    history_seeds |= {str(x) for groups in splits.values() for x in groups}
    metadata = dict(
        preprocess="rgb_bottom_half_yuv_blur3_200x66_chw_div255",
        action_space=ACTION_SPACE,
        output="[linear/max_speed_m_s, angular/max_yaw_rad_s] in [-1, 1]",
        max_speed_m_s=max_speed,
        max_yaw_rad_s=max_yaw,
        experiment=a.experiment,
        seed=a.seed,
        frame_stride=a.frame_stride,
        splits=splits,
        synthetic_seed_history=sorted(history_seeds),
        train_on=train_on,
        select_on=select_on,
        optimizer=dict(name="AdamW", lr=a.lr, weight_decay=a.weight_decay),
        epochs=a.epochs,
        best_epoch=best_epoch,
        optimizer_steps=steps,
        wall_time_s=round(time.monotonic() - started, 1),
        init_checkpoint=None
        if not a.init_checkpoint
        else dict(
            path=str(a.init_checkpoint),
            sha256=sha256(a.init_checkpoint),
            experiment=parent.get("experiment"),
            train_on=parent.get("train_on", "synthetic"),
            real_manifest_id=(parent.get("real_data") or {}).get("manifest_id"),
        ),
        real_data=None
        if not manifest
        else dict(
            manifest=str(a.real_manifest),
            manifest_id=manifest["manifest_id"],
            sessions={k: len(v) for k, v in manifest["assignment"].items()},
            frames={k: len(v) for k, v in real.items()},
            used_for_training=bool(real_train),
        ),
        history=history,
        frames={name: len(loader.dataset) for name, loader in loaders.items()},
        test_evaluated=False,
        warning="Selection used validation only; run evaluate_lane_offline.py --split test "
        "once after fixing settings. Offline MAE is not closed-loop driving performance.",
    )
    (a.output / "policy.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({k: metadata[k] for k in ["experiment", "best_epoch", "frames"]}, indent=2))


if __name__ == "__main__":
    main()
