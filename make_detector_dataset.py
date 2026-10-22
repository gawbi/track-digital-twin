#!/usr/bin/env python3
"""Convert synchronized renderer boxes into a YOLO dataset, split by scene seed."""

import argparse
import json
import os
import shutil
from pathlib import Path

from dtwin.perception import CLASSES, yolo_label
from train_policy import read_episodes, split_groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("--output", type=Path, default=Path("datasets/signs"))
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    episodes = read_episodes(a.data)
    splits = split_groups(episodes, a.seed)
    a.output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for split, groups in splits.items():
        target = "val" if split == "validation" else split
        images = a.output / "images" / target
        labels = a.output / "labels" / target
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        counts[split] = {name: 0 for name in CLASSES}
        for eidx, episode in enumerate(episodes):
            if episode["group"] not in groups:
                continue
            c = episode["metadata"]["config"]["camera"]
            for frame in episode["frames"]:
                if "detections" not in frame:
                    raise ValueError("Missing boxes: recollect using this version of run_sim.py")
                name = f"e{eidx}-{frame['frame']:06d}"
                source = (episode["path"] / frame["image"]).resolve()
                try:
                    os.link(source, images / (name + ".jpg"))
                except OSError:
                    shutil.copy2(source, images / (name + ".jpg"))
                lines = []
                for box in frame["detections"]:
                    values = yolo_label(box, c["width"], c["height"])
                    counts[split][box["class"]] += 1
                    lines.append(str(values[0]) + " " + " ".join(f"{v:.7f}" for v in values[1:]))
                (labels / (name + ".txt")).write_text("\n".join(lines) + "\n" if lines else "")
    # JSON is a valid YAML subset, supported by the YOLO YAML loader.
    data = {
        "path": str(a.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASSES,
    }
    (a.output / "data.yaml").write_text(json.dumps(data, indent=2))
    (a.output / "split.json").write_text(
        json.dumps({"seed": a.seed, "groups": splits, "counts": counts}, indent=2)
    )
    print(json.dumps(counts, indent=2))
    if any(counts["train"][name] == 0 for name in CLASSES):
        print(
            "Some classes have no training examples. Change prop placements/views and collect more data before training."
        )


if __name__ == "__main__":
    main()
