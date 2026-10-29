#!/usr/bin/env python3
"""Build a YOLO dataset: synthetic renderer boxes split by scene seed, or labeled
real frames split by a fixed session manifest (--real-manifest)."""

import argparse
import json
import os
import shutil
from pathlib import Path

from dtwin.perception import CLASSES, yolo_label
from train_policy import read_episodes, split_groups


def link(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def export_real(manifest_path, output):
    from dtwin.real_data import load_manifest, split_frames

    manifest = load_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for split in ["train", "validation", "test"]:
        target = "val" if split == "validation" else split
        (output / "images" / target).mkdir(parents=True)
        (output / "labels" / target).mkdir(parents=True)
        counts[split] = {name: 0 for name in CLASSES}
        counts[split]["_frames"] = 0
        for item in split_frames(manifest, split, need="detector"):
            name = f"{item['session_id']}-{item['row']['frame_id']:06d}"
            link((item["path"] / item["row"]["image"]).resolve(), output / "images" / target / (name + ".jpg"))
            lines = [f"{b[0]} " + " ".join(f"{v:.7f}" for v in b[1:]) for b in item["boxes"]]
            for box in item["boxes"]:
                counts[split][CLASSES[box[0]]] += 1
            counts[split]["_frames"] += 1
            (output / "labels" / target / (name + ".txt")).write_text(
                "\n".join(lines) + "\n" if lines else ""
            )
    return dict(source="real", manifest_id=manifest["manifest_id"], groups={}, counts=counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path, nargs="?", help="Synthetic episodes root")
    p.add_argument("--real-manifest", type=Path, help="Fixed real split (labeled frames only)")
    p.add_argument("--output", type=Path, default=Path("datasets/signs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stride", type=int, default=1, help="Use every Nth synthetic frame")
    a = p.parse_args()
    if bool(a.data) == bool(a.real_manifest):
        p.error("Give either synthetic DATA or --real-manifest")
    if a.real_manifest:
        split = export_real(a.real_manifest, a.output)
        write_yaml(a.output)
        (a.output / "split.json").write_text(json.dumps(split, indent=2))
        print(json.dumps(split["counts"], indent=2))
        return
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
            for frame in episode["frames"][:: a.stride]:
                if "detections" not in frame:
                    raise ValueError("Missing boxes: recollect using this version of run_sim.py")
                name = f"e{eidx}-{frame['frame']:06d}"
                source = (episode["path"] / frame["image"]).resolve()
                link(source, images / (name + ".jpg"))
                lines = []
                for box in frame["detections"]:
                    values = yolo_label(box, c["width"], c["height"])
                    counts[split][box["class"]] += 1
                    lines.append(str(values[0]) + " " + " ".join(f"{v:.7f}" for v in values[1:]))
                (labels / (name + ".txt")).write_text("\n".join(lines) + "\n" if lines else "")
    write_yaml(a.output)
    (a.output / "split.json").write_text(
        json.dumps(
            {"source": "synthetic", "seed": a.seed, "stride": a.stride, "groups": splits,
             "counts": counts}, indent=2
        )
    )
    print(json.dumps(counts, indent=2))
    if any(counts["train"][name] == 0 for name in CLASSES):
        print(
            "Some classes have no training examples. Change prop placements/views and collect more data before training."
        )


def write_yaml(output):
    # JSON is a valid YAML subset, supported by the YOLO YAML loader.
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASSES,
    }
    (output / "data.yaml").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
