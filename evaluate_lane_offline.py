#!/usr/bin/env python3
"""Offline lane-policy error of the EXPORTED ONNX on one fixed split.

Validation is for choosing models/learning rates. The test split needs --final and
every test access is appended to a ledger, so repeated test peeking is visible.
Offline MAE is not closed-loop driving performance.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from dtwin.policy import OnnxPolicy


def synthetic_rows(data, metadata, split):
    from train_policy import read_episodes

    groups = set(metadata["splits"][split])
    rows = []
    for episode in read_episodes(data):
        if episode["group"] not in groups:
            continue
        sim = episode["metadata"]["config"]["simulation"]
        for frame in episode["frames"]:
            target = [
                frame["linear_m_s"] / sim["max_speed_m_s"],
                frame["angular_rad_s"] / sim["max_yaw_rad_s"],
            ]
            rows.append((episode["path"] / frame["image"], target))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("policy", type=Path, help="policy.onnx with policy.json")
    p.add_argument("--split", choices=["validation", "test"], required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", type=Path, help="Synthetic episodes root (seed split)")
    source.add_argument("--real-manifest", type=Path)
    p.add_argument("--final", action="store_true", help="Required to touch the test split")
    p.add_argument("--purpose", default="", help="Recorded in the test ledger")
    p.add_argument("--turn-threshold", type=float, default=0.2,
                   help="|normalized angular label| at or above this counts as a turning frame")
    a = p.parse_args()
    if a.split == "test" and not a.final:
        p.error("Test is reserved for the final evaluation: fix settings, then pass --final")
    policy = OnnxPolicy(a.policy)
    metadata = policy.metadata
    if a.synthetic:
        rows = synthetic_rows(a.synthetic, metadata, a.split)
        manifest_id = None
    else:
        from dtwin.real_data import load_manifest, split_frames

        manifest = load_manifest(a.real_manifest)
        manifest_id = manifest["manifest_id"]
        used = (metadata.get("real_data") or {}).get("manifest_id")
        if used and used != manifest_id:
            p.error(f"Policy trained with manifest {used}, not {manifest_id}")
        rows = [
            (item["path"] / item["row"]["image"], item["row"]["command"]["action"])
            for item in split_frames(manifest, a.split, need="lane")
        ]
    if not rows:
        p.error("No labeled frames in this split")
    predictions = np.asarray([policy.action(np.asarray(Image.open(path).convert("RGB"))) for path, _ in rows])
    targets = np.asarray([target for _, target in rows], dtype=float)
    errors = np.abs(predictions - targets)
    # Turns are rare, so the overall mean hides missed curves: report them separately.
    turning = np.abs(targets[:, 1]) >= a.turn_threshold
    by_label = {}
    for name, mask in [("straight", ~turning), ("turning", turning)]:
        if mask.any():
            by_label[name] = dict(frames=int(mask.sum()), normalized_mae=errors[mask].mean(0).tolist())
    if turning.any():
        t, pr = targets[turning, 1], predictions[turning, 1]
        by_label["turning"].update(
            same_direction=float((np.sign(pr) == np.sign(t)).mean()),
            mean_abs_target_angular=float(np.abs(t).mean()),
            mean_abs_predicted_angular=float(np.abs(pr).mean()),
        )
    report = dict(
        policy=str(a.policy),
        experiment=metadata.get("experiment"),
        source="synthetic" if a.synthetic else "real",
        split=a.split,
        real_manifest_id=manifest_id,
        frames=len(rows),
        normalized_mae=errors.mean(0).tolist(),
        normalized_p95_abs_error=np.percentile(errors, 95, axis=0).tolist(),
        by_label=dict(turn_threshold=a.turn_threshold, **by_label),
        warning="Offline label error only; report closed-loop driving separately",
    )
    if a.synthetic:
        report["physical_mae"] = dict(
            linear_m_s=report["normalized_mae"][0] * metadata["max_speed_m_s"],
            angular_rad_s=report["normalized_mae"][1] * metadata["max_yaw_rad_s"],
        )
    if a.split == "test":
        if a.real_manifest:
            from dtwin.real_data import record_test_access

            ledger, previous = record_test_access(a.real_manifest, a.purpose, a.policy, report)
        else:
            ledger = Path(a.policy).with_name("test_access.jsonl")
            previous = len(ledger.read_text().splitlines()) if ledger.exists() else 0
            with ledger.open("a", encoding="utf-8") as file:
                entry = dict(time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             purpose=a.purpose, result=report)
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        report["test_ledger"] = str(ledger)
        report["previous_test_accesses"] = previous
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
