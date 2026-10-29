#!/usr/bin/env python3
"""Collect or evaluate seeded episodes with Isaac's python.sh in fresh processes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", required=True, help="Absolute path to Isaac Sim 5.1 python.sh")
    p.add_argument("--mode", choices=["collect", "evaluate"], required=True)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--seed-start", type=int)
    p.add_argument("--duration", type=float, default=90)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policy", type=Path)
    p.add_argument("--stress", action="store_true")
    p.add_argument("--route", choices=["outer", "inner"], default="outer")
    p.add_argument("--missions", action="store_true")
    p.add_argument("--detector", type=Path)
    p.add_argument("--calibration", type=Path)
    a = p.parse_args()
    if a.mode == "evaluate" and not a.policy:
        p.error("evaluate requires --policy")
    if a.missions and (a.mode != "evaluate" or not a.detector):
        p.error("--missions requires --mode evaluate --detector ...")
    if a.count <= 0:
        p.error("count must be positive")
    first = a.seed_start if a.seed_start is not None else (1000 if a.mode == "collect" else 9000)
    if a.mode == "evaluate":
        metadata = json.loads(a.policy.with_suffix(".json").read_text())
        used = set(str(seed) for groups in metadata["splits"].values() for seed in groups)
        # Fine-tuned policies keep the synthetic seeds of every ancestor.
        used |= set(str(seed) for seed in metadata.get("synthetic_seed_history", []))
        if a.missions:
            detector_meta = json.loads(a.detector.with_suffix(".json").read_text())
            used |= set(str(seed) for seed in detector_meta.get("training_scene_seeds", []))
        if any(str(seed) in used for seed in range(first, first + a.count)):
            p.error("Evaluation seeds overlap training, validation or offline test scenes")
    a.output.mkdir(parents=True, exist_ok=True)
    sim_mode = "expert" if a.mode == "collect" else "onnx"
    suffix = "-missions" if a.missions else ""
    existing = [
        a.output / f"{sim_mode}-{a.route}-seed{seed}{suffix}"
        for seed in range(first, first + a.count)
        if (a.output / f"{sim_mode}-{a.route}-seed{seed}{suffix}").exists()
    ]
    if existing:
        p.error(
            f"Episode already exists: {existing[0]}; choose a new output directory or seed range"
        )
    statuses = []
    for seed in range(first, first + a.count):
        command = [
            a.python,
            str(Path(__file__).with_name("run_sim.py")),
            "--headless",
            "--mode",
            "expert" if a.mode == "collect" else "onnx",
            "--seed",
            str(seed),
            "--randomize",
            "--route",
            a.route,
            "--duration",
            str(a.duration),
            "--output",
            str(a.output.resolve()),
        ]
        if a.mode == "collect":
            command.append("--record")
        else:
            command += ["--policy", str(a.policy.resolve())]
        if a.stress:
            command.append("--stress")
        if a.missions:
            command += ["--missions", "--detector", str(a.detector.resolve())]
            if a.calibration:
                command += ["--calibration", str(a.calibration.resolve())]
        with (a.output / f"process-seed{seed}.log").open("w") as log:
            try:
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=a.duration * 20 + 600
                )
                status = result.returncode
            except subprocess.TimeoutExpired:
                status = "timeout"
        statuses.append({"seed": seed, "process_exit": status})
        print(json.dumps(statuses[-1]), flush=True)
    (a.output / "batch.json").write_text(json.dumps(statuses, indent=2))
    if any(row["process_exit"] != 0 for row in statuses):
        sys.exit("Some episodes failed; inspect process logs and summary.json")


if __name__ == "__main__":
    main()
