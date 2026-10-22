#!/usr/bin/env python3
"""Aggregate camera-policy closed-loop results, including failed episode folders."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path)
    a = p.parse_args()
    rows = []
    for directory in sorted(a.runs.iterdir()):
        if not directory.is_dir():
            continue
        path = directory / "summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
        else:
            rows.append(
                {
                    "status": "missing_summary",
                    "success": False,
                    "mode": "onnx" if directory.name.startswith("onnx-") else "unknown",
                    "mission_enabled": directory.name.endswith("-missions"),
                }
            )
    if not rows:
        p.error("No episode directories")
    policies = [
        r for r in rows if r.get("mode") == "onnx" and not r.get("policy_uses_ground_truth", False)
    ]
    if len(policies) != len(rows):
        p.error("Teacher runs are present. Evaluate camera-only ONNX runs separately.")
    mission_flags = {bool(r.get("mission_enabled")) for r in rows}
    if len(mission_flags) != 1:
        p.error("Lane-only and mission-controlled runs must be evaluated separately")
    missions = True in mission_flags
    report = {
        "episodes": len(rows),
        "successful_episodes": sum(bool(r.get("success")) for r in rows),
        "success_rate": sum(bool(r.get("success")) for r in rows) / len(rows),
        "statuses": {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}},
        "minimum_clearance_m": min(
            (r["min_clearance_m"] for r in rows if "min_clearance_m" in r), default=None
        ),
        "scope": (
            "lane + configured required missions; RGB/wheel odometry/map control"
            if missions
            else "lane following only; does not establish sign/signal mission performance"
        ),
    }
    if missions:
        report["mission_completions"] = {
            kind: sum(r.get("mission_counts", {}).get(kind, 0) for r in rows)
            for kind in ["horn", "stop", "speed20", "signal"]
        }
        report["mission_faults"] = {
            fault: sum(r.get("mission_fault") == fault for r in rows)
            for fault in {r.get("mission_fault") for r in rows}
            if fault is not None
        }
    print(json.dumps(report, indent=2))
    (a.runs / "evaluation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
