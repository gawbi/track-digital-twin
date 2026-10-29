#!/usr/bin/env python3
"""Summarize Pi runs: latency/period statistics and operator-recorded trials.

trials.jsonl: one JSON object per attempt, e.g.
{"trial": 3, "route": "outer", "attempted_at": "2026-11-03T15:10", "result": "completed",
 "missions": {"stop": {"completed": 1, "duplicate": 0, "failed": 0, "wrong_place": 0}},
 "stop_reason": null, "evidence": "videos/trial3.mp4", "run_dir": "runs/trial3"}
Results without evidence (video/observer note) are listed as unverified.
"""

import argparse
import json
from pathlib import Path

from dtwin.latency import summarize, summarize_trials


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="*", help="run_vehicle.py output folders")
    p.add_argument("--trials", type=Path)
    p.add_argument("--warmup", type=float, default=3.0)
    p.add_argument("--control-hz", type=float, default=50)
    a = p.parse_args()
    report = {}
    for run in a.runs:
        env = json.loads((run / "env.json").read_text()) if (run / "env.json").exists() else {}
        hz = env.get("settings", {}).get("control_hz", a.control_hz)
        report[str(run)] = dict(
            system=env.get("system_start"),
            system_end=env.get("system_end"),
            latency=summarize(run / "telemetry.jsonl", a.warmup, hz),
        )
    if a.trials:
        report["trials"] = summarize_trials(a.trials)
    if not report:
        p.error("Give run folders and/or --trials")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
