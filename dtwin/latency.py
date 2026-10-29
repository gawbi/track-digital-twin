"""Latency and control-period statistics from realtime.Telemetry JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def stats_ms(values_ns):
    if not len(values_ns):
        return None
    values = np.asarray(values_ns, dtype=np.float64) / 1e6
    return dict(
        n=int(len(values)),
        p50=round(float(np.percentile(values, 50)), 3),
        p95=round(float(np.percentile(values, 95)), 3),
        max=round(float(values.max()), 3),
    )


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize(path_or_records, warmup_s=3.0, control_hz=50.0):
    records = load(path_or_records) if isinstance(path_or_records, (str, Path)) else path_or_records
    ticks = [r for r in records if r["kind"] == "tick"]
    if not ticks:
        return dict(error="no control ticks recorded")
    start = ticks[0]["start_ns"] + warmup_s * 1e9
    ticks = [r for r in ticks if r["start_ns"] >= start]
    frames = [r for r in records if r["kind"] == "frame" and r["received_ns"] >= start]
    results = [r for r in records if r["kind"] == "result" and r["started_ns"] >= start]
    period_ns = 1e9 / control_hz
    intervals = np.diff([r["start_ns"] for r in ticks]) if len(ticks) > 1 else []
    lateness = [r["lateness_ns"] for r in ticks]
    clocks = sorted({r["capture_clock"] for r in results})
    first_use = [r for r in ticks if r.get("first_use")]
    stage_names = sorted({k for r in results for k in r.get("stages", {})})
    reasons = {}
    for r in ticks:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    dropped = [r["dropped_total"] for r in frames]
    return dict(
        warmup_excluded_s=warmup_s,
        capture_clock=clocks,
        end_to_end_reference=(
            "camera sensor timestamp"
            if clocks == ["sensor"]
            else "frame RECEIVE time (no sensor timestamp)"
            if clocks == ["received"]
            else f"mixed/simulated clocks {clocks}"
        ),
        camera=dict(
            frames=len(frames),
            interval_ms=stats_ms(np.diff([r["received_ns"] for r in frames])) if len(frames) > 1 else None,
            capture_to_receive_ms=stats_ms([r["received_ns"] - r["capture_ns"] for r in frames]),
            dropped_unprocessed_frames=(dropped[-1] - dropped[0]) if dropped else 0,
        ),
        perception=dict(
            results=len(results),
            queue_wait_ms=stats_ms([r["started_ns"] - r["received_ns"] for r in results]),
            total_ms=stats_ms([r["finished_ns"] - r["started_ns"] for r in results]),
            capture_to_result_ms=stats_ms([r["finished_ns"] - r["capture_ns"] for r in results]),
            stages_ms={
                name: stats_ms([r["stages"][name] for r in results if name in r.get("stages", {})])
                for name in stage_names
            },
        ),
        end_to_end_capture_to_motor_ms=stats_ms([r["applied_ns"] - r["capture_ns"] for r in first_use]),
        control=dict(
            target_period_ms=round(period_ns / 1e6, 3),
            ticks=len(ticks),
            interval_ms=stats_ms(intervals),
            lateness_ms=stats_ms(lateness),
            deadline_misses=int(sum(np.asarray(intervals) > 1.5 * period_ns)),
            skipped_ticks=sum(
                r["count"] for r in records if r["kind"] == "skipped_ticks"
            ),
            reasons=reasons,
            applied_fraction=round(reasons.get("applied", 0) / len(ticks), 4),
        ),
        faults=[r for r in records if r["kind"] == "fault"],
        note="Linux/Python timing is best effort; judge with braking distance and speed",
    )


TRIAL_FIELDS = {"trial", "route", "attempted_at", "result", "evidence"}


def summarize_trials(path):
    """Operator-recorded closed-loop trials. Results without evidence are flagged."""
    rows = load(path)
    for row in rows:
        missing = TRIAL_FIELDS - set(row)
        if missing:
            raise ValueError(f"Trial {row.get('trial')} is missing {sorted(missing)}")
        if row["result"] not in {"completed", "offroad", "stopped", "aborted"}:
            raise ValueError("result must be completed/offroad/stopped/aborted")
    missions = {}
    for row in rows:
        for kind, outcome in row.get("missions", {}).items():
            entry = missions.setdefault(kind, dict(completed=0, duplicate=0, failed=0, wrong_place=0))
            for key in entry:
                entry[key] += int(outcome.get(key, 0))
    by_route = {}
    for row in rows:
        route = by_route.setdefault(row["route"], dict(attempts=0, completed=0))
        route["attempts"] += 1
        route["completed"] += row["result"] == "completed"
    return dict(
        attempts=len(rows),
        completed=sum(r["result"] == "completed" for r in rows),
        results={k: sum(r["result"] == k for r in rows) for k in {r["result"] for r in rows}},
        by_route=by_route,
        missions=missions,
        stop_reasons={
            k: sum(r.get("stop_reason") == k for r in rows)
            for k in {r.get("stop_reason") for r in rows if r.get("stop_reason")}
        },
        without_evidence=[r["trial"] for r in rows if not r["evidence"]],
    )
