#!/usr/bin/env python3
"""Fix the train/validation/test split of real sessions once, by source recording.

Prints frame/session counts, per-split class counts, empty classes, exact/near
duplicates and split limitations. Refuses to overwrite an existing manifest: new
sessions mean a new manifest version, never a silent reshuffle.
"""

import argparse
import json
from pathlib import Path

from dtwin.real_data import build_manifest, write_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sessions", type=Path, nargs="+", help="Session folders or their parent")
    p.add_argument("--output", type=Path, required=True, help="manifest JSON (not overwritten)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--assign",
        action="append",
        default=[],
        metavar="SOURCE_RECORDING=SPLIT",
        help="Pin a recording to train/validation/test (e.g. a different venue to test)",
    )
    p.add_argument("--near-duplicate-bits", type=int, default=2)
    a = p.parse_args()
    explicit = dict(item.split("=", 1) for item in a.assign)
    manifest = build_manifest(a.sessions, a.seed, explicit, a.near_duplicate_bits)
    write_manifest(manifest, a.output)
    report = manifest["report"]
    print(
        json.dumps(
            dict(
                manifest=str(a.output),
                manifest_id=manifest["manifest_id"],
                assignment=manifest["assignment"],
                **{k: report[k] for k in report if k != "class_counts"},
                limitations=manifest["limitations"],
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("class counts per split:")
    for split, counts in report["class_counts"].items():
        print(f"  {split:10s} " + " ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
