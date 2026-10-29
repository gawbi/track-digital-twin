#!/usr/bin/env python3
"""Run an exported detector on real sessions: per-session class counts, overlays
and a proposals.json usable by propose_real_boxes.py --write-labels for review.

Counts without reviewed labels are NOT accuracy; they only show what the model
fires on in real frames.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtwin.perception import OnnxDetector  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("detector", type=Path)
    p.add_argument("sessions", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--confidence", type=float, default=0.15, help="Low: proposals for review")
    p.add_argument("--overlay-every", type=int, default=4)
    a = p.parse_args()
    detector = OnnxDetector(a.detector, confidence=a.confidence)
    a.output.mkdir(parents=True, exist_ok=False)
    proposals, summary = [], {}
    for session in sorted(s for s in a.sessions.iterdir() if (s / "frames.jsonl").exists()):
        rows = [json.loads(line) for line in (session / "frames.jsonl").read_text().splitlines()]
        counts, tiles = {}, []
        for k, row in enumerate(rows):
            bgr = cv2.imread(str(session / row["image"]))
            detections = detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for d in detections:
                proposals.append(dict(id=len(proposals), session=session.name, image=row["image"],
                                      kind=d["class"], confidence=round(d["confidence"], 3),
                                      xyxy=[round(v, 1) for v in d["xyxy"]]))
                if d["confidence"] >= (detector.metadata.get("confidence_threshold") or 0.5):
                    counts[d["class"]] = counts.get(d["class"], 0) + 1
            if k % a.overlay_every == 0:
                for d in detections:
                    x0, y0, x1, y1 = (int(v) for v in d["xyxy"])
                    color = (0, 220, 0) if d["confidence"] >= 0.5 else (0, 160, 255)
                    cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(bgr, f"{d['class']} {d['confidence']:.2f}", (x0, max(10, y0 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                tiles.append(cv2.resize(bgr, (320, 240)))
        summary[session.name] = dict(frames=len(rows), detections_at_threshold=counts)
        for start in range(0, len(tiles), 12):
            chunk = tiles[start : start + 12] + [np.zeros_like(tiles[0])] * (12 - len(tiles[start : start + 12]))
            grid = np.vstack([np.hstack(chunk[r * 4 : r * 4 + 4]) for r in range(3)])
            cv2.imwrite(str(a.output / f"{session.name}-{start // 12:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (a.output / "proposals.json").write_text(json.dumps(proposals, indent=1))
    (a.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
