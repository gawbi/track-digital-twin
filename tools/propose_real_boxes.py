#!/usr/bin/env python3
"""Draft sign/signal boxes on real car-camera frames from color and shape.

Candidates: red or blue sign faces above the horizon (flat floor circles are
rejected by aspect/position), and lit signal lamps grown to their dark housing.
Each candidate gets a crop in a numbered contact sheet; a reviewer assigns the
class (or rejects it) in a JSON file, then --write-labels emits YOLO labels.
These are DRAFT labels until a person has checked the overlays.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtwin.perception import CLASSES  # noqa: E402


def components(mask, min_area):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [stats[i] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def housing(dark, x0, y0, x1, y1):
    """Grow a lamp box over the black signal housing; stop at the thin pole."""
    height, width = dark.shape
    w = x1 - x0
    left, right = max(0, x0 - w), min(width, x1 + w)
    row_dark = lambda y: dark[y, left:right].mean() if 0 <= y < height else 0  # noqa: E731
    top, bottom = y0, y1
    while top > 0 and row_dark(top - 1) > 0.35:
        top -= 1
    while bottom < height - 1 and row_dark(bottom + 1) > 0.35:
        bottom += 1
    cols = dark[top : bottom + 1].mean(0)
    xs = [x for x in range(left, right) if cols[x] > 0.5]
    if xs:
        left, right = min(xs), max(xs) + 1
    return [int(left), int(top), int(right), int(bottom) + 1]


def propose(rgb, horizon=300):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    # Thresholds measured on the 2026-10-14 car photos: the STOP red is pale (S~48).
    red = ((h <= 14) | (h >= 165)) & (s >= 40) & (v >= 90)
    blue = (h >= 96) & (h <= 125) & (s >= 110) & (v >= 50)
    dark = (v < 95).astype(float)
    candidates = []
    for name, mask in [("red", red), ("blue", blue)]:
        # Close gaps from white arrows/text so one sign face is one component.
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        for x, y, w, hh, area in components(mask, 30):
            aspect = w / max(hh, 1)
            if y + hh > horizon + 25 or not 0.55 <= aspect <= 1.6 or w < 7 or w > 160:
                continue  # floor spots, thin edges, walls/clothes
            candidates.append(dict(kind=name, xyxy=[int(x), int(y), int(x + w), int(y + hh)],
                                   fill=round(area / (w * hh), 3)))
    for name, lo, hi in [("lamp_green", 40, 95), ("lamp_red", None, None), ("lamp_yellow", 15, 35)]:
        if lo is None:
            lamp = ((h <= 10) | (h >= 165)) & (s >= 60) & (v >= 200)
        else:
            lamp = (h >= lo) & (h <= hi) & (s >= 60) & (v >= 200)
        for x, y, w, hh, area in components(lamp, 6):
            if y + hh > horizon + 25 or w > 40 or not 0.5 <= w / max(hh, 1) <= 2:
                continue
            box = housing(dark, x, y, x + w, y + hh)
            if (box[3] - box[1]) < 1.6 * hh or (box[2] - box[0]) < w:
                continue  # no dark housing around it
            candidates.append(dict(kind=name, xyxy=box, lamp=[int(x), int(y), int(x + w), int(y + hh)]))
    return candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sessions", type=Path, help="Folder of imported/recorded sessions")
    p.add_argument("--output", type=Path, required=True, help="Review folder")
    p.add_argument("--write-labels", type=Path, help="Reviewed decisions JSON -> YOLO labels")
    a = p.parse_args()
    if a.write_labels:
        decisions = json.loads(a.write_labels.read_text())
        proposals = json.loads((a.output / "proposals.json").read_text())
        frames = {}
        for item in proposals:
            frames.setdefault((item["session"], item["image"]), [])
            label = decisions.get(str(item["id"]))
            if label and label != "reject":
                frames[(item["session"], item["image"])].append((label, item["xyxy"]))
        for extra in decisions.get("added", []):
            frames.setdefault((extra["session"], extra["image"]), []).append((extra["class"], extra["xyxy"]))
        for (session, image), boxes in frames.items():
            path = a.sessions / session / image
            width, height = cv2.imread(str(path)).shape[1::-1]
            labels = a.sessions / session / "labels"
            labels.mkdir(exist_ok=True)
            lines = [
                f"{CLASSES.index(c)} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
                f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"
                for c, (x0, y0, x1, y1) in boxes
            ]
            (labels / (Path(image).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"Wrote labels for {len(frames)} frames (draft until a person checks overlays)")
        return
    a.output.mkdir(parents=True, exist_ok=True)
    proposals, crops = [], []
    for session in sorted(p for p in a.sessions.iterdir() if (p / "frames.jsonl").exists()):
        for line in (session / "frames.jsonl").read_text().splitlines():
            row = json.loads(line)
            bgr = cv2.imread(str(session / row["image"]))
            for c in propose(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)):
                c.update(id=len(proposals), session=session.name, image=row["image"])
                proposals.append(c)
                x0, y0, x1, y1 = c["xyxy"]
                pad = max(4, (x1 - x0) // 3)
                crop = bgr[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad]
                crops.append((c["id"], cv2.resize(crop, (72, 72))))
    (a.output / "proposals.json").write_text(json.dumps(proposals, indent=1))
    per_sheet, cols = 80, 10
    for start in range(0, len(crops), per_sheet):
        sheet = np.full((8 * 88, cols * 76, 3), 255, np.uint8)
        for k, (cid, crop) in enumerate(crops[start : start + per_sheet]):
            r, c = divmod(k, cols)
            sheet[r * 88 + 14 : r * 88 + 86, c * 76 + 2 : c * 76 + 74] = crop
            cv2.putText(sheet, str(cid), (c * 76 + 2, r * 88 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1)
        cv2.imwrite(str(a.output / f"sheet-{start // per_sheet:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    kinds = {}
    for c in proposals:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(json.dumps(dict(proposals=len(proposals), sheets=(len(crops) + per_sheet - 1) // per_sheet, kinds=kinds)))


if __name__ == "__main__":
    main()
