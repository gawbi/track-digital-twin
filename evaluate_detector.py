#!/usr/bin/env python3
"""Per-class precision/recall of the EXPORTED detector through the runtime adapter.

--split validation --select   choose the confidence threshold (macro F1) and store it
                              in the detector .json next to the ONNX file.
--split test --final          final report with the stored threshold; logged to a ledger.
Boxes match when the class agrees and IoU >= --iou.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from dtwin.perception import CLASSES, OnnxDetector, iou

THRESHOLDS = [round(t, 2) for t in np.arange(0.2, 0.96, 0.05)]


def ground_truth(label_path, width, height):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if line.strip():
                cls, cx, cy, w, h = (float(v) for v in line.split())
                boxes.append((CLASSES[int(cls)], [(cx - w / 2) * width, (cy - h / 2) * height,
                                                   (cx + w / 2) * width, (cy + h / 2) * height]))
    return boxes


def score(predictions, truths, threshold, min_iou):
    counts = {c: dict(tp=0, fp=0, fn=0) for c in CLASSES}
    for preds, gts in zip(predictions, truths):
        used = set()
        for p in sorted((p for p in preds if p["confidence"] >= threshold), key=lambda p: -p["confidence"]):
            match = max(
                ((iou(p["xyxy"], box), i) for i, (cls, box) in enumerate(gts)
                 if cls == p["class"] and i not in used),
                default=(0, None),
            )
            if match[0] >= min_iou:
                used.add(match[1])
                counts[p["class"]]["tp"] += 1
            else:
                counts[p["class"]]["fp"] += 1
        for i, (cls, _) in enumerate(gts):
            if i not in used:
                counts[cls]["fn"] += 1
    per_class = {}
    for cls, c in counts.items():
        if c["tp"] + c["fn"] == 0 and c["fp"] == 0:
            continue
        precision = c["tp"] / max(1, c["tp"] + c["fp"])
        recall = c["tp"] / max(1, c["tp"] + c["fn"])
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        per_class[cls] = dict(c, precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4))
    present = [v["f1"] for k, v in per_class.items() if v["tp"] + v["fn"] > 0]
    return dict(macro_f1=round(float(np.mean(present)), 4) if present else 0.0, per_class=per_class)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("detector", type=Path, help="best.onnx with best.json")
    p.add_argument("dataset", type=Path, help="Folder with data.yaml (images/, labels/)")
    p.add_argument("--split", choices=["validation", "test"], required=True)
    p.add_argument("--select", action="store_true", help="Store the best validation threshold")
    p.add_argument("--final", action="store_true")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--purpose", default="")
    a = p.parse_args()
    if a.split == "test" and not a.final:
        p.error("Test is reserved for the final evaluation: fix settings, then pass --final")
    if a.select and a.split != "validation":
        p.error("Thresholds are selected on validation only")
    folder = "val" if a.split == "validation" else "test"
    images = sorted((a.dataset / "images" / folder).glob("*.jpg"))
    if not images:
        p.error("No images in this split")
    detector = OnnxDetector(a.detector, confidence=min(THRESHOLDS))
    predictions, truths = [], []
    for path in images:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        predictions.append(detector.detect(rgb))
        truths.append(ground_truth(a.dataset / "labels" / folder / (path.stem + ".txt"), rgb.shape[1], rgb.shape[0]))
    stored = detector.metadata.get("confidence_threshold")
    report = dict(detector=str(a.detector), dataset=str(a.dataset), split=a.split, images=len(images),
                  experiment=detector.metadata.get("experiment"))
    if a.split == "validation":
        sweep = {t: score(predictions, truths, t, a.iou) for t in THRESHOLDS}
        best = max(sweep, key=lambda t: sweep[t]["macro_f1"])
        report.update(sweep_macro_f1={t: s["macro_f1"] for t, s in sweep.items()}, best_threshold=best,
                      at_best=sweep[best])
        if a.select:
            meta_path = a.detector.with_suffix(".json")
            meta = json.loads(meta_path.read_text())
            meta.update(confidence_threshold=best, threshold_selected_on=dict(
                dataset=str(a.dataset), split="validation", macro_f1=sweep[best]["macro_f1"]))
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    else:
        if stored is None:
            p.error("Select a threshold on validation first (--split validation --select)")
        report.update(threshold=stored, result=score(predictions, truths, stored, a.iou))
        ledger = a.dataset / "test_access.jsonl"
        previous = len(ledger.read_text().splitlines()) if ledger.exists() else 0
        with ledger.open("a", encoding="utf-8") as file:
            file.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                       purpose=a.purpose, result=report), ensure_ascii=False) + "\n")
        report.update(test_ledger=str(ledger), previous_test_accesses=previous)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
