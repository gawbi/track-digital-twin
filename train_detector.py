#!/usr/bin/env python3
"""Train/fine-tune the 9-class YOLO11 detector in a separate CUDA environment.

--weights X.pt   starts a NEW training run from those weights (e.g. synthetic -> real).
--resume RUN/weights/last.pt  continues an interrupted run with its saved settings.
Learning rate is only honored with an explicit optimizer (Ultralytics 'auto' picks its
own); the applied values are read back from the trainer and stored in the metadata.
The test split is never evaluated here: use evaluate_detector.py --split test --final.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lineage(data_yaml, weights):
    split_path = Path(data_yaml).parent / "split.json"
    split = json.loads(split_path.read_text()) if split_path.exists() else {}
    seeds = {str(s) for groups in split.get("groups", {}).values() for s in groups}
    parent = None
    if weights and Path(weights).exists():
        candidates = [Path(weights).with_suffix(".json"), Path(weights).parent / "best.json"]
        meta = next((json.loads(c.read_text()) for c in candidates if c.exists()), None)
        if meta:
            seeds |= set(meta.get("training_scene_seeds", []))
        parent = dict(
            path=str(weights),
            sha256=sha256(weights),
            experiment=(meta or {}).get("experiment"),
            sources=(meta or {}).get("sources"),
        )
    source = dict(data=str(data_yaml), kind=split.get("source", "synthetic"))
    if split.get("manifest_id"):
        source["real_manifest_id"] = split["manifest_id"]
    return sorted(seeds), parent, source


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, nargs="?", help="data.yaml from make_detector_dataset.py")
    p.add_argument("--weights", default="yolo11n.pt", help="Start weights for a NEW run")
    p.add_argument("--resume", type=Path, help="last.pt of an interrupted run")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--optimizer", default="AdamW", choices=["SGD", "Adam", "AdamW", "auto"])
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--project", default="models/detector")
    p.add_argument("--name", default="train", help="Experiment folder name (never overwritten)")
    p.add_argument("--experiment", default="", help="Recorded label, e.g. A/B/C")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    from ultralytics import YOLO

    if a.resume:
        model = YOLO(str(a.resume))
        model.train(resume=True)
        data, init = Path(model.trainer.args.data), None
    else:
        if not a.data:
            p.error("data.yaml is required for a new training run")
        if a.optimizer == "auto":
            print("WARNING: optimizer=auto ignores --lr0; the trainer chooses the learning rate")
        model = YOLO(a.weights)
        if getattr(model.model, "end2end", False):
            p.error("The runtime adapter supports YOLO11 raw detection exports; use yolo11n.pt")
        model.train(
            data=str(a.data),
            epochs=a.epochs,
            imgsz=a.imgsz,
            batch=a.batch_size,
            device=a.device,
            optimizer=a.optimizer,
            lr0=a.lr0,
            seed=a.seed,
            deterministic=True,
            hsv_h=0.01,
            hsv_s=0.25,
            hsv_v=0.4,
            fliplr=0.0,
            flipud=0.0,
            mosaic=0.3,
            close_mosaic=10,
            project=a.project,
            name=a.name,
            exist_ok=False,
        )
        data, init = a.data, a.weights
    trainer = model.trainer
    exported = Path(
        model.export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=False, nms=False, opset=17)
    )
    classes = [model.names[i] for i in range(len(model.names))]
    seeds, parent, source = lineage(data, init)
    metadata = dict(
        format="yolo11_raw_xywh_classes",
        classes=classes,
        input="uint8 RGB -> letterbox {0} -> NCHW float32 /255".format(a.imgsz),
        nms=False,
        experiment=a.experiment,
        training_scene_seeds=seeds,
        sources=source,
        init_weights=parent if parent else str(init) if init else "resumed",
        resumed_from=str(a.resume) if a.resume else None,
        applied_args={
            k: getattr(trainer.args, k)
            for k in ["optimizer", "lr0", "lrf", "momentum", "weight_decay", "epochs", "batch", "imgsz", "seed"]
        },
        validation_metrics={k: float(v) for k, v in (trainer.metrics or {}).items()},
        save_dir=str(trainer.save_dir),
        test_evaluated=False,
        confidence_threshold=None,
        note="Pick confidence_threshold with evaluate_detector.py --split validation",
    )
    exported.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({k: metadata[k] for k in ["experiment", "applied_args", "save_dir"]}, indent=2))


if __name__ == "__main__":
    main()
