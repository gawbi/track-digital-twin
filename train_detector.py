#!/usr/bin/env python3
"""Optional small detector training in a separate CUDA environment, outside Isaac."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path, help="data.yaml from make_detector_dataset.py")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="0")
    a = p.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.model)
    if getattr(model.model, "end2end", False):
        p.error("The runtime adapter supports YOLO11 raw detection exports; use --model yolo11n.pt")
    model.train(
        data=str(a.data),
        epochs=a.epochs,
        imgsz=640,
        batch=a.batch_size,
        device=a.device,
        seed=42,
        deterministic=True,
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.4,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.3,
        close_mosaic=10,
        project="models/detector",
        name="train",
        exist_ok=False,
    )
    model.val(data=str(a.data), split="test", device=a.device)
    exported = Path(
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=False, nms=False, opset=17)
    )
    classes = [model.names[i] for i in range(len(model.names))]
    split_path = a.data.parent / "split.json"
    scenes = json.loads(split_path.read_text())["groups"] if split_path.exists() else {}
    exported.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "yolo11_raw_xywh_classes",
                "classes": classes,
                "input": "uint8 RGB -> letterbox 640 -> NCHW float32 /255",
                "nms": False,
                "training_scene_seeds": sorted(
                    {str(seed) for seeds in scenes.values() for seed in seeds}
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
