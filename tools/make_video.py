#!/usr/bin/env python3
"""Turn a recorded episode (run_sim.py --record) into an H.264 MP4 with overlays.

Shows the car camera with the applied command, teacher/policy mode, lap and
road-clearance status. If the episode also has chase-camera frames
(images_chase/, from run_sim.py --chase-video) they are placed side by side.
Needs imageio-ffmpeg (training environment).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def text(image, line, y, color=(255, 255, 255)):
    cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default="")
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--boxes", action="store_true", help="Draw detections (vision or renderer labels)")
    a = p.parse_args()
    import imageio_ffmpeg

    meta = json.loads((a.episode / "episode.json").read_text())
    summary = json.loads((a.episode / "summary.json").read_text())
    rows = [json.loads(line) for line in (a.episode / "frames.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "image" in r]
    if not rows:
        p.error("Episode has no recorded images (use --record)")
    chase = (a.episode / "images_chase").is_dir()
    mode = {"expert": "teacher (uses true pose)", "onnx": "camera policy (RGB only)"}.get(
        meta["mode"], meta["mode"]
    )
    width = 640 + (640 if chase else 0)
    writer = imageio_ffmpeg.write_frames(
        str(a.output), (width, 480), fps=a.fps, codec="libx264", pix_fmt_out="yuv420p",
        quality=7, macro_block_size=8,
    )
    writer.send(None)
    try:
        for row in rows:
            camera = cv2.imread(str(a.episode / row["image"]))
            if a.boxes:
                boxes = (row.get("autonomy") or {}).get("detections") or row.get("detections", [])
                for box in boxes:
                    x0, y0, x1, y1 = (int(v) for v in box["xyxy"])
                    cv2.rectangle(camera, (x0, y0), (x1, y1), (255, 0, 255), 2)
                    cv2.putText(camera, box["class"], (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 0, 255), 1, cv2.LINE_AA)
            text(camera, f"{a.title}  {mode}", 22)
            text(camera, f"t={row['time_s']:5.1f}s  v={row['linear_m_s']:.2f} m/s  w={row['angular_rad_s']:+.2f} rad/s", 46)
            ok = row.get("footprint_on_road", True)
            text(camera, f"on road: {'yes' if ok else 'NO'}  clearance {row.get('min_clearance_m', 0) * 100:.1f} cm",
                 70, (120, 255, 120) if ok else (80, 80, 255))
            frame = camera
            if chase:
                view = cv2.imread(str(a.episode / "images_chase" / Path(row["image"]).name))
                if view is None:
                    view = np.zeros_like(camera)
                text(view, "overhead view (evaluation only)", 22)
                frame = np.hstack([cv2.resize(view, (640, 480)), camera])
            writer.send(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        last = frame.copy()
        text(last, f"END: {summary.get('status')}  laps={summary.get('laps')}  offroad frames={summary.get('offroad_frames')}",
             470 - 10, (0, 255, 255))
        for _ in range(int(a.fps * 2)):
            writer.send(np.ascontiguousarray(cv2.cvtColor(last, cv2.COLOR_BGR2RGB)))
    finally:
        writer.close()
    print(json.dumps(dict(video=str(a.output), frames=len(rows), summary=summary.get("status"))))


if __name__ == "__main__":
    main()
