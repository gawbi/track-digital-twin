#!/usr/bin/env python3
"""README figures for the sim-to-real write-up (synthetic frames and summary numbers only).

  docs/images/curve_steering_{light,dark}.png  how much each lane policy turns on real
      turning frames vs the human label (from evaluate_lane_offline.py logs)
  docs/images/sim_{clean,harsh}.jpg            synthetic camera frames before/after the
      glare + harsh randomization

Real camera images are never written here: the public repository holds no real data.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = [  # (log name, label)
    ("lane-v4b", "A  Isaac Sim만"),
    ("lane-real-B", "B  실제만"),
    ("lane-real-C", "C  Isaac Sim → 실제"),
    ("lane-v5", "A5  Isaac Sim만 (반사·가혹)"),
    ("lane-real-C5", "C5  A5 → 실제"),
]
THEMES = {  # chart chrome from the dataviz reference palette, series slot 1
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", series="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", series="#3987e5"),
}


def curve_chart(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for path in ["/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
                 "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    rows, target = [], None
    for name, label in MODELS:
        log = ROOT / f"outputs/logs/eval-{name}-real-val-turns.log"
        if not log.exists():
            continue
        turning = json.loads(log.read_text())["by_label"]["turning"]
        rows.append((label, turning["mean_abs_predicted_angular"]))
        target = turning["mean_abs_target_angular"]
    if not rows:
        raise SystemExit("No evaluate_lane_offline.py logs with by_label")
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(rows) + 1.3), dpi=160)
        fig.patch.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        y = np.arange(len(rows))[::-1]
        values = [v for _, v in rows]
        ax.barh(y, values, height=0.5, color=t["series"], zorder=2)
        for yi, v in zip(y, values):
            ax.text(v + 0.012, yi, f"{v:.2f}", va="center", color=t["ink"], fontsize=10)
        ax.axvline(target, color=t["ink2"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
        ax.text(target - 0.01, len(rows) - 0.45, f"사람 운전 정답 {target:.2f}", ha="right", va="bottom",
                color=t["ink2"], fontsize=9.5)
        ax.set_yticks(y, [label for label, _ in rows], color=t["ink"], fontsize=10)
        ax.set_xlim(0, max(target, max(values)) * 1.18)
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.tick_params(axis="x", colors=t["muted"], labelsize=9)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=t["grid"], linewidth=0.8, zorder=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.set_xlabel("실제 커브 사진에서 모델이 꺾는 양 (평균 |회전 명령|, 실제 검증 세션)",
                      color=t["ink2"], fontsize=9.5)
        fig.tight_layout()
        fig.savefig(out / f"curve_steering_{mode}.png", facecolor=t["surface"])
        plt.close(fig)
    print("curve chart:", ", ".join(f"{label.split()[0]} {v:.2f}" for label, v in rows), f"target {target:.2f}")


def frame_strip(paths, out_path):
    from PIL import Image

    tiles = [Image.open(p).convert("RGB").resize((320, 240)) for p in paths]
    strip = Image.new("RGB", (320 * len(tiles) + 6 * (len(tiles) - 1), 240), (255, 255, 255))
    for i, tile in enumerate(tiles):
        strip.paste(tile, (i * 326, 0))
    strip.save(out_path, quality=88)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", nargs="*", type=Path, default=[], help="Synthetic frames before glare")
    p.add_argument("--harsh", nargs="*", type=Path, default=[], help="Synthetic frames with glare")
    a = p.parse_args()
    out = ROOT / "docs/images"
    out.mkdir(parents=True, exist_ok=True)
    curve_chart(out)
    for name, frames in [("sim_clean.jpg", a.clean), ("sim_harsh.jpg", a.harsh)]:
        if frames:
            frame_strip(frames, out / name)


if __name__ == "__main__":
    main()
