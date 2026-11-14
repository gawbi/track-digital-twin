#!/usr/bin/env python3
"""Gate 2: score lane policies in 원철's bench (Edwin29/AutoMobile, branch drive-v4-bench).

Runs the bench's own CNN suites — I_hybrid (17 conditions x 2 directions) and J_xcheck
(field-log camera, 4 seeds x 16) — for every --model, alone and optionally inside the hyb2
hybrid (A5b's seat swapped for the model). The bench checkout is not modified: models are
registered only inside this script's child processes. Same scoring and CNN speed (1.0) as
the bench reports, so the numbers compare with 원철's tables.

    python tools/bench_gate.py --bench ../AutoMobile-bench --model A5b=models/lane-v5b \
        --model A6=models/lane-a6 --hybrid --workers 8 --out outputs/bench_gate/a6
    python tools/bench_gate.py --bench ../AutoMobile-bench --out outputs/bench_gate/a6 --report
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

GROUPS = ("I_hybrid", "J_xcheck")


def specs(bench, models, hybrid):
    sys.path.insert(0, str(bench))
    from bench.matrix import specs as bench_specs

    out = []
    for sp in bench_specs():
        if sp["group"] not in GROUPS or sp["controller"] != "cnnA5b":
            continue
        for name, path in models.items():
            for arm in ("cnn", "hyb2") if hybrid else ("cnn",):
                slug = f"{sp['group']}__{arm}_{name}__{sp['vehicle'].replace(':', '')}__{'+'.join(sp['effects']) or 'none'}__{sp['direction']}__{sp['variation']}__s{sp['seed']}"
                out.append(dict(sp, arm=arm, model=name, model_path=str(path), slug=slug))
    return out


def run_one(spec, bench, out):
    dst = Path(out) / spec["slug"]
    if (dst / "summary.json").exists():
        return spec["slug"], "cached"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec), "--bench", str(bench), "--out", str(dst)]
    env = dict(os.environ, BENCH_CNN_SPEED="1.0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run(cmd, cwd=bench, env=env, capture_output=True, text=True)
    return spec["slug"], (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else "FAIL " + r.stderr[-400:]


def child(spec, bench, out):
    """Register the model as the bench's CNN (and as hyb2's CNN seat), then run one case."""
    sys.path.insert(0, str(bench))
    sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
    import bench.controllers as C
    import bench.run as R
    from dtwin.policy import CONTRACTS

    folder = Path(spec["model_path"]).resolve()
    members = {"model": str(folder)}
    gap = CONTRACTS[json.loads((folder / "policy.json").read_text())["preprocess"]]
    if gap:  # two-frame model: the bench only knows single images, so feed the frame `gap` steps back
        from collections import deque

        import cv2
        import numpy as np

        def init(self, direction, start_hint, timing):
            # TeamCNN.__init__ without its single-image contract check (same loading otherwise)
            self.cv2 = cv2
            self.meta = json.loads((folder / "policy.json").read_text())
            self.net = cv2.dnn.readNetFromONNX(np.frombuffer((folder / "policy.onnx").read_bytes(), np.uint8))
            self.mix = json.loads((C.VENDOR_CNN / "configs" / "real_car.json").read_text())["duty_mixer"]
            self.source = f"track-digital-twin {folder.name} (two-frame), speed_scale {self.speed_scale}"
            self.history = deque(maxlen=gap + 1)

        def preprocess(self, rgb):
            now = C.TeamCNN._preprocess(self, rgb)
            self.history.append(now)
            return np.concatenate([now, self.history[0]], axis=1)

        members.update(__init__=init, _preprocess=preprocess)
    cnn = type(f"TeamCNN_{spec['model']}", (C.TeamCNN,), members)
    C.CONTROLLERS["cnnA5b"] = cnn  # hyb2 builds its CNN from this entry
    controller = "cnnA5b" if spec["arm"] == "cnn" else "hyb2"
    params = R.vehicle_from_args(spec["vehicle"], None)
    s = R.run(controller, params, spec["effects"], spec["direction"], spec["timing"], spec["seed"], 200.0, out,
              False, spec["variation"])
    print(json.dumps({k: s[k] for k in ("outcome", "verified_lap", "min_margin_cm", "outside_s")}, default=float))


def report(bench, out, models, hybrid):
    import numpy as np

    rows = []
    for sp in specs(bench, models, hybrid):
        p = Path(out) / sp["slug"] / "summary.json"
        if p.exists():
            rows.append(sp | {"r": json.loads(p.read_text())})
    lines = []
    for group in GROUPS:
        lines.append(f"\n## {group}\n")
        for arm in dict.fromkeys(r["arm"] for r in rows):
            for name in models:
                g = [r["r"] for r in rows if r["group"] == group and r["arm"] == arm and r["model"] == name]
                if not g:
                    continue
                m = np.array([x["min_margin_cm"] for x in g])
                lines.append(f"- {arm} {name}: laps {sum(x['verified_lap'] for x in g)}/{len(g)}, clean "
                             f"{sum(x['clean_lap'] for x in g)}/{len(g)}, min margin median {np.median(m):+.2f} "
                             f"worst {m.min():+.2f} cm, departed {sum(x['outcome'] == 'severe_departure' for x in g)}")
    text = "\n".join(lines)
    (Path(out) / "REPORT.md").write_text(text)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", type=Path, required=True, help="AutoMobile checkout on branch drive-v4-bench")
    p.add_argument("--model", action="append", default=[], help="NAME=folder with policy.onnx + policy.json")
    p.add_argument("--hybrid", action="store_true", help="also run each model inside hyb2")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", action="store_true")
    p.add_argument("--child", help=argparse.SUPPRESS)
    a = p.parse_args()
    bench = a.bench.resolve()
    if a.child:
        return child(json.loads(a.child), bench, a.out)
    models = dict(m.split("=", 1) for m in a.model) if a.model else json.loads((a.out / "models.json").read_text())
    models = {k: str(Path(v).resolve()) for k, v in models.items()}
    for path in models.values():
        if not (Path(path) / "policy.onnx").is_file():
            p.error(f"{path} has no policy.onnx")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "models.json").write_text(json.dumps(models, indent=1))
    if not a.report:
        todo = specs(bench, models, a.hybrid)
        with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
            for i, (slug, result) in enumerate(pool.map(lambda s: run_one(s, bench, a.out), todo), 1):
                print(f"[{i}/{len(todo)}] {slug}: {result}", flush=True)
    print(report(bench, a.out, models, a.hybrid))


if __name__ == "__main__":
    main()
