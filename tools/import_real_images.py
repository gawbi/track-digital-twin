#!/usr/bin/env python3
"""Import a folder of existing car-camera JPEGs as a real session (no commands).

Image bytes are copied unchanged. Capture times come from file names like
image_YYYYMMDD_HHMMSS_micro.jpg (wall clock, recorded as such). These sessions
carry no lane labels; they are for detector labels/evaluation.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtwin.actions import ACTION_SPACE  # noqa: E402
from dtwin.real_data import SESSION_SCHEMA  # noqa: E402

NAME = re.compile(r"(\d{8})_(\d{6})_(\d{6})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", type=Path)
    p.add_argument("--output", type=Path, required=True, help="Parent folder for sessions")
    p.add_argument("--session-id", required=True)
    p.add_argument("--source-recording", help="Group shared with duplicates/clips")
    p.add_argument("--location", required=True)
    p.add_argument("--lighting", required=True)
    p.add_argument("--notes", default="")
    a = p.parse_args()
    target = a.output / a.session_id
    (target / "images").mkdir(parents=True, exist_ok=False)
    files = sorted(f for f in a.images.iterdir() if f.suffix.lower() in {".jpg", ".jpeg"})
    if not files:
        p.error("No JPEG files")
    from PIL import Image

    with Image.open(files[0]) as first:
        width, height = first.size
    info = dict(
        schema=SESSION_SCHEMA,
        session_id=a.session_id,
        source_recording=a.source_recording or a.session_id,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        action_space=ACTION_SPACE,
        location=a.location,
        lighting=a.lighting,
        notes=a.notes,
        camera=dict(width=width, height=height, stored="original JPEG bytes", source=str(a.images)),
        calibration_version="unmeasured",
        clock=dict(capture="wall-clock time parsed from file name; not a sensor timestamp"),
        encoder=False,
        driver="unknown (imported photos, no commands)",
    )
    (target / "session.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    with (target / "frames.jsonl").open("x", encoding="utf-8") as out:
        for index, path in enumerate(files):
            relative = f"images/{index:06d}.jpg"
            shutil.copyfile(path, target / relative)
            match = NAME.search(path.stem)
            stamp = None
            if match:
                moment = datetime.strptime("".join(match.groups()[:2]), "%Y%m%d%H%M%S")
                stamp = int(moment.timestamp()) * 1_000_000_000 + int(match.group(3)) * 1000
            row = dict(
                frame_id=index,
                capture_ns=stamp,
                capture_clock="filename_wall_clock" if stamp else "unknown",
                received_ns=None,
                image=relative,
                sha1=hashlib.sha1((target / relative).read_bytes()).hexdigest(),
                original_name=path.name,
                command=None,
                encoder=None,
            )
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(dict(session=str(target), frames=len(files))))


if __name__ == "__main__":
    main()
