"""Real-car recording sessions, fixed session-level splits and dataset statistics.

A session folder holds ``session.json``, ``frames.jsonl`` and ``images/``.
Optional detector labels are YOLO text files in ``labels/`` with the frame's
image stem. Sessions are split by ``source_recording``: clips, copies and
augmentations of one recording always share a split. Frames are never shuffled
across splits.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .actions import ACTION_SPACE
from .perception import CLASSES

SESSION_SCHEMA = "real_session_v1"
MANIFEST_SCHEMA = "real_split_v1"
SPLITS = ("train", "validation", "test")


def sha1_file(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


class SessionWriter:
    """Writes one session. Commands are what was SENT; encoder data is separate."""

    def __init__(self, root, session_id, info):
        if not session_id or "/" in session_id or session_id.startswith("."):
            raise ValueError("session_id must be a plain folder name")
        required = {"location", "lighting", "camera", "calibration_version", "clock", "encoder"}
        missing = sorted(required - set(info))
        if missing:
            raise ValueError(f"Session info is missing {missing}")
        self.path = Path(root) / session_id
        (self.path / "images").mkdir(parents=True, exist_ok=False)
        self.info = dict(
            schema=SESSION_SCHEMA,
            session_id=session_id,
            source_recording=info.pop("source_recording", session_id),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            action_space=ACTION_SPACE,
            **info,
        )
        (self.path / "session.json").write_text(
            json.dumps(self.info, ensure_ascii=False, indent=2)
        )
        self.file = (self.path / "frames.jsonl").open("x", encoding="utf-8")
        self.frames = 0

    def write(self, frame_id, rgb, capture_ns, capture_clock, received_ns, command, encoder=None):
        """``command``: dict(action=[a_lin, a_ang], duty=[l, r], applied_ns, source).

        ``encoder`` must be a real measurement (rad/s with timestamp) or None.
        """
        from PIL import Image

        if rgb.ndim != 3 or rgb.shape[2] != 3 or str(rgb.dtype) != "uint8":
            raise ValueError("Expected uint8 RGB frame")
        if command is not None and (
            len(command.get("action", [])) != 2 or len(command.get("duty", [])) != 2
        ):
            raise ValueError("Command needs action [linear, angular] and duty [left, right]")
        if encoder is not None and not {"left_rad_s", "right_rad_s", "measured_ns"} <= set(
            encoder
        ):
            raise ValueError("Encoder entries are measurements with left/right rad/s and time")
        relative = f"images/{frame_id:06d}.jpg"
        Image.fromarray(rgb).save(self.path / relative, quality=95)
        row = dict(
            frame_id=frame_id,
            capture_ns=capture_ns,
            capture_clock=capture_clock,
            received_ns=received_ns,
            image=relative,
            sha1=sha1_file(self.path / relative),
            command=command,
            encoder=encoder,
        )
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()
        self.frames += 1
        return row

    def close(self, **summary):
        self.file.close()
        (self.path / "summary.json").write_text(
            json.dumps(dict(frames=self.frames, **summary), ensure_ascii=False, indent=2)
        )


def read_session(path):
    path = Path(path)
    info = json.loads((path / "session.json").read_text())
    if info.get("schema") != SESSION_SCHEMA:
        raise ValueError(f"{path}: not a {SESSION_SCHEMA} session")
    if info.get("action_space") != ACTION_SPACE:
        raise ValueError(f"{path}: action space {info.get('action_space')} != {ACTION_SPACE}")
    rows = [json.loads(line) for line in (path / "frames.jsonl").read_text().splitlines() if line]
    return info, rows


def find_sessions(roots):
    sessions = []
    for root in roots:
        root = Path(root)
        candidates = [root] if (root / "session.json").exists() else root.rglob("session.json")
        for item in candidates:
            sessions.append(item if item.is_dir() else item.parent)
    unique = sorted({p.resolve() for p in sessions})
    if not unique:
        raise ValueError("No real sessions found (expected session.json folders)")
    return unique


def read_labels(session_path, row):
    """YOLO label lines for a frame, [] for an explicit background, None if unlabeled."""
    label = Path(session_path) / "labels" / (Path(row["image"]).stem + ".txt")
    if not label.exists():
        return None
    boxes = []
    for line in label.read_text().splitlines():
        if not line.strip():
            continue
        values = line.split()
        cls = int(values[0])
        if len(values) != 5 or not 0 <= cls < len(CLASSES):
            raise ValueError(f"{label}: invalid YOLO line {line!r}")
        coords = [float(v) for v in values[1:]]
        if not all(0 <= v <= 1 for v in coords) or coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"{label}: coordinates must be normalized positive boxes")
        boxes.append([cls, *coords])
    return boxes


def dhash(path, size=8):
    """64-bit difference hash for near-duplicate reporting across splits."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = 0
    for value in (pixels[:, :-1] > pixels[:, 1:]).ravel():
        bits = (bits << 1) | int(value)
    return bits


def assign_groups(groups, seed, fractions=(0.6, 0.2, 0.2), explicit=None):
    """Deterministic source-group assignment. ``explicit`` maps group -> split."""
    explicit = dict(explicit or {})
    unknown = set(explicit) - set(groups)
    if unknown:
        raise ValueError(f"Explicit assignment names unknown source recordings: {sorted(unknown)}")
    if any(split not in SPLITS for split in explicit.values()):
        raise ValueError(f"Splits must be one of {SPLITS}")
    remaining = sorted(set(groups) - set(explicit))
    random.Random(seed).shuffle(remaining)
    result = {split: sorted(g for g, s in explicit.items() if s == split) for split in SPLITS}
    total = len(groups)
    for split, fraction in zip(("validation", "test"), fractions[1:]):
        need = max(1, round(total * fraction)) - len(result[split])
        while need > 0 and remaining:
            result[split].append(remaining.pop())
            need -= 1
    result["train"] += remaining
    for split in SPLITS:
        result[split] = sorted(result[split])
    if any(not result[split] for split in SPLITS):
        raise ValueError(
            "Need at least one independent source recording in each of train/validation/test; "
            "record more sessions instead of re-labeling the same recording"
        )
    return result


def build_manifest(roots, seed=42, explicit=None, near_duplicate_bits=2, fractions=None):
    sessions = []
    for path in find_sessions(roots):
        info, rows = read_session(path)
        sessions.append((path, info, rows))
    ids = Counter(info["session_id"] for _, info, _ in sessions)
    repeated = [name for name, count in ids.items() if count > 1]
    if repeated:
        raise ValueError(f"Duplicate session_id values: {repeated}")
    groups = sorted({info["source_recording"] for _, info, _ in sessions})
    assignment = assign_groups(groups, seed, fractions or (0.6, 0.2, 0.2), explicit)
    split_of_group = {g: s for s, members in assignment.items() for g in members}
    report = dict(
        frames={s: 0 for s in SPLITS},
        sessions={s: 0 for s in SPLITS},
        source_recordings={s: len(assignment[s]) for s in SPLITS},
        lane_labeled_frames={s: 0 for s in SPLITS},
        detector_labeled_frames={s: 0 for s in SPLITS},
        background_frames={s: 0 for s in SPLITS},
        class_counts={s: {c: 0 for c in CLASSES} for s in SPLITS},
        encoder_frames={s: 0 for s in SPLITS},
    )
    entries, by_hash, hashes = {}, {}, []
    for path, info, rows in sessions:
        split = split_of_group[info["source_recording"]]
        report["sessions"][split] += 1
        report["frames"][split] += len(rows)
        for row in rows:
            image = path / row["image"]
            digest = sha1_file(image)
            if digest != row["sha1"]:
                raise ValueError(f"{image}: image changed after recording")
            by_hash.setdefault(digest, set()).add((split, info["source_recording"]))
            hashes.append((dhash(image), split, info["session_id"], row["frame_id"]))
            if row.get("command") and row["command"].get("action"):
                report["lane_labeled_frames"][split] += 1
            if row.get("encoder"):
                report["encoder_frames"][split] += 1
            boxes = read_labels(path, row)
            if boxes is not None:
                report["detector_labeled_frames"][split] += 1
                report["background_frames"][split] += not boxes
                for box in boxes:
                    report["class_counts"][split][CLASSES[box[0]]] += 1
        entries[info["session_id"]] = dict(
            path=str(path),
            source_recording=info["source_recording"],
            split=split,
            frames=len(rows),
            frames_sha256=sha256_bytes((path / "frames.jsonl").read_bytes()),
            location=info.get("location"),
            lighting=info.get("lighting"),
            calibration_version=info.get("calibration_version"),
            encoder=info.get("encoder"),
        )
    leaks = sorted(
        digest
        for digest, owners in by_hash.items()
        if len({split for split, _ in owners}) > 1 or len({g for _, g in owners}) > 1
    )
    if leaks:
        raise ValueError(
            f"{len(leaks)} identical images appear in different source recordings/splits; "
            "copies must keep the original source_recording"
        )
    near = []
    for i, (a, split_a, session_a, frame_a) in enumerate(hashes):
        for b, split_b, session_b, frame_b in hashes[i + 1 :]:
            if split_a != split_b and bin(a ^ b).count("1") <= near_duplicate_bits:
                near.append([f"{session_a}:{frame_a}", f"{session_b}:{frame_b}"])
    report["near_duplicates_across_splits"] = dict(
        threshold_bits=near_duplicate_bits, pairs=len(near), examples=near[:20]
    )
    report["empty_classes"] = {
        s: [c for c, n in report["class_counts"][s].items() if n == 0] for s in SPLITS
    }
    limitations = []
    if min(report["source_recordings"].values()) < 2:
        limitations.append(
            "A split has a single source recording: validation/test reflect one session only"
        )
    if near:
        limitations.append("Near-identical frames exist across splits; inspect the examples")
    manifest = dict(
        schema=MANIFEST_SCHEMA,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=seed,
        action_space=ACTION_SPACE,
        assignment=assignment,
        sessions=entries,
        report=report,
        limitations=limitations,
    )
    manifest["manifest_id"] = sha256_bytes(
        canonical({k: manifest[k] for k in ["assignment", "sessions", "action_space"]})
    )[:16]
    return manifest


def write_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return path


def load_manifest(path):
    """Load and verify that no session changed since the split was fixed."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Not a fixed real-data split manifest")
    expected = sha256_bytes(
        canonical({k: manifest[k] for k in ["assignment", "sessions", "action_space"]})
    )[:16]
    if expected != manifest["manifest_id"]:
        raise ValueError("Manifest was edited after it was fixed")
    for session_id, entry in manifest["sessions"].items():
        frames = Path(entry["path"]) / "frames.jsonl"
        if sha256_bytes(frames.read_bytes()) != entry["frames_sha256"]:
            raise ValueError(f"Session {session_id} changed after the split was fixed")
    return manifest


def split_frames(manifest, split, need="lane"):
    """Rows of one split: ``lane`` needs command actions, ``detector`` needs labels."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split}")
    result = []
    for session_id, entry in sorted(manifest["sessions"].items()):
        if entry["split"] != split:
            continue
        path = Path(entry["path"])
        _, rows = read_session(path)
        for row in rows:
            if need == "lane" and row.get("command") and row["command"].get("action"):
                result.append(dict(session_id=session_id, path=path, row=row))
            elif need == "detector":
                boxes = read_labels(path, row)
                if boxes is not None:
                    result.append(dict(session_id=session_id, path=path, row=row, boxes=boxes))
    return result


def record_test_access(manifest_path, purpose, model, result):
    """Append-only ledger so final-test use is visible across experiments."""
    ledger = Path(manifest_path).with_suffix(".test_access.jsonl")
    entry = dict(
        time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        monotonic_s=time.monotonic(),
        purpose=purpose,
        model=str(model),
        result=result,
    )
    with ledger.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    previous = len(ledger.read_text(encoding="utf-8").splitlines()) - 1
    return ledger, previous
