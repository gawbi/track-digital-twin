"""RTX runtime. Import this only AFTER SimulationApp has started."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from .core import ROOT, body_corners, wheel_speeds


class Simulator:
    def __init__(self, app, cfg, layout, params, route, output):
        import carb
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager

        from .scene import build_scene, set_signal

        self.app, self.cfg, self.layout, self.params = app, cfg, layout, params
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.frame = 0
        self.dt = 1 / cfg["simulation"]["camera_hz"]
        self.rng = np.random.default_rng(params["seed"])
        self.delay = deque(maxlen=params["latency_frames"] + 1)
        self.signal_color = "red"
        self._set_signal = set_signal
        x, y = route.points[0]
        nx, ny = route.points[1]
        yaw = math.atan2(ny - y, nx - x)
        offset = params["spawn_lateral_m"]
        spawn = (
            x - math.sin(yaw) * offset,
            y + math.cos(yaw) * offset,
            yaw + math.radians(params["spawn_yaw_deg"]),
        )
        omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()
        # USD transforms are the pose source. Disable Fabric so reads cannot be stale.
        settings = carb.settings.get_settings()
        settings.set_bool("/physics/updateToUsd", True)
        settings.set_bool("/physics/updateVelocitiesToUsd", True)
        self.objects = build_scene(
            self.stage, cfg, layout, params, ROOT, self.output / "assets", spawn
        )
        self.sm = SimulationManager
        self.legacy = not hasattr(SimulationManager, "setup_simulation")
        if self.legacy:
            from isaacsim.core.api import World

            self.world = World(
                physics_dt=1 / cfg["simulation"]["physics_hz"],
                rendering_dt=self.dt,
                stage_units_in_meters=1.0,
                physics_prim_path="/World/Physics",
            )
            self.world.reset()
        else:
            from isaacsim.core.rendering_manager import RenderingManager

            self.renderer = RenderingManager
            SimulationManager.switch_physics_engine("physx")
            SimulationManager.setup_simulation(dt=1 / cfg["simulation"]["physics_hz"], device="cpu")
            SimulationManager.enable_fabric(False)
            SimulationManager.initialize_physics()
        # Replicator RGB annotator works with a USD Camera; avoids 5.x/6.x camera API changes.
        self.render_product = rep.create.render_product(
            self.objects["camera_path"], (cfg["camera"]["width"], cfg["camera"]["height"])
        )
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.render_product])
        self.boxes = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
        self.boxes.attach([self.render_product])
        self.last_boxes = []
        for prop in self.objects["props"]:
            if prop["kind"] != "signal":
                self._semantic(prop["path"] + "/Face", prop["kind"])
        self.road_mask = self._road_mask()
        import cv2

        self.clearance_map = cv2.distanceTransform(self.road_mask, cv2.DIST_L2, 5) / 1000.0
        self.set_signal("red")
        for _ in range(cfg["simulation"]["warmup_frames"]):
            self.step(0, 0, count=False)
        if self.raw_rgb().size == 0:
            raise RuntimeError(
                "RGB renderer did not produce a frame; check RTX/driver/headless settings"
            )
        self.frame = 0
        self.delay.clear()

    def _road_mask(self):
        from PIL import Image, ImageDraw

        meta, dims = self.layout["meta"], self.layout["dims"]
        w, h = (round(s * 1000) for s in meta["board_size"])
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        px, py = meta["printed_size"]
        bx, by = dims["block_size_xy"]
        bandx = dims["outer_green_band_east_west"]
        bandy = dims["outer_green_band_north_south"]

        def box(x0, y0, x1, y1):
            return [
                round((x0 + meta["board_size"][0] / 2) * 1000),
                round((meta["board_size"][1] / 2 - y1) * 1000),
                round((x1 + meta["board_size"][0] / 2) * 1000),
                round((meta["board_size"][1] / 2 - y0) * 1000),
            ]

        draw.rounded_rectangle(
            box(-px / 2 + bandx, -py / 2 + bandy, px / 2 - bandx, py / 2 - bandy),
            round(dims["outer_road_corner_radius"] * 1000),
            fill=255,
        )
        cx = (dims["center_road_ns_width"] + bx) / 2
        cy = (dims["center_road_ew_width"] + by) / 2
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                draw.rounded_rectangle(
                    box(sx * cx - bx / 2, sy * cy - by / 2, sx * cx + bx / 2, sy * cy + by / 2),
                    round(dims["block_corner_radius"] * 1000),
                    fill=0,
                )
        return np.asarray(mask).copy()

    def set_signal(self, color):
        self.signal_color = color
        self._set_signal(self.stage, self.objects["props"], color)
        for prop in self.objects["props"]:
            if prop["kind"] == "signal":
                self._semantic(prop["path"] + "/Front", "signal_" + color)

    def _semantic(self, path, label):
        prim = self.stage.GetPrimAtPath(path)
        if self.legacy:
            from isaacsim.core.utils.semantics import add_update_semantics

            add_update_semantics(prim, label)
        else:
            from isaacsim.core.experimental.utils.semantics import add_labels

            add_labels(prim, [label])

    def detections(self):
        data = self.boxes.get_data()
        labels = data.get("info", {}).get("idToLabels", {})
        width, height = self.cfg["camera"]["width"], self.cfg["camera"]["height"]
        result = []
        from .perception import CLASSES

        for row in data.get("data", []):
            name = labels.get(str(int(row["semanticId"])), {}).get("class", "")
            if name not in CLASSES:
                continue
            x0, y0, x1, y1 = (float(row[k]) for k in ["x_min", "y_min", "x_max", "y_max"])
            x0, x1 = max(0.0, x0), min(float(width), x1)
            y0, y1 = max(0.0, y0), min(float(height), y1)
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                result.append(
                    {"class": name, "xyxy": [x0, y0, x1, y1], "source": "renderer_ground_truth"}
                )
        return result

    def raw_rgb(self):
        data = self.rgb.get_data()
        if isinstance(data, dict):
            data = data["data"]
        return np.asarray(data)[..., :3].copy() if np.asarray(data).size else np.array([])

    def image(self):
        frame = self.raw_rgb()
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Invalid camera frame: {frame.shape}")
        frame = np.clip(
            frame.astype(np.float32) * self.params["image_gain"]
            + self.rng.normal(0, self.params["image_noise_std"], frame.shape),
            0,
            255,
        ).astype(np.uint8)
        # Delayed RGB and renderer boxes must refer to the SAME captured frame.
        self.delay.append((frame, self.detections()))
        self.last_boxes = self.delay[0][1]
        return self.delay[0][0].copy()

    def pose(self):
        from pxr import UsdGeom

        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(self.objects["body_path"])
        )
        pos = matrix.ExtractTranslation()
        forward = matrix.TransformDir((1, 0, 0))
        return float(pos[0]), float(pos[1]), math.atan2(forward[1], forward[0])

    def footprint(self):
        x, y, yaw = self.pose()
        v = self.cfg["vehicle"]
        width = v["track_width_m"] + v["wheel_width_m"]
        # Conservative enclosing rectangle sampled along every edge, in 1mm road mask.
        corners = body_corners(x, y, yaw, v["length_m"], width)
        c, s = math.cos(yaw), math.sin(yaw)
        points = []
        for lx in np.linspace(-v["length_m"] / 2, v["length_m"] / 2, 17):
            for ly in [-width / 2, width / 2]:
                points.append((x + c * lx - s * ly, y + s * lx + c * ly))
        for ly in np.linspace(-width / 2, width / 2, 11):
            for lx in [-v["length_m"] / 2, v["length_m"] / 2]:
                points.append((x + c * lx - s * ly, y + s * lx + c * ly))
        w, h = self.layout["meta"]["board_size"]
        values = []
        for px, py in points + corners:
            col, row = round((px + w / 2) * 1000), round((h / 2 - py) * 1000)
            if not 0 <= row < self.road_mask.shape[0] or not 0 <= col < self.road_mask.shape[1]:
                values.append(0.0)
            else:
                values.append(float(self.clearance_map[row, col]))
        return {"footprint_on_road": all(v > 0 for v in values), "min_clearance_m": min(values)}

    def step(self, linear, angular, count=True, wheel_override=None):
        left, right = wheel_override or wheel_speeds(linear, angular, self.cfg["vehicle"])
        for side, drive in self.objects["wheel_drives"]:
            speed = left if side == "left" else right
            gain = self.params["motor_gain_left" if side == "left" else "motor_gain_right"]
            # USD angular Drive targetVelocity is degrees/s; controller uses rad/s.
            drive.GetTargetVelocityAttr().Set(math.degrees(speed * gain))
        steps = self.cfg["simulation"]["physics_hz"] // self.cfg["simulation"]["camera_hz"]
        if self.legacy:
            for _ in range(steps):
                self.world.step(render=False)
            self.world.render()
        else:
            self.sm.step(steps=steps)
            self.renderer.render()
        if count:
            self.frame += 1

    def save_stage(self):
        path = self.output / "scene.usda"
        self.stage.Export(str(path))
        return path

    def close(self):
        for _, drive in self.objects["wheel_drives"]:
            drive.GetTargetVelocityAttr().Set(0.0)
        self.rgb.detach([self.render_product])
        self.boxes.detach([self.render_product])
        self.render_product.destroy()


class Recorder:
    def __init__(self, output, cfg, params, mode, route_name, save_images=False):
        self.path = Path(output)
        self.path.mkdir(parents=True, exist_ok=True)
        self.images = save_images
        if save_images:
            (self.path / "images").mkdir()
        self.file = (self.path / "frames.jsonl").open("w", encoding="utf-8")
        metadata = dict(
            config=cfg,
            domain=params,
            mode=mode,
            route=route_name,
            teacher_uses_ground_truth=(mode == "expert"),
            labels="linear_m_s, angular_rad_s: command applied AFTER this camera frame",
        )
        (self.path / "episode.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    def write(self, record, rgb):
        if self.images:
            from PIL import Image

            relative = f"images/{record['frame']:06d}.jpg"
            Image.fromarray(rgb).save(self.path / relative, quality=95)
            record["image"] = relative
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self, summary):
        self.file.close()
        (self.path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
