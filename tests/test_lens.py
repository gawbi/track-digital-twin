import unittest

import numpy as np

from dtwin.geometry import CameraGeometry
from dtwin.lens import Lens

CAMERA = dict(width=640, height=480, x_m=0.1, z_m=0.07, pitch_down_deg=-8, yaw_deg=0,
              hfov_deg=88, distortion=[-0.3, 0.08, 0, 0, 0])


class LensTests(unittest.TestCase):
    def test_pixels_and_boxes_move_together(self):
        lens = Lens(CAMERA)
        self.assertGreater(lens.render_hfov_deg, 88)
        for u, v in [(320, 240), (100, 80), (600, 420), (500, 120)]:
            width, height = lens.render_size
            image = np.zeros((height, width, 3), np.uint8)
            # Choose a render-image point, distort, and compare with the box mapping.
            u_p, v_p = width / 2 + (u - 320) * 1.4, height / 2 + (v - 240) * 1.4
            image[round(v_p) - 2 : round(v_p) + 3, round(u_p) - 2 : round(u_p) + 3] = 255
            out = lens.apply(image)[..., 0].astype(float)
            ys, xs = np.nonzero(out > 60)
            box = lens.box([u_p, v_p, u_p, v_p])
            self.assertAlmostEqual(xs.mean(), box[0], delta=1.5)
            self.assertAlmostEqual(ys.mean(), box[1], delta=1.5)

    def test_simulated_distortion_matches_mission_geometry(self):
        lens, geometry = Lens(CAMERA), CameraGeometry(CAMERA)
        # A ground point projected by the mission model lands where the lens puts it.
        u, v = geometry.project(0.6, 0.1, 0.0)
        width, height = lens.render_size
        pinhole = CameraGeometry(
            dict(CAMERA, width=width, height=height, hfov_deg=lens.render_hfov_deg, distortion=[])
        )
        u_p, v_p = pinhole.project(0.6, 0.1, 0.0)
        mapped = lens.box([u_p, v_p, u_p, v_p])
        self.assertAlmostEqual(mapped[0], u, delta=0.5)
        self.assertAlmostEqual(mapped[1], v, delta=0.5)

    def test_no_distortion_is_identity(self):
        lens = Lens(dict(CAMERA, distortion=[]))
        image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), np.uint8)
        self.assertIs(lens.apply(image), image)
        self.assertAlmostEqual(lens.render_hfov_deg, 88, places=6)


if __name__ == "__main__":
    unittest.main()
