import math
import tempfile
import unittest

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None

from dtwin.core import ROOT, load_inputs, sample_episode


@unittest.skipIf(Usd is None, "Requires usd-core or Isaac Sim")
class SceneTests(unittest.TestCase):
    def test_four_dynamic_wheels_camera_axes_and_uvs(self):
        from dtwin.scene import build_scene

        cfg, layout = load_inputs()
        cfg, params = sample_episode(cfg, 1000)
        stage = Usd.Stage.CreateInMemory()
        with tempfile.TemporaryDirectory() as cache:
            scene = build_scene(stage, cfg, layout, params, ROOT, cache, (0, 0, 0))
            self.assertEqual(len(scene["wheel_drives"]), 4)
            for side, drive in scene["wheel_drives"]:
                joint = UsdPhysics.RevoluteJoint(drive.GetPrim())
                self.assertEqual(joint.GetAxisAttr().Get(), "Y")
                self.assertEqual(len(joint.GetBody0Rel().GetTargets()), 1)
                self.assertEqual(len(joint.GetBody1Rel().GetTargets()), 1)
            pose = UsdGeom.XformCache().GetLocalToWorldTransform(
                stage.GetPrimAtPath(scene["camera_path"])
            )
            forward = pose.TransformDir(Gf.Vec3d(0, 0, -1))
            self.assertGreater(forward[0], 0)
            self.assertLess(forward[2], 0)
            self.assertAlmostEqual(
                -math.degrees(math.atan2(forward[2], forward[0])),
                cfg["camera"]["pitch_down_deg"],
                places=4,
            )
            right = pose.TransformDir(Gf.Vec3d(1, 0, 0))
            self.assertLess(right[1], 0)
            self.assertAlmostEqual(pose.ExtractTranslation()[2], cfg["camera"]["z_m"], places=5)
            uv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Track")).GetPrimvar("st").Get()
            self.assertEqual(
                list(uv), [Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]
            )
            self.assertTrue(stage.GetPrimAtPath("/World/Props/signal/green"))
