"""Author a self-contained USD scene, without network robot assets.

Vehicle: +X forward, +Y left, +Z up. Four Y-axis wheel joints.
Camera: USD convention +X right, +Y up, -Z forward.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema
except ImportError:  # permits USD structural validation outside Isaac Sim
    PhysxSchema = None


def transform(prim, xyz=(0, 0, 0), rotation=None, scale=None):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    if rotation is not None:
        xf.AddOrientOp().Set(Gf.Quatf(rotation))
    if scale is not None:
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))


def material(stage, path, rgb=(0.2, 0.2, 0.2), roughness=0.5, texture=None, emission=None):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if emission:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
    if texture:
        reader = UsdShade.Shader.Define(stage, path + "/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, path + "/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(Path(texture).resolve()))
        )
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def rigid(prim, mass):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    UsdPhysics.CollisionAPI.Apply(prim)
    if PhysxSchema:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateSolverPositionIterationCountAttr(16)
        rb.CreateSolverVelocityIterationCountAttr(4)


def cube(stage, path, xyz, size, mat, mass=None, collision=True):
    obj = UsdGeom.Cube.Define(stage, path)
    obj.CreateSizeAttr(1.0)
    transform(obj.GetPrim(), xyz, scale=size)
    bind(obj.GetPrim(), mat)
    if mass is not None:
        rigid(obj.GetPrim(), mass)
    elif collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
    return obj


def textured_quad(stage, path, points, texture):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    # Image top (north) maps to v=1; east is +X. No transpose or mirrored UVs.
    uv.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    mat = material(stage, path + "_Material", roughness=0.45, texture=texture)
    bind(mesh.GetPrim(), mat)
    return mesh, mat


def make_sign_textures(folder):
    """Procedural approximations of the three photographed signs; no people/photos."""
    from PIL import Image, ImageDraw, ImageFont

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]

    def font(size):
        for candidate in candidates:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size)
        return ImageFont.load_default(size=size)

    for name in ["speed20", "stop", "horn"]:
        image = Image.new("RGB", (256, 256), "#eeeeec")
        draw = ImageDraw.Draw(image)
        color = "#a82d25" if name != "horn" else "#123b79"
        draw.ellipse((12, 12, 244, 244), fill=color)
        draw.ellipse((21, 21, 235, 235), outline="white", width=4)
        if name == "speed20":
            draw.ellipse((30, 30, 226, 226), fill="#eeeee8")
            draw.text((128, 128), "20", fill="#182327", font=font(94), anchor="mm")
        elif name == "stop":
            draw.text((128, 128), "STOP", fill="white", font=font(66), anchor="mm")
        else:
            draw.polygon(
                [(48, 92), (88, 115), (173, 115), (173, 133), (88, 133), (48, 157)], fill="white"
            )
            draw.line(
                [
                    (169, 124),
                    (201, 124),
                    (201, 146),
                    (130, 146),
                    (113, 159),
                    (129, 173),
                    (185, 173),
                ],
                fill="white",
                width=13,
            )
        image.save(folder / f"{name}.png")
    return folder


def build_scene(stage, cfg, layout, params, root, cache, spawn):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    if PhysxSchema:
        ps = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        ps.CreateTimeStepsPerSecondAttr(cfg["simulation"]["physics_hz"])
        ps.CreateSolverTypeAttr("TGS")
        ps.CreateEnableGPUDynamicsAttr(False)  # one car; RTX GPU remains dedicated to rendering
        ps.CreateBroadphaseTypeAttr("MBP")
    surface = UsdShade.Material.Define(stage, "/World/Materials/Friction")
    pm = UsdPhysics.MaterialAPI.Apply(surface.GetPrim())
    pm.CreateStaticFrictionAttr(params["friction"])
    pm.CreateDynamicFrictionAttr(params["friction"] * 0.8)
    pm.CreateRestitutionAttr(0.0)

    def physics_bind(prim):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(surface, materialPurpose="physics")

    floor_mat = material(stage, "/World/Materials/Floor", params["background_rgb"])
    floor = cube(stage, "/World/Floor", (0, 0, -0.03), (8, 8, 0.06), floor_mat)
    physics_bind(floor.GetPrim())
    bx, by = layout["meta"]["board_size"]
    board, board_mat = textured_quad(
        stage,
        "/World/Track",
        [
            (-bx / 2, -by / 2, 0.0002),
            (bx / 2, -by / 2, 0.0002),
            (bx / 2, by / 2, 0.0002),
            (-bx / 2, by / 2, 0.0002),
        ],
        Path(root) / "track_texture.png",
    )
    UsdShade.Shader(stage.GetPrimAtPath("/World/Track_Material/Surface")).GetInput("roughness").Set(
        params["roughness"]
    )
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(params["dome_intensity"])
    dome.CreateEnableColorTemperatureAttr(True)
    dome.CreateColorTemperatureAttr(params["light_temperature_k"])
    key = UsdLux.RectLight.Define(stage, "/World/Key")
    transform(key.GetPrim(), (0.8, -0.5, 2.6))
    key.CreateWidthAttr(1.2)
    key.CreateHeightAttr(0.7)
    key.CreateIntensityAttr(params["key_intensity"])
    key.CreateEnableColorTemperatureAttr(True)
    key.CreateColorTemperatureAttr(params["light_temperature_k"])
    black = material(stage, "/World/Materials/Black", (0.025, 0.028, 0.035))
    yellow = material(stage, "/World/Materials/Yellow", (0.85, 0.6, 0.03))
    white = material(stage, "/World/Materials/White", (0.8, 0.8, 0.77))
    v = cfg["vehicle"]
    x, y, yaw = spawn
    r = v["wheel_radius_m"]
    # Root transform sets spawn yaw once; body/wheels are dynamic siblings.
    robot = UsdGeom.Xform.Define(stage, "/World/Car")
    transform(
        robot.GetPrim(), (x, y, 0), Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)).GetQuat()
    )
    body_z = r + 0.012 + v["body_height_m"] / 2
    # Keep dynamic body scale at 1: put geometry below rigid Xform.
    chassis = UsdGeom.Xform.Define(stage, "/World/Car/Chassis")
    transform(chassis.GetPrim(), (0, 0, body_z))
    UsdPhysics.RigidBodyAPI.Apply(chassis.GetPrim())
    UsdPhysics.MassAPI.Apply(chassis.GetPrim()).CreateMassAttr(v["mass_kg"] * 0.8)
    UsdPhysics.ArticulationRootAPI.Apply(chassis.GetPrim())
    cube(
        stage,
        "/World/Car/Chassis/Plate",
        (0, 0, 0),
        (v["length_m"], v["body_width_m"], v["body_height_m"]),
        black,
    )
    wheels = []
    for side, sy in [("left", 1), ("right", -1)]:
        for axle, sx in [("front", 1), ("rear", -1)]:
            path = f"/World/Car/{side}_{axle}"
            pos = (sx * v["wheelbase_m"] / 2, sy * v["track_width_m"] / 2, r)
            wheel = UsdGeom.Cylinder.Define(stage, path)
            wheel.CreateAxisAttr("Y")
            wheel.CreateRadiusAttr(r)
            wheel.CreateHeightAttr(v["wheel_width_m"])
            transform(wheel.GetPrim(), pos)
            rigid(wheel.GetPrim(), v["mass_kg"] * 0.05)
            bind(wheel.GetPrim(), yellow)
            physics_bind(wheel.GetPrim())
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Car/Joints/" + side + "_" + axle)
            joint.CreateBody0Rel().SetTargets([chassis.GetPath()])
            joint.CreateBody1Rel().SetTargets([wheel.GetPath()])
            joint.CreateAxisAttr("Y")
            joint.CreateLocalPos0Attr(Gf.Vec3f(pos[0], pos[1], r - body_z))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(v["drive_damping"])
            drive.CreateMaxForceAttr(v["wheel_torque_nm"])
            drive.CreateTargetVelocityAttr(0.0)
            wheels.append((side, drive))
    camera = UsdGeom.Camera.Define(stage, "/World/Car/Chassis/Camera")
    c = cfg["camera"]
    # USD camera basis: local right=-robot Y, local up=robot Z, forward=robot X.
    basis = Gf.Matrix3d(0, -1, 0, 0, 0, 1, -1, 0, 0)
    rotation = Gf.Rotation(basis.ExtractRotation().GetQuat())
    pitch = Gf.Rotation(Gf.Vec3d(0, 1, 0), c["pitch_down_deg"])
    yaw_rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), c["yaw_deg"])
    orient = (
        (Gf.Matrix3d(rotation) * Gf.Matrix3d(pitch) * Gf.Matrix3d(yaw_rotation))
        .ExtractRotation()
        .GetQuat()
    )
    transform(camera.GetPrim(), (c["x_m"], 0, c["z_m"] - body_z), orient)
    aperture = 20.955
    camera.CreateHorizontalApertureAttr(aperture)
    camera.CreateVerticalApertureAttr(aperture * c["height"] / c["width"])
    camera.CreateFocalLengthAttr(aperture / (2 * math.tan(math.radians(c["hfov_deg"] / 2))))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 30.0))
    textures = make_sign_textures(cache)
    spots = {p["id"]: p["center"] for p in layout["prop_spots"]}
    props = []
    for p in cfg["props"]:
        px, py = spots[p["spot"]]
        path = "/World/Props/" + p["id"]
        parent = UsdGeom.Xform.Define(stage, path)
        transform(
            parent.GetPrim(), (px, py, 0), Gf.Rotation(Gf.Vec3d(0, 0, 1), p["yaw_deg"]).GetQuat()
        )
        cube(stage, path + "/Base", (0, 0, 0.0085), (0.06, 0.06, 0.017), white)
        cube(stage, path + "/Pole", (0, 0, 0.07), (0.012, 0.012, 0.12), black)
        if p["kind"] != "signal":
            # Face normal local -Y, v upwards. Sign is 7cm square, bottom at 10.3cm.
            textured_quad(
                stage,
                path + "/Face",
                [
                    (-0.035, -0.007, 0.103),
                    (0.035, -0.007, 0.103),
                    (0.035, -0.007, 0.173),
                    (-0.035, -0.007, 0.173),
                ],
                textures / f"{p['kind']}.png",
            )
            UsdGeom.Mesh(stage.GetPrimAtPath(path + "/Face")).GetDoubleSidedAttr().Set(False)
            cube(stage, path + "/Backing", (0, 0, 0.138), (0.07, 0.012, 0.07), white)
        else:
            cube(stage, path + "/Housing", (0, 0, 0.16), (0.045, 0.03, 0.17), black)
            front = UsdGeom.Mesh.Define(stage, path + "/Front")
            front.CreatePointsAttr(
                [
                    (-0.0225, -0.016, 0.075),
                    (0.0225, -0.016, 0.075),
                    (0.0225, -0.016, 0.245),
                    (-0.0225, -0.016, 0.245),
                ]
            )
            front.CreateFaceVertexCountsAttr([4])
            front.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            front.CreateSubdivisionSchemeAttr("none")
            front.CreateDoubleSidedAttr(False)
            bind(front.GetPrim(), black)
            # Photographed classroom signal has GREEN on top, RED below.
            for color, height, rgb in [
                ("red", 0.105, (1, 0.01, 0.01)),
                ("yellow", 0.16, (1, 0.6, 0.01)),
                ("green", 0.21, (0.01, 1, 0.01)),
            ]:
                lamp = UsdGeom.Sphere.Define(stage, path + "/" + color)
                lamp.CreateRadiusAttr(0.014)
                transform(lamp.GetPrim(), (0, -0.021, height))
                bind(
                    lamp.GetPrim(), material(stage, path + "/" + color + "_Mat", rgb, emission=rgb)
                )
        props.append({**p, "position": [px, py], "path": path})
    return {
        "camera_path": str(camera.GetPath()),
        "body_path": str(chassis.GetPath()),
        "wheel_drives": wheels,
        "props": props,
        "body_z": body_z,
    }


def set_signal(stage, props, color):
    for p in props:
        if p["kind"] == "signal":
            for name, rgb in [
                ("red", (1, 0.01, 0.01)),
                ("yellow", (1, 0.6, 0.01)),
                ("green", (0.01, 1, 0.01)),
            ]:
                sh = UsdShade.Shader(stage.GetPrimAtPath(p["path"] + "/" + name + "_Mat/Surface"))
                sh.GetInput("emissiveColor").Set(Gf.Vec3f(*(rgb if color == name else (0, 0, 0))))
                sh.GetInput("diffuseColor").Set(
                    Gf.Vec3f(*(rgb if color == name else tuple(v * 0.035 for v in rgb)))
                )
