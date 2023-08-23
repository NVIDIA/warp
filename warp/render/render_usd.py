# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
import numpy as np
import math


def _usd_add_xform(prim):
    from pxr import UsdGeom

    prim = UsdGeom.Xform(prim)
    prim.ClearXformOpOrder()

    t = prim.AddTranslateOp()
    r = prim.AddOrientOp()
    s = prim.AddScaleOp()


def _usd_set_xform(xform, pos: tuple, rot: tuple, scale: tuple, time):
    from pxr import UsdGeom, Gf

    xform = UsdGeom.Xform(xform)

    xform_ops = xform.GetOrderedXformOps()

    xform_ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), time)
    xform_ops[1].Set(Gf.Quatf(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])), time)
    xform_ops[2].Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])), time)


# transforms a cylinder such that it connects the two points pos0, pos1
def _compute_segment_xform(pos0, pos1):
    from pxr import Gf

    mid = (pos0 + pos1) * 0.5
    height = (pos1 - pos0).GetLength()

    dir = (pos1 - pos0) / height

    rot = Gf.Rotation()
    rot.SetRotateInto((0.0, 0.0, 1.0), Gf.Vec3d(dir))

    scale = Gf.Vec3f(1.0, 1.0, height)

    return (mid, Gf.Quath(rot.GetQuat()), scale)


class UsdRenderer:
    """A USD renderer"""

    def __init__(self, stage, up_axis="Y", fps=60, scaling=1.0):
        """Construct a UsdRenderer object

        Args:
            model: A simulation model
            stage (str/Usd.Stage): A USD stage (either in memory or on disk)
            up_axis (str): The upfacing axis of the stage
            fps: The number of frames per second to use in the USD file
            scaling: Scaling factor to use for the entities in the scene
        """

        from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf

        if isinstance(stage, str):
            self.stage = stage = Usd.Stage.CreateNew(stage)
        elif isinstance(stage, Usd.Stage):
            self.stage = stage
        else:
            print("Failed to create stage in renderer. Please construct with stage path or stage object.")
        self.up_axis = up_axis.upper()
        self.fps = float(fps)
        self.time = 0.0

        self.draw_points = True
        self.draw_springs = False
        self.draw_triangles = False

        self.root = UsdGeom.Xform.Define(stage, "/root")

        # mapping from shape ID to UsdGeom class
        self._shape_constructors = {}
        # optional scaling applied to shape instances (e.g. cubes)
        self._shape_custom_scale = {}

        # apply scaling
        self.root.ClearXformOpOrder()
        s = self.root.AddScaleOp()
        s.Set(Gf.Vec3d(float(scaling), float(scaling), float(scaling)), 0.0)

        self.stage.SetDefaultPrim(self.root.GetPrim())
        self.stage.SetStartTimeCode(0.0)
        self.stage.SetEndTimeCode(0.0)
        self.stage.SetTimeCodesPerSecond(self.fps)

        if up_axis == "X":
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.x)
        elif up_axis == "Y":
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        elif up_axis == "Z":
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        # add default lights
        light_0 = UsdLux.DistantLight.Define(stage, "/light_0")
        light_0.GetPrim().CreateAttribute("intensity", Sdf.ValueTypeNames.Float, custom=False).Set(2500.0)
        light_0.GetPrim().CreateAttribute("color", Sdf.ValueTypeNames.Color3f, custom=False).Set(
            Gf.Vec3f(0.98, 0.85, 0.7)
        )

        UsdGeom.Xform(light_0.GetPrim()).AddRotateYOp().Set(value=(70.0))
        UsdGeom.Xform(light_0.GetPrim()).AddRotateXOp().Set(value=(-45.0))

        light_1 = UsdLux.DistantLight.Define(stage, "/light_1")
        light_1.GetPrim().CreateAttribute("intensity", Sdf.ValueTypeNames.Float, custom=False).Set(2500.0)
        light_1.GetPrim().CreateAttribute("color", Sdf.ValueTypeNames.Color3f, custom=False).Set(
            Gf.Vec3f(0.62, 0.82, 0.98)
        )

        UsdGeom.Xform(light_1.GetPrim()).AddRotateYOp().Set(value=(-70.0))
        UsdGeom.Xform(light_1.GetPrim()).AddRotateXOp().Set(value=(-45.0))

    def begin_frame(self, time):
        self.stage.SetEndTimeCode(time * self.fps)
        self.time = time * self.fps

    def end_frame(self):
        pass

    def register_body(self, body_name):
        from pxr import UsdGeom

        xform = UsdGeom.Xform.Define(self.stage, self.root.GetPath().AppendChild(body_name))

        _usd_add_xform(xform)

    def _resolve_path(self, name, parent_body=None, is_template=False):
        # resolve the path to the prim with the given name and optional parent body
        if is_template:
            return self.root.GetPath().AppendChild("_template_shapes").AppendChild(name)
        if parent_body is None:
            return self.root.GetPath().AppendChild(name)
        else:
            return self.root.GetPath().AppendChild(parent_body).AppendChild(name)

    def add_shape_instance(
        self,
        name: str,
        shape,
        body,
        pos: tuple,
        rot: tuple,
        scale: tuple = (1.0, 1.0, 1.0),
        color: tuple = (1.0, 1.0, 1.0),
    ):
        sdf_path = self._resolve_path(name, body)
        instance = self._shape_constructors[shape.name].Define(self.stage, sdf_path)
        instance.GetPrim().GetReferences().AddInternalReference(shape)

        _usd_add_xform(instance)
        if shape.name in self._shape_custom_scale:
            cs = self._shape_custom_scale[shape.name]
            scale = (scale[0] * cs[0], scale[1] * cs[1], scale[2] * cs[2])
        _usd_set_xform(instance, pos, rot, scale, self.time)

    def render_plane(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        width: float,
        length: float,
        color: tuple = None,
        parent_body: str = None,
        is_template: bool = False,
    ):
        """
        Render a plane with the given dimensions.

        Args:
            name: Name of the plane
            pos: Position of the plane
            rot: Rotation of the plane
            width: Width of the plane
            length: Length of the plane
            color: Color of the plane
            parent_body: Name of the parent body
            is_template: Whether the plane is a template
        """
        from pxr import UsdGeom, Sdf

        if is_template:
            prim_path = self._resolve_path(name, parent_body, is_template)
            blueprint = UsdGeom.Scope.Define(self.stage, prim_path)
            blueprint_prim = blueprint.GetPrim()
            blueprint_prim.SetInstanceable(True)
            blueprint_prim.SetSpecifier(Sdf.SpecifierClass)
            plane_path = prim_path.AppendChild("plane")
        else:
            plane_path = self._resolve_path(name, parent_body)
            prim_path = plane_path

        plane = UsdGeom.Mesh.Get(self.stage, plane_path)
        if not plane:
            plane = UsdGeom.Mesh.Define(self.stage, plane_path)
            plane.CreateDoubleSidedAttr().Set(True)

            width = width if width > 0.0 else 100.0
            length = length if length > 0.0 else 100.0
            points = ((-width, 0.0, -length), (width, 0.0, -length), (width, 0.0, length), (-width, 0.0, length))
            normals = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
            counts = (4,)
            indices = [0, 1, 2, 3]

            plane.GetPointsAttr().Set(points)
            plane.GetNormalsAttr().Set(normals)
            plane.GetFaceVertexCountsAttr().Set(counts)
            plane.GetFaceVertexIndicesAttr().Set(indices)
            _usd_add_xform(plane)

        self._shape_constructors[name] = UsdGeom.Mesh

        if not is_template:
            _usd_set_xform(plane, pos, rot, (1.0, 1.0, 1.0), 0.0)

        return prim_path

    def render_ground(self, size: float = 100.0):
        from pxr import UsdGeom

        mesh = UsdGeom.Mesh.Define(self.stage, self.root.GetPath().AppendChild("ground"))
        mesh.CreateDoubleSidedAttr().Set(True)

        if self.up_axis == "X":
            points = ((0.0, -size, -size), (0.0, size, -size), (0.0, size, size), (0.0, -size, size))
            normals = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        elif self.up_axis == "Y":
            points = ((-size, 0.0, -size), (size, 0.0, -size), (size, 0.0, size), (-size, 0.0, size))
            normals = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        elif self.up_axis == "Z":
            points = ((-size, -size, 0.0), (size, -size, 0.0), (size, size, 0.0), (-size, size, 0.0))
            normals = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
        counts = (4,)
        indices = [0, 1, 2, 3]

        mesh.GetPointsAttr().Set(points)
        mesh.GetNormalsAttr().Set(normals)
        mesh.GetFaceVertexCountsAttr().Set(counts)
        mesh.GetFaceVertexIndicesAttr().Set(indices)

    def render_sphere(
        self, name: str, pos: tuple, rot: tuple, radius: float, parent_body: str = None, is_template: bool = False
    ):
        """Debug helper to add a sphere for visualization

        Args:
            pos: The position of the sphere
            radius: The radius of the sphere
            name: A name for the USD prim on the stage
        """

        from pxr import UsdGeom, Sdf

        if is_template:
            prim_path = self._resolve_path(name, parent_body, is_template)
            blueprint = UsdGeom.Scope.Define(self.stage, prim_path)
            blueprint_prim = blueprint.GetPrim()
            blueprint_prim.SetInstanceable(True)
            blueprint_prim.SetSpecifier(Sdf.SpecifierClass)
            sphere_path = prim_path.AppendChild("sphere")
        else:
            sphere_path = self._resolve_path(name, parent_body)
            prim_path = sphere_path

        sphere = UsdGeom.Sphere.Get(self.stage, sphere_path)
        if not sphere:
            sphere = UsdGeom.Sphere.Define(self.stage, sphere_path)
            _usd_add_xform(sphere)

        sphere.GetRadiusAttr().Set(radius, self.time)

        self._shape_constructors[name] = UsdGeom.Sphere

        if not is_template:
            _usd_set_xform(sphere, pos, rot, (1.0, 1.0, 1.0), 0.0)

        return prim_path

    def render_capsule(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        radius: float,
        half_height: float,
        parent_body: str = None,
        is_template: bool = False,
    ):
        """
        Debug helper to add a capsule for visualization

        Args:
            pos: The position of the capsule
            radius: The radius of the capsule
            half_height: The half height of the capsule
            name: A name for the USD prim on the stage
        """

        from pxr import UsdGeom, Sdf

        if is_template:
            prim_path = self._resolve_path(name, parent_body, is_template)
            blueprint = UsdGeom.Scope.Define(self.stage, prim_path)
            blueprint_prim = blueprint.GetPrim()
            blueprint_prim.SetInstanceable(True)
            blueprint_prim.SetSpecifier(Sdf.SpecifierClass)
            capsule_path = prim_path.AppendChild("capsule")
        else:
            capsule_path = self._resolve_path(name, parent_body)
            prim_path = capsule_path

        capsule = UsdGeom.Capsule.Get(self.stage, capsule_path)
        if not capsule:
            capsule = UsdGeom.Capsule.Define(self.stage, capsule_path)
            _usd_add_xform(capsule)

        capsule.GetRadiusAttr().Set(float(radius))
        capsule.GetHeightAttr().Set(float(half_height * 2.0))
        capsule.GetAxisAttr().Set("Y")

        self._shape_constructors[name] = UsdGeom.Capsule

        if not is_template:
            _usd_set_xform(capsule, pos, rot, (1.0, 1.0, 1.0), 0.0)

        return prim_path

    def render_cylinder(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        radius: float,
        half_height: float,
        parent_body: str = None,
        is_template: bool = False,
    ):
        """
        Debug helper to add a cylinder for visualization

        Args:
            pos: The position of the cylinder
            radius: The radius of the cylinder
            half_height: The half height of the cylinder
            name: A name for the USD prim on the stage
        """

        from pxr import UsdGeom, Sdf

        if is_template:
            prim_path = self._resolve_path(name, parent_body, is_template)
            blueprint = UsdGeom.Scope.Define(self.stage, prim_path)
            blueprint_prim = blueprint.GetPrim()
            blueprint_prim.SetInstanceable(True)
            blueprint_prim.SetSpecifier(Sdf.SpecifierClass)
            cylinder_path = prim_path.AppendChild("cylinder")
        else:
            cylinder_path = self._resolve_path(name, parent_body)
            prim_path = cylinder_path

        cylinder = UsdGeom.Cylinder.Get(self.stage, cylinder_path)
        if not cylinder:
            cylinder = UsdGeom.Cylinder.Define(self.stage, cylinder_path)
            _usd_add_xform(cylinder)

        cylinder.GetRadiusAttr().Set(float(radius))
        cylinder.GetHeightAttr().Set(float(half_height * 2.0))
        cylinder.GetAxisAttr().Set("Y")

        self._shape_constructors[name] = UsdGeom.Cylinder

        if not is_template:
            _usd_set_xform(cylinder, pos, rot, (1.0, 1.0, 1.0), 0.0)

        return prim_path

    def render_cone(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        radius: float,
        half_height: float,
        parent_body: str = None,
        is_template: bool = False,
    ):
        """
        Debug helper to add a cone for visualization

        Args:
            pos: The position of the cone
            radius: The radius of the cone
            half_height: The half height of the cone
            name: A name for the USD prim on the stage
        """

        from pxr import UsdGeom, Sdf

        if is_template:
            prim_path = self._resolve_path(name, parent_body, is_template)
            blueprint = UsdGeom.Scope.Define(self.stage, prim_path)
            blueprint_prim = blueprint.GetPrim()
            blueprint_prim.SetInstanceable(True)
            blueprint_prim.SetSpecifier(Sdf.SpecifierClass)
            cone_path = prim_path.AppendChild("cone")
        else:
            cone_path = self._resolve_path(name, parent_body)
            prim_path = cone_path

        cone = UsdGeom.Cone.Get(self.stage, cone_path)
        if not cone:
            cone = UsdGeom.Cone.Define(self.stage, cone_path)
            _usd_add_xform(cone)

        cone.GetRadiusAttr().Set(float(radius))
        cone.GetHeightAttr().Set(float(half_height * 2.0))
        cone.GetAxisAttr().Set("Y")

        self._shape_constructors[name] = UsdGeom.Cone

        if not is_template:
            _usd_set_xform(cone, pos, rot, (1.0, 1.0, 1.0), 0.0)

        return prim_path

    def render_box(
        self, name: str, pos: tuple, rot: tuple, extents: tuple, parent_body: str = None, is_template: bool = False
    ):
        """Debug helper to add a box for visualization

        Args:
            pos: The position of the sphere
            extents: The radius of the sphere
            name: A name for the USD prim on the stage
        """

        from pxr import UsdGeom, Sdf, Gf, Vt

        if is_template:
            prim_path = self._resolve_path(name, parent_body, is_template)
            blueprint = UsdGeom.Scope.Define(self.stage, prim_path)
            blueprint_prim = blueprint.GetPrim()
            blueprint_prim.SetInstanceable(True)
            blueprint_prim.SetSpecifier(Sdf.SpecifierClass)
            cube_path = prim_path.AppendChild("cube")
        else:
            cube_path = self._resolve_path(name, parent_body)
            prim_path = cube_path

        cube = UsdGeom.Cube.Get(self.stage, cube_path)
        if not cube:
            cube = UsdGeom.Cube.Define(self.stage, cube_path)
            _usd_add_xform(cube)

        self._shape_constructors[name] = UsdGeom.Cube
        self._shape_custom_scale[name] = extents

        if not is_template:
            _usd_set_xform(cube, pos, rot, extents, 0.0)

        return prim_path

    def render_ref(self, name: str, path: str, pos: tuple, rot: tuple, scale: tuple):
        from pxr import UsdGeom

        ref_path = "/root/" + name

        ref = UsdGeom.Xform.Get(self.stage, ref_path)
        if not ref:
            ref = UsdGeom.Xform.Define(self.stage, ref_path)
            ref.GetPrim().GetReferences().AddReference(path)
            _usd_add_xform(ref)

        # update transform
        _usd_set_xform(ref, pos, rot, scale, self.time)

    def render_mesh(
        self,
        name: str,
        points,
        indices,
        colors=None,
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
        update_topology=False,
        parent_body: str = None,
        is_template: bool = False,
    ):
        from pxr import UsdGeom, Sdf

        if is_template:
            prim_path = self._resolve_path(name, parent_body, is_template)
            blueprint = UsdGeom.Scope.Define(self.stage, prim_path)
            blueprint_prim = blueprint.GetPrim()
            blueprint_prim.SetInstanceable(True)
            blueprint_prim.SetSpecifier(Sdf.SpecifierClass)
            mesh_path = prim_path.AppendChild("mesh")
        else:
            mesh_path = self._resolve_path(name, parent_body)
            prim_path = mesh_path

        mesh = UsdGeom.Mesh.Get(self.stage, mesh_path)
        if not mesh:
            mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
            UsdGeom.Primvar(mesh.GetDisplayColorAttr()).SetInterpolation("vertex")
            _usd_add_xform(mesh)

            # force topology update on first frame
            update_topology = True

        mesh.GetPointsAttr().Set(points, self.time)

        if update_topology:
            idxs = np.array(indices).reshape(-1, 3)
            mesh.GetFaceVertexIndicesAttr().Set(idxs, self.time)
            mesh.GetFaceVertexCountsAttr().Set([3] * len(idxs), self.time)

        if colors:
            mesh.GetDisplayColorAttr().Set(colors, self.time)

        self._shape_constructors[name] = UsdGeom.Mesh
        self._shape_custom_scale[name] = scale

        if not is_template:
            _usd_set_xform(mesh, pos, rot, scale, self.time)

        return prim_path

    def render_line_list(self, name, vertices, indices, color, radius):
        """Debug helper to add a line list as a set of capsules

        Args:
            vertices: The vertices of the line-strip
            color: The color of the line
            time: The time to update at
        """

        from pxr import UsdGeom, Gf

        num_lines = int(len(indices) / 2)

        if num_lines < 1:
            return

        # look up rope point instancer
        instancer_path = self.root.GetPath().AppendChild(name)
        instancer = UsdGeom.PointInstancer.Get(self.stage, instancer_path)

        if not instancer:
            instancer = UsdGeom.PointInstancer.Define(self.stage, instancer_path)
            instancer_capsule = UsdGeom.Capsule.Define(self.stage, instancer.GetPath().AppendChild("capsule"))
            instancer_capsule.GetRadiusAttr().Set(radius)
            instancer.CreatePrototypesRel().SetTargets([instancer_capsule.GetPath()])
            # instancer.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Float3Array, "constant", 1)

        line_positions = []
        line_rotations = []
        line_scales = []

        for i in range(num_lines):
            pos0 = vertices[indices[i * 2 + 0]]
            pos1 = vertices[indices[i * 2 + 1]]

            (pos, rot, scale) = _compute_segment_xform(
                Gf.Vec3f(float(pos0[0]), float(pos0[1]), float(pos0[2])),
                Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])),
            )

            line_positions.append(pos)
            line_rotations.append(rot)
            line_scales.append(scale)
            # line_colors.append(Gf.Vec3f((float(i)/num_lines, 0.5, 0.5)))

        instancer.GetPositionsAttr().Set(line_positions, self.time)
        instancer.GetOrientationsAttr().Set(line_rotations, self.time)
        instancer.GetScalesAttr().Set(line_scales, self.time)
        instancer.GetProtoIndicesAttr().Set([0] * num_lines, self.time)

    #      instancer.GetPrimvar("displayColor").Set(line_colors, time)

    def render_line_strip(self, name: str, vertices, color: tuple, radius: float = 0.01):
        from pxr import UsdGeom, Gf

        num_lines = int(len(vertices) - 1)

        if num_lines < 1:
            return

        # look up rope point instancer
        instancer_path = self.root.GetPath().AppendChild(name)
        instancer = UsdGeom.PointInstancer.Get(self.stage, instancer_path)

        if not instancer:
            instancer = UsdGeom.PointInstancer.Define(self.stage, instancer_path)
            instancer_capsule = UsdGeom.Capsule.Define(self.stage, instancer.GetPath().AppendChild("capsule"))
            instancer_capsule.GetRadiusAttr().Set(radius)
            instancer.CreatePrototypesRel().SetTargets([instancer_capsule.GetPath()])

        line_positions = []
        line_rotations = []
        line_scales = []

        for i in range(num_lines):
            pos0 = vertices[i]
            pos1 = vertices[i + 1]

            (pos, rot, scale) = _compute_segment_xform(
                Gf.Vec3f(float(pos0[0]), float(pos0[1]), float(pos0[2])),
                Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])),
            )

            line_positions.append(pos)
            line_rotations.append(rot)
            line_scales.append(scale)

        instancer.GetPositionsAttr().Set(line_positions, self.time)
        instancer.GetOrientationsAttr().Set(line_rotations, self.time)
        instancer.GetScalesAttr().Set(line_scales, self.time)
        instancer.GetProtoIndicesAttr().Set([0] * num_lines, self.time)

        instancer_capsule = UsdGeom.Capsule.Get(self.stage, instancer.GetPath().AppendChild("capsule"))
        instancer_capsule.GetDisplayColorAttr().Set([Gf.Vec3f(color)], self.time)

    def render_points(self, name: str, points, radius, colors=None):
        from pxr import UsdGeom, Gf

        instancer_path = self.root.GetPath().AppendChild(name)
        instancer = UsdGeom.PointInstancer.Get(self.stage, instancer_path)
        radius_is_scalar = np.isscalar(radius)
        if not instancer:
            if colors is None:
                instancer = UsdGeom.PointInstancer.Define(self.stage, instancer_path)
                instancer_sphere = UsdGeom.Sphere.Define(self.stage, instancer.GetPath().AppendChild("sphere"))
                if radius_is_scalar:
                    instancer_sphere.GetRadiusAttr().Set(radius)
                else:
                    instancer_sphere.GetRadiusAttr().Set(1.0)
                    instancer.GetScalesAttr().Set(np.tile(radius, (3, 1)).T)

                instancer.CreatePrototypesRel().SetTargets([instancer_sphere.GetPath()])
                instancer.CreateProtoIndicesAttr().Set([0] * len(points))

                # set identity rotations
                quats = [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(points)
                instancer.GetOrientationsAttr().Set(quats, self.time)
            else:
                from pxr import Sdf

                instancer = UsdGeom.Points.Define(self.stage, instancer_path)

                instancer.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Float3Array, "vertex", 1)
                if radius_is_scalar:
                    instancer.GetWidthsAttr().Set([radius] * len(points))
                else:
                    instancer.GetWidthsAttr().Set(radius)

        if colors is None:
            instancer.GetPositionsAttr().Set(points, self.time)
        else:
            instancer.GetPointsAttr().Set(points, self.time)
            instancer.GetDisplayColorAttr().Set(colors, self.time)

    def update_body_transforms(self, body_q):
        from pxr import UsdGeom, Sdf

        if isinstance(body_q, wp.array):
            body_q = body_q.numpy()

        with Sdf.ChangeBlock():
            for b in range(self.model.body_count):
                node_name = self.body_names[b]
                node = UsdGeom.Xform(self.stage.GetPrimAtPath(self.root.GetPath().AppendChild(node_name)))

                # unpack rigid transform
                X_sb = wp.transform_expand(body_q[b])

                _usd_set_xform(node, X_sb.p, X_sb.q, (1.0, 1.0, 1.0), self.time)

    def save(self):
        try:
            self.stage.Save()
            return True
        except:
            print("Failed to save USD stage")
            return False
