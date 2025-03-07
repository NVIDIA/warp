# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import warp as wp


def _usd_add_xform(prim):
    from pxr import UsdGeom

    prim = UsdGeom.Xform(prim)
    prim.ClearXformOpOrder()

    prim.AddTranslateOp()
    prim.AddOrientOp()
    prim.AddScaleOp()


def _usd_set_xform(xform, pos: tuple, rot: tuple, scale: tuple, time):
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform(xform)

    xform_ops = xform.GetOrderedXformOps()

    if pos is not None:
        xform_ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), time)
    if rot is not None:
        xform_ops[1].Set(Gf.Quatf(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])), time)
    if scale is not None:
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
        try:
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
        except ImportError as e:
            raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).") from e

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

        dome_light = UsdLux.DomeLight.Define(stage, "/dome_light")
        dome_light.AddRotateXYZOp().Set((-90.0, -30.0, 0.0))
        dome_light.GetEnableColorTemperatureAttr().Set(True)
        dome_light.GetColorTemperatureAttr().Set(6150.0)
        dome_light.GetIntensityAttr().Set(1.0)
        dome_light.GetExposureAttr().Set(9.0)
        dome_light.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)

        distant_light = UsdLux.DistantLight.Define(stage, "/distant_light")
        distant_light.AddRotateXYZOp().Set((-35.0, 45.0, 0.0))
        distant_light.GetEnableColorTemperatureAttr().Set(True)
        distant_light.GetColorTemperatureAttr().Set(7250.0)
        distant_light.GetIntensityAttr().Set(1.0)
        distant_light.GetExposureAttr().Set(10.0)

    def begin_frame(self, time):
        self.time = round(time * self.fps)
        self.stage.SetEndTimeCode(self.time)

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
        custom_index: int = -1,
        visible: bool = True,
    ):
        if not visible:
            return
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
        from pxr import Sdf, UsdGeom

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

    def render_ground(self, size: float = 100.0, plane=None):
        from pxr import UsdGeom

        mesh = UsdGeom.Mesh.Define(self.stage, self.root.GetPath().AppendChild("ground"))
        mesh.CreateDoubleSidedAttr().Set(True)

        if self.up_axis == "X":
            points = ((0.0, size, -size), (0.0, -size, -size), (0.0, size, size), (0.0, -size, size))
            normals = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        elif self.up_axis == "Y":
            points = ((-size, 0.0, -size), (size, 0.0, -size), (-size, 0.0, size), (size, 0.0, size))
            normals = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        elif self.up_axis == "Z":
            points = ((-size, size, 0.0), (size, size, 0.0), (-size, -size, 0.0), (size, -size, 0.0))
            normals = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
        if plane is not None:
            normal = np.array(plane[:3])
            normal /= np.linalg.norm(normal)
            pos = plane[3] * normal
            axis_up = [0.0, 0.0, 0.0]
            axis_up["XYZ".index(self.up_axis)] = 1.0
            if np.allclose(normal, axis_up):
                # no rotation necessary
                q = (0.0, 0.0, 0.0, 1.0)
            else:
                c = np.cross(normal, axis_up)
                angle = np.arcsin(np.linalg.norm(c))
                axis = np.abs(c) / np.linalg.norm(c)
                q = wp.quat_from_axis_angle(axis, angle)
            tf = wp.transform(pos, q)
            points = [wp.transform_point(tf, wp.vec3(p)) for p in points]
            normals = [wp.transform_vector(tf, wp.vec3(n)) for n in normals]
        counts = (4,)
        indices = [0, 2, 3, 1]

        mesh.GetPointsAttr().Set(points)
        mesh.GetNormalsAttr().Set(normals)
        mesh.GetFaceVertexCountsAttr().Set(counts)
        mesh.GetFaceVertexIndicesAttr().Set(indices)

    def render_sphere(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        radius: float,
        parent_body: str = None,
        is_template: bool = False,
        color: tuple = None,
    ):
        """Debug helper to add a sphere for visualization

        Args:
            pos: The position of the sphere
            radius: The radius of the sphere
            name: A name for the USD prim on the stage
            color: The color of the sphere
        """

        from pxr import Gf, Sdf, UsdGeom

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

        if color is not None:
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(color)], self.time)

        self._shape_constructors[name] = UsdGeom.Sphere

        if not is_template:
            _usd_set_xform(sphere, pos, rot, (1.0, 1.0, 1.0), self.time)

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
        color: tuple = None,
    ):
        """
        Debug helper to add a capsule for visualization

        Args:
            pos: The position of the capsule
            radius: The radius of the capsule
            half_height: The half height of the capsule
            name: A name for the USD prim on the stage
            color: The color of the capsule
        """

        from pxr import Gf, Sdf, UsdGeom

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

        if color is not None:
            capsule.GetDisplayColorAttr().Set([Gf.Vec3f(color)], self.time)

        self._shape_constructors[name] = UsdGeom.Capsule

        if not is_template:
            _usd_set_xform(capsule, pos, rot, (1.0, 1.0, 1.0), self.time)

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
        color: tuple = None,
    ):
        """
        Debug helper to add a cylinder for visualization

        Args:
            pos: The position of the cylinder
            radius: The radius of the cylinder
            half_height: The half height of the cylinder
            name: A name for the USD prim on the stage
            color: The color of the cylinder
        """

        from pxr import Gf, Sdf, UsdGeom

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

        if color is not None:
            cylinder.GetDisplayColorAttr().Set([Gf.Vec3f(color)], self.time)

        self._shape_constructors[name] = UsdGeom.Cylinder

        if not is_template:
            _usd_set_xform(cylinder, pos, rot, (1.0, 1.0, 1.0), self.time)

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
        color: tuple = None,
    ):
        """
        Debug helper to add a cone for visualization

        Args:
            pos: The position of the cone
            radius: The radius of the cone
            half_height: The half height of the cone
            name: A name for the USD prim on the stage
            color: The color of the cone
        """

        from pxr import Gf, Sdf, UsdGeom

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

        if color is not None:
            cone.GetDisplayColorAttr().Set([Gf.Vec3f(color)], self.time)

        self._shape_constructors[name] = UsdGeom.Cone

        if not is_template:
            _usd_set_xform(cone, pos, rot, (1.0, 1.0, 1.0), self.time)

        return prim_path

    def render_box(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        extents: tuple,
        parent_body: str = None,
        is_template: bool = False,
        color: tuple = None,
    ):
        """Debug helper to add a box for visualization

        Args:
            pos: The position of the box
            extents: The radius of the box
            name: A name for the USD prim on the stage
            color: The color of the box
        """

        from pxr import Gf, Sdf, UsdGeom

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

        if color is not None:
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(color)], self.time)

        self._shape_constructors[name] = UsdGeom.Cube
        self._shape_custom_scale[name] = extents

        if not is_template:
            _usd_set_xform(cube, pos, rot, extents, self.time)

        return prim_path

    def render_ref(self, name: str, path: str, pos: tuple, rot: tuple, scale: tuple, color: tuple = None):
        from pxr import Gf, Usd, UsdGeom

        ref_path = "/root/" + name

        ref = UsdGeom.Xform.Get(self.stage, ref_path)
        if not ref:
            ref = UsdGeom.Xform.Define(self.stage, ref_path)
            ref.GetPrim().GetReferences().AddReference(path)
            _usd_add_xform(ref)

        # update transform
        _usd_set_xform(ref, pos, rot, scale, self.time)

        if color is not None:
            it = iter(Usd.PrimRange(ref.GetPrim()))
            for prim in it:
                if prim.IsA(UsdGeom.Gprim):
                    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(color)], self.time)
                    it.PruneChildren()

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
        from pxr import Sdf, UsdGeom

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
            if colors is not None and len(colors) == 3:
                color_interp = "constant"
            else:
                color_interp = "vertex"

            UsdGeom.Primvar(mesh.GetDisplayColorAttr()).SetInterpolation(color_interp)
            _usd_add_xform(mesh)

            # force topology update on first frame
            update_topology = True

        mesh.GetPointsAttr().Set(points, self.time)

        if update_topology:
            idxs = np.array(indices).reshape(-1, 3)
            mesh.GetFaceVertexIndicesAttr().Set(idxs, self.time)
            mesh.GetFaceVertexCountsAttr().Set([3] * len(idxs), self.time)

        if colors is not None:
            if len(colors) == 3:
                colors = (colors,)

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

        from pxr import Gf, UsdGeom

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
        from pxr import Gf, UsdGeom

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
        from pxr import Gf, UsdGeom

        instancer_path = self.root.GetPath().AppendChild(name)
        instancer = UsdGeom.PointInstancer.Get(self.stage, instancer_path)
        radius_is_scalar = np.isscalar(radius)
        if not instancer:
            if colors is None or len(colors) == 3:
                instancer = UsdGeom.PointInstancer.Define(self.stage, instancer_path)
                instancer_sphere = UsdGeom.Sphere.Define(self.stage, instancer.GetPath().AppendChild("sphere"))
                if radius_is_scalar:
                    instancer_sphere.GetRadiusAttr().Set(radius)
                else:
                    instancer_sphere.GetRadiusAttr().Set(1.0)
                    instancer.GetScalesAttr().Set(np.tile(radius, (3, 1)).T)

                if colors is not None:
                    instancer_sphere.GetDisplayColorAttr().Set([Gf.Vec3f(colors)], self.time)

                instancer.CreatePrototypesRel().SetTargets([instancer_sphere.GetPath()])
                instancer.CreateProtoIndicesAttr().Set([0] * len(points))

                # set identity rotations
                quats = [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(points)
                instancer.GetOrientationsAttr().Set(quats, self.time)
            else:
                instancer = UsdGeom.Points.Define(self.stage, instancer_path)

                if radius_is_scalar:
                    instancer.GetWidthsAttr().Set([radius * 2.0] * len(points))
                else:
                    instancer.GetWidthsAttr().Set(radius * 2.0)

        if colors is None or len(colors) == 3:
            instancer.GetPositionsAttr().Set(points, self.time)
        else:
            instancer.GetPointsAttr().Set(points, self.time)
            instancer.GetDisplayColorAttr().Set(colors, self.time)

    def update_body_transforms(self, body_q):
        from pxr import Sdf, UsdGeom

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
        except Exception as e:
            print("Failed to save USD stage:", e)
            return False

        file_path = self.stage.GetRootLayer().realPath
        print(f"Saved the USD stage file at `{file_path}`")
        return True
