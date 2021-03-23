"""This optional module contains a built-in renderer for the USD data
format that can be used to visualize time-sampled simulation data.

Users should create a simulation model and integrator and periodically
call :func:`UsdRenderer.update()` to write time-sampled simulation data to the USD stage.

Example:

    >>> # construct a new USD stage
    >>> stage = Usd.Stage.CreateNew("my_stage.usda")
    >>> renderer = og.render.UsdRenderer(model, stage)
    >>>
    >>> time = 0.0
    >>>
    >>> for i in range(100):
    >>>
    >>>     # update simulation here
    >>>     # ....
    >>>
    >>>     # update renderer
    >>>     stage.update(state, time)
    >>>     time += dt
    >>>
    >>> # write stage to file
    >>> stage.Save()

Note:
    You must have the Pixar USD bindings installed to use this module
    please see https://developer.nvidia.com/usd to obtain precompiled
    USD binaries and installation instructions.
    
"""

from pxr import Usd, UsdGeom, Gf, Sog

import og.sim
import og.util

import math


def _usd_add_xform(prim):

    prim.ClearXformOpOrder()

    t = prim.AddTranslateOp()
    r = prim.AddOrientOp()
    s = prim.AddScaleOp()


def _usd_set_xform(xform, transform, scale, time):

    xform_ops = xform.GetOrderedXformOps()

    pos = tuple(transform[0])
    rot = tuple(transform[1])

    xform_ops[0].Set(Gf.Vec3d(pos), time)
    xform_ops[1].Set(Gf.Quatf(rot[3], rot[0], rot[1], rot[2]), time)
    xform_ops[2].Set(Gf.Vec3d(scale), time)

# transforms a cylinder such that it connects the two points pos0, pos1
def _compute_segment_xform(pos0, pos1):

    mid = (pos0 + pos1) * 0.5
    height = (pos1 - pos0).GetLength()

    dir = (pos1 - pos0) / height

    rot = Gf.Rotation()
    rot.SetRotateInto((0.0, 0.0, 1.0), Gf.Vec3d(dir))

    scale = Gf.Vec3f(1.0, 1.0, height)

    return (mid, Gf.Quath(rot.GetQuat()), scale)


class UsdRenderer:
    """A USD renderer
    """  
    def __init__(self, model: og.model.Model, stage):
        """Construct a UsdRenderer object
        
        Args:
            model: A simulation model
            stage (Usd.Stage): A USD stage (either in memory or on disk)            
        """

        self.stage = stage
        self.model = model

        self.draw_points = True
        self.draw_springs = False
        self.draw_triangles = False

        if (stage.GetPrimAtPath("/root")):
            stage.RemovePrim("/root")

        self.root = UsdGeom.Xform.Define(stage, '/root')

        # add sphere instancer for particles
        self.particle_instancer = UsdGeom.PointInstancer.Define(stage, self.root.GetPath().AppendChild("particle_instancer"))
        self.particle_instancer_sphere = UsdGeom.Sphere.Define(stage, self.particle_instancer.GetPath().AppendChild("sphere"))
        self.particle_instancer_sphere.GetRadiusAttr().Set(model.particle_radius)

        self.particle_instancer.CreatePrototypesRel().SetTargets([self.particle_instancer_sphere.GetPath()])
        self.particle_instancer.CreateProtoIndicesAttr().Set([0] * model.particle_count)

        # add line instancer
        if (self.model.spring_count > 0):
            self.spring_instancer = UsdGeom.PointInstancer.Define(stage, self.root.GetPath().AppendChild("spring_instancer"))
            self.spring_instancer_cylinder = UsdGeom.Capsule.Define(stage, self.spring_instancer.GetPath().AppendChild("cylinder"))
            self.spring_instancer_cylinder.GetRadiusAttr().Set(0.01)

            self.spring_instancer.CreatePrototypesRel().SetTargets([self.spring_instancer_cylinder.GetPath()])
            self.spring_instancer.CreateProtoIndicesAttr().Set([0] * model.spring_count)

        self.stage.SetDefaultPrim(self.root.GetPrim())

        # time codes
        try:
            self.stage.SetStartTimeCode(0.0)
            self.stage.SetEndTimeCode(0.0)
            self.stage.SetTimeCodesPerSecond(1.0)
        except:
            pass

        # add dynamic cloth mesh
        if (model.tri_count > 0):

            self.cloth_mesh = UsdGeom.Mesh.Define(stage, self.root.GetPath().AppendChild("cloth"))

            self.cloth_remap = {}
            self.cloth_verts = []
            self.cloth_indices = []

            # USD needs a contiguous vertex buffer, use a dict to map from simulation indices->render indices
            indices = self.model.tri_indices.flatten().tolist()

            for i in indices:

                if i not in self.cloth_remap:

                    # copy vertex
                    new_index = len(self.cloth_verts)

                    self.cloth_verts.append(self.model.particle_q[i].tolist())
                    self.cloth_indices.append(new_index)

                    self.cloth_remap[i] = new_index

                else:
                    self.cloth_indices.append(self.cloth_remap[i])

            self.cloth_mesh.GetPointsAttr().Set(self.cloth_verts)
            self.cloth_mesh.GetFaceVertexIndicesAttr().Set(self.cloth_indices)
            self.cloth_mesh.GetFaceVertexCountsAttr().Set([3] * model.tri_count)

        else:
            self.cloth_mesh = None

        # built-in ground plane
        if (model.ground):

            size = 10.0

            mesh = UsdGeom.Mesh.Define(stage, self.root.GetPath().AppendChild("plane_0"))
            mesh.CreateDoubleSidedAttr().Set(True)

            points = ((-size, 0.0, -size), (size, 0.0, -size), (size, 0.0, size), (-size, 0.0, size))
            normals = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
            counts = (4, )
            indices = [0, 1, 2, 3]

            mesh.GetPointsAttr().Set(points)
            mesh.GetNormalsAttr().Set(normals)
            mesh.GetFaceVertexCountsAttr().Set(counts)
            mesh.GetFaceVertexIndicesAttr().Set(indices)

        # add rigid bodies xform root
        for b in range(model.link_count):

            xform = UsdGeom.Xform.Define(stage, self.root.GetPath().AppendChild("body_" + str(b)))
            _usd_add_xform(xform)

        # add rigid body shapes
        for s in range(model.shape_count):

            parent_path = self.root.GetPath()
            if model.shape_body[s] >= 0:
                parent_path = parent_path.AppendChild("body_" + str(model.shape_body[s].item()))

            geo_type = model.shape_geo_type[s].item()
            geo_scale = model.shape_geo_scale[s].tolist()
            geo_src = model.shape_geo_src[s]

            # shape transform in body frame
            X_bs = og.util.transform_expand(model.shape_transform[s].tolist())

            if (geo_type == og.sim.GEO_PLANE):

                # plane mesh
                size = 1000.0

                mesh = UsdGeom.Mesh.Define(stage, parent_path.AppendChild("plane_" + str(s)))
                mesh.CreateDoubleSidedAttr().Set(True)

                points = ((-size, 0.0, -size), (size, 0.0, -size), (size, 0.0, size), (-size, 0.0, size))
                normals = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
                counts = (4, )
                indices = [0, 1, 2, 3]

                mesh.GetPointsAttr().Set(points)
                mesh.GetNormalsAttr().Set(normals)
                mesh.GetFaceVertexCountsAttr().Set(counts)
                mesh.GetFaceVertexIndicesAttr().Set(indices)

            elif (geo_type == og.sim.GEO_SPHERE):

                mesh = UsdGeom.Sphere.Define(stage, parent_path.AppendChild("sphere_" + str(s)))
                mesh.GetRadiusAttr().Set(geo_scale[0])

                _usd_add_xform(mesh)
                _usd_set_xform(mesh, X_bs, (1.0, 1.0, 1.0), 0.0)

            elif (geo_type == og.sim.GEO_CAPSULE):
                mesh = UsdGeom.Capsule.Define(stage, parent_path.AppendChild("capsule_" + str(s)))
                mesh.GetRadiusAttr().Set(geo_scale[0])
                mesh.GetHeightAttr().Set(geo_scale[1] * 2.0)

                # geometry transform w.r.t shape, convert USD geometry to physics engine convention
                X_sg = og.util.transform((0.0, 0.0, 0.0), og.util.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi * 0.5))
                X_bg = og.util.transform_multiply(X_bs, X_sg)

                _usd_add_xform(mesh)
                _usd_set_xform(mesh, X_bg, (1.0, 1.0, 1.0), 0.0)

            elif (geo_type == og.sim.GEO_BOX):
                mesh = UsdGeom.Cube.Define(stage, parent_path.AppendChild("box_" + str(s)))
                #mesh.GetSizeAttr().Set((geo_scale[0], geo_scale[1], geo_scale[2]))

                _usd_add_xform(mesh)
                _usd_set_xform(mesh, X_bs, (geo_scale[0], geo_scale[1], geo_scale[2]), 0.0)

            elif (geo_type == og.sim.GEO_MESH):

                mesh = UsdGeom.Mesh.Define(stage, parent_path.AppendChild("mesh_" + str(s)))
                mesh.GetPointsAttr().Set(geo_src.vertices)
                mesh.GetFaceVertexIndicesAttr().Set(geo_src.indices)
                mesh.GetFaceVertexCountsAttr().Set([3] * int(len(geo_src.indices) / 3))

                _usd_add_xform(mesh)
                _usd_set_xform(mesh, X_bs, (geo_scale[0], geo_scale[1], geo_scale[2]), 0.0)

            elif (geo_type == og.sim.GEO_Sog):
                pass

    def update(self, state: og.model.State, time: float):
        """Update the USD stage with latest simulation data
        
        Args:
            state: Current state of the simulation
            time: The current time to update at in seconds
        """

        try:
            self.stage.SetEndTimeCode(time)
        except:
            pass

        # convert to list
        if (self.model.particle_count):
            
            particle_q = state.particle_q.tolist()

            # point instancer 
            if (self.draw_points):

                particle_orientations = [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * self.model.particle_count

                self.particle_instancer.GetPositionsAttr().Set(particle_q, time)
                self.particle_instancer.GetOrientationsAttr().Set(particle_orientations, time)

        # update cloth
        if (self.cloth_mesh):

            for k, v in self.cloth_remap.items():
                self.cloth_verts[v] = particle_q[k]

            self.cloth_mesh.GetPointsAttr().Set(self.cloth_verts, time)

        # update springs
        if (self.model.spring_count > 0):

            line_positions = []
            line_rotations = []
            line_scales = []

            for i in range(self.model.spring_count):

                index0 = self.model.spring_indices[i * 2 + 0]
                index1 = self.model.spring_indices[i * 2 + 1]

                pos0 = particle_q[index0]
                pos1 = particle_q[index1]

                (pos, rot, scale) = _compute_segment_xform(Gf.Vec3f(pos0), Gf.Vec3f(pos1))

                line_positions.append(pos)
                line_rotations.append(rot)
                line_scales.append(scale)

            self.spring_instancer.GetPositionsAttr().Set(line_positions, time)
            self.spring_instancer.GetOrientationsAttr().Set(line_rotations, time)
            self.spring_instancer.GetScalesAttr().Set(line_scales, time)

        # rigids
        for b in range(self.model.link_count):

            #xform = UsdGeom.Xform.Define(self.stage, self.root.GetPath().AppendChild("body_" + str(b)))

            node = UsdGeom.Xform(self.stage.GetPrimAtPath(self.root.GetPath().AppendChild("body_" + str(b))))

            # unpack rigid spatial_transform
            X_sb = og.util.transform_expand(state.body_X_sc[b].tolist())

            _usd_set_xform(node, X_sb, (1.0, 1.0, 1.0), time)

    def add_sphere(self, pos: tuple, radius: float, name: str, time: float=0.0):
        """Debug helper to add a sphere for visualization
        
        Args:
            pos: The position of the sphere
            radius: The radius of the sphere
            name: A name for the USD prim on the stage
        """

        sphere_path = self.root.GetPath().AppendChild(name)
        sphere = UsdGeom.Sphere.Get(self.stage, sphere_path)
        if not sphere:
            sphere = UsdGeom.Sphere.Define(self.stage, sphere_path)
        
        sphere.GetRadiusAttr().Set(radius, time)

        mat = Gf.Matrix4d()
        mat.SetIdentity()
        mat.SetTranslateOnly(Gf.Vec3d(pos))

        op = sphere.MakeMatrixXform()
        op.Set(mat, time)

    def add_box(self, pos: tuple, extents: float, name: str, time: float=0.0):
        """Debug helper to add a box for visualization
        
        Args:
            pos: The position of the sphere
            extents: The radius of the sphere
            name: A name for the USD prim on the stage
        """

        sphere_path = self.root.GetPath().AppendChild(name)
        sphere = UsdGeom.Cube.Get(self.stage, sphere_path)
        if not sphere:
            sphere = UsdGeom.Cube.Define(self.stage, sphere_path)
        
        #sphere.GetSizeAttr().Set((extents[0]*2.0, extents[1]*2.0, extents[2]*2.0), time)

        mat = Gf.Matrix4d()
        mat.SetIdentity()
        mat.SetScale(extents)
        mat.SetTranslateOnly(Gf.Vec3d(pos))

        op = sphere.MakeMatrixXform()
        op.Set(mat, time)        

    def add_mesh(self, name: str, path: str, transform, scale, time: float):

        ref_path = "/root/" + name

        ref = UsdGeom.Xform.Get(self.stage, ref_path)
        if not ref:
            ref = UsdGeom.Xform.Define(self.stage, ref_path)
            ref.GetPrim().GetReferences().AddReference(path)
            
            _usd_add_xform(ref)

        # update transform
        _usd_set_xform(ref, transform, scale, time)

    def add_line_list(self, vertices, color, time, name, radius):
        """Debug helper to add a line list as a set of capsules
        
        Args:
            vertices: The vertices of the line-strip
            color: The color of the line
            time: The time to update at
        """
        
        num_lines = int(len(vertices)/2)

        if (num_lines < 1):
            return

        # look up rope point instancer
        instancer_path = self.root.GetPath().AppendChild(name)
        instancer = UsdGeom.PointInstancer.Get(self.stage, instancer_path)

        if not instancer:
            instancer = UsdGeom.PointInstancer.Define(self.stage, instancer_path)
            instancer_capsule = UsdGeom.Capsule.Define(self.stage, instancer.GetPath().AppendChild("capsule"))
            instancer_capsule.GetRadiusAttr().Set(radius)
            instancer.CreatePrototypesRel().SetTargets([instancer_capsule.GetPath()])
            instancer.CreatePrimvar("displayColor", Sog.ValueTypeNames.Float3Array, "constant", 1)

        line_positions = []
        line_rotations = []
        line_scales = []
#        line_colors = []

        for i in range(num_lines):

            pos0 = vertices[i*2+0]
            pos1 = vertices[i*2+1]

            (pos, rot, scale) = _compute_segment_xform(Gf.Vec3f(pos0), Gf.Vec3f(pos1))

            line_positions.append(pos)
            line_rotations.append(rot)
            line_scales.append(scale)
            #line_colors.append(Gf.Vec3f((float(i)/num_lines, 0.5, 0.5)))

        instancer.GetPositionsAttr().Set(line_positions, time)
        instancer.GetOrientationsAttr().Set(line_rotations, time)
        instancer.GetScalesAttr().Set(line_scales, time)
        instancer.GetProtoIndicesAttr().Set([0] * num_lines, time)
 #      instancer.GetPrimvar("displayColor").Set(line_colors, time)        


    def add_line_strip(self, vertices: og.sim.List[og.sim.Vec3], color: tuple, time: float, name: str, radius: float=0.01):
        """Debug helper to add a line strip as a connected list of capsules
        
        Args:
            vertices: The vertices of the line-strip
            color: The color of the line
            time: The time to update at
        """
        
        num_lines = int(len(vertices)-1)

        if (num_lines < 1):
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
            pos1 = vertices[i+1]

            (pos, rot, scale) = _compute_segment_xform(Gf.Vec3f(pos0), Gf.Vec3f(pos1))

            line_positions.append(pos)
            line_rotations.append(rot)
            line_scales.append(scale)

        instancer.GetPositionsAttr().Set(line_positions, time)
        instancer.GetOrientationsAttr().Set(line_rotations, time)
        instancer.GetScalesAttr().Set(line_scales, time)
        instancer.GetProtoIndicesAttr().Set([0] * num_lines, time)

        instancer_capsule = UsdGeom.Capsule.Get(self.stage, instancer.GetPath().AppendChild("capsule"))
        instancer_capsule.GetDisplayColorAttr().Set([Gf.Vec3f(color)], time)

        


