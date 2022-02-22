from pxr import Usd, UsdGeom, Gf, Sdf

import math

import warp as wp

def _usd_add_xform(prim):

    prim.ClearXformOpOrder()

    t = prim.AddTranslateOp()
    r = prim.AddOrientOp()
    s = prim.AddScaleOp()


def _usd_set_xform(xform, pos: tuple, rot: tuple, scale: tuple, time):

    xform_ops = xform.GetOrderedXformOps()

    xform_ops[0].Set(Gf.Vec3d(pos), time)
    xform_ops[1].Set(Gf.Quatf(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])), time)
    xform_ops[2].Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])), time)

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
    def __init__(self, path):
        """Construct a UsdRenderer object
        
        Args:
            model: A simulation model
            stage (Usd.Stage): A USD stage (either in memory or on disk)            
        """

        self.stage = stage = Usd.Stage.CreateNew(path)

        self.draw_points = True
        self.draw_springs = False
        self.draw_triangles = False

        self.root = UsdGeom.Xform.Define(stage, '/root')

        self.stage.SetDefaultPrim(self.root.GetPrim())
        self.stage.SetStartTimeCode(0.0)
        self.stage.SetEndTimeCode(0.0)
        self.stage.SetTimeCodesPerSecond(1.0)

    def begin_frame(self, time):
        self.stage.SetEndTimeCode(time)
        self.time = time

    def end_frame(self):
        pass

    def render_ground(self, size: float=100.0):

        mesh = UsdGeom.Mesh.Define(self.stage, self.root.GetPath().AppendChild("ground"))
        mesh.CreateDoubleSidedAttr().Set(True)

        points = ((-size, 0.0, -size), (size, 0.0, -size), (size, 0.0, size), (-size, 0.0, size))
        normals = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        counts = (4, )
        indices = [0, 1, 2, 3]

        mesh.GetPointsAttr().Set(points)
        mesh.GetNormalsAttr().Set(normals)
        mesh.GetFaceVertexCountsAttr().Set(counts)
        mesh.GetFaceVertexIndicesAttr().Set(indices)

    def render_sphere(self, name: str, pos: tuple, rot: tuple, radius: float):
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
            _usd_add_xform(sphere)
        
        sphere.GetRadiusAttr().Set(radius, self.time)

        # mat = Gf.Matrix4d()
        # mat.SetIdentity()
        # mat.SetTranslateOnly(Gf.Vec3d(pos[0], pos[1], pos[2]))

        # op = sphere.MakeMatrixXform()
        # op.Set(mat, self.time)
        _usd_set_xform(sphere, pos, rot, (1.0, 1.0, 1.0), self.time)


    def render_box(self, name: str, pos: tuple, rot: tuple, extents: tuple):
        """Debug helper to add a box for visualization
        
        Args:
            pos: The position of the sphere
            extents: The radius of the sphere
            name: A name for the USD prim on the stage
        """

        box_path = self.root.GetPath().AppendChild(name)
        box = UsdGeom.Cube.Get(self.stage, box_path)
        if not box:
            box = UsdGeom.Cube.Define(self.stage, box_path)
            _usd_add_xform(box)

        # update transform        
        _usd_set_xform(box, pos, rot, extents, self.time)
    

    def render_ref(self, name: str, path: str, pos: tuple, rot: tuple, scale: tuple):

        ref_path = "/root/" + name

        ref = UsdGeom.Xform.Get(self.stage, ref_path)
        if not ref:
            ref = UsdGeom.Xform.Define(self.stage, ref_path)
            ref.GetPrim().GetReferences().AddReference(path)
            _usd_add_xform(ref)

        # update transform
        _usd_set_xform(ref, pos, rot, scale, self.time)


    def render_mesh(self, name: str, points, indices, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), scale=(1.0, 1.0, 1.0)):
        
        mesh_path = self.root.GetPath().AppendChild(name)
        mesh = UsdGeom.Mesh.Get(self.stage, mesh_path)
        if not mesh:
            
            mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
            _usd_add_xform(mesh)

        mesh.GetPointsAttr().Set(points, self.time)
        mesh.GetFaceVertexIndicesAttr().Set(indices, self.time)
        mesh.GetFaceVertexCountsAttr().Set([3] * int(len(indices)/3), self.time)

        _usd_set_xform(mesh, pos, rot, scale, self.time)


    def render_line_list(self, name, vertices, indices, color, radius):
        """Debug helper to add a line list as a set of capsules
        
        Args:
            vertices: The vertices of the line-strip
            color: The color of the line
            time: The time to update at
        """
        
        num_lines = int(len(indices)/2)

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
            #instancer.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Float3Array, "constant", 1)

        line_positions = []
        line_rotations = []
        line_scales = []

        for i in range(num_lines):

            pos0 = vertices[indices[i*2+0]]
            pos1 = vertices[indices[i*2+1]]

            (pos, rot, scale) = _compute_segment_xform(Gf.Vec3f(float(pos0[0]), float(pos0[1]), float(pos0[2])), Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])))

            line_positions.append(pos)
            line_rotations.append(rot)
            line_scales.append(scale)
            #line_colors.append(Gf.Vec3f((float(i)/num_lines, 0.5, 0.5)))

        instancer.GetPositionsAttr().Set(line_positions, self.time)
        instancer.GetOrientationsAttr().Set(line_rotations, self.time)
        instancer.GetScalesAttr().Set(line_scales, self.time)
        instancer.GetProtoIndicesAttr().Set([0] * num_lines, self.time)
 #      instancer.GetPrimvar("displayColor").Set(line_colors, time)        


    def render_line_strip(self, name: str, vertices, color: tuple, radius: float=0.01):

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

        instancer.GetPositionsAttr().Set(line_positions, self.time)
        instancer.GetOrientationsAttr().Set(line_rotations, self.time)
        instancer.GetScalesAttr().Set(line_scales, self.time)
        instancer.GetProtoIndicesAttr().Set([0] * num_lines, self.time)

        instancer_capsule = UsdGeom.Capsule.Get(self.stage, instancer.GetPath().AppendChild("capsule"))
        instancer_capsule.GetDisplayColorAttr().Set([Gf.Vec3f(color)], self.time)

    def render_points(self, name: str, points, radius):

        instancer_path = self.root.GetPath().AppendChild(name)
        instancer = UsdGeom.PointInstancer.Get(self.stage, instancer_path)

        if not instancer:

            instancer = UsdGeom.PointInstancer.Define(self.stage, instancer_path)
            instancer_sphere = UsdGeom.Sphere.Define(self.stage, instancer.GetPath().AppendChild("sphere"))
            instancer_sphere.GetRadiusAttr().Set(radius)

            instancer.CreatePrototypesRel().SetTargets([instancer_sphere.GetPath()])
            instancer.CreateProtoIndicesAttr().Set([0] * len(points))

        quats = [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(points)

        instancer.GetPositionsAttr().Set(points, self.time)
        instancer.GetOrientationsAttr().Set(quats, self.time)


    def save(self):
        try:
            self.stage.Save()
        except:
            print("Failed to save USD stage")





