"""
This is the implementation of the OGN node defined in OgnParticleVolume.ogn
"""

# Array or tuple values are accessed as numpy arrays so you probably need this import
import math
from warp.types import uint64

import numpy as np
import warp as wp

from pxr import Usd, UsdGeom, Gf, Sdf

import omni.timeline
import omni.appwindow
import omni.usd

import omni.graph.core as og

profile_enabled = False

# # helper to get the transform for a bundle prim
# def read_transform_bundle(bundle):   
#     # xform = bundle.attribute_by_name("transform").value.reshape(4,4)
#     # return Gf.Matrix4d(xform)
#     stage = omni.usd.get_context().get_stage()
#     prim = UsdGeom.Xformable(stage.GetPrimAtPath(bundle.bundle.get_prim_path()))
#     return prim.ComputeLocalToWorldTransform(0.0)

# def read_world_bounds_bundle(bundle):
#     stage = omni.usd.get_context().get_stage()
#     prim = UsdGeom.Xformable(stage.GetPrimAtPath(bundle.bundle.get_prim_path()))
#     return prim.ComputeLocalToWorldTransform(0.0)

#     # lower = bundle.attribute_by_name("bboxMinCorner").value
#     # upper = bundle.attribute_by_name("bboxMaxCorner").value
#     # bbox_xform = bundle.attribute_by_name("bboxTransform").value.reshape(4,4)

#     # box = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(lower[0], lower[1], lower[2]),
#     #                            Gf.Vec3d(upper[0], upper[1], upper[2])),
#     #                            Gf.Matrix4d(bbox_xform))
    
#     return box


def read_transform_bundle(bundle):
    timeline =  omni.timeline.get_timeline_interface()
    time = timeline.get_current_time()*timeline.get_time_codes_per_seconds()

    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Xformable(stage.GetPrimAtPath(bundle.bundle.get_prim_path()))
    return prim.ComputeLocalToWorldTransform(time)

def read_bounds_bundle(bundle):
    timeline =  omni.timeline.get_timeline_interface()
    time = timeline.get_current_time()*timeline.get_time_codes_per_seconds()

    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Xformable(stage.GetPrimAtPath(bundle.bundle.get_prim_path()))
    
    return prim.ComputeWorldBound(time, purpose1="default")

def triangulate(counts, indices):

    # triangulate
    num_tris = np.sum(np.subtract(counts, 2))
    num_tri_vtx = num_tris * 3
    tri_indices = np.zeros(num_tri_vtx, dtype=int)
    ctr = 0
    wedgeIdx = 0
    for nb in counts:
        for i in range(nb-2):
            tri_indices[ctr] = indices[wedgeIdx]
            tri_indices[ctr + 1] = indices[wedgeIdx + i + 1]
            tri_indices[ctr + 2] = indices[wedgeIdx + i + 2]
            ctr+=3
        wedgeIdx+=nb

    return tri_indices


def add_bundle_data(bundle, name, data, type):
    
    attr = bundle.attribute_by_name(name)
    if attr is None:
        attr = bundle.insert((type, name))

    attr.size = len(data)
    attr.cpu_value = data.astype(np.float32)


# transform points from local space to world space given a mat44
@wp.kernel
def transform_points(src: wp.array(dtype=wp.vec3),
                     dest: wp.array(dtype=wp.vec3),
                     xform: wp.mat44):

    tid = wp.tid()

    p = src[tid]
    m = wp.transform_point(xform, p)

    dest[tid] = m

 
   

@wp.kernel
def sample_mesh(mesh: wp.uint64,
                lower: wp.vec3,
                spacing: float,
                dim_x: int,
                dim_y: int,
                dim_z: int,
                sdf_min: float,
                sdf_max: float,
                points: wp.array(dtype=wp.vec3),
                point_count: wp.array(dtype=int),
                point_max: int):

    tid = wp.tid()

    # generate grid coord
    x = tid%dim_x
    tid /= dim_x

    y = tid%dim_y
    tid /= dim_y

    z = tid

    grid_pos = lower + wp.vec3(float(x), float(y), float(z))*spacing

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 1000.0

    if (wp.mesh_query_point(mesh, grid_pos, max_dist, sign, face_index, face_u, face_v)):

        p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

        delta = grid_pos-p
        
        dist = wp.length(delta)*sign

        # mesh collision
        #if (dist < 10.0):
        if (dist >= sdf_min and dist <= sdf_max):
            point_index = wp.atomic_add(point_count, 0, 1)
            
            if (point_index < point_max):
                points[point_index] = grid_pos



class OgnParticleVolumeState:

    def __init__(self):

        self.initialized = False

class OgnParticleVolume:

    @staticmethod
    def internal_state():

        return OgnParticleVolumeState()

    """
    """
    @staticmethod
    def compute(db) -> bool:
        """Run simulation"""

        state = db.internal_state
        mesh = db.inputs.shape

        device = "cuda"

        if mesh.valid and (state.initialized == False or db.inputs.execIn):

            mesh_points = mesh.attribute_by_name(db.tokens.points).value
            mesh_indices = mesh.attribute_by_name(db.tokens.faceVertexIndices).value
            mesh_counts =  mesh.attribute_by_name(db.tokens.faceVertexCounts).value
            mesh_xform = read_transform_bundle(mesh)

            num_points = len(mesh_points)

            if (num_points):
                
                with wp.ScopedTimer("Prepare Mesh", active=profile_enabled):

                    # triangulate
                    mesh_tri_indices = triangulate(mesh_counts, mesh_indices)
                    
                    # transform to world space
                    mesh_points_local = wp.array(mesh_points, dtype=wp.vec3, device=device)
                    mesh_points_world = wp.empty(num_points, dtype=wp.vec3, device=device)
                    
                    wp.launch(
                        kernel=transform_points, 
                        dim=num_points, 
                        inputs=[
                            mesh_points_local,
                            mesh_points_world, 
                            np.array(mesh_xform).T], 
                        device=device)

                    # create Warp mesh
                    state.mesh = wp.Mesh(
                        points=mesh_points_world,
                        velocities=wp.zeros(num_points, dtype=wp.vec3, device=device),
                        indices=wp.array(mesh_tri_indices, dtype=int, device=device))

                with wp.ScopedTimer("Sample Mesh", active=profile_enabled):

                    bounds = read_bounds_bundle(mesh).ComputeAlignedBox()
                    lower = bounds.GetMin()
                    size = bounds.GetSize()

                    if (db.inputs.spacing <= 0.0):
                        print("Spacing must be positive")
                        return False

                    dim_x = int(size[0]/db.inputs.spacing)+1
                    dim_y = int(size[1]/db.inputs.spacing)+1
                    dim_z = int(size[2]/db.inputs.spacing)+1

                    points_max = db.inputs.max_points

                    #print(f"Creating particle grid with dim {dim_x}x{dim_y}x{dim_z}, lower {lower[0]}, {lower[1]}, {lower[2]} size {size[0]}, {size[1]}, {size[2]}, spacing {db.inputs.spacing}")

                    if (dim_x*dim_y*dim_z > points_max):
                        print(f"Trying to create particle volume with > {points_max} particles, increase spacing or geometry size")
                        return False

                    points = wp.zeros(points_max, dtype=wp.vec3, device=device)
                    points_counter = wp.zeros(1, dtype=int, device=device)
                
                    wp.launch(kernel=sample_mesh, 
                            dim=dim_x*dim_y*dim_z, 
                            inputs=[
                                state.mesh.id,
                                lower,
                                db.inputs.spacing,
                                dim_x,
                                dim_y,
                                dim_z,
                                db.inputs.sdf_min,
                                db.inputs.sdf_max,
                                points,
                                points_counter,
                                points_max], 
                            device="cuda")
                    
                    num_points = min(int(points_counter.numpy()[0]), points_max)
                    
                    # bring back to host
                    points = points.numpy()[0:num_points]

                    # jitter
                    points = points + np.random.rand(*points.shape)*db.inputs.spacing*db.inputs.spacing_jitter
                    velocities = np.tile(db.inputs.velocity, (len(points), 1))

                    state.initialized = True

                    add_bundle_data(db.outputs.particles, name="points", data=points, type=og.Type(og.BaseDataType.FLOAT, tuple_count=3, array_depth=1))
                    add_bundle_data(db.outputs.particles, name="velocities", data=velocities, type=og.Type(og.BaseDataType.FLOAT, tuple_count=3, array_depth=1))


        return True
