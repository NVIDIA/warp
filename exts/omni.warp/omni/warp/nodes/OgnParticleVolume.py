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

# helper to get the transform for a bundle prim
def read_transform_bundle(bundle):
    timeline =  omni.timeline.get_timeline_interface()
    time = timeline.get_current_time()*timeline.get_time_codes_per_seconds()

    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Xformable(stage.GetPrimAtPath(bundle.bundle.get_prim_path()))
    return prim.ComputeLocalToWorldTransform(time)

def read_world_bounds_bundle(bundle):
    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Xformable(stage.GetPrimAtPath(bundle.bundle.get_prim_path()))
    
    return prim.ComputeWorldBound(0.0, purpose1="default")


# # transform mesh from local space to world space given a mat44
# @wp.kernel
# def transform_mesh(src: wp.array(dtype=wp.vec3),
#                      dest: wp.array(dtype=wp.vec3),
#                      xform: wp.mat44):

#     tid = wp.tid()

#     p = wp.load(src, tid)
#     m = wp.transform_point(xform, p)

#     wp.store(dest, tid, m)


@wp.kernel
def sample_mesh(mesh: wp.uint64,
                lower: wp.vec3,
                spacing: float,
                dim_x: int,
                dim_y: int,
                dim_z: int,
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
        if (sign < 0.0):
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
        mesh = db.inputs.volume

        device = "cuda"


        if mesh.valid and (state.initialized == False or db.inputs.execIn):

            mesh_points = mesh.attribute_by_name(db.tokens.points).value
            mesh_indices = mesh.attribute_by_name(db.tokens.faceVertexIndices).value
            mesh_counts =  mesh.attribute_by_name(db.tokens.faceVertexCounts).value
            mesh_xform = read_transform_bundle(mesh)

            if (len(mesh_points) and len(mesh_indices)):

                
                with wp.ScopedTimer("Prepare Mesh", active=False):

                    # triangulate
                    num_tris = np.sum(np.subtract(mesh_counts, 2))
                    num_tri_vtx = num_tris * 3
                    tri_indices = np.zeros(num_tri_vtx, dtype=int)
                    ctr = 0
                    wedgeIdx = 0
                    for nb in mesh_counts:
                        for i in range(nb-2):
                            tri_indices[ctr] = mesh_indices[wedgeIdx]
                            tri_indices[ctr + 1] = mesh_indices[wedgeIdx + i + 1]
                            tri_indices[ctr + 2] = mesh_indices[wedgeIdx + i + 2]
                            ctr+=3
                        wedgeIdx+=nb

                    # transform to world space
                    mesh_points_world = []
                    for i in range(len(mesh_points)):
                        mesh_points_world.append(mesh_xform.Transform(Gf.Vec3f(tuple(mesh_points[i]))))

                # create Warp mesh
                state.mesh = wp.Mesh(
                    points=wp.array(mesh_points_world, dtype=wp.vec3, device=device),
                    velocities=wp.zeros(len(mesh_points_world), dtype=wp.vec3, device=device),
                    indices=wp.array(tri_indices, dtype=int, device=device))

            bounds = read_world_bounds_bundle(mesh).ComputeAlignedBox()
            lower = bounds.GetMin()
            size = bounds.GetSize()

            if (db.inputs.spacing <= 0.0):
                print("Spacing must be positive")
                return False

            dim_x = int(size[0]/db.inputs.spacing)+1
            dim_y = int(size[1]/db.inputs.spacing)+1
            dim_z = int(size[2]/db.inputs.spacing)+1

            #print(f"Creating particle grid with dim {dim_x}x{dim_y}x{dim_z}, bounds {size[0]}, {size[1]}, {size[2]}, spacing {db.inputs.spacing}")

            if (dim_x*dim_y*dim_z > 1024*1024):
                print("Trying to create particle volume with > 1M particles, increase spacing or geometry size")
                return False

            points_max = 1024*1024
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
                          points,
                          points_counter,
                          points_max], 
                      device="cuda")
            
            num_points = min(int(points_counter.numpy()[0]), points_max)
            
            # bring back to host
            points = points.numpy()

            # jitter
            points = points + np.random.rand(*points.shape)*db.inputs.spacing*db.inputs.spacing_jitter
            #print(num_points)
            
            # points_local = particle_grid(dim_x, dim_y, dim_z, bounds.GetRange().GetMin(), db.inputs.spacing, db.inputs.spacing_jitter)
            # points_world = []

            # # transform to world space
            # xform = read_transform_bundle(db.inputs.volume)

            # for p in points_local:
            #     points_world.append(xform.Transform(Gf.Vec3f(p[0], p[1], p[2])))

            # point_count = len(points_world)

            state.initialized = True

            db.outputs.positions_size = num_points
            db.outputs.positions[:] = points[0:num_points]

            db.outputs.protoIndices_size = num_points
            db.outputs.protoIndices[:] = np.zeros(num_points)

        return True
