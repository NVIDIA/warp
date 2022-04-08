# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#############################################################################
# Example Mesh Intersection
#
# Show how to use built-in BVH query to test if two triangle meshes intersect.
#
##############################################################################

import warp as wp
import warp.render

import numpy as np
np.random.seed(42)

import os

wp.init()

@wp.func
def cw_min(a: wp.vec3, b: wp.vec3):
    
    return wp.vec3(wp.min(a[0], b[0]), 
                   wp.min(a[1], b[1]),
                   wp.min(a[2], b[2]))

@wp.func
def cw_max(a: wp.vec3, b: wp.vec3):
    
    return wp.vec3(wp.max(a[0], b[0]), 
                   wp.max(a[1], b[1]),
                   wp.max(a[2], b[2]))


@wp.kernel
def intersect(mesh_0: wp.uint64,
              mesh_1: wp.uint64,
              num_faces: int,
              xforms: wp.array(dtype=wp.transform),
              result: wp.array(dtype=int)):
    
    tid = wp.tid()

    # mesh_0 is assumed to be the query mesh, we launch one thread
    # for each face in mesh_0 and test it against the opposing mesh's BVH
    face = tid%num_faces
    batch = tid//num_faces

    # transforms from mesh_0 -> mesh_1 space
    xform = xforms[batch]

    # load query triangles points and transform to mesh_1's space
    v0 = wp.transform_point(xform, wp.mesh_eval_position(mesh_0, face, 1.0, 0.0))
    v1 = wp.transform_point(xform, wp.mesh_eval_position(mesh_0, face, 0.0, 1.0))
    v2 = wp.transform_point(xform, wp.mesh_eval_position(mesh_0, face, 0.0, 0.0))

    # compute bounds of the query triangle 
    lower = cw_min(cw_min(v0, v1), v2)
    upper = cw_max(cw_max(v0, v1), v2)

    query = wp.mesh_query_aabb(mesh_1, lower, upper)

    for f in query:

        u0 = wp.mesh_eval_position(mesh_1, f, 1.0, 0.0)
        u1 = wp.mesh_eval_position(mesh_1, f, 0.0, 1.0)
        u2 = wp.mesh_eval_position(mesh_1, f, 0.0, 0.0)

        # test for triangle intersection
        i = wp.intersect_tri_tri(v0, v1, v2,
                                 u0, u1, u2)

        if i > 0:
            result[batch] = 1
            return
        
        # use if you want to count all intersections
        #wp.atomic_add(result, batch, i)


device = wp.get_preferred_device()

# create collision meshes
from pxr import Usd, UsdGeom, Gf, Sdf

def load_mesh(path, prim, device):

    usd_path = os.path.join(os.path.dirname(__file__), path)
    usd_stage = Usd.Stage.Open(usd_path)
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(prim)) 

    mesh = wp.Mesh(
        points=wp.array(usd_geom.GetPointsAttr().Get(), dtype=wp.vec3, device=device),
        indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int, device=device))

    return mesh

path_0 = "assets/cube.usda"
path_1 = "assets/sphere.usda"

mesh_0 = load_mesh(path_0, "/Cube/Cube_001", device)
mesh_1 = load_mesh(path_1, "/Sphere/Sphere", device)

query_num_faces = int(len(mesh_0.indices)/3)
query_num_points = len(mesh_0.points)
query_count = 1024

# generate random relative transforms
xforms = []

for i in range(query_count):
    
    # random offset
    p = (np.random.rand(3)*0.5 - 0.5)*5.0
    
    # random orientation
    axis = wp.normalize(np.random.rand(3)*0.5 - 0.5)
    angle = float(np.random.rand(1)[0])

    q = wp.quat_from_axis_angle(wp.normalize(axis), angle)

    xforms.append(wp.transform(p, q))


array_result = wp.zeros(query_count, dtype=int, device=device)
array_xforms = wp.array(xforms, dtype=wp.transform, device=device)

# force module load (for accurate profiling)
wp.force_load()

###############
# Test

with wp.ScopedTimer("intersect"):
    wp.launch(kernel=intersect, dim=query_num_faces*query_count, inputs=[mesh_0.id, mesh_1.id, query_num_faces, array_xforms, array_result], device=device)
    wp.synchronize()


###############
# Visualization

render = warp.render.UsdRenderer(os.path.join(os.path.dirname(__file__), "outputs/example_mesh_interesct.usd"))

# bring results back to host
result = array_result.numpy()
print(result)

render.begin_frame(0.0)

for i in range(query_count):

    spacing = 8.0
    offset = i*spacing

    xform = xforms[i]
    render.render_ref(f"mesh_{i}_0", "../" + path_0, pos=wp.vec3(xform.p[0] + offset, xform.p[1], xform.p[2]), rot=xform.q, scale=(1.0, 1.0, 1.0))
    render.render_ref(f"mesh_{i}_1", "../" + path_1, pos=(offset, 0.0, 0.0), rot=wp.quat_identity(), scale=(1.0, 1.0, 1.0))

    # if pair intersects then draw a small box above the pair
    if result[i] > 0:
        render.render_box(f"result_{i}", pos=wp.vec3(xform.p[0] + offset, xform.p[1] + 5.0, xform.p[2]), rot=wp.quat_identity(), extents=(0.1, 0.1, 0.1))
    
render.end_frame()

render.save()

