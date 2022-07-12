# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import numpy as np
import math

import warp as wp
from warp.tests.test_base import *

np.random.seed(42)

wp.init()


# triangulate a list of polygon face indices
def triangulate(face_counts, face_indices):
    num_tris = np.sum(np.subtract(face_counts, 2))
    num_tri_vtx = num_tris * 3
    tri_indices = np.zeros(num_tri_vtx, dtype=int)
    ctr = 0
    wedgeIdx = 0

    for nb in face_counts:
        for i in range(nb-2):
            tri_indices[ctr] = face_indices[wedgeIdx]
            tri_indices[ctr + 1] = face_indices[wedgeIdx + i + 1]
            tri_indices[ctr + 2] = face_indices[wedgeIdx + i + 2]
            ctr+=3
        wedgeIdx+=nb

    return tri_indices


@wp.kernel
def mesh_query_ray_loss(mesh: wp.uint64,
                query_points: wp.array(dtype=wp.vec3),
                query_dirs: wp.array(dtype=wp.vec3),
                intersection_points: wp.array(dtype=wp.vec3),
                loss: wp.array(dtype=float)):

    tid = wp.tid()

    p = query_points[tid]
    D = query_dirs[tid]

    max_t = 10012.0
    t = float(0.0)
    bary_u = float(0.0)
    bary_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()
    face_index = int(0)

    q = wp.vec3()

    if wp.mesh_query_ray(mesh, p, D, max_t, t, bary_u, bary_v, sign, normal, face_index):
        q = wp.mesh_eval_position(mesh, face_index, bary_u, bary_v)

    intersection_points[tid] = q
    l = q[0]
    loss[tid] = l


def test_mesh_query_ray_grad(test, device):

    from pxr import Usd, UsdGeom, Gf, Sdf

    # test tri
    # print("Testing Single Triangle")
    # mesh_points = wp.array(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), dtype=wp.vec3, device=device)
    # mesh_indices = wp.array(np.array([0,1,2]), dtype=int, device=device)

    mesh = Usd.Stage.Open(os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/torus.usda")))
    mesh_geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/World/Torus"))

    mesh_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
    mesh_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()

    tri_indices = triangulate(mesh_counts, mesh_indices)

    mesh_points = wp.array(np.array(mesh_geom.GetPointsAttr().Get()), dtype=wp.vec3, device=device)
    mesh_indices = wp.array(np.array(tri_indices), dtype=int, device=device)

    p = wp.vec3(50.0, 50.0, 0.0)
    D = wp.vec3(0.0, -1.0, 0.0)

    # create mesh
    mesh = wp.Mesh(
        points=mesh_points, 
        velocities=None,
        indices=mesh_indices)

    tape = wp.Tape()

    # analytic gradients
    with tape:

        query_points = wp.array(p, dtype=wp.vec3, device=device, requires_grad=True)
        query_dirs = wp.array(D, dtype=wp.vec3, device=device, requires_grad=True)
        intersection_points = wp.zeros(n=1, dtype=wp.vec3, device=device)
        loss = wp.zeros(n=1, dtype=float, device=device, requires_grad=True)

        wp.launch(kernel=mesh_query_ray_loss, dim=1, inputs=[mesh.id, query_points, query_dirs, intersection_points, loss], device=device)

    tape.backward(loss=loss)
    q = intersection_points.numpy().flatten()
    analytic_p = tape.gradients[query_points].numpy().flatten()
    analytic_D = tape.gradients[query_dirs].numpy().flatten()

    # numeric gradients

    # ray origin
    eps = 1.e-3
    loss_values_p = []
    numeric_p = np.zeros(3)

    offset_query_points = [
        wp.vec3(p[0] - eps, p[1], p[2]), wp.vec3(p[0] + eps, p[1], p[2]),
        wp.vec3(p[0], p[1] - eps, p[2]), wp.vec3(p[0], p[1] + eps, p[2]),
        wp.vec3(p[0], p[1], p[2] - eps), wp.vec3(p[0], p[1], p[2] + eps)]

    for i in range(6):
        q = offset_query_points[i]

        query_points = wp.array(q, dtype=wp.vec3, device=device)
        query_dirs = wp.array(D, dtype=wp.vec3, device=device)
        intersection_points = wp.zeros(n=1, dtype=wp.vec3, device=device)
        loss = wp.zeros(n=1, dtype=float, device=device)

        wp.launch(kernel=mesh_query_ray_loss, dim=1, inputs=[mesh.id, query_points, query_dirs, intersection_points, loss], device=device)

        loss_values_p.append(loss.numpy()[0])

    for i in range(3):
        l_0 = loss_values_p[i*2]
        l_1 = loss_values_p[i*2+1]
        gradient = (l_1 - l_0) / (2.0*eps)
        numeric_p[i] = gradient

    # ray dir
    loss_values_D = []
    numeric_D = np.zeros(3)

    offset_query_dirs = [
        wp.vec3(D[0] - eps, D[1], D[2]), wp.vec3(D[0] + eps, D[1], D[2]),
        wp.vec3(D[0], D[1] - eps, D[2]), wp.vec3(D[0], D[1] + eps, D[2]),
        wp.vec3(D[0], D[1], D[2] - eps), wp.vec3(D[0], D[1], D[2] + eps)]

    for i in range(6):
        q = offset_query_dirs[i]

        query_points = wp.array(p, dtype=wp.vec3, device=device)
        query_dirs = wp.array(q, dtype=wp.vec3, device=device)
        intersection_points = wp.zeros(n=1, dtype=wp.vec3, device=device)
        loss = wp.zeros(n=1, dtype=float, device=device)

        wp.launch(kernel=mesh_query_ray_loss, dim=1, inputs=[mesh.id, query_points, query_dirs, intersection_points, loss], device=device)

        loss_values_D.append(loss.numpy()[0])

    for i in range(3):
        l_0 = loss_values_D[i*2]
        l_1 = loss_values_D[i*2+1]
        gradient = (l_1 - l_0) / (2.0*eps)
        numeric_D[i] = gradient
    
    error_p = ((analytic_p - numeric_p) * (analytic_p - numeric_p)).sum(axis=0)
    error_D = ((analytic_D - numeric_D) * (analytic_D - numeric_D)).sum(axis=0)

    tolerance = 1.e-3
    test.assertTrue(error_p < tolerance, f"error is {error_p} which is >= {tolerance}")
    test.assertTrue(error_D < tolerance, f"error is {error_D} which is >= {tolerance}")



@wp.kernel
def raycast_kernel(mesh: wp.uint64,
                ray_starts: wp.array(dtype=wp.vec3),
                ray_directions: wp.array(dtype=wp.vec3),
                ray_hits: wp.array(dtype=wp.vec3),
                count: wp.array(dtype=int)):

    t = float(0.0)                   # hit distance along ray
    u = float(0.0)                   # hit face barycentric u
    v = float(0.0)                   # hit face barycentric v
    sign = float(0.0)                # hit face sign
    n = wp.vec3()                    # hit face normal
    f = int(0)                       # hit face index
    max_dist = 1e6                   # max raycast disance

    # ray cast against the mesh
    tid = wp.tid()

    if wp.mesh_query_ray(mesh, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f):
        wp.atomic_add(count, 0, 1)


# tests rays against a quad of two connected triangles
# with rays exactly falling on the edge, tests that
# there are no leaks

def test_mesh_query_ray_edge(test, device):

    # Create raycast starts and directions
    xx, yy = np.meshgrid(np.arange(0.1, 0.4, 0.01), np.arange(0.1, 0.4, 0.01))
    xx = xx.flatten().reshape(-1, 1)
    yy = yy.flatten().reshape(-1, 1)
    zz = np.ones_like(xx)

    ray_starts = np.concatenate((xx, yy, zz), axis=1)
    ray_dirs = np.zeros_like(ray_starts)
    ray_dirs[:, 2] = -1.0


    # Create simple square mesh
    vertices = np.array([[0., 0., 0.],
                        [0., 0.5, 0.],
                        [0.5, 0., 0.],
                        [0.5, 0.5, 0.]], dtype=np.float32)

    triangles = np.array([[1, 0, 2],
                          [1, 2, 3]], dtype=np.int32)


    mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=device),
                indices=wp.array(triangles.flatten(), dtype=int, device=device))
    

    counts = wp.zeros(1, dtype=int, device=device)

    n = len(ray_starts)
    
    ray_starts = wp.array(ray_starts, shape=(n,), dtype=wp.vec3, device=device)
    ray_dirs = wp.array(ray_dirs, shape=(n,), dtype=wp.vec3, device=device)
    ray_hits = wp.zeros((n, ), dtype=wp.vec3, device=device)
    
    wp.launch(kernel=raycast_kernel,
            dim=n,
            inputs=[mesh.id, ray_starts, ray_dirs, ray_hits, counts],
            device=device)
    wp.synchronize()

    test.assertEqual(counts.numpy()[0], n)


def register(parent):

    devices = wp.get_devices()

    class TestMeshQueryRay(parent):
        pass

    add_function_test(TestMeshQueryRay, "test_mesh_query_ray_edge", test_mesh_query_ray_edge, devices=devices)
    add_function_test(TestMeshQueryRay, "test_mesh_query_ray_grad", test_mesh_query_ray_grad, devices=devices)
    
    return TestMeshQueryRay

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
