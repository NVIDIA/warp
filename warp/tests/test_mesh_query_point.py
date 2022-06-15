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

@wp.kernel
def sample_mesh_query(mesh: wp.uint64,
                query_points: wp.array(dtype=wp.vec3),
                query_faces: wp.array(dtype=int),
                query_signs: wp.array(dtype=float),
                query_dist: wp.array(dtype=float)):
    
    tid = wp.tid()

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 10012.0

    p = query_points[tid]
    
    wp.mesh_query_point(mesh, p, max_dist, sign, face_index, face_u, face_v)
        
    cp = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

    query_signs[tid] = sign
    query_faces[tid] = face_index
    query_dist[tid] = wp.length(cp-p)


@wp.func
def triangle_closest_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)

    if (d1 <= 0.0 and d2 <= 0.0):
        return wp.vec2(1.0, 0.0)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)

    if (d3 >= 0.0 and d4 <= d3):
        return wp.vec2(0.0, 1.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
        return wp.vec2(1.0 - v, v)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)

    if (d6 >= 0.0 and d5 <= d6):
        return wp.vec2(0.0, 0.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
        return wp.vec2(1.0 - w, 0.0)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
        return wp.vec2(0.0, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w

    return wp.vec2(u, v)


@wp.func
def solid_angle(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, p: wp.vec3):
    a = v0 - p
    b = v1 - p
    c = v2 - p

    a_len = wp.length(a)
    b_len = wp.length(b)
    c_len = wp.length(c)

    det = wp.dot(a, wp.cross(b, c))
    den = a_len*b_len*c_len + wp.dot(a, b)*c_len + wp.dot(b, c)*a_len + wp.dot(c, a)*b_len

    return wp.atan2(det, den)



@wp.kernel
def sample_mesh_brute(
                tri_points: wp.array(dtype=wp.vec3),
                tri_indices: wp.array(dtype=int),
                tri_count: int,
                query_points: wp.array(dtype=wp.vec3),
                query_faces: wp.array(dtype=int),
                query_signs: wp.array(dtype=float),
                query_dist: wp.array(dtype=float)):
    
    tid = wp.tid()

    min_face = int(0)
    min_dist = float(1.e+6)

    winding_angle = float(0.0)

    p = query_points[tid]

    for i in range(0, tri_count):
        
        a = tri_points[tri_indices[i*3+0]]
        b = tri_points[tri_indices[i*3+1]]
        c = tri_points[tri_indices[i*3+2]]

        winding_angle += solid_angle(a, b, c, p)

        bary = triangle_closest_point(a, b, c, p)
        u = bary[0]
        v = bary[1]

        cp = u*a + v*b + (1.0-u-v)*c
        cp_dist = wp.length(cp - p)

        if (cp_dist < min_dist):
            min_dist = cp_dist
            min_face = i
            
    query_faces[tid] = min_face
    query_signs[tid] = winding_angle 
    query_dist[tid] = min_dist


# constructs a grid of evenly spaced particles
def particle_grid(dim_x, dim_y, dim_z, lower, radius, jitter):
    points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
    points_t = np.array((points[0], points[1], points[2])).T*radius*2.0 + np.array(lower)
    points_t = points_t + np.random.rand(*points_t.shape)*radius*jitter

    return points_t.reshape((-1, 3))

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


def test_mesh_query_point(test, device):

    from pxr import Usd, UsdGeom, Gf, Sdf

    mesh = Usd.Stage.Open(os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/torus.usda")))
    mesh_geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/World/Torus"))

    mesh_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
    mesh_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()

    tri_indices = triangulate(mesh_counts, mesh_indices)

    mesh_points = wp.array(np.array(mesh_geom.GetPointsAttr().Get()), dtype=wp.vec3, device=device)
    mesh_indices = wp.array(np.array(tri_indices), dtype=int, device=device)

    # create mesh
    mesh = wp.Mesh(
        points=mesh_points, 
        velocities=None,
        indices=mesh_indices)

    p = particle_grid(32, 32, 32, np.array([-5.0, -5.0, -5.0]), 0.1, 0.1)*100.0

    query_count = len(p)
    query_points = wp.array(p, dtype=wp.vec3, device=device)

    signs_query = wp.zeros(query_count, dtype=float, device=device)
    faces_query = wp.zeros(query_count, dtype=int, device=device)
    dist_query = wp.zeros(query_count, dtype=float, device=device)

    signs_brute = wp.zeros(query_count, dtype=float, device=device)
    faces_brute = wp.zeros(query_count, dtype=int, device=device)
    dist_brute = wp.zeros(query_count, dtype=float, device=device)

    wp.launch(kernel=sample_mesh_query, dim=query_count, inputs=[mesh.id, query_points, faces_query, signs_query, dist_query], device=device)
    wp.launch(kernel=sample_mesh_brute, dim=query_count, inputs=[mesh_points, mesh_indices, int(len(mesh_indices)/3), query_points, faces_brute, signs_brute, dist_brute], device=device)

    signs_query = signs_query.numpy()
    faces_query = faces_query.numpy()
    dist_query = dist_query.numpy()

    signs_brute = signs_brute.numpy()
    faces_brute = faces_brute.numpy()
    dist_brute = dist_brute.numpy()

    query_points = query_points.numpy()

    inside_query = []
    inside_brute = []

    for i in range(query_count):

        if (signs_query[i] < 0.0):
            inside_query.append(query_points[i].tolist())
        
        if (signs_brute[i] > 6.0):
            inside_brute.append(query_points[i].tolist())

    inside_query = np.array(inside_query)
    inside_brute = np.array(inside_brute)

    dist_error = np.max(np.abs(dist_query - dist_brute))
    sign_error = np.max(np.abs(inside_query - inside_brute))

    tolerance = 1.5e-4
    test.assertTrue(dist_error < tolerance, f"dist_error is {dist_error} which is >= {tolerance}")
    test.assertTrue(sign_error < tolerance, f"sign_error is {sign_error} which is >= {tolerance}")

    # import warp.render

    # stage = warp.render.UsdRenderer("tests/outputs/test_mesh_query_point.usd")

    # radius = 10.0
    # stage.begin_frame(0.0)
    # stage.render_mesh(points=mesh_points.numpy(), indices=mesh_indices.numpy(), name="mesh")
    # stage.render_points(points=inside_query, radius=radius, name="query")
    # stage.render_points(points=inside_brute, radius=radius, name="brute")
    # stage.render_points(points=query_points, radius=radius, name="all")
    # stage.end_frame()

    # stage.save()


@wp.kernel
def mesh_query_point_loss(mesh: wp.uint64,
                query_points: wp.array(dtype=wp.vec3),
                projected_points: wp.array(dtype=wp.vec3),
                loss: wp.array(dtype=float)):
    
    tid = wp.tid()

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 10012.0

    p = query_points[tid]
    
    wp.mesh_query_point(mesh, p, max_dist, sign, face_index, face_u, face_v)
    q = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

    projected_points[tid] = q

    dist = wp.length(wp.sub(p, q))
    loss[tid] = dist


def test_adj_mesh_query_point(test, device):

    from pxr import Usd, UsdGeom, Gf, Sdf

    mesh = Usd.Stage.Open(os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/torus.usda")))
    mesh_geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/World/Torus"))

    mesh_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
    mesh_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()

    tri_indices = triangulate(mesh_counts, mesh_indices)

    mesh_points = wp.array(np.array(mesh_geom.GetPointsAttr().Get()), dtype=wp.vec3, device=device)
    mesh_indices = wp.array(np.array(tri_indices), dtype=int, device=device)

    # test tri
    # print("Testing Single Triangle")
    # mesh_points = wp.array(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), dtype=wp.vec3, device=device)
    # mesh_indices = wp.array(np.array([0,1,2]), dtype=int, device=device)

    # create mesh
    mesh = wp.Mesh(
        points=mesh_points, 
        velocities=None,
        indices=mesh_indices)

    # p = particle_grid(32, 32, 32, np.array([-5.0, -5.0, -5.0]), 0.1, 0.1)*100.0
    p = wp.vec3(50.0, 50.0, 50.0)

    tape = wp.Tape()

    # analytic gradients
    with tape:

        query_points = wp.array(p, dtype=wp.vec3, device=device, requires_grad=True)
        projected_points = wp.zeros(n=1, dtype=wp.vec3, device=device)
        loss = wp.zeros(n=1, dtype=float, device=device, requires_grad=True)

        wp.launch(kernel=mesh_query_point_loss, dim=1, inputs=[mesh.id, query_points, projected_points, loss], device=device)

    tape.backward(loss=loss)
    analytic = tape.gradients[query_points].numpy().flatten()

    # numeric gradients
    eps = 1.e-3
    loss_values = []
    numeric = np.zeros(3)

    offset_query_points = [
        wp.vec3(p[0] - eps, p[1], p[2]), wp.vec3(p[0] + eps, p[1], p[2]),
        wp.vec3(p[0], p[1] - eps, p[2]), wp.vec3(p[0], p[1] + eps, p[2]),
        wp.vec3(p[0], p[1], p[2] - eps), wp.vec3(p[0], p[1], p[2] + eps)]

    for i in range(6):
        q = offset_query_points[i]

        query_points = wp.array(q, dtype=wp.vec3, device=device)
        projected_points = wp.zeros(n=1, dtype=wp.vec3, device=device)
        loss = wp.zeros(n=1, dtype=float, device=device)

        wp.launch(kernel=mesh_query_point_loss, dim=1, inputs=[mesh.id, query_points, projected_points, loss], device=device)

        loss_values.append(loss.numpy()[0])

    for i in range(3):
        l_0 = loss_values[i*2]
        l_1 = loss_values[i*2+1]
        gradient = (l_1 - l_0) / (2.0*eps)
        numeric[i] = gradient

    error = ((analytic - numeric) * (analytic - numeric)).sum(axis=0)

    tolerance = 1.e-3
    test.assertTrue(error < tolerance, f"error is {error} which is >= {tolerance}")


def register(parent):

    devices = wp.get_devices()

    class TestMeshQuery(parent):
        pass

    add_function_test(TestMeshQuery, "test_mesh_query_point", test_mesh_query_point, devices=devices)
    add_function_test(TestMeshQuery, "test_adj_mesh_query_point", test_adj_mesh_query_point, devices=devices)

    return TestMeshQuery

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
