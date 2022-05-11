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


def test_adj_mesh_query_ray(test, device):

    # test tri
    print("Testing Single Triangle")
    mesh_points = wp.array(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), dtype=wp.vec3, device=device)
    mesh_indices = wp.array(np.array([0,1,2]), dtype=int, device=device)

    p = wp.vec3(0.5, 0.5, 2.0)
    D = wp.vec3(0.0, 0.0, -1.0)

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
        loss = wp.zeros(n=1, dtype=float, device=device)

        wp.launch(kernel=mesh_query_ray_loss, dim=1, inputs=[mesh.id, query_points, query_dirs, intersection_points, loss], device=device)

    tape.backward(loss=loss)
    q = intersection_points.numpy().flatten()
    analytic_p = tape.gradients[query_points].numpy().flatten()
    analytic_D = tape.gradients[query_dirs].numpy().flatten()

    print("intersection point")
    print(q)
    print("analytic_p:")
    print(analytic_p)
    print("analytic_D:")
    print(analytic_D)

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

    print("numeric_p")
    print(numeric_p)
    print("numeric_D")
    print(numeric_D)
    

def register(parent):

    devices = wp.get_devices()

    class TestMeshQueryRay(parent):
        pass

    add_function_test(TestMeshQueryRay, "test_adj_mesh_query_ray", test_adj_mesh_query_ray, devices=devices)

    return TestMeshQueryRay

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
