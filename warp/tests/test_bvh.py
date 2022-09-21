# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import numpy as np

import warp as wp
from warp.tests.test_base import *

np.random.seed(42)

wp.init()


@wp.kernel
def bvh_query_aabb(bvh_id: wp.uint64,
                lower: wp.vec3,
                upper: wp.vec3,
                intersected_bounds: wp.array(dtype=int),
                max_intersections):

    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    nr = 0
    bounds_nr = int(0)

    while (nr < max_intersections and wp.bvh_query_next(query, bounds_nr)):
        intersected_bounds[nr] = bounds_nr
        nr = nr + 1


@wp.kernel
def bvh_query_ray(bvh_id: wp.uint64,
                start: wp.vec3,
                dir: wp.vec3,
                intersected_bounds: wp.array(dtype=int),
                max_intersections):

    query = wp.bvh_query_ray(bvh_id, start, dir)
    nr = 0
    bounds_nr = int(0)

    while (nr < max_intersections and wp.bvh_query_next(query, bounds_nr)):
        intersected_bounds[nr] = bounds_nr
        nr = nr + 1

        

def test_bvh_queries(test, device):
    
    num_bounds = 100
    lowers = np.random.rand(num_bounds, 3) * 10.0
    uppers = lowers + np.random.rand(num_bounds, 3)
            
    bvh = wp.Bvh(lowers=wp.array(lowers, dtype=wp.vec3, device=device),
                 uppers=wp.array(uppers, dtype=wp.vec3, device=device))

    max_intersections = 100
    intersected_bounds = wp.array([-1] * max_intersections, dtype=int, device=device)
    
    query_lower = wp.vec3(*(np.random.rand(3)*0.5))
    query_upper = wp.add(query_lower, wp.vec3(*(np.random.rand(3)*0.5)))
    
    wp.launch(kernel=bvh_query_aabb, dim=1, 
              inputs=[bvh.id, query_lower, query_upper, intersected_bounds, max_intersections], device=device)

    print("aabb overlaps", intersected_bounds.numpy())

    intersected_bounds = wp.array([-1] * max_intersections, dtype=int, device=device)
    
    query_start = wp.vec3(0.0, 0.0, 0.0)
    query_dir = wp.normalize(wp.vec3(*np.random.rand(3)))
    
    wp.launch(kernel=bvh_query_ray, dim=1, 
              inputs=[bvh.id, query_start, query_dir, intersected_bounds, max_intersections], device=device)
    
    print("ray overlaps", intersected_bounds.numpy())
    

test_bvh_queries(None, device="cpu")


# def register(parent):

#     devices = wp.get_devices()

#     class TestMeshQueryRay(parent):
#         pass

#     add_function_test(TestMeshQueryRay, "test_mesh_query_ray_edge", test_mesh_query_ray_edge, devices=devices)

#     # USD import failures should not count as a test failure
#     try:
#         from pxr import Usd, UsdGeom
#         have_usd = True
#     except:
#         have_usd = False

#     if have_usd:
#         add_function_test(TestMeshQueryRay, "test_mesh_query_ray_grad", test_mesh_query_ray_grad, devices=devices)

#     return TestMeshQueryRay

# if __name__ == '__main__':
#     c = register(unittest.TestCase)
#     unittest.main(verbosity=2)
