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
                bounds_intersected: wp.array(dtype=int)):

    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    bounds_nr = int(0)

    while (wp.bvh_query_next(query, bounds_nr)):
        bounds_intersected[bounds_nr] = 1


@wp.kernel
def bvh_query_ray(bvh_id: wp.uint64,
                start: wp.vec3,
                dir: wp.vec3,
                bounds_intersected: wp.array(dtype=int)):

    query = wp.bvh_query_ray(bvh_id, start, dir)
    bounds_nr = int(0)

    while (wp.bvh_query_next(query, bounds_nr)):
        bounds_intersected[bounds_nr] = 1

        
def aabb_overlap(a_lower, a_upper, b_lower, b_upper):
    
    if a_lower[0] > b_upper[0] or \
       a_lower[1] > b_upper[1] or \
       a_lower[2] > b_upper[2] or \
       a_upper[0] < b_lower[0] or \
       a_upper[1] < b_lower[1] or \
       a_upper[2] < b_lower[2]:
        return 0
    else:
        return 1
      
        
def intersect_ray_aabb(start, dir, lower, upper):
    
    l1 = (lower[0] - start[0]) * dir[0]
    l2 = (upper[0] - start[0]) * dir[0]
    lmin = min(l1,l2)
    lmax = max(l1,l2)

    l1 = (lower[1] - start[1]) * dir[1]
    l2 = (upper[1] - start[1]) * dir[1]
    lmin = max(min(l1,l2), lmin)
    lmax = min(max(l1,l2), lmax)

    l1 = (lower[2] - start[2]) * dir[2]
    l2 = (upper[2] - start[2]) * dir[2]
    lmin = max(min(l1,l2), lmin)
    lmax = min(max(l1,l2), lmax)

    if lmax >= 0.0 and lmax >= lmin:
        return 1
    else:
        return 0
      
        
def test_bvh_query_aabb(test, device):
    
    num_bounds = 100
    lowers = np.random.rand(num_bounds, 3) * 5.0
    uppers = lowers + np.random.rand(num_bounds, 3) * 5.0
            
    bvh = wp.Bvh(lowers=wp.array(lowers, dtype=wp.vec3, device=device),
                 uppers=wp.array(uppers, dtype=wp.vec3, device=device))

    bounds_intersected = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    
    query_lower = wp.vec3(2.0, 2.0, 2.0)
    query_upper = wp.vec3(8.0, 8.0, 8.0)
    
    wp.launch(kernel=bvh_query_aabb, dim=1, 
              inputs=[bvh.id, query_lower, query_upper, bounds_intersected], device=device)
    
    device_intersected = bounds_intersected.numpy()
    
    for i in range(num_bounds):
        lower = lowers[i]
        upper = uppers[i]
        host_intersected = aabb_overlap(lower, upper, query_lower, query_upper);
        test.assertEqual(host_intersected, device_intersected[i])
    

def test_bvh_query_ray(test, device):
    
    num_bounds = 100
    lowers = np.random.rand(num_bounds, 3) * 5.0
    uppers = lowers + np.random.rand(num_bounds, 3) * 5.0
            
    bvh = wp.Bvh(lowers=wp.array(lowers, dtype=wp.vec3, device=device),
                 uppers=wp.array(uppers, dtype=wp.vec3, device=device))

    bounds_intersected = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    
    query_start = wp.vec3(0.0, 0.0, 0.0)
    query_dir = wp.normalize(wp.vec3(1.0, 1.0, 1.0))
    
    wp.launch(kernel=bvh_query_ray, dim=1, 
              inputs=[bvh.id, query_start, query_dir, bounds_intersected], device=device)
    
    device_intersected = bounds_intersected.numpy()
    
    for i in range(num_bounds):
        lower = lowers[i]
        upper = uppers[i]
        host_intersected = intersect_ray_aabb(query_start, query_dir, lower, upper);
        test.assertEqual(host_intersected, device_intersected[i])    


def register(parent):

    devices = wp.get_devices()

    class TestBvh(parent):
        pass

    add_function_test(TestBvh, "test_bvh_aabb", test_bvh_query_aabb, devices=devices)
    add_function_test(TestBvh, "test_bvh_ray", test_bvh_query_ray, devices=devices)

    return TestBvh

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
