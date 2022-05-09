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

def test_adj_mesh_query_ray(test, device):

    # test tri
    print("Testing Single Triangle")
    mesh_points = wp.array(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), dtype=wp.vec3, device=device)
    mesh_indices = wp.array(np.array([0,1,2]), dtype=int, device=device)

def register(parent):

    devices = wp.get_devices()

    class TestRayQuery(parent):
        pass

    add_function_test(TestRayQuery, "test_mesh_query_ray", test_mesh_query_ray, devices=devices)

    return TestMeshQuery

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
