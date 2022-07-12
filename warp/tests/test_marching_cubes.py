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

import unittest

wp.init()

@wp.kernel
def make_field(field: wp.array3d(dtype=float),
               center: wp.vec3,
               radius: float):

    i, j, k = wp.tid()

    p = wp.vec3(float(i), float(j), float(k))

    d = wp.length(p - center) - radius

    field[i,j,k] = d
    

def test_marching_cubes(test, device):

    dim = 64
    max_verts = 10**6
    max_tris = 10**6

    field = wp.zeros(shape=(dim, dim, dim), dtype=float, device=device)
    
    iso = wp.MarchingCubes(nx=dim,
                           ny=dim,
                           nz=dim,
                           max_verts=max_verts,
                           max_tris=max_tris,
                           device=device)

    radius = dim/4.0

    wp.launch(make_field, dim=field.shape, inputs=[field, wp.vec3(dim/2, dim/2, dim/2), radius], device="cuda")

    iso.surface(field=field, threshold=0.0)

    # check that all returned vertices lie on the surface of the sphere
    length = np.linalg.norm(iso.verts.numpy() - np.array([dim/2, dim/2, dim/2]), axis=1)
    error = np.abs(length - radius)

    test.assertTrue(np.max(error) < 1.0)


def register(parent):

    devices = ["cuda"]

    class TestMarchingCubes(parent):
        pass

    add_function_test(TestMarchingCubes, "test_marching_cubes", test_marching_cubes, devices=devices)

    return TestMarchingCubes

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)