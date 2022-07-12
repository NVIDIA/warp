# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest
from unittest import runner

import warp as wp

import warp.tests.test_codegen
import warp.tests.test_mesh_query_aabb
import warp.tests.test_mesh_query_point
import warp.tests.test_mesh_query_ray
import warp.tests.test_conditional
import warp.tests.test_operators
import warp.tests.test_rounding
import warp.tests.test_hash_grid
import warp.tests.test_ctypes
import warp.tests.test_rand
import warp.tests.test_noise
import warp.tests.test_tape
import warp.tests.test_compile_consts
import warp.tests.test_volume
import warp.tests.test_mlp
import warp.tests.test_grad
import warp.tests.test_intersect
import warp.tests.test_array
import warp.tests.test_launch
import warp.tests.test_import
import warp.tests.test_func
import warp.tests.test_fp16
import warp.tests.test_reload
import warp.tests.test_struct
import warp.tests.test_closest_point_edge_edge

def run():

    tests = unittest.TestSuite()
    result = unittest.TestResult()
   
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_codegen.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_mesh_query_aabb.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_mesh_query_point.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_mesh_query_ray.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_conditional.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_operators.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_rounding.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_hash_grid.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_ctypes.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_rand.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_noise.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_tape.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_compile_consts.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_volume.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_mlp.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_grad.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_intersect.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_array.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_launch.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_import.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_func.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_fp16.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_reload.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_struct.register(unittest.TestCase)))
    tests.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(warp.tests.test_closest_point_edge_edge.register(unittest.TestCase)))

    # force rebuild of all kernels
    wp.build.clear_kernel_cache()

    # load all modules
    wp.force_load()

    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    ret = not runner.run(tests).wasSuccessful()
    return ret


if __name__ == '__main__':
    ret = run()

    import sys
    sys.exit(ret)

