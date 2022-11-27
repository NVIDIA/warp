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
import warp.tests.test_multigpu
import warp.tests.test_quat
import warp.tests.test_atomic
import warp.tests.test_adam
import warp.tests.test_transient_module
import warp.tests.test_lerp
import warp.tests.test_smoothstep
import warp.tests.test_model
import warp.tests.test_fast_math
import warp.tests.test_streams
import warp.tests.test_torch
import warp.tests.test_pinned


def register_tests(parent):

    tests = []

    tests.append(warp.tests.test_codegen.register(parent))
    tests.append(warp.tests.test_mesh_query_aabb.register(parent))
    tests.append(warp.tests.test_mesh_query_point.register(parent))
    tests.append(warp.tests.test_mesh_query_ray.register(parent))
    tests.append(warp.tests.test_conditional.register(parent))
    tests.append(warp.tests.test_operators.register(parent))
    tests.append(warp.tests.test_rounding.register(parent))
    tests.append(warp.tests.test_hash_grid.register(parent))
    tests.append(warp.tests.test_ctypes.register(parent))
    tests.append(warp.tests.test_rand.register(parent))
    tests.append(warp.tests.test_noise.register(parent))
    tests.append(warp.tests.test_tape.register(parent))
    tests.append(warp.tests.test_compile_consts.register(parent))
    tests.append(warp.tests.test_volume.register(parent))
    tests.append(warp.tests.test_mlp.register(parent))
    tests.append(warp.tests.test_grad.register(parent))
    tests.append(warp.tests.test_intersect.register(parent))
    tests.append(warp.tests.test_array.register(parent))
    tests.append(warp.tests.test_launch.register(parent))
    tests.append(warp.tests.test_import.register(parent))
    tests.append(warp.tests.test_func.register(parent))
    tests.append(warp.tests.test_fp16.register(parent))
    tests.append(warp.tests.test_reload.register(parent))
    tests.append(warp.tests.test_struct.register(parent))
    tests.append(warp.tests.test_closest_point_edge_edge.register(parent))
    tests.append(warp.tests.test_multigpu.register(parent))
    tests.append(warp.tests.test_quat.register(parent))
    tests.append(warp.tests.test_atomic.register(parent))
    tests.append(warp.tests.test_adam.register(parent))
    tests.append(warp.tests.test_transient_module.register(parent))
    tests.append(warp.tests.test_lerp.register(parent))
    tests.append(warp.tests.test_smoothstep.register(parent))
    tests.append(warp.tests.test_model.register(parent))
    tests.append(warp.tests.test_fast_math.register(parent))
    tests.append(warp.tests.test_streams.register(parent))
    tests.append(warp.tests.test_torch.register(parent))
    tests.append(warp.tests.test_pinned.register(parent))

    return tests

def run():

    test_suite = unittest.TestSuite()
    result = unittest.TestResult()

    tests = register_tests(unittest.TestCase)

    for test in tests:
        test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(test))

    # force rebuild of all kernels
    wp.build.clear_kernel_cache()

    # load all modules
    wp.force_load()

    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    ret = not runner.run(test_suite).wasSuccessful()
    return ret


if __name__ == '__main__':

    ret = run()

    import sys
    sys.exit(ret)

