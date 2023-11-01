# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for the Warp library itself.

Only a trimmed down list of tests is run since the full suite is too slow.
"""

import omni.kit

import warp as wp
import warp.tests.test_array
import warp.tests.test_array_reduce
import warp.tests.test_bvh
import warp.tests.test_codegen
import warp.tests.test_compile_consts
import warp.tests.test_conditional
import warp.tests.test_ctypes
import warp.tests.test_devices
import warp.tests.test_dlpack
import warp.tests.test_fabricarray
import warp.tests.test_func
import warp.tests.test_generics
import warp.tests.test_grad_customs
import warp.tests.test_hash_grid
import warp.tests.test_indexedarray
import warp.tests.test_launch
import warp.tests.test_marching_cubes
import warp.tests.test_mat_lite
import warp.tests.test_math
import warp.tests.test_matmul_lite
import warp.tests.test_mesh
import warp.tests.test_mesh_query_aabb
import warp.tests.test_mesh_query_point
import warp.tests.test_mesh_query_ray
import warp.tests.test_modules_lite
import warp.tests.test_noise
import warp.tests.test_operators
import warp.tests.test_quat
import warp.tests.test_rand
import warp.tests.test_reload
import warp.tests.test_rounding
import warp.tests.test_runlength_encode
import warp.tests.test_sparse
import warp.tests.test_streams
import warp.tests.test_tape
import warp.tests.test_transient_module
import warp.tests.test_types
import warp.tests.test_utils
import warp.tests.test_vec_lite
import warp.tests.test_volume
import warp.tests.test_volume_write

initialized = False


class BaseTestCase(omni.kit.test.AsyncTestCase):
    @classmethod
    def setUpClass(cls):
        global initialized

        if not initialized:
            # Load all the Warp modules just once. This needs to be done
            # within the `setUpClass` method instead of at the module level
            # to avoid the reload to be done as soon as this module is imported,
            # which might happen when scanning for existing tests without
            # actually needing to run these ones.
            wp.force_load()
            initialized = True


test_clss = (
    warp.tests.test_array.register(BaseTestCase),
    warp.tests.test_array_reduce.register(BaseTestCase),
    warp.tests.test_bvh.register(BaseTestCase),
    warp.tests.test_codegen.register(BaseTestCase),
    warp.tests.test_compile_consts.register(BaseTestCase),
    warp.tests.test_conditional.register(BaseTestCase),
    warp.tests.test_ctypes.register(BaseTestCase),
    warp.tests.test_devices.register(BaseTestCase),
    warp.tests.test_dlpack.register(BaseTestCase),
    warp.tests.test_fabricarray.register(BaseTestCase),
    warp.tests.test_func.register(BaseTestCase),
    warp.tests.test_generics.register(BaseTestCase),
    warp.tests.test_grad_customs.register(BaseTestCase),
    warp.tests.test_hash_grid.register(BaseTestCase),
    warp.tests.test_indexedarray.register(BaseTestCase),
    warp.tests.test_launch.register(BaseTestCase),
    warp.tests.test_marching_cubes.register(BaseTestCase),
    warp.tests.test_mat_lite.register(BaseTestCase),
    warp.tests.test_math.register(BaseTestCase),
    warp.tests.test_matmul_lite.register(BaseTestCase),
    warp.tests.test_mesh.register(BaseTestCase),
    warp.tests.test_mesh_query_aabb.register(BaseTestCase),
    warp.tests.test_mesh_query_point.register(BaseTestCase),
    warp.tests.test_mesh_query_ray.register(BaseTestCase),
    warp.tests.test_modules_lite.register(BaseTestCase),
    warp.tests.test_noise.register(BaseTestCase),
    warp.tests.test_operators.register(BaseTestCase),
    warp.tests.test_quat.register(BaseTestCase),
    warp.tests.test_rand.register(BaseTestCase),
    #warp.tests.test_reload.register(BaseTestCase),
    warp.tests.test_rounding.register(BaseTestCase),
    warp.tests.test_runlength_encode.register(BaseTestCase),
    warp.tests.test_sparse.register(BaseTestCase),
    warp.tests.test_streams.register(BaseTestCase),
    warp.tests.test_tape.register(BaseTestCase),
    warp.tests.test_transient_module.register(BaseTestCase),
    warp.tests.test_types.register(BaseTestCase),
    warp.tests.test_utils.register(BaseTestCase),
    warp.tests.test_vec_lite.register(BaseTestCase),
    warp.tests.test_volume_write.register(BaseTestCase),
    warp.tests.test_volume.register(BaseTestCase)
)

# Each test class needs to be defined at the module level to be found by
# the test runners.
locals().update({str(i): x for i, x in enumerate(test_clss)})
