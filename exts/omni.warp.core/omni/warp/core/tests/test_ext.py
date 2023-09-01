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
    warp.tests.test_codegen.register(BaseTestCase),
    warp.tests.test_mesh_query_aabb.register(BaseTestCase),
    warp.tests.test_mesh_query_point.register(BaseTestCase),
    warp.tests.test_mesh_query_ray.register(BaseTestCase),
    warp.tests.test_conditional.register(BaseTestCase),
    warp.tests.test_operators.register(BaseTestCase),
    warp.tests.test_rounding.register(BaseTestCase),
    warp.tests.test_hash_grid.register(BaseTestCase),
    warp.tests.test_ctypes.register(BaseTestCase),
    warp.tests.test_rand.register(BaseTestCase),
    warp.tests.test_noise.register(BaseTestCase),
    warp.tests.test_tape.register(BaseTestCase),
)

# Each test class needs to be defined at the module level to be found by
# the test runners.
locals().update({str(i): x for i, x in enumerate(test_clss)})
