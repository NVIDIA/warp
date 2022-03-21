# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test

import numpy as np
import warp as wp

import warp.tests.test_codegen 
import warp.tests.test_mesh_query_aabb
import warp.tests.test_mesh_query_point
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

# build test cases using Kit async test base class
test_codegen = warp.tests.test_codegen.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_mesh_query_aabb = warp.tests.test_mesh_query_aabb.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_mesh_query_point = warp.tests.test_mesh_query_point.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_conditional = warp.tests.test_conditional.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_operators = warp.tests.test_operators.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_rounding = warp.tests.test_rounding.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_hash_grid = warp.tests.test_hash_grid.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_ctypes = warp.tests.test_ctypes.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_rand = warp.tests.test_rand.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_noise = warp.tests.test_noise.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_tape = warp.tests.test_tape.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_compile_consts = warp.tests.test_compile_consts.register(omni.kit.test.AsyncTestCaseFailOnLogError)
test_volume = warp.tests.test_volume.register(omni.kit.test.AsyncTestCaseFailOnLogError)

# ensure all modules loaded
wp.force_load()