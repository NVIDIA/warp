# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test

import numpy as np
import warp as wp

# build test cases using Kit async test base class
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

# register a subset of Warp unit tests since the full suite is too slow
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
    
    return tests

tests = register_tests(omni.kit.test.AsyncTestCase)

# test classes must be stored as local variables to be recognized
for i, test in enumerate(tests):
    locals()[f"{i}"] = test

# ensure all modules loaded
wp.force_load()
