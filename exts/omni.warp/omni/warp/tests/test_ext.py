# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test

import numpy as np
import warp as wp

from warp.tests.test_all import register_tests

# build test cases using Kit async test base class
tests = register_tests(omni.kit.test.AsyncTestCaseFailOnLogError)

# ensure all modules loaded
wp.force_load()