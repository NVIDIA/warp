# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test that validation failures don't contaminate shared modules with invalid code."""

import unittest

import warp as wp
from warp.tests.unittest_utils import *


def test_function_validation_failure_contamination(test, device):
    """Test that function validation failures don't contaminate modules.

    This test creates two scenarios in the same test module:
    1. A kernel that calls a function with invalid return type annotation
    2. A valid kernel that should work

    Without the fix, both kernels end up in the same module hash, and when
    the first kernel's function fails validation, it leaves undefined
    references that break C++ compilation of the entire module, causing
    the second kernel to fail too.
    """

    # First kernel: calls a function that will fail validation
    @wp.func
    def bad_return_type(x: int) -> tuple[int, int, int]:
        # Returns 2 values but annotation says 3 - validation will fail
        return (x + x, x * x)

    def bad_kernel_fn():
        _x, _y, _z = bad_return_type(123)

    # Second kernel: completely valid, should always work
    @wp.kernel
    def good_kernel():
        x = 1.0
        y = 2.0
        wp.expect_eq(x + y, 3.0)

    # The bad kernel should fail with WarpCodegenError
    bad_kernel = wp.Kernel(func=bad_kernel_fn)
    with test.assertRaisesRegex(
        wp.WarpCodegenError,
        r"has its return type annotated as a tuple of 3 elements but the code returns 2 values",
    ):
        wp.launch(bad_kernel, dim=1, device=device)

    # After the codegen failure, bad_kernel.adj.skip_build=True is set, which changes the
    # module hash (the failed kernel is excluded from the hash). Calling mark_modified()
    # clears the cached hash so the next load recomputes it and uses a different cache path.
    # Without this, on multi-GPU systems the second device would find the binary written
    # by the first device's successful good_kernel compilation and skip codegen entirely,
    # so the WarpCodegenError would never be raised for the subsequent devices.
    bad_kernel.module.mark_modified()

    # The good kernel should still work despite the bad kernel failure
    # This is the key test - without the fix, this will fail with
    # "use of undeclared identifier 'bad_return_type_1'" because both
    # kernels ended up in the same module and bad_return_type was never defined
    try:
        wp.launch(good_kernel, dim=1, device=device)
    except Exception as e:
        test.fail(f"good_kernel should not fail due to bad_kernel contamination, but got: {type(e).__name__}: {e}")


class TestModuleContamination(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestModuleContamination,
    func=test_function_validation_failure_contamination,
    name="test_function_validation_failure_contamination",
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
