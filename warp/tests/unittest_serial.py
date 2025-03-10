# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import warp as wp
import warp.tests.unittest_suites


def run_suite() -> bool:
    """Run a test suite"""

    # force rebuild of all kernels
    wp.clear_lto_cache()
    wp.clear_kernel_cache()
    print("Cleared Warp kernel cache")

    runner = unittest.TextTestRunner(verbosity=2, failfast=True)

    # Can swap out different suites
    suite = warp.tests.unittest_suites.default_suite()
    # suite = warp.tests.unittest_suites.auto_discover_suite()
    # suite = warp.tests.unittest_suites.kit_suite()

    print(f"Test suite has {suite.countTestCases()} tests")

    ret = not runner.run(suite).wasSuccessful()
    return ret


if __name__ == "__main__":
    ret = run_suite()
    import sys

    sys.exit(ret)
