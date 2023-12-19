# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
from warp.tests.unittest_utils import TeamCityTestRunner


def run_suite() -> bool:
    """Run a test suite"""

    import warp.tests.unittest_suites

    # force rebuild of all kernels
    wp.build.clear_kernel_cache()
    print("Cleared Warp kernel cache")

    runner = TeamCityTestRunner(verbosity=2, failfast=False)

    # Can swap out warp.tests.unittest_suites.explicit_suite()
    suite = warp.tests.unittest_suites.auto_discover_suite()
    print(f"Test suite has {suite.countTestCases()} tests")

    ret = not runner.run(suite, "WarpTests").wasSuccessful()
    return ret


if __name__ == "__main__":
    ret = run_suite()
    import sys

    sys.exit(ret)
