# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Serial code-coverage runner

This script is used to generate code-coverage reports by running Warp tests.
It runs in serial so can take over an hour to finish. To generate a coverage
report in parallel, use the warp/thirdparty./unittest_parallel.py script
instead with the --coverage option, e.g. python -m warp.tests --coverage
"""

import coverage

cover = coverage.Coverage(config_file=True, messages=True)

cover.start()

with cover.collect():
    import unittest_serial  # noqa: E402

    unittest_serial.run_specified()

cover.save()

cover.report()

cover.html_report(title="Warp Testing Code Coverage Report")
