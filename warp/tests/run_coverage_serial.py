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
