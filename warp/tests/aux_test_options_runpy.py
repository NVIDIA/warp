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

"""Auxiliary module for testing wp.set_module_options under runpy.

This module is executed via runpy.run_module() from test_options.py to
verify that set_module_options/get_module_options work correctly even
when the calling module is not yet registered in sys.modules.
"""

import warp as wp

# This call will exercise the _get_caller_module_name fallback chain when
# this module is run via runpy.run_module().
wp.set_module_options({"enable_backward": False})

options = wp.get_module_options()

# Store the result so the test can inspect it.
_result = {
    "enable_backward": options.get("enable_backward"),
    "success": True,
}
