# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# isort: skip_file

from warp._src.context import Device as Device
from warp._src.context import Devicelike as Devicelike
from warp._src.context import Module as Module
from warp._src.context import assert_conditional_graph_support as assert_conditional_graph_support
from warp._src.context import get_module as get_module
from warp._src.context import type_str as type_str


# TODO: Remove after cleaning up the public API.

from warp._src import context as _context


def __getattr__(name):
    from warp._src.utils import get_deprecated_api

    return get_deprecated_api(_context, "wp", name)
