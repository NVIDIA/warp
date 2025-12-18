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

# TODO: Remove after cleaning up the public API.

from warp._src import context as _context
from warp._src.utils import warn_deprecated_namespace as _warn_deprecated_namespace


def __getattr__(name):
    from warp._src.utils import get_deprecated_api, warn  # noqa: PLC0415

    # Handle special case: Devicelike -> DeviceLike (capital L)
    # This is both a rename AND a relocation to warp.DeviceLike
    if name == "Devicelike":
        warn(
            "The symbol `warp.context.Devicelike` will soon be removed from the public API. Use `warp.DeviceLike` instead.",
            DeprecationWarning,
        )
        return _context.DeviceLike

    return get_deprecated_api(_context, "warp", name)


_warn_deprecated_namespace(__name__)
