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

from warp._src.render.render_opengl import OpenGLRenderer as OpenGLRenderer

from warp._src.render.render_usd import UsdRenderer as UsdRenderer


# TODO: Remove after cleaning up the public API.

from warp._src import render as _render
from warp._src.render import utils as _render_utils


def __getattr__(name):
    from warp._src.utils import get_deprecated_api, warn  # noqa: PLC0415

    # Symbols from warp._src.render.utils that were previously accessible from warp.render
    if name in ("bourke_color_map", "tab10_color_map", "solidify_mesh"):
        warn(
            f"The symbol `warp.render.{name}` will soon be removed from the public API. "
            f"It can still be accessed from `warp._src.render.utils.{name}` but might be changed or removed without notice.",
            DeprecationWarning,
        )
        return getattr(_render_utils, name)

    return get_deprecated_api(_render, "warp", name)
