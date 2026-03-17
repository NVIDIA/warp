# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendering utilities for visualizing Warp simulations.

This module provides a set of renderers that can be used for visualizing scenes
involving shapes of various types.

The :class:`OpenGLRenderer` provides an interactive renderer to play back animations
in real time and is mostly intended for debugging, whereas more sophisticated rendering
can be achieved with the help of the :class:`UsdRenderer`, which allows exporting the
scene to a USD file that can then be rendered in an external 3D application or renderer
of your choice.

Usage:
    This module must be explicitly imported::

        import warp.render
"""

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
