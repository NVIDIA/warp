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

# The source-to-public Warp module declarations for `warp.render` live in the
# top-level `warp/__init__.py`, so they are in effect before these imports run.

from warp._src.render.render_opengl import OpenGLRenderer as OpenGLRenderer

from warp._src.render.render_usd import UsdRenderer as UsdRenderer
