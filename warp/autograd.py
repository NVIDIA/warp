# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for debugging automatic differentiation.

This module provides functions to evaluate and verify gradients computed by Warp's
automatic differentiation engine. Typical workflows involve computing Jacobian matrices
using both automatic differentiation and finite differences, then comparing them to
verify gradient accuracy.

Usage:
    This module must be explicitly imported::

        import warp.autograd
"""

# isort: skip_file

from warp._src.autograd import gradcheck as gradcheck
from warp._src.autograd import gradcheck_tape as gradcheck_tape
from warp._src.autograd import jacobian as jacobian
from warp._src.autograd import jacobian_fd as jacobian_fd
from warp._src.autograd import jacobian_plot as jacobian_plot


# TODO: Remove after cleaning up the public API.

from warp._src import autograd as _autograd


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_autograd, "warp", name)
