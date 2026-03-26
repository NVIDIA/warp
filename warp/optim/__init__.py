# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimization algorithms for gradient descent and linear systems.

This module provides gradient-based optimizers (:class:`Adam`, :class:`SGD`) for
updating arrays based on computed gradients. The :mod:`warp.optim.linear` submodule
provides iterative linear solvers.

Usage:
    This module must be explicitly imported::

        import warp.optim
"""

# isort: skip_file

from warp._src.optim.adam import Adam as Adam
from warp._src.optim.sgd import SGD as SGD

from . import linear as linear


# TODO: Remove after cleaning up the public API.

from warp._src import optim as _optim


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_optim, "warp", name)
