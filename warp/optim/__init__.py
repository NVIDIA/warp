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

# The source-to-public Warp module declarations for `warp.optim` live in the
# top-level `warp/__init__.py`, so they are in effect before these imports run.

from warp._src.optim.adam import Adam as Adam
from warp._src.optim.sgd import SGD as SGD

from . import linear as linear
