# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Iterative linear solvers for sparse linear systems.

This module provides GPU-accelerated iterative methods for solving linear systems,
including conjugate gradient (CG), biconjugate gradient stabilized (BiCGSTAB), conjugate
residual (CR), and generalized minimal residual (GMRES) methods.
"""

# isort: skip_file

from warp._src.optim.linear import LinearOperator as LinearOperator
from warp._src.optim.linear import aslinearoperator as aslinearoperator
from warp._src.optim.linear import bicgstab as bicgstab
from warp._src.optim.linear import cg as cg
from warp._src.optim.linear import cr as cr
from warp._src.optim.linear import gmres as gmres
from warp._src.optim.linear import preconditioner as preconditioner


# TODO: Remove after cleaning up the public API.

from warp._src.optim import linear as _linear


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_linear, "warp.optim", name)
