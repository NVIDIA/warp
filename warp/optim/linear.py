# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Iterative linear solvers for sparse linear systems.

This module provides GPU-accelerated iterative methods for solving linear systems,
including conjugate gradient (CG), biconjugate gradient stabilized (BiCGSTAB), conjugate
residual (CR), and generalized minimal residual (GMRES) methods.
"""

# isort: skip_file

from warp._src.optim.linear import BiCGSTAB as BiCGSTAB
from warp._src.optim.linear import CG as CG
from warp._src.optim.linear import CR as CR
from warp._src.optim.linear import GMRES as GMRES
from warp._src.optim.linear import LinearOperator as LinearOperator
from warp._src.optim.linear import LinearSolverState as LinearSolverState
from warp._src.optim.linear import aslinearoperator as aslinearoperator
from warp._src.optim.linear import bicgstab as bicgstab
from warp._src.optim.linear import cg as cg
from warp._src.optim.linear import cr as cr
from warp._src.optim.linear import gmres as gmres
from warp._src.optim.linear import preconditioner as preconditioner
