# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linear algebra utilities for FEM computations.

This module provides kernel-compatible linear algebra functions including matrix
decompositions (QR factorization, tensor decompositions), eigenvalue solvers,
matrix inversion, and array operations commonly needed in finite element computations.
"""

# isort: skip_file

from warp._src.fem.linalg import array_axpy as array_axpy
from warp._src.fem.linalg import generalized_inner as generalized_inner
from warp._src.fem.linalg import generalized_outer as generalized_outer
from warp._src.fem.linalg import householder_make_hessenberg as householder_make_hessenberg
from warp._src.fem.linalg import householder_qr_decomposition as householder_qr_decomposition
from warp._src.fem.linalg import inverse_qr as inverse_qr
from warp._src.fem.linalg import skew_part as skew_part
from warp._src.fem.linalg import solve_triangular as solve_triangular
from warp._src.fem.linalg import spherical_part as spherical_part
from warp._src.fem.linalg import symmetric_eigenvalues_qr as symmetric_eigenvalues_qr
from warp._src.fem.linalg import symmetric_part as symmetric_part
from warp._src.fem.linalg import tridiagonal_symmetric_eigenvalues_qr as tridiagonal_symmetric_eigenvalues_qr


# TODO: Remove after cleaning up the public API.

from warp._src.fem import linalg as _linalg


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_linalg, "warp.fem", name)
