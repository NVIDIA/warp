# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Polynomial utilities for finite element interpolation and quadrature.

This module provides utilities for working with polynomial families used in FEM,
including quadrature rules (Gauss-Legendre, Lobatto-Gauss-Legendre, Newton-Cotes)
and Lagrange polynomial scaling factors.
"""

# isort: skip_file

from warp._src.fem.polynomial import quadrature_1d as quadrature_1d
from warp._src.fem.polynomial import lagrange_scales as lagrange_scales
