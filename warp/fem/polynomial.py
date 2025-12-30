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

"""Polynomial utilities for finite element interpolation and quadrature.

This module provides utilities for working with polynomial families used in FEM,
including quadrature rules (Gauss-Legendre, Lobatto-Gauss-Legendre, Newton-Cotes)
and Lagrange polynomial scaling factors.
"""

# isort: skip_file

from warp._src.fem.polynomial import quadrature_1d as quadrature_1d
from warp._src.fem.polynomial import lagrange_scales as lagrange_scales


# TODO: Remove after cleaning up the public API.

from warp._src.fem import polynomial as _polynomial


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_polynomial, "warp.fem", name)
