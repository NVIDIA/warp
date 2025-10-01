# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
from enum import Enum

import numpy as np


class Polynomial(Enum):
    """Polynomial family defining interpolation nodes over an interval"""

    GAUSS_LEGENDRE = "GL"
    """Gauss--Legendre 1D polynomial family (does not include endpoints)"""

    LOBATTO_GAUSS_LEGENDRE = "LGL"
    """Lobatto--Gauss--Legendre 1D polynomial family (includes endpoints)"""

    EQUISPACED_CLOSED = "closed"
    """Closed 1D polynomial family with uniformly distributed nodes (includes endpoints)"""

    EQUISPACED_OPEN = "open"
    """Open 1D polynomial family with uniformly distributed nodes (does not include endpoints)"""

    def __str__(self):
        return self.value


def is_closed(family: Polynomial):
    """Whether the polynomial roots include interval endpoints"""
    return family == Polynomial.LOBATTO_GAUSS_LEGENDRE or family == Polynomial.EQUISPACED_CLOSED


def _gauss_legendre_quadrature_1d(n: int):
    if n == 1:
        coords = [0.0]
        weights = [2.0]
    elif n == 2:
        coords = [-math.sqrt(1.0 / 3), math.sqrt(1.0 / 3)]
        weights = [1.0, 1.0]
    elif n == 3:
        coords = [0.0, -math.sqrt(3.0 / 5.0), math.sqrt(3.0 / 5.0)]
        weights = [8.0 / 9.0, 5.0 / 9.0, 5.0 / 9.0]
    elif n == 4:
        c_a = math.sqrt(3.0 / 7.0 - 2.0 / 7.0 * math.sqrt(6.0 / 5.0))
        c_b = math.sqrt(3.0 / 7.0 + 2.0 / 7.0 * math.sqrt(6.0 / 5.0))
        w_a = (18.0 + math.sqrt(30.0)) / 36.0
        w_b = (18.0 - math.sqrt(30.0)) / 36.0
        coords = [c_a, -c_a, c_b, -c_b]
        weights = [w_a, w_a, w_b, w_b]
    elif n == 5:
        c_a = 1.0 / 3.0 * math.sqrt(5.0 - 2.0 * math.sqrt(10.0 / 7.0))
        c_b = 1.0 / 3.0 * math.sqrt(5.0 + 2.0 * math.sqrt(10.0 / 7.0))
        w_a = (322.0 + 13.0 * math.sqrt(70.0)) / 900.0
        w_b = (322.0 - 13.0 * math.sqrt(70.0)) / 900.0
        coords = [0.0, c_a, -c_a, c_b, -c_b]
        weights = [128.0 / 225.0, w_a, w_a, w_b, w_b]
    else:
        raise NotImplementedError

    # Shift from [-1, 1] to [0, 1]
    weights = 0.5 * np.array(weights)
    coords = 0.5 * np.array(coords) + 0.5

    return coords, weights


def _lobatto_gauss_legendre_quadrature_1d(n: int):
    if n == 2:
        coords = [-1.0, 1.0]
        weights = [1.0, 1.0]
    elif n == 3:
        coords = [-1.0, 0.0, 1.0]
        weights = [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]
    elif n == 4:
        coords = [-1.0, -1.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0), 1.0]
        weights = [1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0]
    elif n == 5:
        coords = [-1.0, -math.sqrt(3.0 / 7.0), 0.0, math.sqrt(3.0 / 7.0), 1.0]
        weights = [1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0]
    else:
        raise NotImplementedError

    # Shift from [-1, 1] to [0, 1]
    weights = 0.5 * np.array(weights)
    coords = 0.5 * np.array(coords) + 0.5

    return coords, weights


def _uniform_open_quadrature_1d(n: int):
    step = 1.0 / (n + 1)
    coords = np.linspace(step, 1.0 - step, n)
    weights = np.full(n, 1.0 / (n + 1))

    # Boundaries have 3/2 the weight
    weights[0] = 1.5 / (n + 1)
    weights[-1] = 1.5 / (n + 1)

    return coords, weights


def _uniform_closed_quadrature_1d(n: int):
    coords = np.linspace(0.0, 1.0, n)
    weights = np.full(n, 1.0 / (n - 1))

    # Boundaries have half the weight
    weights[0] = 0.5 / (n - 1)
    weights[-1] = 0.5 / (n - 1)

    return coords, weights


def _open_newton_cotes_quadrature_1d(n: int):
    step = 1.0 / (n + 1)
    coords = np.linspace(step, 1.0 - step, n)

    # Weisstein, Eric W. "Newton-Cotes Formulas." From MathWorld--A Wolfram Web Resource.
    # https://mathworld.wolfram.com/Newton-CotesFormulas.html

    if n == 1:
        weights = np.array([1.0])
    elif n == 2:
        weights = np.array([0.5, 0.5])
    elif n == 3:
        weights = np.array([2.0, -1.0, 2.0]) / 3.0
    elif n == 4:
        weights = np.array([11.0, 1.0, 1.0, 11.0]) / 24.0
    elif n == 5:
        weights = np.array([11.0, -14.0, 26.0, -14.0, 11.0]) / 20.0
    elif n == 6:
        weights = np.array([611.0, -453.0, 562.0, 562.0, -453.0, 611.0]) / 1440.0
    elif n == 7:
        weights = np.array([460.0, -954.0, 2196.0, -2459.0, 2196.0, -954.0, 460.0]) / 945.0
    else:
        raise NotImplementedError

    return coords, weights


def _closed_newton_cotes_quadrature_1d(n: int):
    coords = np.linspace(0.0, 1.0, n)

    # OEIS: A093735, A093736

    if n == 2:
        weights = np.array([1.0, 1.0]) / 2.0
    elif n == 3:
        weights = np.array([1.0, 4.0, 1.0]) / 3.0
    elif n == 4:
        weights = np.array([3.0, 9.0, 9.0, 3.0]) / 8.0
    elif n == 5:
        weights = np.array([14.0, 64.0, 24.0, 64.0, 14.0]) / 45.0
    elif n == 6:
        weights = np.array([95.0 / 288.0, 125.0 / 96.0, 125.0 / 144.0, 125.0 / 144.0, 125.0 / 96.0, 95.0 / 288.0])
    elif n == 7:
        weights = np.array([41, 54, 27, 68, 27, 54, 41], dtype=float) / np.array(
            [140, 35, 140, 35, 140, 35, 140], dtype=float
        )
    elif n == 8:
        weights = np.array(
            [
                5257,
                25039,
                343,
                20923,
                20923,
                343,
                25039,
                5257,
            ]
        ) / np.array(
            [
                17280,
                17280,
                640,
                17280,
                17280,
                640,
                17280,
                17280,
            ],
            dtype=float,
        )
    else:
        raise NotImplementedError

    # Normalize with interval length
    weights = weights / (n - 1)

    return coords, weights


def quadrature_1d(point_count: int, family: Polynomial):
    """Return quadrature points and weights for the given family and point count"""

    if family == Polynomial.GAUSS_LEGENDRE:
        return _gauss_legendre_quadrature_1d(point_count)
    if family == Polynomial.LOBATTO_GAUSS_LEGENDRE:
        return _lobatto_gauss_legendre_quadrature_1d(point_count)
    if family == Polynomial.EQUISPACED_CLOSED:
        return _closed_newton_cotes_quadrature_1d(point_count)
    if family == Polynomial.EQUISPACED_OPEN:
        return _open_newton_cotes_quadrature_1d(point_count)

    raise NotImplementedError


def lagrange_scales(coords: np.array):
    """Return the scaling factors for Lagrange polynomials with roots at coords"""
    lagrange_scale = np.empty_like(coords)
    for i in range(len(coords)):
        deltas = coords[i] - coords
        deltas[i] = 1.0
        lagrange_scale[i] = 1.0 / np.prod(deltas)

    return lagrange_scale
