# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

import warp as wp

"""
Vector norm functions
"""

__all__ = [
    "norm_l1",
    "norm_l2",
    "norm_huber",
    "norm_pseudo_huber",
    "smooth_normalize",
]


@wp.func
def norm_l1(v: Any):
    """
    Computes the L1 norm of a vector v.

    .. math:: \\|v\\|_1 = \\sum_i |v_i|

    Args:
        v (Vector[Any,Float]): The vector to compute the L1 norm of.

    Returns:
        float: The L1 norm of the vector.
    """
    n = float(0.0)
    for i in range(len(v)):
        n += wp.abs(v[i])
    return n


@wp.func
def norm_l2(v: Any):
    """
    Computes the L2 norm of a vector v.

    .. math:: \\|v\\|_2 = \\sqrt{\\sum_i v_i^2}

    Args:
        v (Vector[Any,Float]): The vector to compute the L2 norm of.

    Returns:
        float: The L2 norm of the vector.
    """
    return wp.length(v)


@wp.func
def norm_huber(v: Any, delta: float = 1.0):
    """
    Computes the Huber norm of a vector v with a given delta.

    .. math::
        H(v) = \\begin{cases} \\frac{1}{2} \\|v\\|^2 & \\text{if } \\|v\\| \\leq \\delta \\\\ \\delta(\\|v\\| - \\frac{1}{2}\\delta) & \\text{otherwise} \\end{cases}

    .. image:: /img/norm_huber.svg
        :align: center

    Args:
        v (Vector[Any,Float]): The vector to compute the Huber norm of.
        delta (float): The threshold value, defaults to 1.0.

    Returns:
        float: The Huber norm of the vector.
    """
    a = wp.dot(v, v)
    if a <= delta * delta:
        return 0.5 * a
    return delta * (wp.sqrt(a) - 0.5 * delta)


@wp.func
def norm_pseudo_huber(v: Any, delta: float = 1.0):
    """
    Computes the "pseudo" Huber norm of a vector v with a given delta.

    .. math::
        H^\\prime(v) = \\delta \\sqrt{1 + \\frac{\\|v\\|^2}{\\delta^2}}

    .. image:: /img/norm_pseudo_huber.svg
        :align: center

    Args:
        v (Vector[Any,Float]): The vector to compute the Huber norm of.
        delta (float): The threshold value, defaults to 1.0.

    Returns:
        float: The Huber norm of the vector.
    """
    a = wp.dot(v, v)
    return delta * wp.sqrt(1.0 + a / (delta * delta))


@wp.func
def smooth_normalize(v: Any, delta: float = 1.0):
    """
    Normalizes a vector using the pseudo-Huber norm.

    See :func:`norm_pseudo_huber`.

    .. math::
        \\frac{v}{H^\\prime(v)}

    Args:
        v (Vector[Any,Float]): The vector to normalize.
        delta (float): The threshold value, defaults to 1.0.

    Returns:
        Vector[Any,Float]: The normalized vector.
    """
    return v / norm_pseudo_huber(v, delta)


# register API functions so they appear in the documentation

wp.context.register_api_function(
    norm_l1,
    group="Vector Math",
)
wp.context.register_api_function(
    norm_l2,
    group="Vector Math",
)
wp.context.register_api_function(
    norm_huber,
    group="Vector Math",
)
wp.context.register_api_function(
    norm_pseudo_huber,
    group="Vector Math",
)
wp.context.register_api_function(
    smooth_normalize,
    group="Vector Math",
)
