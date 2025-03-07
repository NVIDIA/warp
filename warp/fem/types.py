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

from enum import Enum

import warp as wp

# kept to avoid breaking existing example code, no longer used internally
vec2i = wp.vec2i
vec3i = wp.vec3i
vec4i = wp.vec4i

Coords = wp.vec3
OUTSIDE = wp.constant(-1.0e8)

ElementIndex = int
QuadraturePointIndex = int
NodeIndex = int

NULL_ELEMENT_INDEX = wp.constant(-1)
NULL_QP_INDEX = wp.constant(-1)
NULL_NODE_INDEX = wp.constant((1 << 31) - 1)  # this should be larger than normal nodes when sorting

DofIndex = wp.vec2i
"""Opaque descriptor for indexing degrees of freedom within elements"""
NULL_DOF_INDEX = wp.constant(DofIndex(-1, -1))


@wp.func
def get_node_index_in_element(dof_idx: DofIndex):
    return dof_idx[0]


@wp.func
def get_node_coord(dof_idx: DofIndex):
    return dof_idx[1]


class ElementKind(Enum):
    CELL = 0
    SIDE = 1


@wp.struct
class NodeElementIndex:
    domain_element_index: ElementIndex
    node_index_in_element: int


@wp.struct
class Sample:
    """Per-sample point context for evaluating fields and related operators in integrands"""

    element_index: ElementIndex
    """Index of the geometry element the sample point is in"""
    element_coords: Coords
    """Coordinates of the sample point inside the element"""
    qp_index: QuadraturePointIndex = NULL_QP_INDEX
    """If the sample corresponds to a quadrature point, its global index"""
    qp_weight: float = 0.0
    """If the sample corresponds to a quadrature point, its weight"""
    test_dof: DofIndex = NULL_DOF_INDEX
    """For linear of bilinear form assembly, index of the test degree-of-freedom currently being considered"""
    trial_dof: DofIndex = NULL_DOF_INDEX
    """For bilinear form assembly, index of the trial degree-of-freedom currently being considered"""


@wp.func
def make_free_sample(element_index: ElementIndex, element_coords: Coords):
    """Returns a :class:`Sample` that is not associated to any quadrature point or dof"""
    return Sample(element_index, element_coords, NULL_QP_INDEX, 0.0, NULL_DOF_INDEX, NULL_DOF_INDEX)


class Field:
    """
    Tag for field-like integrand arguments
    """

    call_operator: "warp.fem.operator.Operator" = None  # noqa: F821 Set in operator.py


class Domain:
    """
    Tag for domain-like integrand arguments
    """

    call_operator: "warp.fem.operator.Operator" = None  # noqa: F821 Set in operator.py
