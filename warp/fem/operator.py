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

from typing import Any, Callable, Dict, Optional, Set

import warp as wp
from warp.fem.linalg import skew_part, symmetric_part
from warp.fem.types import Coords, Domain, ElementIndex, Field, NodeIndex, Sample, make_free_sample


class Integrand:
    """An integrand is a device function containing arbitrary expressions over Field and Domain variables.
    It will get transformed to a proper warp.Function by resolving concrete Field types at call time.
    """

    def __init__(self, func: Callable, kernel_options: Optional[Dict[str, Any]] = None):
        self.func = func
        self.name = wp.codegen.make_full_qualified_name(self.func)
        self.module = wp.get_module(self.func.__module__)
        self.argspec = wp.codegen.get_full_arg_spec(self.func)
        self.kernel_options = {} if kernel_options is None else kernel_options

        # Operators for each field argument. This will be populated at first integrate call
        self.operators: Dict[str, Set[Operator]] = None


class Operator:
    """
    Operators provide syntactic sugar over Field and Domain evaluation functions and arguments
    """

    def __init__(self, func: Callable, resolver: Callable, field_result: Callable = None):
        self.func = func
        self.name = func.__name__
        self.resolver = resolver
        self.field_result = field_result


def integrand(func: Callable = None, kernel_options: Optional[Dict[str, Any]] = None):
    """Decorator for functions to be integrated (or interpolated) using warp.fem

    Args:
        func: Decorated function
        kernel_options: Supplemental code-generation options to be passed to the generated kernel.
    """

    if func is not None:
        itg = Integrand(func)
        itg.__doc__ = func.__doc__
        return itg

    def wrap_integrand(func: Callable):
        itg = Integrand(func, kernel_options)
        itg.__doc__ = func.__doc__
        return itg

    return wrap_integrand


def operator(**kwargs):
    """Decorator for functions operating on Field-like or Domain-like data inside warp.fem integrands"""

    def wrap_operator(func: Callable):
        op = Operator(func, **kwargs)
        op.__doc__ = func.__doc__
        return op

    return wrap_operator


# Domain operators


@operator(resolver=lambda dmn: dmn.element_position)
def position(domain: Domain, s: Sample):
    """Evaluates the world position of the sample point `s`"""
    pass


@operator(resolver=lambda dmn: dmn.element_normal)
def normal(domain: Domain, s: Sample):
    """Evaluates the element normal at the sample point `s`. Non zero if the element is a side or the geometry is embedded in a higher-dimensional space (e.g. :class:`Trimesh3D`)"""
    pass


@operator(resolver=lambda dmn: dmn.element_deformation_gradient)
def deformation_gradient(domain: Domain, s: Sample):
    """Evaluates the gradient of the domain position with respect to the element reference space at the sample point `s`"""
    pass


@operator(resolver=lambda dmn: dmn.element_lookup)
def lookup(domain: Domain, x: Any) -> Sample:
    """Looks-up the sample point corresponding to a world position `x`, projecting to the closest point on the domain.

    Args:
        x: world position of the point to look-up in the geometry
        guess: (optional) :class:`Sample` initial guess, may help perform the query

    Note:
        Currently this operator is unsupported for :class:`Hexmesh`, :class:`Quadmesh2D`, :class:`Quadmesh3D` and deformed geometries.
    """
    pass


@operator(resolver=lambda dmn: dmn.element_measure)
def measure(domain: Domain, s: Sample) -> float:
    """Returns the measure (volume, area, or length) determinant of an element at a sample point `s`"""
    pass


@operator(resolver=lambda dmn: dmn.element_measure_ratio)
def measure_ratio(domain: Domain, s: Sample) -> float:
    """Returns the maximum ratio between the measure of this element and that of higher-dimensional neighbors."""
    pass


# Operators for evaluating cell-level quantities on domains defined on sides


@operator(
    resolver=lambda dmn: dmn.domain_cell_arg, field_result=lambda dmn: (dmn.cell_domain(), Domain, dmn.geometry.CellArg)
)
def cells(domain: Domain) -> Domain:
    """Converts a domain defined on geometry sides to a domain defined of cells."""
    pass


@operator(resolver=lambda dmn: dmn.element_inner_cell_index)
def _inner_cell_index(domain: Domain, side_index: ElementIndex, side_coords: Coords) -> Sample:
    pass


@operator(resolver=lambda dmn: dmn.element_outer_cell_index)
def _outer_cell_index(domain: Domain, side_index: ElementIndex, side_coords: Coords) -> Sample:
    pass


@operator(resolver=lambda dmn: dmn.element_inner_cell_coords)
def _inner_cell_coords(domain: Domain, side_index: ElementIndex, side_coords: Coords) -> Sample:
    pass


@operator(resolver=lambda dmn: dmn.element_outer_cell_coords)
def _outer_cell_coords(domain: Domain, side_index: ElementIndex, side_coords: Coords) -> Sample:
    pass


@operator(resolver=lambda dmn: dmn.cell_to_element_coords)
def _cell_to_element_coords(
    domain: Domain, side_index: ElementIndex, cell_index: ElementIndex, cell_coords: Coords
) -> Sample:
    pass


@integrand
def to_inner_cell(domain: Domain, s: Sample):
    """Converts a :class:`Sample` defined on a side to a sample defined on the side's inner cell"""
    return make_free_sample(
        _inner_cell_index(domain, s.element_index), _inner_cell_coords(domain, s.element_index, s.element_coords)
    )


@integrand
def to_outer_cell(domain: Domain, s: Sample):
    """Converts a :class:`Sample` defined on a side to a sample defined on the side's outer cell"""
    return make_free_sample(
        _outer_cell_index(domain, s.element_index), _outer_cell_coords(domain, s.element_index, s.element_coords)
    )


@integrand
def to_cell_side(domain: Domain, cell_s: Sample, side_index: ElementIndex):
    """Converts a :class:`Sample` defined on a cell to a sample defined on one of its side.
    If the result does not lie on the side `side_index`, the resulting coordinates will be set to ``OUTSIDE``."""
    return make_free_sample(
        side_index, _cell_to_element_coords(domain, side_index, cell_s.element_index, cell_s.element_coords)
    )


# Field operators
# On a side, inner and outer are such that normal goes from inner to outer


@operator(resolver=lambda f: f.eval_inner)
def inner(f: Field, s: Sample):
    """Evaluates the field at a sample point `s`. On oriented sides, uses the inner element"""
    pass


@operator(resolver=lambda f: f.eval_grad_inner)
def grad(f: Field, s: Sample):
    """Evaluates the field gradient at a sample point `s`. On oriented sides, uses the inner element"""
    pass


@operator(resolver=lambda f: f.eval_div_inner)
def div(f: Field, s: Sample):
    """Evaluates the field divergence at a sample point `s`. On oriented sides, uses the inner element"""
    pass


@operator(resolver=lambda f: f.eval_outer)
def outer(f: Field, s: Sample):
    """Evaluates the field at a sample point `s`. On oriented sides, uses the outer element. On interior points and on domain boundaries, this is equivalent to :func:`inner`."""
    pass


@operator(resolver=lambda f: f.eval_grad_outer)
def grad_outer(f: Field, s: Sample):
    """Evaluates the field gradient at a sample point `s`. On oriented sides, uses the outer element. On interior points and on domain boundaries, this is equivalent to :func:`grad`."""
    pass


@operator(resolver=lambda f: f.eval_grad_outer)
def div_outer(f: Field, s: Sample):
    """Evaluates the field divergence at a sample point `s`. On oriented sides, uses the outer element. On interior points and on domain boundaries, this is equivalent to :func:`div`."""
    pass


@operator(resolver=lambda f: f.eval_degree)
def degree(f: Field):
    """Polynomial degree of a field"""
    pass


@operator(resolver=lambda f: f.at_node)
def at_node(f: Field, s: Sample):
    """For a Test or Trial field `f`, returns a copy of the Sample `s` moved to the coordinates of the node being evaluated"""
    pass


@operator(resolver=lambda f: f.node_partition_index)
def node_partition_index(f: Field, node_index: NodeIndex):
    """For a NodalField `f`, returns the index of a given node in the fields's space partition,
    or ``NULL_NODE_INDEX`` if it does not exists"""
    pass


# Common derived operators, for convenience


@integrand
def D(f: Field, s: Sample):
    """Symmetric part of the (inner) gradient of the field at `s`"""
    return symmetric_part(grad(f, s))


@integrand
def curl(f: Field, s: Sample):
    """Skew part of the (inner) gradient of the field at `s`, as a vector such that ``wp.cross(curl(u), v) = skew(grad(u)) v``"""
    return skew_part(grad(f, s))


@integrand
def jump(f: Field, s: Sample):
    """Jump between inner and outer element values on an interior side. Zero for interior points or domain boundaries"""
    return inner(f, s) - outer(f, s)


@integrand
def average(f: Field, s: Sample):
    """Average between inner and outer element values"""
    return 0.5 * (inner(f, s) + outer(f, s))


@integrand
def grad_jump(f: Field, s: Sample):
    """Jump between inner and outer element gradients on an interior side. Zero for interior points or domain boundaries"""
    return grad(f, s) - grad_outer(f, s)


@integrand
def grad_average(f: Field, s: Sample):
    """Average between inner and outer element gradients"""
    return 0.5 * (grad(f, s) + grad_outer(f, s))


# Set default call operators for argument types, so that field(s) = inner(field, s) and domain(s) = position(domain, s)
Field.call_operator = inner
Domain.call_operator = position
