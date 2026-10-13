# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any

import numpy as np

import warp as wp
from warp._src.fem import cache
from warp._src.fem.geometry import Element
from warp._src.fem.types import cached_coords_type


class ShapeFunction:
    """Interface class for defining scalar-valued shape functions over a single element."""

    ORDER: int
    """Maximum degree of the polynomials used to define the shape function."""

    NODES_PER_ELEMENT: int
    """Number of shape function nodes."""

    class Value(Enum):
        Scalar = 0
        """Scalar-valued shape function."""
        CovariantVector = 1
        """Covariant vector-valued shape function."""
        ContravariantVector = 2
        """Contravariant vector-valued shape function."""

    value: Value = Value.Scalar
    """Value type of the shape function."""

    @property
    def _precision_suffix(self) -> str:
        """Suffix for cache key differentiation between fp32 and fp64 shape function variants."""
        return "_f64" if getattr(self, "scalar_type", wp.float32) == wp.float64 else ""

    @property
    def name(self) -> str:
        """Unique name encoding all parameters defining the shape function"""
        raise NotImplementedError()

    def make_node_coords_in_element(self):
        """Create a device function returning the coordinates of each node."""
        raise NotImplementedError()

    def make_node_quadrature_weight(self):
        """Create a device function returning the weight of each node when used as a quadrature point over the element."""
        raise NotImplementedError()

    def make_trace_node_quadrature_weight(self):
        """Create a device function returning the weight of each node when used as a quadrature point over the element boundary."""
        raise NotImplementedError()

    def make_element_inner_weight(self):
        """Create a device function returning the value of the shape function associated to a given node at given coordinates."""
        raise NotImplementedError()

    def make_element_inner_weight_gradient(self):
        """Create a device function returning the gradient of the shape function associated to a given node at given coordinates."""
        raise NotImplementedError()

    def make_node_dual_weight(self):
        """Create a device function returning the reference-space dual weight of each node.

        The degree of freedom of node ``i`` reproducing a reference-space value ``v`` is ``v * d_i`` for scalar-valued
        shape functions, or ``dot(v, d_i)`` for vector-valued ones, with ``d_i`` the returned dual weight.
        For vector-valued shape functions, ``d_i`` is aligned with the edge tangent (covariant) or side normal (contravariant)
        associated to the node, and scaled such that ``dot(d_i, w_i) = 1``, with ``w_i`` the node weight at the node coordinates.

        The default implementation returns ``1 / w_i`` for scalar-valued shape functions, and must be overridden
        for vector-valued ones.
        """
        if self.value != ShapeFunction.Value.Scalar:
            raise NotImplementedError()

        node_coords_in_element = self.make_node_coords_in_element()
        element_inner_weight = self.make_element_inner_weight()
        ONE = wp.constant(getattr(self, "scalar_type", wp.float32)(1.0))

        @cache.dynamic_func(suffix=self.name)
        def node_dual_weight(node_index_in_elt: int):
            return ONE / element_inner_weight(node_coords_in_element(node_index_in_elt), node_index_in_elt)

        return node_dual_weight

    def _make_unit_node_dual_weight(self):
        """Unit dual weights, using node values as degrees of freedom.

        Exact for shape functions that are one at their own node and zero at the other nodes.
        """
        ONE = wp.constant(self.scalar_type(1.0))

        @cache.dynamic_func(suffix=self.name)
        def node_dual_weight(node_index_in_elt: int):
            return ONE

        return node_dual_weight

    def _make_normalized_node_dual_weight(self, node_dof_axis: wp.Function):
        """Scale node degree-of-freedom axes so that their dot product with the node weight at the node is one."""
        node_coords_in_element = self.make_node_coords_in_element()
        element_inner_weight = self.make_element_inner_weight()

        @cache.dynamic_func(suffix=self.name)
        def node_dual_weight(node_index_in_elt: int):
            axis = node_dof_axis(node_index_in_elt)
            node_weight = element_inner_weight(node_coords_in_element(node_index_in_elt), node_index_in_elt)
            return axis / wp.dot(axis, node_weight)

        return node_dual_weight


class ConstantShapeFunction(ShapeFunction):
    """Shape function that is constant over the element."""

    def __init__(self, element: Element, scalar_type: type = wp.float32):
        self._element_prototype = element.prototype
        self.scalar_type = scalar_type

        self.ORDER = wp.constant(0)
        self.NODES_PER_ELEMENT = wp.constant(1)

        coords, _ = self._element_prototype.instantiate_quadrature(order=0, family=None)
        CoordsType = cached_coords_type(scalar_type)
        self.COORDS = wp.constant(CoordsType(*coords[0]))

    @property
    def name(self) -> str:
        return f"{self._element_prototype.__name__}{self._precision_suffix}"

    def make_node_coords_in_element(self):
        COORDS = self.COORDS

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            return COORDS

        return node_coords_in_element

    def _make_weight_func(self):
        scalar = self.scalar_type
        ONE = wp.constant(scalar(1.0))

        @cache.dynamic_func(suffix=self.name)
        def _node_quadrature_weight(
            node_index_in_elt: int,
        ):
            return ONE

        return _node_quadrature_weight

    def make_node_quadrature_weight(self):
        return self._make_weight_func()

    def make_trace_node_quadrature_weight(self):
        return self._make_weight_func()

    def make_element_inner_weight(self):
        scalar = self.scalar_type
        ONE = wp.constant(scalar(1.0))

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Any,
            node_index_in_elt: int,
        ):
            return ONE

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        scalar = self.scalar_type
        grad_type = cache.cached_vec_type(length=self._element_prototype.dimension, dtype=scalar)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Any,
            node_index_in_elt: int,
        ):
            return grad_type(scalar(0.0))

        return element_inner_weight_gradient

    def make_node_dual_weight(self):
        return self._make_unit_node_dual_weight()

    def element_vtk_cells(self):
        cell_type = 1  # VTK_VERTEX

        return np.zeros((1, 1), dtype=int), np.full(1, cell_type, dtype=np.int8)
