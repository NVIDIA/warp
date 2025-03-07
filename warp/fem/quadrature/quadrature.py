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

from typing import Any, Optional

import warp as wp
from warp.fem import cache
from warp.fem.domain import GeometryDomain
from warp.fem.geometry import Element
from warp.fem.space import FunctionSpace
from warp.fem.types import NULL_ELEMENT_INDEX, Coords, ElementIndex, QuadraturePointIndex

from ..polynomial import Polynomial


@wp.struct
class QuadraturePointElementIndex:
    domain_element_index: ElementIndex
    qp_index_in_element: int


class Quadrature:
    """Interface class for quadrature rules"""

    @wp.struct
    class Arg:
        """Structure containing arguments to be passed to device functions"""

        pass

    def __init__(self, domain: GeometryDomain):
        self._domain = domain

    @property
    def domain(self):
        """Domain over which this quadrature is defined"""
        return self._domain

    def arg_value(self, device) -> "Arg":
        """
        Value of the argument to be passed to device
        """
        arg = Quadrature.Arg()
        return arg

    def total_point_count(self):
        """Number of unique quadrature points that can be indexed by this rule.
        Returns a number such that `point_index()` is always smaller than this number.
        """
        raise NotImplementedError()

    def evaluation_point_count(self):
        """Number of quadrature points that needs to be evaluated, mostly for internal purposes.
        If the indexing scheme is sparse, or if a quadrature point is shared among multiple elements
        (e.g, nodal quadrature), `evaluation_point_count` may be different than `total_point_count()`.
        Returns a number such that `evaluation_point_index()` is always smaller than this number.
        """
        return self.total_point_count()

    def max_points_per_element(self):
        """Maximum number of points per element if known, or ``None`` otherwise"""
        return None

    @staticmethod
    def point_count(
        elt_arg: "GeometryDomain.ElementArg",
        qp_arg: Arg,
        domain_element_index: ElementIndex,
        geo_element_index: ElementIndex,
    ):
        """Number of quadrature points for a given element"""
        raise NotImplementedError()

    @staticmethod
    def point_coords(
        elt_arg: "GeometryDomain.ElementArg",
        qp_arg: Arg,
        domain_element_index: ElementIndex,
        geo_element_index: ElementIndex,
        element_qp_index: int,
    ):
        """Coordinates in element of the element's qp_index'th quadrature point"""
        raise NotImplementedError()

    @staticmethod
    def point_weight(
        elt_arg: "GeometryDomain.ElementArg",
        qp_arg: Arg,
        domain_element_index: ElementIndex,
        geo_element_index: ElementIndex,
        element_qp_index: int,
    ):
        """Weight of the element's qp_index'th quadrature point"""
        raise NotImplementedError()

    @staticmethod
    def point_index(
        elt_arg: "GeometryDomain.ElementArg",
        qp_arg: Arg,
        domain_element_index: ElementIndex,
        geo_element_index: ElementIndex,
        element_qp_index: int,
    ):
        """
        Global index of the element's qp_index'th  quadrature point.
        May be shared among elements.
        This is what determines `qp_index` in integrands' `Sample` arguments.
        """
        raise NotImplementedError()

    @staticmethod
    def point_evaluation_index(
        elt_arg: "GeometryDomain.ElementArg",
        qp_arg: Arg,
        domain_element_index: ElementIndex,
        geo_element_index: ElementIndex,
        element_qp_index: int,
    ):
        """Quadrature point index according to evaluation order.
        Quadrature points for distinct elements must have different evaluation indices.
        Mostly for internal/parallelization purposes.
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.name

    # By default cache the mapping from evaluation point indices to domain elements

    ElementIndexArg = wp.array(dtype=QuadraturePointElementIndex)

    @cache.cached_arg_value
    def element_index_arg_value(self, device):
        """Builds a map from quadrature point evaluation indices to their index in the element to which they belong"""

        @cache.dynamic_kernel(f"{self.name}{self.domain.name}")
        def quadrature_point_element_indices(
            qp_arg: self.Arg,
            domain_arg: self.domain.ElementArg,
            domain_index_arg: self.domain.ElementIndexArg,
            result: wp.array(dtype=QuadraturePointElementIndex),
        ):
            domain_element_index = wp.tid()
            element_index = self.domain.element_index(domain_index_arg, domain_element_index)

            qp_point_count = self.point_count(domain_arg, qp_arg, domain_element_index, element_index)
            for k in range(qp_point_count):
                qp_eval_index = self.point_evaluation_index(domain_arg, qp_arg, domain_element_index, element_index, k)
                result[qp_eval_index] = QuadraturePointElementIndex(domain_element_index, k)

        null_qp_index = QuadraturePointElementIndex()
        null_qp_index.domain_element_index = NULL_ELEMENT_INDEX
        result = wp.full(
            value=null_qp_index,
            shape=(self.evaluation_point_count()),
            dtype=QuadraturePointElementIndex,
            device=device,
        )
        wp.launch(
            quadrature_point_element_indices,
            device=result.device,
            dim=self.domain.element_count(),
            inputs=[
                self.arg_value(result.device),
                self.domain.element_arg_value(result.device),
                self.domain.element_index_arg_value(result.device),
                result,
            ],
        )

        return result

    @wp.func
    def evaluation_point_element_index(
        element_index_arg: wp.array(dtype=QuadraturePointElementIndex),
        qp_eval_index: QuadraturePointIndex,
    ):
        """Maps from quadrature point evaluation indices to their index in the element to which they belong
        If the quadrature point does not exist, should return NULL_ELEMENT_INDEX as the domain element index
        """

        element_index = element_index_arg[qp_eval_index]
        return element_index.domain_element_index, element_index.qp_index_in_element


class _QuadratureWithRegularEvaluationPoints(Quadrature):
    """Helper subclass for quadrature formulas which use a uniform number of
    evaluations points per element. Avoids building explicit mapping"""

    def __init__(self, domain: GeometryDomain, N: int):
        super().__init__(domain)
        self._EVALUATION_POINTS_PER_ELEMENT = N

        self.point_evaluation_index = self._make_regular_point_evaluation_index()
        self.evaluation_point_element_index = self._make_regular_evaluation_point_element_index()

    ElementIndexArg = Quadrature.Arg
    element_index_arg_value = Quadrature.arg_value

    def evaluation_point_count(self):
        return self.domain.element_count() * self._EVALUATION_POINTS_PER_ELEMENT

    def _make_regular_point_evaluation_index(self):
        N = self._EVALUATION_POINTS_PER_ELEMENT

        @cache.dynamic_func(suffix=f"{self.name}")
        def evaluation_point_index(
            elt_arg: self.domain.ElementArg,
            qp_arg: self.Arg,
            domain_element_index: ElementIndex,
            element_index: ElementIndex,
            qp_index: int,
        ):
            return N * domain_element_index + qp_index

        return evaluation_point_index

    def _make_regular_evaluation_point_element_index(self):
        N = self._EVALUATION_POINTS_PER_ELEMENT

        @cache.dynamic_func(suffix=f"{N}")
        def quadrature_evaluation_point_element_index(
            qp_arg: Quadrature.Arg,
            qp_index: QuadraturePointIndex,
        ):
            domain_element_index = qp_index // N
            index_in_element = qp_index - domain_element_index * N
            return domain_element_index, index_in_element

        return quadrature_evaluation_point_element_index


class RegularQuadrature(_QuadratureWithRegularEvaluationPoints):
    """Regular quadrature formula, using a constant set of quadrature points per element"""

    @wp.struct
    class Arg:
        # Quadrature points and weights used to be passed as Warp constants,
        # but this tended to incur register spilling for high point counts
        points: wp.array(dtype=Coords)
        weights: wp.array(dtype=float)

    # Cache common formulas so we do dot have to do h2d transfer for each call
    class CachedFormula:
        _cache = {}

        def __init__(self, element: Element, order: int, family: Polynomial):
            self.points, self.weights = element.instantiate_quadrature(order, family)
            self.count = wp.constant(len(self.points))

        @cache.cached_arg_value
        def arg_value(self, device):
            arg = RegularQuadrature.Arg()
            arg.points = wp.array(self.points, device=device, dtype=Coords)
            arg.weights = wp.array(self.weights, device=device, dtype=float)
            return arg

        @staticmethod
        def get(element: Element, order: int, family: Polynomial):
            key = (element.__class__.__name__, order, family)
            try:
                return RegularQuadrature.CachedFormula._cache[key]
            except KeyError:
                quadrature = RegularQuadrature.CachedFormula(element, order, family)
                RegularQuadrature.CachedFormula._cache[key] = quadrature
                return quadrature

    def __init__(
        self,
        domain: GeometryDomain,
        order: int,
        family: Polynomial = None,
    ):
        self._formula = RegularQuadrature.CachedFormula.get(domain.reference_element(), order, family)
        self.family = family
        self.order = order

        super().__init__(domain, self._formula.count)

        self.point_count = self._make_point_count()
        self.point_index = self._make_point_index()
        self.point_coords = self._make_point_coords()
        self.point_weight = self._make_point_weight()

    @property
    def name(self):
        return f"{self.__class__.__name__}_{self.domain.name}_{self.family}_{self.order}"

    def total_point_count(self):
        return self._formula.count * self.domain.element_count()

    def max_points_per_element(self):
        return self._formula.count

    @property
    def points(self):
        return self._formula.points

    @property
    def weights(self):
        return self._formula.weights

    def arg_value(self, device):
        return self._formula.arg_value(device)

    def _make_point_count(self):
        N = self._formula.count

        @cache.dynamic_func(suffix=self.name)
        def point_count(
            elt_arg: self.domain.ElementArg,
            qp_arg: self.Arg,
            domain_element_index: ElementIndex,
            element_index: ElementIndex,
        ):
            return N

        return point_count

    def _make_point_coords(self):
        @cache.dynamic_func(suffix=self.name)
        def point_coords(
            elt_arg: self.domain.ElementArg,
            qp_arg: self.Arg,
            domain_element_index: ElementIndex,
            element_index: ElementIndex,
            qp_index: int,
        ):
            return qp_arg.points[qp_index]

        return point_coords

    def _make_point_weight(self):
        @cache.dynamic_func(suffix=self.name)
        def point_weight(
            elt_arg: self.domain.ElementArg,
            qp_arg: self.Arg,
            domain_element_index: ElementIndex,
            element_index: ElementIndex,
            qp_index: int,
        ):
            return qp_arg.weights[qp_index]

        return point_weight

    def _make_point_index(self):
        N = self._formula.count

        @cache.dynamic_func(suffix=self.name)
        def point_index(
            elt_arg: self.domain.ElementArg,
            qp_arg: self.Arg,
            domain_element_index: ElementIndex,
            element_index: ElementIndex,
            qp_index: int,
        ):
            return N * domain_element_index + qp_index

        return point_index


class NodalQuadrature(Quadrature):
    """Quadrature using space node points as quadrature points

    Note that in contrast to the `nodal=True` flag for :func:`integrate`, using this quadrature does not imply
    any assumption about orthogonality of shape functions, and is thus safe to use for arbitrary integrands.
    """

    def __init__(self, domain: Optional[GeometryDomain], space: FunctionSpace):
        self._space = space

        super().__init__(domain)

        self.Arg = self._make_arg()

        self.point_count = self._make_point_count()
        self.point_index = self._make_point_index()
        self.point_coords = self._make_point_coords()
        self.point_weight = self._make_point_weight()
        self.point_evaluation_index = self._make_point_evaluation_index()

    @property
    def name(self):
        return f"{self.__class__.__name__}_{self._space.name}"

    def total_point_count(self):
        return self._space.node_count()

    def max_points_per_element(self):
        return self._space.topology.MAX_NODES_PER_ELEMENT

    def _make_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class Arg:
            space_arg: self._space.SpaceArg
            topo_arg: self._space.topology.TopologyArg

        return Arg

    @cache.cached_arg_value
    def arg_value(self, device):
        arg = self.Arg()
        arg.space_arg = self._space.space_arg_value(device)
        arg.topo_arg = self._space.topology.topo_arg_value(device)
        return arg

    def _make_point_count(self):
        @cache.dynamic_func(suffix=self.name)
        def point_count(
            elt_arg: self.domain.ElementArg,
            qp_arg: self.Arg,
            domain_element_index: ElementIndex,
            element_index: ElementIndex,
        ):
            return self._space.topology.element_node_count(elt_arg, qp_arg.topo_arg, element_index)

        return point_count

    def _make_point_coords(self):
        @cache.dynamic_func(suffix=self.name)
        def point_coords(
            elt_arg: self.domain.ElementArg,
            qp_arg: self.Arg,
            domain_element_index: ElementIndex,
            element_index: ElementIndex,
            qp_index: int,
        ):
            return self._space.node_coords_in_element(elt_arg, qp_arg.space_arg, element_index, qp_index)

        return point_coords

    def _make_point_weight(self):
        @cache.dynamic_func(suffix=self.name)
        def point_weight(
            elt_arg: self.domain.ElementArg,
            qp_arg: self.Arg,
            domain_element_index: ElementIndex,
            element_index: ElementIndex,
            qp_index: int,
        ):
            return self._space.node_quadrature_weight(elt_arg, qp_arg.space_arg, element_index, qp_index)

        return point_weight

    def _make_point_index(self):
        @cache.dynamic_func(suffix=self.name)
        def point_index(
            elt_arg: self.domain.ElementArg,
            qp_arg: self.Arg,
            domain_element_index: ElementIndex,
            element_index: ElementIndex,
            qp_index: int,
        ):
            return self._space.topology.element_node_index(elt_arg, qp_arg.topo_arg, element_index, qp_index)

        return point_index

    def evaluation_point_count(self):
        return self.domain.element_count() * self._space.topology.MAX_NODES_PER_ELEMENT

    def _make_point_evaluation_index(self):
        N = self._space.topology.MAX_NODES_PER_ELEMENT

        @cache.dynamic_func(suffix=self.name)
        def evaluation_point_index(
            elt_arg: self.domain.ElementArg,
            qp_arg: self.Arg,
            domain_element_index: ElementIndex,
            element_index: ElementIndex,
            qp_index: int,
        ):
            return N * domain_element_index + qp_index

        return evaluation_point_index


class ExplicitQuadrature(_QuadratureWithRegularEvaluationPoints):
    """Quadrature using explicit per-cell points and weights.

    The number of quadrature points per cell is assumed to be constant and deduced from the shape of the points and weights arrays.
    Quadrature points may be provided for either the whole geometry or just the domain's elements.

    Args:
        domain: Domain of definition of the quadrature formula
        points: 2d array of shape  ``(domain.element_count(), points_per_cell)`` or ``(domain.geometry_element_count(), points_per_cell)`` containing the coordinates of each quadrature point.
        weights: 2d array of shape ``(domain.element_count(), points_per_cell)`` or ``(domain.geometry_element_count(), points_per_cell)`` containing the weight for each quadrature point.

    See also: :class:`PicQuadrature`
    """

    @wp.struct
    class Arg:
        points_per_cell: int
        points: wp.array2d(dtype=Coords)
        weights: wp.array2d(dtype=float)

    def __init__(self, domain: GeometryDomain, points: "wp.array2d(dtype=Coords)", weights: "wp.array2d(dtype=float)"):
        if points.shape != weights.shape:
            raise ValueError("Points and weights arrays must have the same shape")

        if points.shape[0] == domain.geometry_element_count():
            self.point_index = ExplicitQuadrature._point_index_geo
            self.point_coords = ExplicitQuadrature._point_coords_geo
            self.point_weight = ExplicitQuadrature._point_weight_geo
        elif points.shape[0] == domain.element_count():
            self.point_index = ExplicitQuadrature._point_index_domain
            self.point_coords = ExplicitQuadrature._point_coords_domain
            self.point_weight = ExplicitQuadrature._point_weight_domain
        else:
            raise NotImplementedError(
                "The number of rows of points and weights must match the element count of either the domain or the geometry"
            )

        self._points_per_cell = points.shape[1]

        self._whole_geo = points.shape[0] == domain.geometry_element_count()

        super().__init__(domain, self._points_per_cell)
        self._points = points
        self._weights = weights

    @property
    def name(self):
        return f"{self.__class__.__name__}_{self._whole_geo}"

    def total_point_count(self):
        return self._weights.size

    def max_points_per_element(self):
        return self._points_per_cell

    @cache.cached_arg_value
    def arg_value(self, device):
        arg = self.Arg()
        arg.points_per_cell = self._points_per_cell
        arg.points = self._points.to(device)
        arg.weights = self._weights.to(device)

        return arg

    @wp.func
    def point_count(elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex):
        return qp_arg.points.shape[1]

    @wp.func
    def _point_coords_domain(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, qp_index: int
    ):
        return qp_arg.points[domain_element_index, qp_index]

    @wp.func
    def _point_weight_domain(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, qp_index: int
    ):
        return qp_arg.weights[domain_element_index, qp_index]

    @wp.func
    def _point_index_domain(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, qp_index: int
    ):
        return qp_arg.points_per_cell * domain_element_index + qp_index

    @wp.func
    def _point_coords_geo(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, qp_index: int
    ):
        return qp_arg.points[element_index, qp_index]

    @wp.func
    def _point_weight_geo(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, qp_index: int
    ):
        return qp_arg.weights[element_index, qp_index]

    @wp.func
    def _point_index_geo(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, qp_index: int
    ):
        return qp_arg.points_per_cell * element_index + qp_index
