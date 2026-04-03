# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from typing import Any, ClassVar, Optional

import numpy as np

import warp as wp
from warp._src.context import capture_pause, capture_resume
from warp._src.fem import cache
from warp._src.fem.domain import GeometryDomain
from warp._src.fem.geometry import Element
from warp._src.fem.space.function_space import BasisSpace, FunctionSpace
from warp._src.fem.space.partition import SpacePartition, WholeSpacePartition
from warp._src.fem.types import NULL_ELEMENT_INDEX, ElementIndex, QuadraturePointIndex, cached_coords_type

from ..polynomial import Polynomial

_wp_module_name_ = "warp.fem.quadrature.quadrature"


@wp.struct
class QuadraturePointElementIndex:
    domain_element_index: ElementIndex
    qp_index_in_element: int


class Quadrature:
    """Interface class for quadrature rules."""

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

    @cache.cached_arg_value
    def arg_value(self, device) -> "Arg":
        """
        Value of the argument to be passed to device
        """
        arg = self.Arg()
        self.fill_arg(arg, device)
        return arg

    def fill_arg(self, arg: Arg, device):
        """
        Fill the argument with the value of the argument to be passed to device
        """
        if self.arg_value is __class__.arg_value:
            raise NotImplementedError()
        arg.assign(self.arg_value(device))

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
        """Number of quadrature points for a given element."""
        raise NotImplementedError()

    @staticmethod
    def point_coords(
        elt_arg: "GeometryDomain.ElementArg",
        qp_arg: Arg,
        domain_element_index: ElementIndex,
        geo_element_index: ElementIndex,
        element_qp_index: int,
    ):
        """Coordinate values in element of the element's qp_index'th quadrature point."""
        raise NotImplementedError()

    @staticmethod
    def point_weight(
        elt_arg: "GeometryDomain.ElementArg",
        qp_arg: Arg,
        domain_element_index: ElementIndex,
        geo_element_index: ElementIndex,
        element_qp_index: int,
    ):
        """Weight of the element's qp_index'th quadrature point."""
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
        Only required if ``evaluation_point_element_index`` is not overloaded.
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.name

    # By default cache the mapping from evaluation point indices to domain elements

    ElementIndexArg = wp.array(dtype=QuadraturePointElementIndex)
    """Mapping from evaluation point indices to element indices."""

    @cache.cached_arg_value
    def element_index_arg_value(self, device):
        """Build a map from quadrature point evaluation indices to their index in the element to which they belong."""

        @cache.dynamic_kernel(f"{self.name}{self.domain.name}")
        def quadrature_point_element_indices(
            qp_arg: self.Arg,
            domain_arg: self.domain.ElementArg,
            domain_index_arg: self.domain.ElementIndexArg,
            result: wp.array(dtype=QuadraturePointElementIndex),
        ):
            domain_element_index = wp.tid()
            element_index = self.domain.element_index(domain_index_arg, domain_element_index)
            if element_index == NULL_ELEMENT_INDEX:
                return

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
        element_index_arg: ElementIndexArg,
        qp_eval_index: QuadraturePointIndex,
    ):
        """Map from quadrature point evaluation indices to their index in the element to which they belong.

        If the quadrature point does not exist, should return :data:`NULL_ELEMENT_INDEX` as the domain element index.
        """

        element_index = element_index_arg[qp_eval_index]
        return element_index.domain_element_index, element_index.qp_index_in_element


class _QuadratureWithRegularEvaluationPoints(Quadrature):
    """Helper subclass for quadrature formulas which use a uniform number of
    evaluations points per element.

    Avoids building explicit mapping.
    """

    _dynamic_attribute_constructors: ClassVar = {
        "point_evaluation_index": lambda obj: obj._make_regular_point_evaluation_index(),
        "evaluation_point_element_index": lambda obj: obj._make_regular_evaluation_point_element_index(),
    }

    def __init__(self, domain: GeometryDomain, N: int):
        super().__init__(domain)
        self._EVALUATION_POINTS_PER_ELEMENT = N

        cache.setup_dynamic_attributes(self, cls=__class__)

    ElementIndexArg = Quadrature.Arg

    def element_index_arg_value(self, device):
        return Quadrature.Arg()

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
    """Regular quadrature formula, using a constant set of quadrature points per element."""

    # Cache common formulas so we do dot have to do h2d transfer for each call
    class CachedFormula:
        _cache: ClassVar = {}

        def __init__(self, element: Element, order: int, family: Polynomial, scalar_type: type):
            raw_points, raw_weights = element.prototype.instantiate_quadrature(order, family)

            self._scalar_type = scalar_type
            self._coords_type = cached_coords_type(scalar_type)

            self.ArgType = self._make_arg()

            self.points = np.array([self._coords_type(*p) for p in raw_points])
            self.weights = np.array([self._scalar_type(w) for w in raw_weights])
            self.count = wp.constant(len(raw_points))

        @cache.cached_arg_value
        def arg_value(self, device):
            arg = self.ArgType()

            # pause graph capture while we copy from host
            # we want the cached result to be available outside of the graph
            if device.is_capturing:
                graph = capture_pause()
            else:
                graph = None

            arg.points = wp.array(self.points, device=device, dtype=self._coords_type)
            arg.weights = wp.array(self.weights, device=device, dtype=self._scalar_type)

            if graph is not None:
                capture_resume(graph)
            return arg

        def fill_arg(self, arg, device):
            arg.assign(self.arg_value(device))

        @staticmethod
        def get(element: Element, order: int, family: Polynomial, scalar_type: type):
            key = (element.value, order, family, scalar_type)
            try:
                return RegularQuadrature.CachedFormula._cache[key]
            except KeyError:
                quadrature = RegularQuadrature.CachedFormula(element, order, family, scalar_type)
                RegularQuadrature.CachedFormula._cache[key] = quadrature
                return quadrature

        def _make_arg(self):
            @cache.dynamic_struct(suffix=self._scalar_type)
            class Arg:
                """Structure containing arguments to be passed to device functions."""

                # Quadrature points and weights used to be passed as Warp constants,
                # but this tended to incur register spilling for high point counts
                points: wp.array(dtype=self._coords_type)
                weights: wp.array(dtype=self._scalar_type)

            return Arg

    _dynamic_attribute_constructors: ClassVar = {
        "point_count": lambda obj: obj._make_point_count(),
        "point_index": lambda obj: obj._make_point_index(),
        "point_coords": lambda obj: obj._make_point_coords(),
        "point_weight": lambda obj: obj._make_point_weight(),
    }

    def __init__(
        self,
        domain: GeometryDomain,
        order: int,
        family: Polynomial = None,
    ):
        scalar_type = domain.geometry.scalar_type
        self._formula = RegularQuadrature.CachedFormula.get(domain.reference_element(), order, family, scalar_type)
        self.family = family
        self.order = order

        self.Arg = self._formula.ArgType
        self._scalar_type = scalar_type

        super().__init__(domain, self._formula.count)

        cache.setup_dynamic_attributes(self)

    @cached_property
    def name(self):
        """Unique name of the quadrature rule."""
        return f"{self.__class__.__name__}_{self.domain.name}_{self.family}_{self.order}"

    def total_point_count(self):
        """Total number of quadrature points."""
        return self._formula.count * self.domain.element_count()

    def max_points_per_element(self):
        """Maximum number of quadrature points per element."""
        return self._formula.count

    @property
    def points(self):
        """Quadrature point coordinates in reference space."""
        return self._formula.points

    @property
    def weights(self):
        """Quadrature weights for the reference element."""
        return self._formula.weights

    def fill_arg(self, arg, device):
        """Fill the quadrature argument structure for device functions."""
        self._formula.fill_arg(arg, device)

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


class NodalQuadrature(_QuadratureWithRegularEvaluationPoints):
    """Quadrature using space node points as quadrature points

    Note that in contrast to the `assembly="nodal"` flag for :func:`integrate`, using this quadrature does not imply
    any assumption about orthogonality of shape functions, and is thus safe to use for arbitrary integrands.
    """

    _dynamic_attribute_constructors: ClassVar = {
        "point_count": lambda obj: obj._make_point_count(),
        "point_index": lambda obj: obj._make_point_index(),
        "point_coords": lambda obj: obj._make_point_coords(),
        "point_weight": lambda obj: obj._make_point_weight(),
    }

    def __init__(
        self,
        domain: Optional[GeometryDomain],
        space: Optional[FunctionSpace] = None,
        basis_space: Optional[BasisSpace] = None,
        space_partition: Optional[SpacePartition] = None,
    ):
        if basis_space is None:
            if space is None:
                raise ValueError("One of space, basis_space or space_partition must be provided")
            basis_space = space.basis

        if space_partition is None:
            space_partition = WholeSpacePartition(basis_space.topology)

        self._basis_space = basis_space
        self._space_partition = space_partition

        self.Arg = self._make_arg()
        super().__init__(domain, self.max_points_per_element())
        cache.setup_dynamic_attributes(self)

    @cached_property
    def name(self):
        """Unique name of the quadrature rule."""
        return f"{self.__class__.__name__}_{self._basis_space.name}_{self._space_partition.name}"

    def total_point_count(self):
        """Total number of quadrature points."""
        return self._space_partition.node_count()

    def max_points_per_element(self):
        """Maximum number of quadrature points per element."""
        return self._basis_space.topology.MAX_NODES_PER_ELEMENT

    def _make_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class Arg:
            basis_arg: self._basis_space.BasisArg
            topo_arg: self._basis_space.topology.TopologyArg
            partition_arg: self._space_partition.PartitionArg

        return Arg

    def fill_arg(self, arg: "NodalQuadrature.Arg", device):
        """Fill the quadrature argument structure for device functions."""
        self._basis_space.fill_basis_arg(arg.basis_arg, device)
        self._basis_space.topology.fill_topo_arg(arg.topo_arg, device)
        self._space_partition.fill_partition_arg(arg.partition_arg, device)

    def _make_point_count(self):
        @cache.dynamic_func(suffix=self.name)
        def point_count(
            elt_arg: self.domain.ElementArg,
            qp_arg: self.Arg,
            domain_element_index: ElementIndex,
            element_index: ElementIndex,
        ):
            topo_arg = qp_arg.topo_arg
            return self._basis_space.topology.element_node_count(elt_arg, topo_arg, element_index)

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
            return self._basis_space.node_coords_in_element(
                elt_arg, qp_arg.topo_arg, qp_arg.basis_arg, element_index, qp_index
            )

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
            return self._basis_space.node_quadrature_weight(
                elt_arg, qp_arg.topo_arg, qp_arg.basis_arg, element_index, qp_index
            )

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
            topo_arg = qp_arg.topo_arg
            node_index = self._basis_space.topology.element_node_index(elt_arg, topo_arg, element_index, qp_index)

            return self._space_partition.partition_node_index(qp_arg.partition_arg, node_index)

        return point_index


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
    class _Arg_f32:
        points: wp.array2d(dtype=wp.vec3)
        weights: wp.array2d(dtype=wp.float32)

    @wp.struct
    class _Arg_f64:
        points: wp.array2d(dtype=wp.vec3d)
        weights: wp.array2d(dtype=wp.float64)

    def __init__(
        self,
        domain: GeometryDomain,
        points: "wp.array2d",
        weights: "wp.array2d",
    ):
        if points.shape != weights.shape:
            raise ValueError("Points and weights arrays must have the same shape")

        # Generate precision-appropriate Arg struct based on the domain's geometry
        scalar_type = domain.geometry.scalar_type
        self._scalar_type = scalar_type
        self.Arg = ExplicitQuadrature._Arg_f32 if scalar_type == wp.float32 else ExplicitQuadrature._Arg_f64

        if points.shape[0] == domain.geometry_element_count():
            self.point_index = ExplicitQuadrature._point_index_geo
            self.point_coords = ExplicitQuadrature._point_coords_geo
            self.point_weight = ExplicitQuadrature._point_weight_geo
            self._whole_geo = True
        elif points.shape[0] == domain.element_count():
            self.point_index = ExplicitQuadrature._point_index_domain
            self.point_coords = ExplicitQuadrature._point_coords_domain
            self.point_weight = ExplicitQuadrature._point_weight_domain
            self._whole_geo = False
        else:
            raise NotImplementedError(
                "The number of rows of points and weights must match the element count of either the domain or the geometry"
            )

        self._points_per_cell = points.shape[1]

        super().__init__(domain, self._points_per_cell)

        self._points = points
        self._weights = weights

    @cached_property
    def name(self):
        """Unique name of the quadrature rule."""
        return f"{self.__class__.__name__}_{self._scalar_type.__name__}_{self._whole_geo}_{self._EVALUATION_POINTS_PER_ELEMENT}"

    def total_point_count(self):
        """Total number of quadrature points."""
        return self._weights.size

    def max_points_per_element(self):
        """Maximum number of quadrature points per element."""
        return self._points_per_cell

    def fill_arg(self, arg: "ExplicitQuadrature.Arg", device):
        """Fill the quadrature argument structure for device functions."""
        arg.points = self._points.to(device)
        arg.weights = self._weights.to(device)

    @wp.func
    def point_count(
        elt_arg: Any,
        qp_arg: Any,
        domain_element_index: ElementIndex,
        element_index: ElementIndex,
    ):
        """Return the number of quadrature points for a given element."""
        return qp_arg.points.shape[1]

    @wp.func
    def _point_coords_domain(
        elt_arg: Any,
        qp_arg: Any,
        domain_element_index: ElementIndex,
        element_index: ElementIndex,
        qp_index: int,
    ):
        """Return quadrature point coordinates for a domain element."""
        return qp_arg.points[domain_element_index, qp_index]

    @wp.func
    def _point_weight_domain(
        elt_arg: Any,
        qp_arg: Any,
        domain_element_index: ElementIndex,
        element_index: ElementIndex,
        qp_index: int,
    ):
        """Return quadrature point weights for a domain element."""
        return qp_arg.weights[domain_element_index, qp_index]

    @wp.func
    def _point_index_domain(
        elt_arg: Any,
        qp_arg: Any,
        domain_element_index: ElementIndex,
        element_index: ElementIndex,
        qp_index: int,
    ):
        """Return quadrature point indices for a domain element."""
        return qp_arg.points.shape[1] * domain_element_index + qp_index

    @wp.func
    def _point_coords_geo(
        elt_arg: Any,
        qp_arg: Any,
        domain_element_index: ElementIndex,
        element_index: ElementIndex,
        qp_index: int,
    ):
        """Return quadrature point coordinates for a geometry element."""
        return qp_arg.points[element_index, qp_index]

    @wp.func
    def _point_weight_geo(
        elt_arg: Any,
        qp_arg: Any,
        domain_element_index: ElementIndex,
        element_index: ElementIndex,
        qp_index: int,
    ):
        """Return quadrature point weights for a geometry element."""
        return qp_arg.weights[element_index, qp_index]

    @wp.func
    def _point_index_geo(
        elt_arg: Any,
        qp_arg: Any,
        domain_element_index: ElementIndex,
        element_index: ElementIndex,
        qp_index: int,
    ):
        """Return quadrature point indices for a geometry element."""
        return qp_arg.points.shape[1] * element_index + qp_index
