from typing import Any

import warp as wp

from warp.fem import domain, cache
from warp.fem.types import ElementIndex, Coords
from warp.fem.space import FunctionSpace

from ..polynomial import Polynomial


class Quadrature:
    """Interface class for quadrature rules"""

    @wp.struct
    class Arg:
        """Structure containing arguments to be passed to device functions"""

        pass

    def __init__(self, domain: domain.GeometryDomain):
        self._domain = domain

    @property
    def domain(self):
        """Domain over which this quadrature is defined"""
        return self._domain

    def arg_value(self, device) -> "Arg":
        """
        Value of the argument to be passed to device
        """
        arg = RegularQuadrature.Arg()
        return arg

    def total_point_count(self):
        """Total number of quadrature points over the domain"""
        raise NotImplementedError()

    def points_per_element(self):
        """Number of points per element if constant, or ``None`` if varying"""
        return None

    @staticmethod
    def point_count(elt_arg: "domain.GeometryDomain.ElementArg", qp_arg: Arg, element_index: ElementIndex):
        """Number of quadrature points for a given element"""
        raise NotImplementedError()

    @staticmethod
    def point_coords(
        elt_arg: "domain.GeometryDomain.ElementArg", qp_arg: Arg, element_index: ElementIndex, qp_index: int
    ):
        """Coordinates in element of the element's qp_index'th quadrature point"""
        raise NotImplementedError()

    @staticmethod
    def point_weight(
        elt_arg: "domain.GeometryDomain.ElementArg", qp_arg: Arg, element_index: ElementIndex, qp_index: int
    ):
        """Weight of the element's qp_index'th quadrature point"""
        raise NotImplementedError()

    @staticmethod
    def point_index(
        elt_arg: "domain.GeometryDomain.ElementArg", qp_arg: Arg, element_index: ElementIndex, qp_index: int
    ):
        """Global index of the element's qp_index'th  quadrature point"""
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.name


class RegularQuadrature(Quadrature):
    """Regular quadrature formula, using a constant set of quadrature points per element"""

    def __init__(
        self,
        domain: domain.GeometryDomain,
        order: int,
        family: Polynomial = None,
    ):
        super().__init__(domain)

        self.family = family
        self.order = order

        self._element_quadrature = domain.reference_element().instantiate_quadrature(order, family)

        self._N = wp.constant(len(self.points))

        WeightVec = wp.vec(length=self._N, dtype=wp.float32)
        CoordMat = wp.mat(shape=(self._N, 3), dtype=wp.float32)

        self._POINTS = wp.constant(CoordMat(self.points))
        self._WEIGHTS = wp.constant(WeightVec(self.weights))

        self.point_count = self._make_point_count()
        self.point_index = self._make_point_index()
        self.point_coords = self._make_point_coords()
        self.point_weight = self._make_point_weight()

    @property
    def name(self):
        return f"{self.__class__.__name__}_{self.domain.name}_{self.family}_{self.order}"

    def total_point_count(self):
        return len(self.points) * self.domain.geometry_element_count()

    def points_per_element(self):
        return self._N

    @property
    def points(self):
        return self._element_quadrature[0]

    @property
    def weights(self):
        return self._element_quadrature[1]

    def _make_point_count(self):
        N = self._N

        @cache.dynamic_func(suffix=self.name)
        def point_count(elt_arg: self.domain.ElementArg, qp_arg: self.Arg, element_index: ElementIndex):
            return N

        return point_count

    def _make_point_coords(self):
        POINTS = self._POINTS

        @cache.dynamic_func(suffix=self.name)
        def point_coords(elt_arg: self.domain.ElementArg, qp_arg: self.Arg, element_index: ElementIndex, qp_index: int):
            return Coords(POINTS[qp_index, 0], POINTS[qp_index, 1], POINTS[qp_index, 2])

        return point_coords

    def _make_point_weight(self):
        WEIGHTS = self._WEIGHTS

        @cache.dynamic_func(suffix=self.name)
        def point_weight(elt_arg: self.domain.ElementArg, qp_arg: self.Arg, element_index: ElementIndex, qp_index: int):
            return WEIGHTS[qp_index]

        return point_weight

    def _make_point_index(self):
        N = self._N

        @cache.dynamic_func(suffix=self.name)
        def point_index(elt_arg: self.domain.ElementArg, qp_arg: self.Arg, element_index: ElementIndex, qp_index: int):
            return N * element_index + qp_index

        return point_index


class NodalQuadrature(Quadrature):
    """Quadrature using space node points as quadrature points

    Note that in contrast to the `nodal=True` flag for :func:`integrate`, this quadrature odes not make any assumption
    about orthogonality of shape functions, and is thus safe to use for arbitrary integrands.
    """

    def __init__(self, domain: domain.GeometryDomain, space: FunctionSpace):
        super().__init__(domain)

        self._space = space

        self.Arg = self._make_arg()

        self.point_count = self._make_point_count()
        self.point_index = self._make_point_index()
        self.point_coords = self._make_point_coords()
        self.point_weight = self._make_point_weight()

    @property
    def name(self):
        return f"{self.__class__.__name__}_{self._space.name}"

    def total_point_count(self):
        return self._space.node_count()

    def points_per_element(self):
        return self._space.topology.NODES_PER_ELEMENT

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
        N = self._space.topology.NODES_PER_ELEMENT

        @cache.dynamic_func(suffix=self.name)
        def point_count(elt_arg: self.domain.ElementArg, qp_arg: self.Arg, element_index: ElementIndex):
            return N

        return point_count

    def _make_point_coords(self):
        @cache.dynamic_func(suffix=self.name)
        def point_coords(elt_arg: self.domain.ElementArg, qp_arg: self.Arg, element_index: ElementIndex, qp_index: int):
            return self._space.node_coords_in_element(elt_arg, qp_arg.space_arg, element_index, qp_index)

        return point_coords

    def _make_point_weight(self):
        @cache.dynamic_func(suffix=self.name)
        def point_weight(elt_arg: self.domain.ElementArg, qp_arg: self.Arg, element_index: ElementIndex, qp_index: int):
            return self._space.node_quadrature_weight(elt_arg, qp_arg.space_arg, element_index, qp_index)

        return point_weight

    def _make_point_index(self):
        @cache.dynamic_func(suffix=self.name)
        def point_index(elt_arg: self.domain.ElementArg, qp_arg: self.Arg, element_index: ElementIndex, qp_index: int):
            return self._space.topology.element_node_index(elt_arg, qp_arg.topo_arg, element_index, qp_index)

        return point_index


class ExplicitQuadrature(Quadrature):
    """Quadrature using explicit per-cell points and weights. The number of quadrature points per cell is assumed
    to be constant and deduced from the shape of the points and weights arrays.

    Args:
        domain: Domain of definition of the quadrature formula
        points: 2d array of shape ``(domain.geometry_element-count(), points_per_cell)`` containing the coordinates of each quadrature point.
        weights: 2d array of shape ``(domain.geometry_element-count(), points_per_cell)`` containing the weight for each quadrature point.

    See also: :class:`PicQuadrature`
    """

    @wp.struct
    class Arg:
        points_per_cell: int
        points: wp.array2d(dtype=Coords)
        weights: wp.array2d(dtype=float)

    def __init__(self, domain: domain.GeometryDomain, points: "wp.array2d(dtype=Coords)", weights: "wp.array2d(dtype=float)"):
        super().__init__(domain)

        if points.shape != weights.shape:
            raise ValueError("Points and weights arrays must have the same shape")

        self._points_per_cell = points.shape[1]
        self._points = points
        self._weights = weights

    @property
    def name(self):
        return f"{self.__class__.__name__}"

    def total_point_count(self):
        return self._weights.size

    def points_per_element(self):
        return self._points_per_cell

    @cache.cached_arg_value
    def arg_value(self, device):
        arg = self.Arg()
        arg.points_per_cell = self._points_per_cell
        arg.points = self._points.to(device)
        arg.weights = self._weights.to(device)

        return arg

    @wp.func
    def point_count(elt_arg: Any, qp_arg: Arg, element_index: ElementIndex):
        return qp_arg.points_per_cell

    @wp.func
    def point_coords(elt_arg: Any, qp_arg: Arg, element_index: ElementIndex, qp_index: int):
        return qp_arg.points[element_index, qp_index]

    @wp.func
    def point_weight(elt_arg: Any, qp_arg: Arg, element_index: ElementIndex, qp_index: int):
        return qp_arg.weights[element_index, qp_index]

    @wp.func
    def point_index(elt_arg: Any, qp_arg: Arg, element_index: ElementIndex, qp_index: int):
        return qp_arg.points_per_cell * element_index + qp_index
