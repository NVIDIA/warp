from typing import Any

import warp as wp

from warp.fem import domain
from warp.fem.types import ElementIndex, Coords

from ..polynomial import Polynomial


class Quadrature:
    """Interface class for quadrature rules"""

    Arg: wp.codegen.Struct
    """Structure containing arguments to be passed to device functions"""

    def __init__(self, domain: domain.GeometryDomain):
        self._domain = domain

    @property
    def domain(self):
        """Domain over which this quadrature is defined"""
        return self._domain

    def eval_arg_value(self, device) -> wp.codegen.StructInstance:
        """
        Value of the argument to be passed to device
        """
        pass

    def total_point_count(self):
        """Total number of quadrature points over the domain"""
        pass

    def point_count(arg: Any, element_index: ElementIndex):
        """Number of quadrature points for a given element"""
        pass

    def point_coords(arg: Any, element_index: ElementIndex, qp_index: int):
        """Coordinates in element of the qp_index'th quadrature point"""
        pass

    def point_weight(arg: Any, element_index: ElementIndex, qp_index: int):
        """Weight of the qp_index'th quadrature point"""
        pass

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

        N = wp.constant(len(self.points))

        WeightVec = wp.vec(length=N, dtype=wp.float32)
        CoordMat = wp.mat(shape=(N, 3), dtype=wp.float32)

        POINTS = wp.constant(CoordMat(self.points))
        WEIGHTS = wp.constant(WeightVec(self.weights))

        self.point_count = self._make_point_count(N)
        self.point_index = self._make_point_index(N)
        self.point_coords = self._make_point_coords(POINTS, self.name)
        self.point_weight = self._make_point_weight(WEIGHTS, self.name)

    @property
    def name(self):
        return (
            f"{self.__class__.__name__}_{self.domain.reference_element().__class__.__name__}_{self.family}_{self.order}"
        )

    def __str__(self) -> str:
        return self.name

    def total_point_count(self):
        return len(self.points) * self.domain.geometry_element_count()

    @property
    def points(self):
        return self._element_quadrature[0]

    @property
    def weights(self):
        return self._element_quadrature[1]

    @wp.struct
    class Arg:
        pass

    def arg_value(self, device) -> Arg:
        arg = RegularQuadrature.Arg()
        return arg

    @staticmethod
    def _make_point_count(N):
        def point_count(arg: RegularQuadrature.Arg, element_index: ElementIndex):
            return N

        from warp.fem.cache import get_func

        return get_func(point_count, str(N))

    @staticmethod
    def _make_point_coords(POINTS, name):
        def point_coords(arg: RegularQuadrature.Arg, element_index: ElementIndex, index: int):
            return Coords(POINTS[index, 0], POINTS[index, 1], POINTS[index, 2])

        from warp.fem.cache import get_func

        return get_func(point_coords, name)

    @staticmethod
    def _make_point_weight(WEIGHTS, name):
        def point_weight(arg: RegularQuadrature.Arg, element_index: ElementIndex, index: int):
            return WEIGHTS[index]

        from warp.fem.cache import get_func

        return get_func(point_weight, name)

    @staticmethod
    def _make_point_index(N):
        def point_index(arg: RegularQuadrature.Arg, element_index: ElementIndex, index: int):
            return N * element_index + index

        from warp.fem.cache import get_func

        return get_func(point_index, str(N))
