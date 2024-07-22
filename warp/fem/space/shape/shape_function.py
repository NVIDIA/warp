import numpy as np

import warp as wp
from warp.fem import cache
from warp.fem.geometry import Element
from warp.fem.types import Coords


class ShapeFunction:
    """Interface class for defining scalar-valued shape functions over a single element"""

    ORDER: int
    """Maximum degree of the polynomials used to define the shape function"""

    NODES_PER_ELEMENT: int
    """Number of shape function nodes"""

    @property
    def name(self) -> str:
        """Unique name encoding all parameters defining the shape function"""
        raise NotImplementedError()

    def make_node_coords_in_element(self):
        """Creates a device function returning the coordinates of each node"""
        raise NotImplementedError()

    def make_node_quadrature_weight(self):
        """Creates a device function returning the weight of each node when use as a quadrature point over the element"""
        raise NotImplementedError()

    def make_trace_node_quadrature_weight(self):
        """Creates a device function returning the weight of each node when use as a quadrature point over the element boundary"""
        raise NotImplementedError()

    def make_element_inner_weight(self):
        """Creates a device function returning the value of the shape function associated to a given node at given coordinates"""
        raise NotImplementedError()

    def make_element_inner_weight_gradient(self):
        """Creates a device function returning the gradient of the shape function associated to a given node at given coordinates"""
        raise NotImplementedError()


class ConstantShapeFunction:
    """Shape function that is constant over the element"""

    def __init__(self, element: Element, space_dimension: int):
        self._element = element
        self._dimension = space_dimension

        self.ORDER = wp.constant(0)
        self.NODES_PER_ELEMENT = wp.constant(1)

        coords, _ = element.instantiate_quadrature(order=0, family=None)
        self.COORDS = wp.constant(coords[0])

    @property
    def name(self) -> str:
        return f"{self._element.__class__.__name__}{self._dimension}"

    def make_node_coords_in_element(self):
        COORDS = self.COORDS

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            return COORDS

        return node_coords_in_element

    @wp.func
    def _node_quadrature_weight(
        node_index_in_elt: int,
    ):
        return 1.0

    def make_node_quadrature_weight(self):
        return ConstantShapeFunction._node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        return ConstantShapeFunction._node_quadrature_weight

    @wp.func
    def _element_inner_weight(
        coords: Coords,
        node_index_in_elt: int,
    ):
        return 1.0

    def make_element_inner_weight(self):
        return ConstantShapeFunction._element_inner_weight

    def make_element_inner_weight_gradient(self):
        grad_type = wp.vec(length=self._dimension, dtype=float)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Coords,
            node_index_in_elt: int,
        ):
            return grad_type(0.0)

        return element_inner_weight_gradient

    def element_vtk_cells(self):
        cell_type = 1  # VTK_VERTEX

        return np.zeros((1, 1), dtype=int), np.full(1, cell_type, dtype=np.int8)
