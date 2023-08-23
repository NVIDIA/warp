from typing import Any

import warp as wp

from warp.fem.types import DofIndex, ElementIndex, Coords
from warp.fem import geometry


class FunctionSpace:
    """
    Interface class for function spaces, i.e. geometry + interpolation basis
    """

    DIMENSION: int
    """Input dimension of the function space"""

    NODES_PER_ELEMENT: int
    """Number of interpolation nodes per element of the geometry"""

    ORDER: int
    """Order of the interpolation basis"""

    VALUE_DOF_COUNT: int
    """Number of degrees of freedom per node"""

    dtype: type
    """Value type of the interpolation functions"""

    SpaceArg: wp.codegen.Struct
    """Structure containing arguments to be passed to device function"""

    def node_count(self) -> int:
        """Number of nodes in the interpolation basis"""
        raise NotImplementedError

    def node_positions(self) -> Any:
        """Node positions, for visualization purposes only"""
        raise NotImplementedError

    def geometry(self) -> geometry.Geometry:
        """Underlying geometry"""
        raise NotImplementedError

    def space_arg_value(self, device) -> wp.codegen.StructInstance:
        """Value of the arguments to be passed to device functions"""
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def degree(self):
        return self.ORDER

    def __str__(self):
        return self.name

    def trace(self):
        """Trace of the function space over lower-dimensional elements of the geometry"""
        raise NotImplementedError

    def make_field(self, space_partition=None, geometry_partition=None):
        """Returns a zero-initialized field over this function space"""
        raise NotImplementedError

    def unit_dof_value(args: Any, dof: DofIndex):
        """Unit value for a given degree of freedom. Typically a rank-1 tensor"""
        raise NotImplementedError

    def element_node_index(args: Any, element_index: ElementIndex, node_index_in_elt: int):
        """Global node index for a given node in a given element"""
        raise NotImplementedError

    def node_coords_in_element(args: Any, element_index: ElementIndex, node_index_in_elt: int):
        """Coordinates inside element of a given node"""
        raise NotImplementedError

    def element_inner_weight(args: Any, element_index: ElementIndex, coords: Coords, node_index_in_elt: int):
        """Inner weight for a node at given coordinates"""
        raise NotImplementedError

    def element_inner_weight_gradient(args: Any, element_index: ElementIndex, coords: Coords, node_index_in_elt: int):
        """Inner weight gradient for a node at given coordinates"""
        raise NotImplementedError

    def element_outer_weight(args: Any, element_index: ElementIndex, coords: Coords, node_index_in_elt: int):
        """Outer weight for a node at given coordinates"""
        raise NotImplementedError

    @wp.func
    def element_outer_weight_gradient(args: Any, element_index: ElementIndex, coords: Coords, node_index_in_elt: int):
        """Outer weight gradient for a node at given coordinates"""
        raise NotImplementedError
