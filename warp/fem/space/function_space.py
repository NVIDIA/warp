import warp as wp
from warp.fem.geometry import Geometry
from warp.fem.types import Coords, DofIndex, ElementIndex, ElementKind

from .topology import SpaceTopology


class FunctionSpace:
    """
    Interface class for function spaces, i.e. geometry + interpolation basis
    """

    dtype: type
    """Value type of the interpolation functions"""

    SpaceArg: wp.codegen.Struct
    """Structure containing arguments to be passed to device function"""

    VALUE_DOF_COUNT: int
    """Number of degrees of freedom per node, as a Warp constant"""

    def __init__(self, topology: SpaceTopology):
        self._topology = topology

        if self._topology.is_trace:
            self.element_inner_reference_gradient_transform = self.geometry.side_inner_inverse_deformation_gradient
            self.element_outer_reference_gradient_transform = self.geometry.side_outer_inverse_deformation_gradient
        else:
            self.element_inner_reference_gradient_transform = self.geometry.cell_inverse_deformation_gradient
            self.element_outer_reference_gradient_transform = self.geometry.cell_inverse_deformation_gradient

    def node_count(self) -> int:
        """Number of nodes in the interpolation basis"""
        raise NotImplementedError

    def space_arg_value(self, device) -> wp.codegen.StructInstance:
        """Value of the arguments to be passed to device functions"""
        raise NotImplementedError

    @property
    def topology(self) -> SpaceTopology:
        """Underlying geometry"""
        return self._topology

    @property
    def geometry(self) -> Geometry:
        """Underlying geometry"""
        return self.topology.geometry

    @property
    def element_kind(self) -> ElementKind:
        """Kind of element the function space is expressed over"""
        return ElementKind.CELL if self.dimension == self.geometry.dimension else ElementKind.SIDE

    @property
    def dimension(self) -> int:
        """Function space embedding dimension"""
        return self.topology.dimension

    @property
    def degree(self) -> int:
        """Maximum polynomial degree of the underlying basis"""
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def __str__(self):
        return self.name

    def trace(self) -> "FunctionSpace":
        """Trace of the function space over lower-dimensional elements of the geometry"""
        raise NotImplementedError

    def make_field(self, space_partition=None):
        """Creates a zero-initialized discrete field over the function space holding values for all degrees of freedom of nodes in a space partition

        Args:
            space_partition: If provided, the subset of nodes to consider

        See also: :func:`make_space_partition`
        """
        raise NotImplementedError

    @staticmethod
    def unit_dof_value(elt_arg: "SpaceTopology.ElementArg", space_arg: "SpaceArg", dof: DofIndex):  # noqa: F821
        """Unit value for a given degree of freedom. Typically a rank-1 tensor"""
        raise NotImplementedError

    @staticmethod
    def node_coords_in_element(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        """Coordinates inside element of a given node"""
        raise NotImplementedError

    @staticmethod
    def node_quadrature_weight(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        """Weight of a given node when used as a quadrature point"""
        raise NotImplementedError

    @staticmethod
    def element_inner_weight(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        """Inner weight for a node at given coordinates"""
        raise NotImplementedError

    @staticmethod
    def element_inner_weight_gradient(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        """Inner weight gradient w.r.t. reference space for a node at given coordinates"""
        raise NotImplementedError

    @staticmethod
    def element_outer_weight(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        """Outer weight for a node at given coordinates"""
        raise NotImplementedError

    @staticmethod
    def element_outer_weight_gradient(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        """Outer weight gradient w.r.t reference space for a node at given coordinates"""
        raise NotImplementedError
