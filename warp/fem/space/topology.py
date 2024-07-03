from typing import Optional, Tuple, Type

import warp as wp
from warp.fem import cache
from warp.fem.geometry import DeformedGeometry, Geometry
from warp.fem.types import NULL_ELEMENT_INDEX, NULL_NODE_INDEX, ElementIndex


class SpaceTopology:
    """
    Interface class for defining the topology of a function space.

    The topology only considers the indices of the nodes in each element, and as such,
    the connectivity pattern of the function space.
    It does not specify the actual location of the nodes within the elements, or the valuation function.
    """

    dimension: int
    """Embedding dimension of the function space"""

    MAX_NODES_PER_ELEMENT: int
    """maximum number of interpolation nodes per element of the geometry.

    .. note:: This will change to be defined per-element in future versions
    """

    @wp.struct
    class TopologyArg:
        """Structure containing arguments to be passed to device functions"""

        pass

    def __init__(self, geometry: Geometry, max_nodes_per_element: int):
        self._geometry = geometry
        self.dimension = geometry.dimension
        self.MAX_NODES_PER_ELEMENT = wp.constant(max_nodes_per_element)
        self.ElementArg = geometry.CellArg

        self._make_constant_element_node_count()

    @property
    def geometry(self) -> Geometry:
        """Underlying geometry"""
        return self._geometry

    def node_count(self) -> int:
        """Number of nodes in the interpolation basis"""
        raise NotImplementedError

    def topo_arg_value(self, device) -> "TopologyArg":
        """Value of the topology argument structure to be passed to device functions"""
        return SpaceTopology.TopologyArg()

    @property
    def name(self):
        return f"{self.__class__.__name__}_{self.MAX_NODES_PER_ELEMENT}"

    def __str__(self):
        return self.name

    @staticmethod
    def element_node_count(
        geo_arg: "ElementArg",  # noqa: F821
        topo_arg: "TopologyArg",
        element_index: ElementIndex,
    ) -> int:
        """Returns the actual number of nodes in a given element"""
        raise NotImplementedError

    @staticmethod
    def element_node_index(
        geo_arg: "ElementArg",  # noqa: F821
        topo_arg: "TopologyArg",
        element_index: ElementIndex,
        node_index_in_elt: int,
    ) -> int:
        """Global node index for a given node in a given element"""
        raise NotImplementedError

    @staticmethod
    def side_neighbor_node_counts(
        side_arg: "ElementArg",  # noqa: F821
        side_index: ElementIndex,
    ) -> Tuple[int, int]:
        """Returns the number of nodes for both the inner and outer cells of a given sides"""
        raise NotImplementedError

    def element_node_indices(self, out: Optional[wp.array] = None) -> wp.array:
        """Returns a temporary array containing the global index for each node of each element"""

        MAX_NODES_PER_ELEMENT = self.MAX_NODES_PER_ELEMENT

        @cache.dynamic_kernel(suffix=self.name)
        def fill_element_node_indices(
            geo_cell_arg: self.geometry.CellArg,
            topo_arg: self.TopologyArg,
            element_node_indices: wp.array2d(dtype=int),
        ):
            element_index = wp.tid()
            element_node_count = self.element_node_count(geo_cell_arg, topo_arg, element_index)
            for n in range(element_node_count):
                element_node_indices[element_index, n] = self.element_node_index(
                    geo_cell_arg, topo_arg, element_index, n
                )

        shape = (self.geometry.cell_count(), MAX_NODES_PER_ELEMENT)
        if out is None:
            element_node_indices = wp.empty(
                shape=shape,
                dtype=int,
            )
        else:
            if out.shape != shape or out.dtype != wp.int32:
                raise ValueError(f"Out element node indices array must have shape {shape} and data type 'int32'")
            element_node_indices = out

        wp.launch(
            dim=element_node_indices.shape[0],
            kernel=fill_element_node_indices,
            inputs=[
                self.geometry.cell_arg_value(device=element_node_indices.device),
                self.topo_arg_value(device=element_node_indices.device),
                element_node_indices,
            ],
            device=element_node_indices.device,
        )

        return element_node_indices

    # Interface generating trace space topology

    def trace(self) -> "TraceSpaceTopology":
        """Trace of the function space over lower-dimensional elements of the geometry"""

        return TraceSpaceTopology(self)

    @property
    def is_trace(self) -> bool:
        """Whether this topology is defined on the trace of the geometry"""
        return self.dimension == self.geometry.dimension - 1

    def full_space_topology(self) -> "SpaceTopology":
        """Returns the full space topology from which this topology is derived"""
        return self

    def __eq__(self, other: "SpaceTopology") -> bool:
        """Checks whether two topologies are compatible"""
        return self.geometry == other.geometry and self.name == other.name

    def is_derived_from(self, other: "SpaceTopology") -> bool:
        """Checks whether two topologies are equal, or `self` is the trace of `other`"""
        if self.dimension == other.dimension:
            return self == other
        if self.dimension + 1 == other.dimension:
            return self.full_space_topology() == other
        return False

    def _make_constant_element_node_count(self):
        NODES_PER_ELEMENT = wp.constant(self.MAX_NODES_PER_ELEMENT)

        @cache.dynamic_func(suffix=self.name)
        def constant_element_node_count(
            geo_arg: self.geometry.CellArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
        ):
            return NODES_PER_ELEMENT

        @cache.dynamic_func(suffix=self.name)
        def constant_side_neighbor_node_counts(
            side_arg: self.geometry.SideArg,
            element_index: ElementIndex,
        ):
            return NODES_PER_ELEMENT, NODES_PER_ELEMENT

        self.element_node_count = constant_element_node_count
        self.side_neighbor_node_counts = constant_side_neighbor_node_counts


class TraceSpaceTopology(SpaceTopology):
    """Auto-generated trace topology defining the node indices associated to the geometry sides"""

    def __init__(self, topo: SpaceTopology):
        self._topo = topo

        super().__init__(topo.geometry, 2 * topo.MAX_NODES_PER_ELEMENT)

        self.dimension = topo.dimension - 1
        self.ElementArg = topo.geometry.SideArg

        self.TopologyArg = topo.TopologyArg
        self.topo_arg_value = topo.topo_arg_value

        self.inner_cell_index = self._make_inner_cell_index()
        self.outer_cell_index = self._make_outer_cell_index()
        self.neighbor_cell_index = self._make_neighbor_cell_index()

        self.element_node_index = self._make_element_node_index()
        self.element_node_count = self._make_element_node_count()
        self.side_neighbor_node_counts = None

    def node_count(self) -> int:
        return self._topo.node_count()

    @property
    def name(self):
        return f"{self._topo.name}_Trace"

    def _make_inner_cell_index(self):
        @cache.dynamic_func(suffix=self.name)
        def inner_cell_index(side_arg: self.geometry.SideArg, element_index: ElementIndex, node_index_in_elt: int):
            inner_count, outer_count = self._topo.side_neighbor_node_counts(side_arg, element_index)
            if node_index_in_elt >= inner_count:
                return NULL_ELEMENT_INDEX, NULL_NODE_INDEX
            return self.geometry.side_inner_cell_index(side_arg, element_index), node_index_in_elt

        return inner_cell_index

    def _make_outer_cell_index(self):
        @cache.dynamic_func(suffix=self.name)
        def outer_cell_index(side_arg: self.geometry.SideArg, element_index: ElementIndex, node_index_in_elt: int):
            inner_count, outer_count = self._topo.side_neighbor_node_counts(side_arg, element_index)
            if node_index_in_elt < inner_count:
                return NULL_ELEMENT_INDEX, NULL_NODE_INDEX
            return self.geometry.side_outer_cell_index(side_arg, element_index), node_index_in_elt - inner_count

        return outer_cell_index

    def _make_neighbor_cell_index(self):
        @cache.dynamic_func(suffix=self.name)
        def neighbor_cell_index(side_arg: self.geometry.SideArg, element_index: ElementIndex, node_index_in_elt: int):
            inner_count, outer_count = self._topo.side_neighbor_node_counts(side_arg, element_index)
            if node_index_in_elt < inner_count:
                return self.geometry.side_inner_cell_index(side_arg, element_index), node_index_in_elt

            return (
                self.geometry.side_outer_cell_index(side_arg, element_index),
                node_index_in_elt - inner_count,
            )

        return neighbor_cell_index

    def _make_element_node_count(self):
        @cache.dynamic_func(suffix=self.name)
        def trace_element_node_count(
            geo_side_arg: self.geometry.SideArg,
            topo_arg: self._topo.TopologyArg,
            element_index: ElementIndex,
        ):
            inner_count, outer_count = self._topo.side_neighbor_node_counts(geo_side_arg, element_index)
            return inner_count + outer_count

        return trace_element_node_count

    def _make_element_node_index(self):
        @cache.dynamic_func(suffix=self.name)
        def trace_element_node_index(
            geo_side_arg: self.geometry.SideArg,
            topo_arg: self._topo.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            cell_index, index_in_cell = self.neighbor_cell_index(geo_side_arg, element_index, node_index_in_elt)

            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return self._topo.element_node_index(geo_cell_arg, topo_arg, cell_index, index_in_cell)

        return trace_element_node_index

    def full_space_topology(self) -> SpaceTopology:
        """Returns the full space topology from which this topology is derived"""
        return self._topo

    def __eq__(self, other: "TraceSpaceTopology") -> bool:
        return self._topo == other._topo


class RegularDiscontinuousSpaceTopologyMixin:
    """Helper for defining discontinuous topologies (per-element nodes)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.element_node_index = self._make_element_node_index()

    def node_count(self):
        return self.geometry.cell_count() * self.MAX_NODES_PER_ELEMENT

    @property
    def name(self):
        return f"{self.geometry.name}_D{self.MAX_NODES_PER_ELEMENT}"

    def _make_element_node_index(self):
        NODES_PER_ELEMENT = self.MAX_NODES_PER_ELEMENT

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return NODES_PER_ELEMENT * element_index + node_index_in_elt

        return element_node_index


class RegularDiscontinuousSpaceTopology(RegularDiscontinuousSpaceTopologyMixin, SpaceTopology):
    """Topology for generic discontinuous spaces"""

    pass


class DeformedGeometrySpaceTopology(SpaceTopology):
    def __init__(self, geometry: DeformedGeometry, base_topology: SpaceTopology):
        self.base = base_topology
        super().__init__(geometry, base_topology.MAX_NODES_PER_ELEMENT)

        self.node_count = self.base.node_count
        self.topo_arg_value = self.base.topo_arg_value
        self.TopologyArg = self.base.TopologyArg

        self._make_passthrough_functions()

    @property
    def name(self):
        return f"{self.base.name}_{self.geometry.field.name}"

    def _make_passthrough_functions(self):
        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return self.base.element_node_index(elt_arg.elt_arg, topo_arg, element_index, node_index_in_elt)

        @cache.dynamic_func(suffix=self.name)
        def element_node_count(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.TopologyArg,
            element_count: ElementIndex,
        ):
            return self.base.element_node_count(elt_arg.elt_arg, topo_arg, element_count)

        @cache.dynamic_func(suffix=self.name)
        def side_neighbor_node_counts(
            side_arg: self.geometry.SideArg,
            element_index: ElementIndex,
        ):
            inner_count, outer_count = self.base.side_neighbor_node_counts(side_arg.base_arg, element_index)
            return inner_count, outer_count

        self.element_node_index = element_node_index
        self.element_node_count = element_node_count
        self.side_neighbor_node_counts = side_neighbor_node_counts


def forward_base_topology(topology_class: Type[SpaceTopology], geometry: Geometry, *args, **kwargs) -> SpaceTopology:
    """
    If `geometry` is *not* a :class:`DeformedGeometry`, constructs a normal instance of `topology_class` over `geometry`, forwarding additional arguments.

    If `geometry` *is* a :class:`DeformedGeometry`, constructs an instance of `topology_class` over the base (undeformed) geometry of `geometry`, then warp it
    in a :class:`DeformedGeometrySpaceTopology` forwarding the calls to the underlying topology.
    """

    if isinstance(geometry, DeformedGeometry):
        base_topo = topology_class(geometry.base, *args, **kwargs)
        return DeformedGeometrySpaceTopology(geometry, base_topo)

    return topology_class(geometry, *args, **kwargs)
