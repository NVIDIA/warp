from typing import Any, Optional

import warp as wp
import warp.fem.cache as cache
from warp.fem.geometry import GeometryPartition, WholeGeometryPartition
from warp.fem.types import NULL_NODE_INDEX
from warp.fem.utils import _iota_kernel, compress_node_indices

from .function_space import FunctionSpace
from .topology import SpaceTopology

wp.set_module_options({"enable_backward": False})


class SpacePartition:
    class PartitionArg:
        pass

    def __init__(self, space_topology: SpaceTopology, geo_partition: GeometryPartition):
        self.space_topology = space_topology
        self.geo_partition = geo_partition

    def node_count(self):
        """Returns number of nodes in this partition"""

    def owned_node_count(self) -> int:
        """Returns number of nodes in this partition, excluding exterior halo"""

    def interior_node_count(self) -> int:
        """Returns number of interior nodes in this partition"""

    def space_node_indices(self) -> wp.array:
        """Return the global function space indices for nodes in this partition"""

    def partition_arg_value(self, device):
        pass

    @staticmethod
    def partition_node_index(args: "PartitionArg", space_node_index: int):
        """Returns the index in the partition of a function space node, or ``NULL_NODE_INDEX`` if it does not exist"""

    def __str__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}"


class WholeSpacePartition(SpacePartition):
    @wp.struct
    class PartitionArg:
        pass

    def __init__(self, space_topology: SpaceTopology):
        super().__init__(space_topology, WholeGeometryPartition(space_topology.geometry))
        self._node_indices = None

    def node_count(self):
        """Returns number of nodes in this partition"""
        return self.space_topology.node_count()

    def owned_node_count(self) -> int:
        """Returns number of nodes in this partition, excluding exterior halo"""
        return self.space_topology.node_count()

    def interior_node_count(self) -> int:
        """Returns number of interior nodes in this partition"""
        return self.space_topology.node_count()

    def space_node_indices(self):
        """Return the global function space indices for nodes in this partition"""
        if self._node_indices is None:
            self._node_indices = cache.borrow_temporary(temporary_store=None, shape=(self.node_count(),), dtype=int)
            wp.launch(kernel=_iota_kernel, dim=self.node_count(), inputs=[self._node_indices.array, 1])
        return self._node_indices.array

    def partition_arg_value(self, device):
        return WholeSpacePartition.PartitionArg()

    @wp.func
    def partition_node_index(args: Any, space_node_index: int):
        return space_node_index

    def __eq__(self, other: SpacePartition) -> bool:
        return isinstance(other, SpacePartition) and self.space_topology == other.space_topology

    @property
    def name(self) -> str:
        return "Whole"


class NodeCategory:
    OWNED_INTERIOR = wp.constant(0)
    """Node is touched exclusively by this partition, not touched by frontier side"""
    OWNED_FRONTIER = wp.constant(1)
    """Node is touched by a frontier side, but belongs to an element of this partition"""
    HALO_LOCAL_SIDE = wp.constant(2)
    """Node belongs to an element of another partition, but is touched by one of our frontier side"""
    HALO_OTHER_SIDE = wp.constant(3)
    """Node belongs to an element of another partition, and is not touched by one of our frontier side"""
    EXTERIOR = wp.constant(4)
    """Node is never referenced by this partition"""

    COUNT = 5


class NodePartition(SpacePartition):
    @wp.struct
    class PartitionArg:
        space_to_partition: wp.array(dtype=int)

    def __init__(
        self,
        space_topology: SpaceTopology,
        geo_partition: GeometryPartition,
        with_halo: bool = True,
        device=None,
        temporary_store: cache.TemporaryStore = None,
    ):
        super().__init__(space_topology=space_topology, geo_partition=geo_partition)

        self._compute_node_indices_from_sides(device, with_halo, temporary_store)

    def node_count(self) -> int:
        """Returns number of nodes referenced by this partition, including exterior halo"""
        return int(self._category_offsets.array.numpy()[NodeCategory.HALO_OTHER_SIDE + 1])

    def owned_node_count(self) -> int:
        """Returns number of nodes in this partition, excluding exterior halo"""
        return int(self._category_offsets.array.numpy()[NodeCategory.OWNED_FRONTIER + 1])

    def interior_node_count(self) -> int:
        """Returns number of interior nodes in this partition"""
        return int(self._category_offsets.array.numpy()[NodeCategory.OWNED_INTERIOR + 1])

    def space_node_indices(self):
        """Return the global function space indices for nodes in this partition"""
        return self._node_indices.array

    @cache.cached_arg_value
    def partition_arg_value(self, device):
        arg = NodePartition.PartitionArg()
        arg.space_to_partition = self._space_to_partition.array.to(device)
        return arg

    @wp.func
    def partition_node_index(args: PartitionArg, space_node_index: int):
        return args.space_to_partition[space_node_index]

    def _compute_node_indices_from_sides(self, device, with_halo: bool, temporary_store: cache.TemporaryStore):
        from warp.fem import cache

        trace_topology = self.space_topology.trace()

        @cache.dynamic_kernel(suffix=f"{self.geo_partition.name}_{self.space_topology.name}")
        def node_category_from_cells_kernel(
            geo_arg: self.geo_partition.geometry.CellArg,
            geo_partition_arg: self.geo_partition.CellArg,
            space_arg: self.space_topology.TopologyArg,
            node_mask: wp.array(dtype=int),
        ):
            partition_cell_index = wp.tid()

            cell_index = self.geo_partition.cell_index(geo_partition_arg, partition_cell_index)

            cell_node_count = self.space_topology.element_node_count(geo_arg, space_arg, cell_index)
            for n in range(cell_node_count):
                space_nidx = self.space_topology.element_node_index(geo_arg, space_arg, cell_index, n)
                node_mask[space_nidx] = NodeCategory.OWNED_INTERIOR

        @cache.dynamic_kernel(suffix=f"{self.geo_partition.name}_{self.space_topology.name}")
        def node_category_from_owned_sides_kernel(
            geo_arg: self.geo_partition.geometry.SideArg,
            geo_partition_arg: self.geo_partition.SideArg,
            space_arg: trace_topology.TopologyArg,
            node_mask: wp.array(dtype=int),
        ):
            partition_side_index = wp.tid()

            side_index = self.geo_partition.side_index(geo_partition_arg, partition_side_index)

            side_node_count = trace_topology.element_node_count(geo_arg, space_arg, side_index)
            for n in range(side_node_count):
                space_nidx = trace_topology.element_node_index(geo_arg, space_arg, side_index, n)

                if node_mask[space_nidx] == NodeCategory.EXTERIOR:
                    node_mask[space_nidx] = NodeCategory.HALO_LOCAL_SIDE

        @cache.dynamic_kernel(suffix=f"{self.geo_partition.name}_{self.space_topology.name}")
        def node_category_from_frontier_sides_kernel(
            geo_arg: self.geo_partition.geometry.SideArg,
            geo_partition_arg: self.geo_partition.SideArg,
            space_arg: trace_topology.TopologyArg,
            node_mask: wp.array(dtype=int),
        ):
            frontier_side_index = wp.tid()

            side_index = self.geo_partition.frontier_side_index(geo_partition_arg, frontier_side_index)

            side_node_count = trace_topology.element_node_count(geo_arg, space_arg, side_index)
            for n in range(side_node_count):
                space_nidx = trace_topology.element_node_index(geo_arg, space_arg, side_index, n)
                if node_mask[space_nidx] == NodeCategory.EXTERIOR:
                    node_mask[space_nidx] = NodeCategory.HALO_OTHER_SIDE
                elif node_mask[space_nidx] == NodeCategory.OWNED_INTERIOR:
                    node_mask[space_nidx] = NodeCategory.OWNED_FRONTIER

        node_category = cache.borrow_temporary(
            temporary_store,
            shape=(self.space_topology.node_count(),),
            dtype=int,
            device=device,
        )
        node_category.array.fill_(value=NodeCategory.EXTERIOR)

        wp.launch(
            dim=self.geo_partition.cell_count(),
            kernel=node_category_from_cells_kernel,
            inputs=[
                self.geo_partition.geometry.cell_arg_value(device),
                self.geo_partition.cell_arg_value(device),
                self.space_topology.topo_arg_value(device),
                node_category.array,
            ],
            device=device,
        )

        if with_halo:
            wp.launch(
                dim=self.geo_partition.side_count(),
                kernel=node_category_from_owned_sides_kernel,
                inputs=[
                    self.geo_partition.geometry.side_arg_value(device),
                    self.geo_partition.side_arg_value(device),
                    self.space_topology.topo_arg_value(device),
                    node_category.array,
                ],
                device=device,
            )

            wp.launch(
                dim=self.geo_partition.frontier_side_count(),
                kernel=node_category_from_frontier_sides_kernel,
                inputs=[
                    self.geo_partition.geometry.side_arg_value(device),
                    self.geo_partition.side_arg_value(device),
                    self.space_topology.topo_arg_value(device),
                    node_category.array,
                ],
                device=device,
            )

        self._finalize_node_indices(node_category.array, temporary_store)

        node_category.release()

    def _finalize_node_indices(self, node_category: wp.array(dtype=int), temporary_store: cache.TemporaryStore):
        category_offsets, node_indices = compress_node_indices(
            NodeCategory.COUNT, node_category, temporary_store=temporary_store
        )

        # Copy offsets to cpu
        device = node_category.device
        with wp.ScopedDevice(device):
            self._category_offsets = cache.borrow_temporary(
                temporary_store,
                shape=category_offsets.array.shape,
                dtype=category_offsets.array.dtype,
                pinned=device.is_cuda,
                device="cpu",
            )
            wp.copy(src=category_offsets.array, dest=self._category_offsets.array)
            copy_event = cache.capture_event()

            # Compute global to local indices
            self._space_to_partition = cache.borrow_temporary_like(node_indices, temporary_store)
            wp.launch(
                kernel=NodePartition._scatter_partition_indices,
                dim=self.space_topology.node_count(),
                device=device,
                inputs=[category_offsets.array, node_indices.array, self._space_to_partition.array],
            )

            # Copy to shrinked-to-fit array
            cache.synchronize_event(copy_event)  # Transfer to host must be finished to access node_count()
            self._node_indices = cache.borrow_temporary(
                temporary_store, shape=(self.node_count()), dtype=int, device=device
            )
            wp.copy(dest=self._node_indices.array, src=node_indices.array, count=self.node_count())

            node_indices.release()

    @wp.kernel
    def _scatter_partition_indices(
        category_offsets: wp.array(dtype=int),
        node_indices: wp.array(dtype=int),
        space_to_partition_indices: wp.array(dtype=int),
    ):
        local_idx = wp.tid()
        space_idx = node_indices[local_idx]

        local_node_count = category_offsets[NodeCategory.EXTERIOR]  # all but exterior nodes
        if local_idx < local_node_count:
            space_to_partition_indices[space_idx] = local_idx
        else:
            space_to_partition_indices[space_idx] = NULL_NODE_INDEX


def make_space_partition(
    space: Optional[FunctionSpace] = None,
    geometry_partition: Optional[GeometryPartition] = None,
    space_topology: Optional[SpaceTopology] = None,
    with_halo: bool = True,
    device=None,
    temporary_store: cache.TemporaryStore = None,
) -> SpacePartition:
    """Computes the subset of nodes from a function space topology that touch a geometry partition

    Either `space_topology` or `space` must be provided (and will be considered in that order).

    Args:
        space: (deprecated) the function space defining the topology if `space_topology` is ``None``.
        geometry_partition: The subset of the space geometry.  If not provided, use the whole geometry.
        space_topology: the topology of the function space to consider. If ``None``, deduced from `space`.
        with_halo: if True, include the halo nodes (nodes from exterior frontier cells to the partition)
        device: Warp device on which to perform and store computations

    Returns:
        the resulting space partition
    """

    if space_topology is None:
        space_topology = space.topology

    space_topology = space_topology.full_space_topology()

    if geometry_partition is not None:
        if geometry_partition.cell_count() < geometry_partition.geometry.cell_count():
            return NodePartition(
                space_topology=space_topology,
                geo_partition=geometry_partition,
                with_halo=with_halo,
                device=device,
                temporary_store=temporary_store,
            )

    return WholeSpacePartition(space_topology)
