from typing import Any, Optional

import warp as wp

from warp.fem.geometry import GeometryPartition, WholeGeometryPartition
from warp.fem.utils import compress_node_indices, _iota_kernel
from warp.fem.types import NULL_NODE_INDEX

from .function_space import FunctionSpace


wp.set_module_options({"enable_backward": False})


class SpacePartition:
    class PartitionArg:
        pass

    def __init__(self, space: FunctionSpace, geo_partition: GeometryPartition):
        self.space = space
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

    def partition_node_index(args: Any, space_node_index: int):
        """Returns the index in the partition of a function space node, or -1 if it does not exist"""

    def __str__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}"


class WholeSpacePartition(SpacePartition):
    @wp.struct
    class PartitionArg:
        pass

    def __init__(self, space: FunctionSpace):
        super().__init__(space, WholeGeometryPartition(space.geometry))
        self._node_indices = None

    def node_count(self):
        """Returns number of nodes in this partition"""
        return self.space.node_count()

    def owned_node_count(self) -> int:
        """Returns number of nodes in this partition, excluding exterior halo"""
        return self.space.node_count()

    def interior_node_count(self) -> int:
        """Returns number of interior nodes in this partition"""
        return self.space.node_count()

    def space_node_indices(self):
        """Return the global function space indices for nodes in this partition"""
        if self._node_indices is None:
            self._node_indices = wp.empty(shape=(self.node_count(),), dtype=int)
            wp.launch(kernel=_iota_kernel, dim=self._node_indices.shape, inputs=[self._node_indices, 1])
        return self._node_indices

    def partition_arg_value(self, device):
        return WholeSpacePartition.PartitionArg()

    @wp.func
    def partition_node_index(args: Any, space_node_index: int):
        return space_node_index

    def __eq__(self, other: SpacePartition) -> bool:
        return isinstance(other, SpacePartition) and self.space == other.space


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

    def __init__(self, space: FunctionSpace, geo_partition: GeometryPartition, with_halo: bool = True, device=None):
        super().__init__(space, geo_partition=geo_partition)

        self._compute_node_indices_from_sides(device, with_halo)

    def node_count(self) -> int:
        """Returns number of nodes referenced by this partition, including exterior halo"""
        return int(self._category_offsets[NodeCategory.HALO_OTHER_SIDE + 1])

    def owned_node_count(self) -> int:
        """Returns number of nodes in this partition, excluding exterior halo"""
        return int(self._category_offsets[NodeCategory.OWNED_FRONTIER + 1])

    def interior_node_count(self) -> int:
        """Returns number of interior nodes in this partition"""
        return int(self._category_offsets[NodeCategory.OWNED_INTERIOR + 1])

    def space_node_indices(self):
        """Return the global function space indices for nodes in this partition"""
        return self._node_indices

    def partition_arg_value(self, device):
        arg = NodePartition.PartitionArg()
        arg.space_to_partition = self._space_to_partition.to(device)
        return arg

    @wp.func
    def partition_node_index(args: PartitionArg, space_node_index: int):
        return args.space_to_partition[space_node_index]

    def _compute_node_indices_from_sides(self, device, with_halo: bool):
        from warp.fem import cache

        trace_space = self.space.trace()
        NODES_PER_CELL = self.space.NODES_PER_ELEMENT
        NODES_PER_SIDE = trace_space.NODES_PER_ELEMENT

        def node_category_from_cells_fn(
            geo_partition_arg: self.geo_partition.CellArg,
            space_arg: self.space.SpaceArg,
            node_mask: wp.array(dtype=int),
        ):
            partition_cell_index = wp.tid()

            cell_index = self.geo_partition.cell_index(geo_partition_arg, partition_cell_index)

            for n in range(NODES_PER_CELL):
                space_nidx = self.space.element_node_index(space_arg, cell_index, n)
                node_mask[space_nidx] = NodeCategory.OWNED_INTERIOR

        def node_category_from_owned_sides_fn(
            geo_partition_arg: self.geo_partition.SideArg,
            space_arg: trace_space.SpaceArg,
            node_mask: wp.array(dtype=int),
        ):
            partition_side_index = wp.tid()

            side_index = self.geo_partition.side_index(geo_partition_arg, partition_side_index)

            for n in range(NODES_PER_SIDE):
                space_nidx = trace_space.element_node_index(space_arg, side_index, n)
                if node_mask[space_nidx] == NodeCategory.EXTERIOR:
                    node_mask[space_nidx] = NodeCategory.HALO_LOCAL_SIDE

        def node_category_from_frontier_sides_fn(
            geo_partition_arg: self.geo_partition.SideArg,
            space_arg: trace_space.SpaceArg,
            node_mask: wp.array(dtype=int),
        ):
            frontier_side_index = wp.tid()

            side_index = self.geo_partition.frontier_side_index(geo_partition_arg, frontier_side_index)

            for n in range(NODES_PER_SIDE):
                space_nidx = trace_space.element_node_index(space_arg, side_index, n)
                if node_mask[space_nidx] == NodeCategory.EXTERIOR:
                    node_mask[space_nidx] = NodeCategory.HALO_OTHER_SIDE
                elif node_mask[space_nidx] == NodeCategory.OWNED_INTERIOR:
                    node_mask[space_nidx] = NodeCategory.OWNED_FRONTIER

        node_category_from_cells_kernel = cache.get_kernel(
            node_category_from_cells_fn,
            suffix=f"{self.geo_partition.name}_{self.space.name}",
        )
        node_category_from_owned_sides_kernel = cache.get_kernel(
            node_category_from_owned_sides_fn,
            suffix=f"{self.geo_partition.name}_{self.space.name}",
        )
        node_category_from_frontier_sides_kernel = cache.get_kernel(
            node_category_from_frontier_sides_fn,
            suffix=f"{self.geo_partition.name}_{self.space.name}",
        )

        node_category = wp.empty(
            shape=(self.space.node_count(),),
            dtype=int,
            device=device,
        )
        node_category.fill_(value=NodeCategory.EXTERIOR)

        wp.launch(
            dim=self.geo_partition.cell_count(),
            kernel=node_category_from_cells_kernel,
            inputs=[
                self.geo_partition.cell_arg_value(device),
                self.space.space_arg_value(device),
                node_category,
            ],
            device=device,
        )

        if with_halo:
            wp.launch(
                dim=self.geo_partition.side_count(),
                kernel=node_category_from_owned_sides_kernel,
                inputs=[
                    self.geo_partition.side_arg_value(device),
                    self.space.space_arg_value(device),
                    node_category,
                ],
                device=device,
            )

            wp.launch(
                dim=self.geo_partition.frontier_side_count(),
                kernel=node_category_from_frontier_sides_kernel,
                inputs=[
                    self.geo_partition.side_arg_value(device),
                    self.space.space_arg_value(device),
                    node_category,
                ],
                device=device,
            )

        self._finalize_node_indices(node_category)

    def _finalize_node_indices(self, node_category: wp.array(dtype=int)):
        category_offsets, node_indices, _, __ = compress_node_indices(NodeCategory.COUNT, node_category)
        self._category_offsets = category_offsets.numpy()

        # Compute globla to local indices
        self._space_to_partition = node_category  # Reuse array storage
        wp.launch(
            kernel=NodePartition._scatter_partition_indices,
            dim=self.space.node_count(),
            device=self._space_to_partition.device,
            inputs=[self.node_count(), node_indices, self._space_to_partition],
        )

        # Copy to shrinked-to-fit array, save on memory
        self._node_indices = wp.empty(shape=(self.node_count()), dtype=int, device=node_indices.device)
        wp.copy(dest=self._node_indices, src=node_indices, count=self.node_count())

    @wp.kernel
    def _scatter_partition_indices(
        local_node_count: int,
        node_indices: wp.array(dtype=int),
        space_to_partition_indices: wp.array(dtype=int),
    ):
        local_idx = wp.tid()
        space_idx = node_indices[local_idx]

        if local_idx < local_node_count:
            space_to_partition_indices[space_idx] = local_idx
        else:
            space_to_partition_indices[space_idx] = NULL_NODE_INDEX


def make_space_partition(
    space: FunctionSpace,
    geometry_partition: Optional[GeometryPartition] = None,
    with_halo: bool = True,
    device=None,
) -> SpacePartition:
    """Computes the substep of nodes from a function space that touch a geometry partition

    Args:
        space: the function space to consider
        geometry_partition: The subset of the space geometry.  If not provided, use the whole geometry.
        with_halo: if True, include the halo nodes (nodes from exterior frontier cells to the partition)
        device: Warp device on which to perform and store computations

    Returns:
        the resulting space partition
    """

    if geometry_partition is not None and geometry_partition.cell_count() < geometry_partition.geometry.cell_count():
        return NodePartition(space, geometry_partition, with_halo=with_halo, device=device)

    return WholeSpacePartition(space)
