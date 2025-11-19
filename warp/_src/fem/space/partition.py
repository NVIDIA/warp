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
from warp._src.fem import cache
from warp._src.fem.geometry import GeometryPartition, WholeGeometryPartition
from warp._src.fem.types import NULL_ELEMENT_INDEX, NULL_NODE_INDEX
from warp._src.fem.utils import compress_node_indices

from .function_space import FunctionSpace
from .topology import SpaceTopology

_wp_module_name_ = "warp.fem.space.partition"

wp.set_module_options({"enable_backward": False})


class SpacePartition:
    class PartitionArg:
        pass

    space_topology: SpaceTopology
    """Topology of the function space being partitioned"""

    geo_partition: GeometryPartition
    """Partition of the geometry controlling how to partition the space"""

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

    def rebuild(self, device: Optional = None, temporary_store: Optional[cache.TemporaryStore] = None):
        """Rebuild the space partition indices"""
        pass

    @cache.cached_arg_value
    def partition_arg_value(self, device):
        arg = self.PartitionArg()
        self.fill_partition_arg(arg, device)
        return arg

    def fill_partition_arg(self, arg, device):
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
            wp.launch(kernel=self._iota_kernel, dim=self.node_count(), inputs=[self._node_indices])
        return self._node_indices

    def partition_arg_value(self, device):
        return WholeSpacePartition.PartitionArg()

    def fill_partition_arg(self, arg, device):
        pass

    @wp.func
    def partition_node_index(args: Any, space_node_index: int):
        return space_node_index

    def __eq__(self, other: SpacePartition) -> bool:
        return isinstance(other, WholeSpacePartition) and self.space_topology == other.space_topology

    @property
    def name(self) -> str:
        return "Whole"

    @wp.kernel
    def _iota_kernel(indices: wp.array(dtype=int)):
        indices[wp.tid()] = wp.tid()


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
        max_node_count: int = -1,
        device=None,
        temporary_store: Optional[cache.TemporaryStore] = None,
    ):
        super().__init__(space_topology=space_topology, geo_partition=geo_partition)

        if max_node_count >= 0:
            max_node_count = min(max_node_count, space_topology.node_count())

        self._max_node_count = max_node_count
        self._with_halo = with_halo

        self._category_offsets: wp.array = None
        """Offsets for each node category"""
        self._node_indices: wp.array = None
        """Mapping from local partition node indices to global space node indices"""
        self._space_to_partition: wp.array = None
        """Mapping from global space node indices to local partition node indices"""

        self.rebuild(device, temporary_store)

    def rebuild(self, device: Optional = None, temporary_store: Optional[cache.TemporaryStore] = None):
        self._compute_node_indices_from_sides(device, self._with_halo, self._max_node_count, temporary_store)

    def node_count(self) -> int:
        """Returns number of nodes referenced by this partition, including exterior halo"""
        return int(self._category_offsets.numpy()[NodeCategory.HALO_OTHER_SIDE + 1])

    def owned_node_count(self) -> int:
        """Returns number of nodes in this partition, excluding exterior halo"""
        return int(self._category_offsets.numpy()[NodeCategory.OWNED_FRONTIER + 1])

    def interior_node_count(self) -> int:
        """Returns number of interior nodes in this partition"""
        return int(self._category_offsets.numpy()[NodeCategory.OWNED_INTERIOR + 1])

    def space_node_indices(self):
        """Return the global function space indices for nodes in this partition"""
        return self._node_indices

    def fill_partition_arg(self, arg, device):
        arg.space_to_partition = self._space_to_partition.to(device)

    @wp.func
    def partition_node_index(args: PartitionArg, space_node_index: int):
        return args.space_to_partition[space_node_index]

    def _compute_node_indices_from_sides(
        self, device, with_halo: bool, max_node_count: int, temporary_store: cache.TemporaryStore
    ):
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
            if cell_index == NULL_ELEMENT_INDEX:
                return

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
            if side_index == NULL_ELEMENT_INDEX:
                return

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
            if side_index == NULL_ELEMENT_INDEX:
                return

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
        node_category.fill_(value=NodeCategory.EXTERIOR)

        wp.launch(
            dim=self.geo_partition.cell_count(),
            kernel=node_category_from_cells_kernel,
            inputs=[
                self.geo_partition.geometry.cell_arg_value(device),
                self.geo_partition.cell_arg_value(device),
                self.space_topology.topo_arg_value(device),
                node_category,
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
                    node_category,
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
                    node_category,
                ],
                device=device,
            )

        with wp.ScopedDevice(device):
            self._finalize_node_indices(node_category, max_node_count, temporary_store)

        node_category.release()

    def _finalize_node_indices(
        self, node_category: wp.array(dtype=int), max_node_count: int, temporary_store: cache.TemporaryStore
    ):
        category_offsets, node_indices = compress_node_indices(
            NodeCategory.COUNT, node_category, temporary_store=temporary_store
        )
        device = node_category.device

        if max_node_count >= 0:
            # If max_node_count is provided, we do not bring back the actual node count to the host;
            # instead, we use the provided value as an upper bound to dimension launches and allocations.
            # In this case, all nodes are classified as possible "owned" nodes

            if self._category_offsets is None:
                self._category_offsets = cache.borrow_temporary(
                    temporary_store,
                    shape=(NodeCategory.COUNT + 1,),
                    dtype=category_offsets.dtype,
                    device="cpu",
                )
            self._category_offsets.fill_(max_node_count)
            copy_event = None
        else:
            # Copy offsets to cpu
            if self._category_offsets is None:
                self._category_offsets = cache.borrow_temporary(
                    temporary_store,
                    shape=(NodeCategory.COUNT + 1,),
                    dtype=category_offsets.dtype,
                    pinned=device.is_cuda,
                    device="cpu",
                )
            wp.copy(src=category_offsets, dest=self._category_offsets, count=NodeCategory.COUNT + 1)
            copy_event = cache.capture_event()

        # Compute global to local indices
        if self._space_to_partition is None or self._space_to_partition.shape != node_indices.shape:
            self._space_to_partition = cache.borrow_temporary_like(node_indices, temporary_store)

        wp.launch(
            kernel=NodePartition._scatter_partition_indices,
            dim=self.space_topology.node_count(),
            device=device,
            inputs=[max_node_count, category_offsets, node_indices, self._space_to_partition],
        )

        if copy_event is not None:
            cache.synchronize_event(copy_event)  # Transfer to host must be finished to access node_count()

        # Copy to shrunk-to-fit array
        if self._node_indices is None or self._node_indices.shape[0] != self.node_count():
            self._node_indices = cache.borrow_temporary(
                temporary_store, shape=(self.node_count(),), dtype=int, device=device
            )

        wp.copy(dest=self._node_indices, src=node_indices, count=self.node_count())
        node_indices.release()

    @wp.kernel
    def _scatter_partition_indices(
        max_node_count: int,
        category_offsets: wp.array(dtype=int),
        node_indices: wp.array(dtype=int),
        space_to_partition_indices: wp.array(dtype=int),
    ):
        local_idx = wp.tid()
        space_idx = node_indices[local_idx]

        local_node_count = category_offsets[NodeCategory.EXTERIOR]  # all but exterior nodes
        if max_node_count >= 0:
            if local_node_count > max_node_count:
                if local_idx == 0:
                    wp.printf(
                        "Number of space partition nodes exceeded the %d limit; increase `max_node_count` to %d.\n",
                        max_node_count,
                        local_node_count,
                    )

                local_node_count = max_node_count

        if local_idx < local_node_count:
            space_to_partition_indices[space_idx] = local_idx
        else:
            space_to_partition_indices[space_idx] = NULL_NODE_INDEX


def make_space_partition(
    space: Optional[FunctionSpace] = None,
    geometry_partition: Optional[GeometryPartition] = None,
    space_topology: Optional[SpaceTopology] = None,
    with_halo: bool = True,
    max_node_count: int = -1,
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
        max_node_count: if positive, will be used to limit the number of nodes to avoid device/host synchronization.
        device: Warp device on which to perform and store computations

    Returns:
        the resulting space partition
    """

    if space_topology is None:
        space_topology = space.topology

    space_topology = space_topology.full_space_topology()

    if geometry_partition is not None and not isinstance(geometry_partition, WholeGeometryPartition):
        return NodePartition(
            space_topology=space_topology,
            geo_partition=geometry_partition,
            with_halo=with_halo,
            max_node_count=max_node_count,
            device=device,
            temporary_store=temporary_store,
        )

    return WholeSpacePartition(space_topology)
