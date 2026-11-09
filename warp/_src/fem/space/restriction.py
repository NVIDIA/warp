# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import warp as wp
from warp._src.fem import cache
from warp._src.fem.domain import Cells, GeometryDomain
from warp._src.fem.quadrature.pic_quadrature import PicQuadrature
from warp._src.fem.types import NULL_ELEMENT_INDEX, NULL_NODE_INDEX, NodeElementIndex
from warp._src.fem.utils import compress_node_indices, host_read_at_index
from warp._src.utils import array_scan

from .partition import SpacePartition
from .point_basis_space import UnstructuredPointTopology
from .topology import SpaceTopology

wp.set_module_options({"enable_backward": False})


class SpaceRestriction:
    """Restriction of a space partition to a given :class:`GeometryDomain`."""

    space_partition: SpacePartition
    """Partition of the function space nodes that will be restricted."""
    space_topology: SpaceTopology
    """Topology of the full function space."""
    domain: GeometryDomain
    """Domain to which the space partition is being restricted."""

    def __init__(
        self,
        space_partition: SpacePartition,
        domain: GeometryDomain,
        device=None,
        temporary_store: cache.TemporaryStore | None = None,
    ):
        space_topology = space_partition.space_topology

        if domain.dimension == space_topology.dimension - 1:
            space_topology = space_topology.trace()

        if domain.dimension != space_topology.dimension:
            raise ValueError("Incompatible space and domain dimensions")

        self.space_partition = space_partition
        self.space_topology = space_topology

        self.domain = domain

        self._node_count_dev: wp.array = None
        """Number of unique partition node indices"""
        self._dof_partition_indices: wp.array = None
        """Array of unique partition node indices"""

        self._dof_partition_element_offsets: wp.array = None
        """Mapping from partition node to offset in the per-node element indices array"""
        self._dof_element_indices: wp.array = None
        """Concatenation of neighboring elements indices for each partition node"""
        self._dof_indices_in_element: wp.array = None
        """Concatenation of node index in element for each partition node"""

        self.rebuild(device=device, temporary_store=temporary_store)

    def rebuild(self, device: Optional["wp.DeviceLike"] = None, temporary_store: cache.TemporaryStore | None = None):
        """Rebuild internal indices for the space restriction."""
        self.node_arg_value.invalidate(self)

        topology = self.space_topology
        if (
            type(topology) is UnstructuredPointTopology
            and topology._use_evaluation_point_index
            and isinstance(topology._quadrature, PicQuadrature)
            and type(self.domain) is Cells
            and self.domain == topology._quadrature.domain
        ):
            self._rebuild_point_restriction(device, temporary_store)
            return

        max_nodes_per_element = self.space_topology.MAX_NODES_PER_ELEMENT

        @cache.dynamic_kernel(
            suffix=(self.domain.name, self.space_topology.name, self.space_partition.name),
            kernel_options={"max_unroll": 8},
        )
        def fill_element_node_indices(
            element_arg: self.domain.ElementArg,
            domain_index_arg: self.domain.ElementIndexArg,
            topo_arg: self.space_topology.TopologyArg,
            partition_arg: self.space_partition.PartitionArg,
            element_node_indices: wp.array2d(dtype=int),
        ):
            domain_element_index = wp.tid()
            element_index = self.domain.element_index(domain_index_arg, domain_element_index)

            max_nodes_per_element = element_node_indices.shape[1]
            if element_index == NULL_ELEMENT_INDEX:
                element_node_count = 0
            else:
                element_node_count = self.space_topology.element_node_count(element_arg, topo_arg, element_index)

            for n in range(element_node_count):
                space_nidx = self.space_topology.element_node_index(element_arg, topo_arg, element_index, n)
                partition_nidx = self.space_partition.partition_node_index(partition_arg, space_nidx)
                element_node_indices[domain_element_index, n] = partition_nidx
            for n in range(element_node_count, max_nodes_per_element):
                element_node_indices[domain_element_index, n] = NULL_NODE_INDEX

        element_node_indices = cache.borrow_temporary(
            temporary_store,
            shape=(self.domain.element_count(), max_nodes_per_element),
            dtype=int,
            device=device,
        )
        wp.launch(
            dim=element_node_indices.shape[0],
            kernel=fill_element_node_indices,
            inputs=[
                self.domain.element_arg_value(device),
                self.domain.element_index_arg_value(device),
                self.space_topology.topo_arg_value(device),
                self.space_partition.partition_arg_value(device),
                element_node_indices,
            ],
            device=device,
        )

        # Build compressed map from node to element indices
        partition_node_count = self.space_partition.node_count()
        flattened_node_indices = element_node_indices.flatten()
        (
            self._dof_partition_element_offsets,
            node_array_indices,
            self._node_count_dev,
            self._dof_partition_indices,
        ) = compress_node_indices(
            partition_node_count,
            flattened_node_indices,
            node_offsets=self._dof_partition_element_offsets,
            unique_node_count=self._node_count_dev,
            unique_node_indices=self._dof_partition_indices,
            return_unique_nodes=True,
            temporary_store=temporary_store,
        )

        # Extract element index and index in element
        if self._dof_element_indices is None or self._dof_element_indices.shape != flattened_node_indices.shape:
            self._dof_element_indices = cache.borrow_temporary_like(flattened_node_indices, temporary_store)
            self._dof_indices_in_element = cache.borrow_temporary_like(flattened_node_indices, temporary_store)

        wp.launch(
            kernel=SpaceRestriction._split_vertex_element_index,
            dim=flattened_node_indices.shape,
            inputs=[
                max_nodes_per_element,
                partition_node_count,
                self._dof_partition_element_offsets,
                node_array_indices,
                self._dof_element_indices,
                self._dof_indices_in_element,
            ],
            device=flattened_node_indices.device,
        )

        node_array_indices.release()
        element_node_indices.release()

        # Upper bound on node count, use `node_count_sync` to get the actual value
        self._node_count = min(self.space_partition.node_count(), self._dof_partition_indices.shape[0])

    def _require_point_array(self, name, shape, device, temporary_store):
        array = getattr(self, name)
        if array is None or array.shape != shape or array.device != device or array.dtype != wp.int32:
            if array is not None:
                array.release()
            array = cache.borrow_temporary(temporary_store, shape=shape, dtype=int, device=device)
            setattr(self, name, array)
        return array

    def _rebuild_point_restriction(self, device=None, temporary_store=None):
        """Build unique evaluation-node incidences without a padded table or sort."""
        device = wp.get_device(device)
        topology = self.space_topology
        partition = self.space_partition
        domain = self.domain
        node_count = partition.node_count()
        # Keep the original unsynchronized launch upper bound for clamped spaces.
        self._node_count = min(node_count, domain.element_count() * topology.MAX_NODES_PER_ELEMENT)

        offsets = self._require_point_array(
            "_dof_partition_element_offsets", (node_count + 1,), device, temporary_store
        )
        indices = self._require_point_array("_dof_partition_indices", (node_count,), device, temporary_store)
        elements = self._require_point_array("_dof_element_indices", (node_count,), device, temporary_store)
        local = self._require_point_array("_dof_indices_in_element", (node_count,), device, temporary_store)
        count = self._require_point_array("_node_count_dev", (1,), device, temporary_store)
        node_element = cache.borrow_temporary(temporary_store, shape=(node_count,), dtype=int, device=device)
        node_local = cache.borrow_temporary(temporary_store, shape=(node_count,), dtype=int, device=device)
        offsets.zero_()
        indices.fill_(NULL_NODE_INDEX)
        elements.fill_(NULL_ELEMENT_INDEX)
        local.zero_()

        @cache.dynamic_kernel(
            suffix=(domain.name, topology.name, partition.name, "compact_pic_restriction"),
            kernel_options={"max_unroll": 8, "enable_backward": False},
        )
        def mark_point_incidence(
            element_arg: domain.ElementArg,
            domain_index_arg: domain.ElementIndexArg,
            topo_arg: topology.TopologyArg,
            partition_arg: partition.PartitionArg,
            node_element: wp.array[int],
            node_local: wp.array[int],
            counts: wp.array[int],
        ):
            cell = wp.tid()
            element = domain.element_index(domain_index_arg, cell)
            if element != NULL_ELEMENT_INDEX:
                # This preserves the original explicit picN clipping semantics.
                points = topology.element_node_count(element_arg, topo_arg, element)
                for local_node in range(points):
                    space_node = topology.element_node_index(element_arg, topo_arg, element, local_node)
                    partition_node = partition.partition_node_index(partition_arg, space_node)
                    # NULL_NODE_INDEX is INT_MAX, so check before adding one.
                    if partition_node >= 0 and partition_node < counts.shape[0] - 1:
                        counts[partition_node + 1] = 1
                        node_element[partition_node] = cell
                        node_local[partition_node] = local_node

        wp.launch(
            mark_point_incidence,
            dim=domain.element_count(),
            inputs=[
                domain.element_arg_value(device),
                domain.element_index_arg_value(device),
                topology.topo_arg_value(device),
                partition.partition_arg_value(device),
                node_element,
                node_local,
                offsets,
            ],
            device=device,
        )
        array_scan(offsets, offsets, inclusive=True)
        wp.launch(
            SpaceRestriction._finish_point_restriction,
            dim=max(1, node_count),
            inputs=[offsets, node_element, node_local, indices, elements, local, count],
            device=device,
        )
        node_element.release()
        node_local.release()

    @wp.kernel
    def _finish_point_restriction(
        offsets: wp.array[int],
        node_element: wp.array[int],
        node_local_index: wp.array[int],
        partition_indices: wp.array[int],
        element_indices: wp.array[int],
        local_indices: wp.array[int],
        node_count: wp.array[int],
    ):
        n = wp.tid()
        if n == 0:
            node_count[0] = offsets[offsets.shape[0] - 1]
        if n + 1 < offsets.shape[0] and offsets[n + 1] > offsets[n]:
            destination = offsets[n]
            partition_indices[destination] = n
            element_indices[destination] = node_element[n]
            local_indices[destination] = node_local_index[n]

    def node_count_sync(self) -> int:
        """Ensure that the node count is synchronized with the device and return it."""
        if self._node_count_dev is not None:
            self._node_count = int(host_read_at_index(self._node_count_dev, index=0))
            self._node_count_dev = None
        return self.node_count()

    def node_count(self) -> int:
        """Upper bound for the node count, use ``node_count_sync`` to get the actual value."""
        return self._node_count

    def partition_element_offsets(self):
        """Return offsets into the per-node element index arrays."""
        return self._dof_partition_element_offsets

    def node_partition_indices(self):
        """Return partition node indices referenced by this restriction."""
        return self._dof_partition_indices

    def total_node_element_count(self):
        """Return the total number of node-element references."""
        return self._dof_element_indices.size

    @wp.struct
    class NodeArg:
        """Structure containing node restriction indices for device functions."""

        dof_element_offsets: wp.array(dtype=int)
        dof_element_indices: wp.array(dtype=int)
        dof_partition_indices: wp.array(dtype=int)
        dof_indices_in_element: wp.array(dtype=int)

    @cache.cached_arg_value
    def node_arg_value(self, device):
        """Return the node argument structure for device functions."""
        arg = SpaceRestriction.NodeArg()
        self.fill_node_arg(arg, device)
        return arg

    def fill_node_arg(self, arg: NodeArg, device):
        """Fill node arguments for device functions."""
        arg.dof_element_offsets = self._dof_partition_element_offsets.to(device)
        arg.dof_element_indices = self._dof_element_indices.to(device)
        arg.dof_partition_indices = self._dof_partition_indices.to(device)
        arg.dof_indices_in_element = self._dof_indices_in_element.to(device)

    @wp.func
    def node_partition_index(args: NodeArg, restriction_node_index: int):
        """Return the partition node index for a restriction node."""
        return args.dof_partition_indices[restriction_node_index]

    @wp.func
    def node_partition_index_from_element_offset(args: NodeArg, element_offset: int):
        """Return the partition node index for a given element offset."""
        return wp.lower_bound(args.dof_element_offsets, element_offset + 1) - 1

    @wp.func
    def node_element_range(args: NodeArg, partition_node_index: int):
        """Return the element offset range for a partition node."""
        return args.dof_element_offsets[partition_node_index], args.dof_element_offsets[partition_node_index + 1]

    @wp.func
    def node_element_index(args: NodeArg, node_element_offset: int):
        """Return the node-element index pair for a given element offset."""
        domain_element_index = args.dof_element_indices[node_element_offset]
        index_in_element = args.dof_indices_in_element[node_element_offset]
        return NodeElementIndex(domain_element_index, index_in_element)

    @wp.kernel
    def _split_vertex_element_index(
        nodes_per_element: int,
        partition_node_count: int,
        dof_element_offsets: wp.array(dtype=int),
        sorted_indices: wp.array(dtype=int),
        node_element_index: wp.array(dtype=int),
        node_index_in_element: wp.array(dtype=int),
    ):
        offset = wp.tid()

        # If some elements have less than nodes_per_element real nodes, we will have
        # invalid elements at the end of the sorted array. Make sure to mark them as such
        last_offset = dof_element_offsets[partition_node_count]
        if offset >= last_offset:
            node_element_index[offset] = NULL_ELEMENT_INDEX
            node_index_in_element[offset] = 0
            return

        flat_idx = sorted_indices[offset]
        element_index = flat_idx // nodes_per_element
        node_element_index[offset] = element_index
        node_index_in_element[offset] = flat_idx - nodes_per_element * element_index
