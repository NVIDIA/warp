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

from typing import Optional

import warp as wp
from warp._src.fem import cache
from warp._src.fem.domain import GeometryDomain
from warp._src.fem.types import NULL_ELEMENT_INDEX, NULL_NODE_INDEX, NodeElementIndex
from warp._src.fem.utils import compress_node_indices, host_read_at_index

from .partition import SpacePartition

_wp_module_name_ = "warp.fem.space.restriction"

wp.set_module_options({"enable_backward": False})


class SpaceRestriction:
    """Restriction of a space partition to a given GeometryDomain"""

    def __init__(
        self,
        space_partition: SpacePartition,
        domain: GeometryDomain,
        device=None,
        temporary_store: cache.TemporaryStore = None,
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
        """Number of unique partition node indices (owned temporary borrowed from cache; released once synchronized or on destruction)"""
        self._dof_partition_indices: wp.array = None
        """Array of unique partition node indices (owned temporary borrowed from cache; released on resize or destruction)"""

        self._dof_partition_element_offsets: wp.array = None
        """Mapping from partition node to offset in the per-node element indices array (owned temporary borrowed from cache; released on resize/destruction)"""
        self._dof_element_indices: wp.array = None
        """Concatenation of neighboring elements indices for each partition node (owned temporary borrowed from cache; reused across rebuilds)"""
        self._dof_indices_in_element: wp.array = None
        """Concatenation of node index in element for each partition node (owned temporary borrowed from cache; reused across rebuilds)"""

        self.rebuild(device=device, temporary_store=temporary_store)

    def __del__(self):
        # SpaceRestriction instances are not expected to participate in reference cycles, so explicitly
        # releasing owned temporaries here is safe and prevents holding onto cache buffers until GC runs.
        if hasattr(self, "_release_owned_temporaries"):
            self._release_owned_temporaries()

    def rebuild(self, device: Optional = None, temporary_store: Optional[cache.TemporaryStore] = None):
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

            if element_index == NULL_ELEMENT_INDEX:
                element_node_count = 0
            else:
                element_node_count = self.space_topology.element_node_count(element_arg, topo_arg, element_index)

            for n in range(element_node_count):
                space_nidx = self.space_topology.element_node_index(element_arg, topo_arg, element_index, n)
                partition_nidx = self.space_partition.partition_node_index(partition_arg, space_nidx)
                element_node_indices[domain_element_index, n] = partition_nidx
            for n in range(element_node_count, element_node_indices.shape[1]):
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
        flattened_node_indices = element_node_indices.flatten()
        (
            new_partition_element_offsets,
            node_array_indices,
            new_node_count_dev,
            new_partition_indices,
        ) = compress_node_indices(
            self.space_partition.node_count(),
            flattened_node_indices,
            node_offsets=self._dof_partition_element_offsets,
            unique_node_count=self._node_count_dev,
            unique_node_indices=self._dof_partition_indices,
            return_unique_nodes=True,
            temporary_store=temporary_store,
        )

        self._replace_owned_temporary("_dof_partition_element_offsets", new_partition_element_offsets)
        self._replace_owned_temporary("_node_count_dev", new_node_count_dev)
        self._replace_owned_temporary("_dof_partition_indices", new_partition_indices)

        # Extract element index and index in element
        if self._dof_element_indices is None or self._dof_element_indices.shape != flattened_node_indices.shape:
            new_dof_element_indices = cache.borrow_temporary_like(flattened_node_indices, temporary_store)
            new_dof_indices_in_element = cache.borrow_temporary_like(flattened_node_indices, temporary_store)

            self._replace_owned_temporary("_dof_element_indices", new_dof_element_indices)
            self._replace_owned_temporary("_dof_indices_in_element", new_dof_indices_in_element)

        wp.launch(
            kernel=SpaceRestriction._split_vertex_element_index,
            dim=flattened_node_indices.shape,
            inputs=[
                max_nodes_per_element,
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

    def node_count_sync(self) -> int:
        """Ensures that the node count is synchronized with the device and returns it"""
        if self._node_count_dev is not None:
            self._node_count = int(host_read_at_index(self._node_count_dev, index=0))
            if hasattr(self._node_count_dev, "release"):
                self._node_count_dev.release()
            self._node_count_dev = None
        return self.node_count()

    def node_count(self) -> int:
        """Upper bound for the node count, use `node_count_sync` to get the actual value"""
        return self._node_count

    def partition_element_offsets(self):
        return self._dof_partition_element_offsets

    def node_partition_indices(self):
        return self._dof_partition_indices

    def total_node_element_count(self):
        return self._dof_element_indices.size

    @wp.struct
    class NodeArg:
        dof_element_offsets: wp.array(dtype=int)
        dof_element_indices: wp.array(dtype=int)
        dof_partition_indices: wp.array(dtype=int)
        dof_indices_in_element: wp.array(dtype=int)

    @cache.cached_arg_value
    def node_arg_value(self, device):
        arg = SpaceRestriction.NodeArg()
        self.fill_node_arg(arg, device)
        return arg

    def fill_node_arg(self, arg: NodeArg, device):
        arg.dof_element_offsets = self._dof_partition_element_offsets.to(device)
        arg.dof_element_indices = self._dof_element_indices.to(device)
        arg.dof_partition_indices = self._dof_partition_indices.to(device)
        arg.dof_indices_in_element = self._dof_indices_in_element.to(device)

    @wp.func
    def node_partition_index(args: NodeArg, restriction_node_index: int):
        return args.dof_partition_indices[restriction_node_index]

    @wp.func
    def node_partition_index_from_element_offset(args: NodeArg, element_offset: int):
        return wp.lower_bound(args.dof_element_offsets, element_offset + 1) - 1

    @wp.func
    def node_element_range(args: NodeArg, partition_node_index: int):
        return args.dof_element_offsets[partition_node_index], args.dof_element_offsets[partition_node_index + 1]

    @wp.func
    def node_element_index(args: NodeArg, node_element_offset: int):
        domain_element_index = args.dof_element_indices[node_element_offset]
        index_in_element = args.dof_indices_in_element[node_element_offset]
        return NodeElementIndex(domain_element_index, index_in_element)

    @wp.kernel
    def _split_vertex_element_index(
        vertex_per_element: int,
        sorted_indices: wp.array(dtype=int),
        vertex_element_index: wp.array(dtype=int),
        vertex_index_in_element: wp.array(dtype=int),
    ):
        idx = sorted_indices[wp.tid()]
        element_index = idx // vertex_per_element
        vertex_element_index[wp.tid()] = element_index
        vertex_index_in_element[wp.tid()] = idx - vertex_per_element * element_index

    def _replace_owned_temporary(self, attr_name: str, new_value):
        """Return previously owned temporaries to the cache before overwriting them."""
        if hasattr(self, attr_name):
            old_value = getattr(self, attr_name)
            if old_value is not None and old_value is not new_value and hasattr(old_value, "release"):
                old_value.release()
        setattr(self, attr_name, new_value)

    def _release_owned_temporaries(self):
        for attr in (
            "_dof_partition_element_offsets",
            "_dof_partition_indices",
            "_dof_element_indices",
            "_dof_indices_in_element",
            "_node_count_dev",
        ):
            if hasattr(self, attr):
                self._replace_owned_temporary(attr, None)
