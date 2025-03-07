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

import warp as wp
from warp.fem.cache import TemporaryStore, borrow_temporary, borrow_temporary_like, cached_arg_value
from warp.fem.domain import GeometryDomain
from warp.fem.types import NULL_NODE_INDEX, NodeElementIndex
from warp.fem.utils import compress_node_indices

from .partition import SpacePartition

wp.set_module_options({"enable_backward": False})


class SpaceRestriction:
    """Restriction of a space partition to a given GeometryDomain"""

    def __init__(
        self,
        space_partition: SpacePartition,
        domain: GeometryDomain,
        device=None,
        temporary_store: TemporaryStore = None,
    ):
        space_topology = space_partition.space_topology

        if domain.dimension == space_topology.dimension - 1:
            space_topology = space_topology.trace()

        if domain.dimension != space_topology.dimension:
            raise ValueError("Incompatible space and domain dimensions")

        self.space_partition = space_partition
        self.space_topology = space_topology
        self.domain = domain

        self._compute_node_element_indices(device=device, temporary_store=temporary_store)

    def _compute_node_element_indices(self, device, temporary_store: TemporaryStore):
        from warp.fem import cache

        MAX_NODES_PER_ELEMENT = self.space_topology.MAX_NODES_PER_ELEMENT

        @cache.dynamic_kernel(
            suffix=f"{self.domain.name}_{self.space_topology.name}_{self.space_partition.name}",
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
            element_node_count = self.space_topology.element_node_count(element_arg, topo_arg, element_index)
            for n in range(element_node_count):
                space_nidx = self.space_topology.element_node_index(element_arg, topo_arg, element_index, n)
                partition_nidx = self.space_partition.partition_node_index(partition_arg, space_nidx)
                element_node_indices[domain_element_index, n] = partition_nidx
            for n in range(element_node_count, MAX_NODES_PER_ELEMENT):
                element_node_indices[domain_element_index, n] = NULL_NODE_INDEX

        element_node_indices = borrow_temporary(
            temporary_store,
            shape=(self.domain.element_count(), MAX_NODES_PER_ELEMENT),
            dtype=int,
            device=device,
        )
        wp.launch(
            dim=element_node_indices.array.shape[0],
            kernel=fill_element_node_indices,
            inputs=[
                self.domain.element_arg_value(device),
                self.domain.element_index_arg_value(device),
                self.space_topology.topo_arg_value(device),
                self.space_partition.partition_arg_value(device),
                element_node_indices.array,
            ],
            device=device,
        )

        # Build compressed map from node to element indices
        flattened_node_indices = element_node_indices.array.flatten()
        (
            self._dof_partition_element_offsets,
            node_array_indices,
            self._node_count,
            self._dof_partition_indices,
        ) = compress_node_indices(
            self.space_partition.node_count(),
            flattened_node_indices,
            return_unique_nodes=True,
            temporary_store=temporary_store,
        )

        # Extract element index and index in element
        self._dof_element_indices = borrow_temporary_like(flattened_node_indices, temporary_store)
        self._dof_indices_in_element = borrow_temporary_like(flattened_node_indices, temporary_store)
        wp.launch(
            kernel=SpaceRestriction._split_vertex_element_index,
            dim=flattened_node_indices.shape,
            inputs=[
                MAX_NODES_PER_ELEMENT,
                node_array_indices.array,
                self._dof_element_indices.array,
                self._dof_indices_in_element.array,
            ],
            device=flattened_node_indices.device,
        )

        node_array_indices.release()

    def node_count(self):
        return self._node_count

    def partition_element_offsets(self):
        return self._dof_partition_element_offsets.array

    def node_partition_indices(self):
        return self._dof_partition_indices.array

    def total_node_element_count(self):
        return self._dof_element_indices.array.size

    @wp.struct
    class NodeArg:
        dof_element_offsets: wp.array(dtype=int)
        dof_element_indices: wp.array(dtype=int)
        dof_partition_indices: wp.array(dtype=int)
        dof_indices_in_element: wp.array(dtype=int)

    @cached_arg_value
    def node_arg(self, device):
        arg = SpaceRestriction.NodeArg()
        arg.dof_element_offsets = self._dof_partition_element_offsets.array.to(device)
        arg.dof_element_indices = self._dof_element_indices.array.to(device)
        arg.dof_partition_indices = self._dof_partition_indices.array.to(device)
        arg.dof_indices_in_element = self._dof_indices_in_element.array.to(device)
        return arg

    @wp.func
    def node_partition_index(args: NodeArg, restriction_node_index: int):
        return args.dof_partition_indices[restriction_node_index]

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
