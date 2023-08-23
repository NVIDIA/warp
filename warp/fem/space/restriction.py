import warp as wp

from warp.fem.domain import GeometryDomain
from warp.fem.types import NodeElementIndex
from warp.fem.utils import compress_node_indices

from .function_space import FunctionSpace
from .partition import SpacePartition

wp.set_module_options({"enable_backward": False})


class SpaceRestriction:
    """Restriction of a space to a given GeometryDomain"""

    def __init__(
        self,
        space: FunctionSpace,
        domain: GeometryDomain,
        space_partition: SpacePartition,
        device=None,
    ):
        if domain.dimension() == space.DIMENSION - 1:
            space = space.trace()

        if domain.dimension() != space.DIMENSION:
            raise ValueError("Incompatible space and domain dimensions")

        self.space = space
        self.space_partition = space_partition
        self.domain = domain

        self._compute_node_element_indices(device=device)

    def _compute_node_element_indices(self, device):
        from warp.fem import cache

        NODES_PER_ELEMENT = self.space.NODES_PER_ELEMENT

        def fill_element_node_indices_fn(
            domain_index_arg: self.domain.ElementIndexArg,
            space_arg: self.space.SpaceArg,
            partition_arg: self.space_partition.PartitionArg,
            element_node_indices: wp.array2d(dtype=int),
        ):
            domain_element_index = wp.tid()
            element_index = self.domain.element_index(domain_index_arg, domain_element_index)
            for n in range(NODES_PER_ELEMENT):
                space_nidx = self.space.element_node_index(space_arg, element_index, n)
                partition_nidx = self.space_partition.partition_node_index(partition_arg, space_nidx)
                element_node_indices[domain_element_index, n] = partition_nidx

        fill_element_node_indices = cache.get_kernel(
            fill_element_node_indices_fn,
            suffix=f"{self.domain.name}_{self.space.name}_{self.space_partition.name}",
        )

        element_node_indices = wp.empty(
            shape=(self.domain.element_count(), NODES_PER_ELEMENT),
            dtype=int,
            device=device,
        )
        wp.launch(
            dim=element_node_indices.shape[0],
            kernel=fill_element_node_indices,
            inputs=[
                self.domain.element_index_arg_value(device),
                self.space.space_arg_value(device),
                self.space_partition.partition_arg_value(device),
                element_node_indices,
            ],
            device=device,
        )

        # Build compressed map from node to element indices
        flattened_node_indices = element_node_indices.reshape(element_node_indices.size)
        (
            self._dof_partition_element_offsets,
            node_array_indices,
            self._node_count,
            self._dof_partition_indices,
        ) = compress_node_indices(self.space_partition.node_count(), flattened_node_indices)

        # Extract element index and index in element
        self._dof_element_indices = wp.empty_like(flattened_node_indices)
        self._dof_indices_in_element = wp.empty_like(flattened_node_indices)
        wp.launch(
            kernel=SpaceRestriction._split_vertex_element_index,
            dim=flattened_node_indices.shape,
            inputs=[NODES_PER_ELEMENT, node_array_indices, self._dof_element_indices, self._dof_indices_in_element],
            device=flattened_node_indices.device,
        )

    def node_count(self):
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

    def node_arg(self, device):
        arg = SpaceRestriction.NodeArg()
        arg.dof_element_offsets = self._dof_partition_element_offsets.to(device)
        arg.dof_element_indices = self._dof_element_indices.to(device)
        arg.dof_partition_indices = self._dof_partition_indices.to(device)
        arg.dof_indices_in_element = self._dof_indices_in_element.to(device)
        return arg

    @wp.func
    def node_partition_index(args: NodeArg, node_index: int):
        return args.dof_partition_indices[node_index]

    @wp.func
    def node_element_count(args: NodeArg, node_index: int):
        partition_node_index = SpaceRestriction.node_partition_index(args, node_index)
        return args.dof_element_offsets[partition_node_index + 1] - args.dof_element_offsets[partition_node_index]

    @wp.func
    def node_element_index(args: NodeArg, node_index: int, element_index: int):
        partition_node_index = SpaceRestriction.node_partition_index(args, node_index)
        offset = args.dof_element_offsets[partition_node_index] + element_index
        domain_element_index = args.dof_element_indices[offset]
        index_in_element = args.dof_indices_in_element[offset]
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
