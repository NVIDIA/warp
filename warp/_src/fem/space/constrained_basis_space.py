# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basis spaces constrained by a sparse prolongation, such as conforming adaptive or periodic spaces."""

from functools import cached_property
from typing import Any, ClassVar

import warp as wp
from warp._src.fem import cache
from warp._src.fem.types import (
    NULL_ELEMENT_INDEX,
    NULL_NODE_INDEX,
    OUTSIDE,
    ElementIndex,
    QuadraturePointIndex,
)
from warp._src.sparse import BsrMatrix, _bsr_row_end, bsr_diag, bsr_get_diag, bsr_mm, bsr_zeros
from warp._src.types import (
    float_types,
    get_type_code,
    type_is_matrix,
    type_is_vector,
    type_scalar_type,
    type_size,
    type_to_warp,
)
from warp._src.utils import array_scan

from .basis_space import BasisSpace, ShapeBasisSpace, TraceBasisSpace
from .constraints import (
    _NO_KEY,
    _add_sorted_range,
    _find_sorted,
    _find_sorted_range,
    _hanging_node_constraints,
    _insert_sorted_key,
    _Prolongation,
    _prolongation_matrix,
)
from .shape import ShapeFunction
from .topology import SpaceTopology


@wp.struct
class ConstrainedTopologyArg:
    cell_counts: wp.array(dtype=int)
    cell_nodes: wp.array2d(dtype=int)
    side_counts: wp.array(dtype=int)
    side_nodes: wp.array2d(dtype=int)
    side_inner_slots: wp.array2d(dtype=int)
    side_outer_slots: wp.array2d(dtype=int)


def _make_constrained_basis_arg(scalar_type, coords_type):
    @cache.dynamic_struct(suffix=(scalar_type, coords_type))
    class ConstrainedBasisArg:
        coefficients: wp.array3d(dtype=scalar_type)
        node_coords: wp.array2d(dtype=coords_type)
        copy_coords: wp.array2d(dtype=coords_type)
        side_copy_coords: wp.array2d(dtype=coords_type)
        owner_cell: wp.array(dtype=int)
        owner_cell_slot: wp.array(dtype=int)
        owner_side: wp.array(dtype=int)
        owner_side_slot: wp.array(dtype=int)
        owner_side_coords: wp.array(dtype=coords_type)

    return ConstrainedBasisArg


class ConstrainedSpaceTopology(SpaceTopology):
    """Cell topology whose local slots enumerate the independent nodes acting on each cell."""

    TopologyArg = ConstrainedTopologyArg

    _dynamic_attribute_constructors: ClassVar = {
        "element_node_count": lambda obj: obj._make_element_node_count(),
        "element_node_index": lambda obj: obj._make_element_node_index(),
        "side_neighbor_node_counts": lambda obj: obj._make_side_neighbor_node_counts(),
    }

    def __init__(
        self,
        raw_topology: SpaceTopology,
        cell_nodes: wp.array2d[int],
        cell_counts: wp.array[int],
        side_nodes: wp.array2d[int],
        side_counts: wp.array[int],
        side_inner_slots: wp.array2d[int],
        side_outer_slots: wp.array2d[int],
        independent_count: int,
    ):
        self.raw_topology = raw_topology
        self._independent_count = independent_count
        self._cell_counts = cell_counts
        self._cell_nodes = cell_nodes
        self._side_counts = side_counts
        self._side_nodes = side_nodes
        self._side_inner_slots = side_inner_slots
        self._side_outer_slots = side_outer_slots
        self._side_max_nodes = side_nodes.shape[1]
        super().__init__(raw_topology.geometry, cell_nodes.shape[1])
        cache.setup_dynamic_attributes(self, cls=__class__)
        self._trace = ConstrainedTraceSpaceTopology(self)

    @cached_property
    def name(self):
        return f"{self.raw_topology.name}_Constrained_{self.MAX_NODES_PER_ELEMENT}_{self._side_max_nodes}"

    def node_count(self) -> int:
        return self._independent_count

    def fill_topo_arg(self, arg, device):
        arg.cell_counts = self._cell_counts.to(device)
        arg.cell_nodes = self._cell_nodes.to(device)
        arg.side_counts = self._side_counts.to(device)
        arg.side_nodes = self._side_nodes.to(device)
        arg.side_inner_slots = self._side_inner_slots.to(device)
        arg.side_outer_slots = self._side_outer_slots.to(device)

    def trace(self):
        """Return the eagerly built side topology."""
        return self._trace

    def _make_element_node_count(self):
        @cache.dynamic_func(suffix=self.name)
        def element_node_count(geo_arg: self.geometry.CellArg, topo_arg: self.TopologyArg, element_index: ElementIndex):
            return topo_arg.cell_counts[element_index]

        return element_node_count

    def _make_element_node_index(self):
        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            geo_arg: self.geometry.CellArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return topo_arg.cell_nodes[element_index, node_index_in_elt]

        return element_node_index

    def _make_side_neighbor_node_counts(self):
        @cache.dynamic_func(suffix=self.name)
        def side_neighbor_node_counts(
            side_arg: self.geometry.SideArg, topo_arg: self.TopologyArg, side_index: ElementIndex
        ):
            inner = int(0)
            outer = int(0)
            for n in range(topo_arg.side_counts[side_index]):
                if topo_arg.side_inner_slots[side_index, n] != NULL_NODE_INDEX:
                    inner += 1
                if topo_arg.side_outer_slots[side_index, n] != NULL_NODE_INDEX:
                    outer += 1
            return inner, outer

        return side_neighbor_node_counts


class ConstrainedTraceSpaceTopology(SpaceTopology):
    """Side topology listing each independent node of both adjacent cells once, with inner and outer slots."""

    TopologyArg = ConstrainedTopologyArg

    _dynamic_attribute_constructors: ClassVar = {
        "element_node_count": lambda obj: obj._make_element_node_count(),
        "element_node_index": lambda obj: obj._make_element_node_index(),
        "inner_cell_index": lambda obj: obj._make_inner_cell_index(),
        "outer_cell_index": lambda obj: obj._make_outer_cell_index(),
        "neighbor_cell_index": lambda obj: obj._make_neighbor_cell_index(),
    }

    def __init__(self, topology: ConstrainedSpaceTopology):
        self._topo = topology
        super().__init__(topology.geometry, topology._side_max_nodes)
        self.dimension = topology.dimension - 1
        self.ElementArg = topology.geometry.SideArg
        self.topo_arg_value = topology.topo_arg_value
        self.fill_topo_arg = topology.fill_topo_arg
        cache.setup_dynamic_attributes(self, cls=__class__)

    @cached_property
    def name(self):
        return f"{self._topo.name}_CompactTrace"

    def node_count(self):
        return self._topo.node_count()

    def full_space_topology(self):
        return self._topo

    def element_node_indices(self, out: wp.array | None = None) -> wp.array:
        """Return independent node indices for each side."""

        @cache.dynamic_kernel(suffix=self.name)
        def fill_side_node_indices(
            side_arg: self.geometry.SideArg, topo_arg: self.TopologyArg, node_indices: wp.array2d(dtype=int)
        ):
            side = wp.tid()
            for n in range(node_indices.shape[1]):
                node_indices[side, n] = NULL_NODE_INDEX
            for n in range(topo_arg.side_counts[side]):
                node_indices[side, n] = topo_arg.side_nodes[side, n]

        shape = (self.geometry.side_count(), self.MAX_NODES_PER_ELEMENT)
        if out is None:
            out = wp.empty(shape=shape, dtype=int, device=self._topo._cell_nodes.device)
        elif out.shape != shape or out.dtype != wp.int32:
            raise ValueError(f"Output side node indices must have shape {shape} and data type 'int32'")
        wp.launch(
            fill_side_node_indices,
            dim=shape[0],
            inputs=[self.geometry.side_arg_value(device=out.device), self.topo_arg_value(device=out.device), out],
            device=out.device,
        )
        return out

    def _make_element_node_count(self):
        @cache.dynamic_func(suffix=self.name)
        def element_node_count(
            side_arg: self.geometry.SideArg, topo_arg: self.TopologyArg, element_index: ElementIndex
        ):
            return topo_arg.side_counts[element_index]

        return element_node_count

    def _make_element_node_index(self):
        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            side_arg: self.geometry.SideArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return topo_arg.side_nodes[element_index, node_index_in_elt]

        return element_node_index

    def _make_inner_cell_index(self):
        @cache.dynamic_func(suffix=self.name)
        def inner_cell_index(
            side_arg: self.geometry.SideArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            slot = topo_arg.side_inner_slots[element_index, node_index_in_elt]
            if slot == NULL_NODE_INDEX:
                return NULL_ELEMENT_INDEX, NULL_NODE_INDEX
            return self.geometry.side_inner_cell_index(side_arg, element_index), slot

        return inner_cell_index

    def _make_outer_cell_index(self):
        @cache.dynamic_func(suffix=self.name)
        def outer_cell_index(
            side_arg: self.geometry.SideArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            slot = topo_arg.side_outer_slots[element_index, node_index_in_elt]
            if slot == NULL_NODE_INDEX:
                return NULL_ELEMENT_INDEX, NULL_NODE_INDEX
            return self.geometry.side_outer_cell_index(side_arg, element_index), slot

        return outer_cell_index

    def _make_neighbor_cell_index(self):
        @cache.dynamic_func(suffix=self.name)
        def neighbor_cell_index(
            side_arg: self.geometry.SideArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            inner_slot = topo_arg.side_inner_slots[element_index, node_index_in_elt]
            if inner_slot != NULL_NODE_INDEX:
                return self.geometry.side_inner_cell_index(side_arg, element_index), inner_slot
            return (
                self.geometry.side_outer_cell_index(side_arg, element_index),
                topo_arg.side_outer_slots[element_index, node_index_in_elt],
            )

        return neighbor_cell_index


@wp.kernel(enable_backward=False)
def _gather_independent_values(raw_values: wp.array[Any], independent_raw_nodes: wp.array[int], values: wp.array[Any]):
    node = wp.tid()
    values[node] = raw_values[independent_raw_nodes[node]]


@wp.kernel(enable_backward=False)
def _gather_independent_rows(
    raw_values: wp.array2d[Any], independent_raw_nodes: wp.array[int], values: wp.array2d[Any]
):
    node, component = wp.tid()
    values[node, component] = raw_values[independent_raw_nodes[node], component]


class ConstrainedBasisSpace(BasisSpace):
    """Raw basis transformed by a cell-local prolongation into independent nodes.

    Built by :func:`make_constrained_basis_space` or
    :func:`make_conforming_basis_space`. Each independent node is anchored at
    a raw node whose prolongation row is the unit vector of that node, which
    gives its position and its ``at_node()`` samples.
    """

    supports_nodal_assembly = False
    supports_nodal_quadrature = False
    moves_node_samples = True

    _dynamic_attribute_constructors: ClassVar = {
        "node_coords_in_element": lambda obj: obj.make_node_coords_in_element(),
        "node_sample": lambda obj: obj.make_node_sample(),
        "node_quadrature_weight": lambda obj: obj.make_node_quadrature_weight(),
        "element_inner_weight": lambda obj: obj.make_element_inner_weight(),
        "element_inner_weight_gradient": lambda obj: obj.make_element_inner_weight_gradient(),
        "element_outer_weight": lambda obj: obj.make_element_outer_weight(),
        "element_outer_weight_gradient": lambda obj: obj.make_element_outer_weight_gradient(),
    }

    def __init__(
        self,
        raw_basis: ShapeBasisSpace,
        topology: ConstrainedSpaceTopology,
        coefficients: wp.array3d,
        node_coords: wp.array2d,
        copy_coords: wp.array2d,
        side_copy_coords: wp.array2d,
        independent_raw_nodes: wp.array[int],
        prolongation_offsets: wp.array[int],
        prolongation_indices: wp.array[int],
        prolongation_weights: wp.array,
        owner_cell: wp.array[int],
        owner_cell_slot: wp.array[int],
        owner_side: wp.array[int],
        owner_side_slot: wp.array[int],
        owner_side_coords: wp.array,
    ):
        self.raw_basis = raw_basis
        self.ORDER = raw_basis.ORDER
        self._coefficients = coefficients
        self._node_coords = node_coords
        self._copy_coords = copy_coords
        self._side_copy_coords = side_copy_coords
        self.independent_raw_nodes = independent_raw_nodes
        """Anchor raw node of each independent node, whose prolongation row is that node's unit vector."""
        self.prolongation_offsets = prolongation_offsets
        """CSR row offsets of the raw-to-independent prolongation, one row per raw node."""
        self.prolongation_indices = prolongation_indices
        """Independent node index of each prolongation entry, sorted within each row."""
        self.prolongation_weights = prolongation_weights
        """Weight of each prolongation entry, in the geometry scalar type."""
        self._owner_cell = owner_cell
        self._owner_cell_slot = owner_cell_slot
        self._owner_side = owner_side
        self._owner_side_slot = owner_side_slot
        self._owner_side_coords = owner_side_coords
        self.BasisArg = _make_constrained_basis_arg(raw_basis.geometry.scalar_type, raw_basis.geometry.coords_type)
        super().__init__(topology)
        self._trace = ConstrainedTraceBasisSpace(self)

    @cached_property
    def name(self):
        return f"{self.topology.name}_Basis"

    @property
    def value(self):
        return ShapeFunction.Value.Scalar

    def fill_basis_arg(self, arg, device):
        arg.coefficients = self._coefficients.to(device)
        arg.node_coords = self._node_coords.to(device)
        arg.copy_coords = self._copy_coords.to(device)
        arg.side_copy_coords = self._side_copy_coords.to(device)
        arg.owner_cell = self._owner_cell.to(device)
        arg.owner_cell_slot = self._owner_cell_slot.to(device)
        arg.owner_side = self._owner_side.to(device)
        arg.owner_side_slot = self._owner_side_slot.to(device)
        arg.owner_side_coords = self._owner_side_coords.to(device)

    def trace(self):
        """Return the eagerly prepared trace basis."""
        return self._trace

    def node_positions(self, out: wp.array | None = None) -> wp.array:
        pos_type = cache.cached_vec_type(length=self.geometry.dimension, dtype=self.geometry.scalar_type)
        raw_positions = self.raw_basis.node_positions(
            out=wp.empty(
                shape=(self.raw_basis.topology.node_count(),), dtype=pos_type, device=self.independent_raw_nodes.device
            )
        )

        @cache.dynamic_kernel(suffix=self.name, kernel_options={"enable_backward": False})
        def gather_positions(
            raw: wp.array(dtype=raw_positions.dtype),
            raw_indices: wp.array(dtype=int),
            positions: wp.array(dtype=raw_positions.dtype),
        ):
            i = wp.tid()
            positions[i] = raw[raw_indices[i]]

        if out is None:
            out = wp.empty(shape=(self.topology.node_count(),), dtype=raw_positions.dtype, device=raw_positions.device)
        elif out.shape != (self.topology.node_count(),) or out.dtype != raw_positions.dtype:
            raise ValueError("Output node positions must match the independent node count and coordinate type")
        wp.launch(
            gather_positions,
            dim=self.topology.node_count(),
            inputs=[raw_positions.to(out.device), self.independent_raw_nodes.to(out.device), out],
            device=out.device,
        )
        return out

    def make_node_coords_in_element(self):
        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return basis_arg.node_coords[element_index, node_index_in_elt]

        return node_coords_in_element

    def make_node_sample(self):
        @cache.dynamic_func(suffix=self.name)
        def node_sample(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            # Stay in the current cell when it holds the anchor or a periodic copy of the node
            coords = basis_arg.copy_coords[element_index, node_index_in_elt]
            if coords[0] != OUTSIDE:
                return element_index, coords, node_index_in_elt
            node = topo_arg.cell_nodes[element_index, node_index_in_elt]
            owner = basis_arg.owner_cell[node]
            owner_slot = basis_arg.owner_cell_slot[node]
            return owner, basis_arg.node_coords[owner, owner_slot], owner_slot

        return node_sample

    def make_node_quadrature_weight(self):
        return None

    def make_trace_node_quadrature_weight(self, trace_basis):
        return None

    def make_element_inner_weight(self):
        raw_weight = self.raw_basis.shape.make_element_inner_weight()
        scalar = self.geometry.scalar_type
        raw_node_count = self.raw_basis.topology.MAX_NODES_PER_ELEMENT

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Any,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            weight = scalar(0.0)
            for raw_slot in range(raw_node_count):
                weight += basis_arg.coefficients[element_index, node_index_in_elt, raw_slot] * raw_weight(
                    coords, raw_slot
                )
            return weight

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        raw_gradient = self.raw_basis.shape.make_element_inner_weight_gradient()
        gradient_type = self.weight_gradient_type
        scalar = self.geometry.scalar_type
        raw_node_count = self.raw_basis.topology.MAX_NODES_PER_ELEMENT

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Any,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            gradient = gradient_type(scalar(0.0))
            for raw_slot in range(raw_node_count):
                gradient += basis_arg.coefficients[element_index, node_index_in_elt, raw_slot] * raw_gradient(
                    coords, raw_slot
                )
            return gradient

        return element_inner_weight_gradient

    def prolongation_matrix(self, dof_dtype: type | None = None) -> BsrMatrix:
        """Return the sparse prolongation matrix ``P`` from independent nodes to raw nodes.

        ``P`` has one row of blocks per node of ``raw_basis`` and one column
        of blocks per independent node of this basis. For degrees of freedom
        ``u_independent`` of a field on this basis, ``u_raw = P @ u_independent``
        gives the matching degrees of freedom on the raw basis: independent nodes
        keep their value, and constrained nodes, such as hanging nodes or the
        high copies of periodic nodes, take the weighted sum of their masters.

        The transpose reduces raw-space quantities: ``P.T @ f_raw`` maps a raw
        right-hand side to the independent nodes, and ``P.T @ K_raw @ P`` reduces
        a raw system matrix. :func:`warp.sparse.bsr_transposed` builds ``P.T``
        explicitly as a restriction matrix.

        The returned matrix owns copies of the prolongation arrays, so it may be
        modified without affecting this basis.

        Args:
            dof_dtype: Degree-of-freedom type of the fields ``P`` acts on.
              ``None`` uses the geometry scalar type. A floating-point scalar
              type gives scalar blocks. A vector or matrix type with ``n``
              components gives ``n x n`` blocks equal to the weight times the
              identity.

        Returns:
            A compact BSR matrix with ``raw_basis.topology.node_count()`` rows
            and ``topology.node_count()`` columns of blocks, on the device of
            this basis.
        """
        if dof_dtype is None:
            dof_dtype = self.geometry.scalar_type
        dof_dtype = type_to_warp(dof_dtype)
        if dof_dtype in float_types:
            scalar = dof_dtype
            block_type = dof_dtype
        elif (type_is_vector(dof_dtype) or type_is_matrix(dof_dtype)) and type_scalar_type(dof_dtype) in float_types:
            scalar = type_scalar_type(dof_dtype)
            block_type = cache.cached_mat_type(shape=(type_size(dof_dtype), type_size(dof_dtype)), dtype=scalar)
        else:
            raise TypeError(
                f"Prolongation degree-of-freedom type must be a floating-point scalar, vector, or matrix type, got {dof_dtype}"
            )

        weight_type = self.prolongation_weights.dtype
        kernel_suffix = f"{get_type_code(weight_type)}_{get_type_code(block_type)}"
        if block_type == scalar:

            @cache.dynamic_kernel(suffix=kernel_suffix, kernel_options={"enable_backward": False})
            def fill_prolongation_blocks(weights: wp.array[weight_type], blocks: wp.array[block_type]):
                k = wp.tid()
                blocks[k] = scalar(weights[k])

        else:
            block_rows = block_type._shape_[0]

            @cache.dynamic_kernel(suffix=kernel_suffix, kernel_options={"enable_backward": False})
            def fill_prolongation_blocks(weights: wp.array[weight_type], blocks: wp.array[block_type]):
                k = wp.tid()
                blocks[k] = scalar(weights[k]) * wp.identity(n=block_rows, dtype=scalar)

        device = self.prolongation_offsets.device
        nnz = self.prolongation_indices.shape[0]
        matrix = bsr_zeros(
            self.raw_basis.topology.node_count(), self.topology.node_count(), block_type=block_type, device=device
        )
        # Rows are already sorted by column, so the CSR arrays are used as is.
        wp.copy(dest=matrix.offsets, src=self.prolongation_offsets)
        matrix.columns = wp.clone(self.prolongation_indices)
        matrix.values = wp.empty(nnz, dtype=block_type, device=device)
        wp.launch(fill_prolongation_blocks, dim=nnz, inputs=[self.prolongation_weights, matrix.values], device=device)
        matrix.notify_nnz_changed(nnz=nnz)
        return matrix

    def reduce_dirichlet_projector(self, raw_projector: BsrMatrix) -> BsrMatrix:
        """Restrict a Dirichlet projector on the raw nodes to the independent nodes.

        Nodal assembly is not available on this basis, but it is on
        ``raw_basis``. Assemble the boundary projector with
        ``assembly="nodal"`` on a function space built from ``raw_basis``,
        then call this method to obtain the projector for the independent
        degrees of freedom. Prescribed values assembled the same way can be
        restricted with :meth:`gather_independent_values`.

        The projector block of each independent node is taken from its raw
        anchor node. A dependent node, such as a hanging node or a periodic
        copy, gets its value from its masters, so its own projector block is
        dropped: a Dirichlet region should contain the masters of the
        dependent nodes it constrains.

        The result is not normalized. Pass it to
        :func:`warp.fem.normalize_dirichlet_projector`, or to
        :func:`warp.fem.project_linear_system` with ``normalize_projector=True``,
        before using it as an orthogonal projector.

        Args:
            raw_projector: Square block-diagonal projector with one row of
              blocks per node of ``raw_basis``, such as one assembled with
              ``assembly="nodal"``. Only its diagonal blocks are read; the
              structure is not checked, and off-diagonal blocks are ignored.

        Returns:
            A diagonal BSR matrix with one block per independent node and the
            block type of ``raw_projector``.

        Raises:
            ValueError: If the shape does not match ``raw_basis``.
        """
        raw_count = self.raw_basis.topology.node_count()
        if raw_projector.nrow != raw_count or raw_projector.ncol != raw_count:
            raise ValueError(
                f"Raw projector must have {raw_count} rows and columns of blocks, "
                f"got {raw_projector.nrow} and {raw_projector.ncol}"
            )

        device = raw_projector.device
        diagonal = bsr_get_diag(raw_projector)
        independent_raw_nodes = self.independent_raw_nodes.to(device)

        independent_count = self.topology.node_count()
        projector_blocks = wp.empty(independent_count, dtype=diagonal.dtype, device=device)
        wp.launch(
            _gather_independent_values,
            dim=independent_count,
            inputs=[diagonal, independent_raw_nodes, projector_blocks],
            device=device,
        )
        return bsr_diag(projector_blocks)

    def gather_independent_values(self, raw_values: wp.array) -> wp.array:
        """Return the entries of a raw nodal array at the independent nodes.

        This picks out the value of each independent node from an array with
        one entry or row per node of ``raw_basis``; values at dependent nodes,
        such as hanging nodes, are dropped. It is the counterpart of :meth:`prolongation_matrix`,
        which expands independent values to all raw nodes.

        A typical use is restricting Dirichlet values assembled with
        ``assembly="nodal"`` on the raw space, alongside
        :meth:`reduce_dirichlet_projector`. Such values must be scaled like the
        projector, as expected by :func:`warp.fem.normalize_dirichlet_projector`
        (for instance, assembled with the same nodal form); interpolated
        values are not scaled that way.

        Args:
            raw_values: One-dimensional array with one entry per node of
              ``raw_basis``, or two-dimensional array with one row per node.

        Returns:
            An array on the device of ``raw_values`` with one entry or row per
            independent node and the data type of ``raw_values``.

        Raises:
            ValueError: If ``raw_values`` does not have one entry or row per
              node of ``raw_basis``, or has more than two dimensions.
        """
        raw_count = self.raw_basis.topology.node_count()
        if raw_values.ndim > 2:
            raise ValueError(f"Raw values must have one or two dimensions, got {raw_values.ndim}")
        if raw_values.shape[0] != raw_count:
            raise ValueError(f"Raw values must have {raw_count} entries, got {raw_values.shape[0]}")

        device = raw_values.device
        values = wp.empty((self.topology.node_count(), *raw_values.shape[1:]), dtype=raw_values.dtype, device=device)
        wp.launch(
            _gather_independent_rows if raw_values.ndim == 2 else _gather_independent_values,
            dim=values.shape,
            inputs=[raw_values, self.independent_raw_nodes.to(device), values],
            device=device,
        )
        return values


class ConstrainedTraceBasisSpace(TraceBasisSpace):
    """Trace of a constrained basis, with each independent node listed once per side."""

    supports_nodal_assembly = False
    supports_nodal_quadrature = False
    moves_node_samples = True

    def node_positions(self, out: wp.array | None = None) -> wp.array:
        """Return positions of the shared independent master nodes."""
        return self._basis.node_positions(out=out)

    def make_node_sample(self):
        @cache.dynamic_func(suffix=self.name)
        def node_sample(
            side_arg: self.geometry.SideArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            # Stay on the current side when it holds the anchor or a periodic copy of the node
            coords = basis_arg.side_copy_coords[element_index, node_index_in_elt]
            if coords[0] != OUTSIDE:
                return element_index, coords, node_index_in_elt
            node = topo_arg.side_nodes[element_index, node_index_in_elt]
            owner = basis_arg.owner_side[node]
            if owner == NULL_ELEMENT_INDEX:
                # The node's anchor lies on no side; its trace is not attached to a side point
                return element_index, basis_arg.owner_side_coords[node], node_index_in_elt
            return owner, basis_arg.owner_side_coords[node], basis_arg.owner_side_slot[node]

        return node_sample


# Construction of a constrained basis from a final prolongation

# Slots of the builder status array. Error slots hold the smallest offending
# column or row (``_NO_KEY`` when there is no error); the others hold flags and
# sizes read back by the host.
_CELL_MAX_COUNT = wp.constant(0)
_SIDE_MAX_COUNT = wp.constant(1)
_EMPTY_COLUMN = wp.constant(2)
_UNANCHORED_COLUMN = wp.constant(3)
_INVALID_ANCHOR = wp.constant(4)
_COLUMN_RANGE_ERROR = wp.constant(5)
_COMPACT_NNZ = wp.constant(6)
_BUILD_STATUS_SIZE = 7

_DEFAULT_MAX_NODES_PER_ELEMENT = 256
"""Default limit on the number of independent nodes acting on one cell."""


def _new_build_status(device) -> wp.array:
    status = wp.empty(_BUILD_STATUS_SIZE, dtype=int, device=device)
    status.fill_(_NO_KEY)
    status[_CELL_MAX_COUNT:_EMPTY_COLUMN].zero_()
    status[_COMPACT_NNZ : _COMPACT_NNZ + 1].zero_()
    return status


@wp.kernel(enable_backward=False)
def _compact_count_kernel(
    row_count: int,
    column_count: int,
    offsets: wp.array[int],
    row_counts: wp.array[int],
    columns: wp.array[int],
    values: wp.array3d[Any],
    counts: wp.array[int],
    status: wp.array[int],
):
    # One extra trailing entry so that exclusive scans also produce totals
    row = wp.tid()
    counts[row] = 0
    if row == row_count:
        return

    count = int(0)
    begin = offsets[row]
    end = _bsr_row_end(offsets, row_counts, row)
    for entry in range(begin, end):
        value = values[entry, 0, 0]
        if value != type(value)(0.0):
            column = columns[entry]
            if column < 0 or column >= column_count:
                wp.atomic_min(status, _COLUMN_RANGE_ERROR, row)
            else:
                # Repeated columns are summed into one entry
                repeated = int(0)
                for previous in range(begin, entry):
                    if columns[previous] == column and values[previous, 0, 0] != type(value)(0.0):
                        repeated = 1
                count += 1 - repeated
    counts[row] = count


@wp.kernel(enable_backward=False)
def _compact_fill_kernel(
    column_count: int,
    offsets: wp.array[int],
    row_counts: wp.array[int],
    columns: wp.array[int],
    values: wp.array3d[Any],
    compact_offsets: wp.array[int],
    compact_columns: wp.array[int],
    compact_weights: wp.array[wp.float64],
):
    row = wp.tid()
    start = compact_offsets[row]
    count = int(0)
    for entry in range(offsets[row], _bsr_row_end(offsets, row_counts, row)):
        value = values[entry, 0, 0]
        column = columns[entry]
        if value != type(value)(0.0) and column >= 0 and column < column_count:
            count = _add_sorted_range(compact_columns, compact_weights, start, count, column, wp.float64(value))


def _compact_prolongation(matrix: BsrMatrix):
    """Return the nonzero entries of a scalar BSR matrix as sorted CSR rows with double-precision weights."""
    device = matrix.device
    row_count = matrix.nrow
    status = _new_build_status(device)
    counts = wp.empty(row_count + 1, dtype=int, device=device)
    offsets = wp.empty(row_count + 1, dtype=int, device=device)
    values = matrix.scalar_values
    wp.launch(
        _compact_count_kernel,
        dim=row_count + 1,
        inputs=[row_count, matrix.ncol, matrix.offsets, matrix.row_counts, matrix.columns, values, counts, status],
        device=device,
    )
    array_scan(counts, offsets, inclusive=False)
    wp.copy(dest=status, src=offsets, dest_offset=_COMPACT_NNZ, src_offset=row_count, count=1)
    host_status = status.numpy()
    if host_status[_COLUMN_RANGE_ERROR] != _NO_KEY:
        raise ValueError(
            f"Prolongation row {int(host_status[_COLUMN_RANGE_ERROR])} refers to a column outside [0, {matrix.ncol})"
        )
    nnz = int(host_status[_COMPACT_NNZ])

    columns = wp.empty(nnz, dtype=int, device=device)
    weights = wp.empty(nnz, dtype=wp.float64, device=device)
    wp.launch(
        _compact_fill_kernel,
        dim=row_count,
        inputs=[matrix.ncol, matrix.offsets, matrix.row_counts, matrix.columns, values, offsets, columns, weights],
        device=device,
    )
    return offsets, columns, weights, nnz


@wp.kernel(enable_backward=False)
def _gather_anchor_kernel(base_anchors: wp.array[int], anchors: wp.array[int]):
    column = wp.tid()
    base_node = anchors[column]
    if base_node >= 0 and base_node < base_anchors.shape[0]:
        anchors[column] = base_anchors[base_node]
    else:
        anchors[column] = -1


@wp.kernel(enable_backward=False)
def _column_usage_kernel(columns: wp.array[int], used: wp.array[int]):
    used[columns[wp.tid()]] = 1


@wp.kernel(enable_backward=False)
def _detect_anchors_kernel(
    offsets: wp.array[int], columns: wp.array[int], weights: wp.array[Any], anchors: wp.array[int]
):
    # The smallest raw node whose row is a unit vector anchors that column
    row = wp.tid()
    begin = offsets[row]
    if offsets[row + 1] - begin == 1:
        weight = weights[begin]
        if weight == type(weight)(1.0):
            wp.atomic_min(anchors, columns[begin], row)


@wp.kernel(enable_backward=False)
def _check_anchors_kernel(
    offsets: wp.array[int],
    columns: wp.array[int],
    weights: wp.array[Any],
    used: wp.array[int],
    anchors: wp.array[int],
    status: wp.array[int],
):
    column = wp.tid()
    if used[column] == 0:
        wp.atomic_min(status, _EMPTY_COLUMN, column)
        return
    row = anchors[column]
    if row == _NO_KEY:
        wp.atomic_min(status, _UNANCHORED_COLUMN, column)
        return
    valid = row >= 0 and row < offsets.shape[0] - 1
    if valid:
        # Check the row length first: an empty row may start past the last entry
        begin = offsets[row]
        valid = offsets[row + 1] - begin == 1
        if valid:
            weight = weights[begin]
            valid = columns[begin] == column and weight == type(weight)(1.0)
    if not valid:
        wp.atomic_min(status, _INVALID_ANCHOR, column)


@wp.kernel(enable_backward=False)
def _anchor_of_raw_kernel(anchors: wp.array[int], anchor_of_raw: wp.array[int]):
    column = wp.tid()
    row = anchors[column]
    if row >= 0 and row < anchor_of_raw.shape[0]:
        anchor_of_raw[row] = column


@wp.kernel(enable_backward=False)
def _cell_count_kernel(
    raw_nodes: wp.array2d[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    cell_counts: wp.array[int],
    status: wp.array[int],
):
    cell = wp.tid()
    count = int(0)
    for slot in range(raw_nodes.shape[1]):
        raw_node = raw_nodes[cell, slot]
        if raw_node >= 0:
            for entry in range(offsets[raw_node], offsets[raw_node + 1]):
                key = columns[entry]
                # Count each key at its first slot only
                first = int(1)
                for previous in range(slot):
                    other = raw_nodes[cell, previous]
                    if other >= 0:
                        if _find_sorted_range(columns, offsets[other], offsets[other + 1], key) >= 0:
                            first = 0
                count += first
    cell_counts[cell] = count
    wp.atomic_max(status, _CELL_MAX_COUNT, count)


@wp.func
def _is_unit_row(offsets: wp.array[int], columns: wp.array[int], weights: wp.array[Any], row: int, column: int):
    """Whether prolongation row ``row`` is the unit vector of ``column``."""
    begin = offsets[row]
    if offsets[row + 1] - begin != 1:
        return False
    if columns[begin] != column:
        return False
    return weights[begin] == type(weights[begin])(1.0)


@wp.func
def _side_raw_node(
    raw_nodes: wp.array2d[int], side_cells: wp.array2d[int], side_slots: wp.array2d[int], side: int, entry: int
):
    """Raw node of a side entry, from the inner cell in the first half of the entries and the outer cell after; or -1."""
    slot = side_slots[side, entry]
    if slot < 0:
        return -1
    cell = side_cells[side, wp.where(entry < side_slots.shape[1] // 2, 0, 1)]
    return raw_nodes[cell, slot]


@wp.kernel(enable_backward=False)
def _side_count_kernel(
    raw_nodes: wp.array2d[int],
    side_cells: wp.array2d[int],
    side_slots: wp.array2d[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    side_counts: wp.array[int],
    status: wp.array[int],
):
    side = wp.tid()
    count = int(0)
    for entry in range(side_slots.shape[1]):
        raw_node = _side_raw_node(raw_nodes, side_cells, side_slots, side, entry)
        if raw_node >= 0:
            for j in range(offsets[raw_node], offsets[raw_node + 1]):
                key = columns[j]
                first = int(1)
                for previous in range(entry):
                    other = _side_raw_node(raw_nodes, side_cells, side_slots, side, previous)
                    if other >= 0:
                        if _find_sorted_range(columns, offsets[other], offsets[other + 1], key) >= 0:
                            first = 0
                count += first
    side_counts[side] = count
    wp.atomic_max(status, _SIDE_MAX_COUNT, count)


@wp.func
def _side_half_contains(
    raw_nodes: wp.array2d[int],
    side_cells: wp.array2d[int],
    side_slots: wp.array2d[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    side: int,
    half: int,
    key: int,
):
    """Whether ``key`` appears in the rows of the inner (``half == 0``) or outer cell entries of a side."""
    half_size = side_slots.shape[1] // 2
    for entry in range(half * half_size, (half + 1) * half_size):
        raw_node = _side_raw_node(raw_nodes, side_cells, side_slots, side, entry)
        if raw_node >= 0:
            if _find_sorted_range(columns, offsets[raw_node], offsets[raw_node + 1], key) >= 0:
                return True
    return False


@wp.kernel(enable_backward=False)
def _side_fill_kernel(
    raw_nodes: wp.array2d[int],
    side_cells: wp.array2d[int],
    side_slots: wp.array2d[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    cell_nodes: wp.array2d[int],
    cell_counts: wp.array[int],
    side_nodes: wp.array2d[int],
    side_inner_slots: wp.array2d[int],
    side_outer_slots: wp.array2d[int],
):
    side = wp.tid()
    for slot in range(side_nodes.shape[1]):
        side_nodes[side, slot] = NULL_NODE_INDEX
        side_inner_slots[side, slot] = NULL_NODE_INDEX
        side_outer_slots[side, slot] = NULL_NODE_INDEX

    count = int(0)
    for entry in range(side_slots.shape[1]):
        raw_node = _side_raw_node(raw_nodes, side_cells, side_slots, side, entry)
        if raw_node >= 0:
            for j in range(offsets[raw_node], offsets[raw_node + 1]):
                count = _insert_sorted_key(side_nodes, side, count, columns[j])

    inner = side_cells[side, 0]
    outer = side_cells[side, 1]
    for slot in range(count):
        node = side_nodes[side, slot]
        if _side_half_contains(raw_nodes, side_cells, side_slots, offsets, columns, side, 0, node):
            side_inner_slots[side, slot] = _find_sorted(cell_nodes, inner, cell_counts[inner], node)
        if outer >= 0:
            if _side_half_contains(raw_nodes, side_cells, side_slots, offsets, columns, side, 1, node):
                side_outer_slots[side, slot] = _find_sorted(cell_nodes, outer, cell_counts[outer], node)
        else:
            # Geometry evaluates the outer trace of a boundary side in the
            # same cell as its inner trace.
            side_outer_slots[side, slot] = side_inner_slots[side, slot]


@wp.kernel(enable_backward=False)
def _owner_cell_kernel(raw_nodes: wp.array2d[int], anchor_of_raw: wp.array[int], owner_cell: wp.array[int]):
    cell, slot = wp.tid()
    raw_node = raw_nodes[cell, slot]
    if raw_node >= 0:
        node = anchor_of_raw[raw_node]
        if node >= 0:
            wp.atomic_min(owner_cell, node, cell)


@wp.kernel(enable_backward=False)
def _owner_side_kernel(
    raw_nodes: wp.array2d[int],
    side_cells: wp.array2d[int],
    side_slots: wp.array2d[int],
    side_valid: wp.array2d[int],
    anchor_of_raw: wp.array[int],
    owner_side: wp.array[int],
):
    side, entry = wp.tid()
    if side_valid[side, entry] == 0:
        return
    raw_node = _side_raw_node(raw_nodes, side_cells, side_slots, side, entry)
    if raw_node >= 0:
        node = anchor_of_raw[raw_node]
        if node >= 0:
            wp.atomic_min(owner_side, node, side)


@wp.kernel(enable_backward=False)
def _owner_slot_kernel(
    cell_nodes: wp.array2d[int],
    cell_counts: wp.array[int],
    side_nodes: wp.array2d[int],
    side_counts: wp.array[int],
    owner_cell: wp.array[int],
    owner_cell_slot: wp.array[int],
    owner_side: wp.array[int],
    owner_side_slot: wp.array[int],
):
    node = wp.tid()
    cell = owner_cell[node]
    owner_cell_slot[node] = NULL_NODE_INDEX
    if cell == _NO_KEY:
        owner_cell[node] = NULL_ELEMENT_INDEX
    else:
        owner_cell_slot[node] = _find_sorted(cell_nodes, cell, cell_counts[cell], node)

    side = owner_side[node]
    owner_side_slot[node] = NULL_NODE_INDEX
    if side == _NO_KEY:
        owner_side[node] = NULL_ELEMENT_INDEX
    else:
        owner_side_slot[node] = _find_sorted(side_nodes, side, side_counts[side], node)


def _scalar_kernels(scalar_type):
    """Kernels writing prolongation weights in the geometry scalar type."""

    @cache.dynamic_kernel(suffix=scalar_type.__name__, kernel_options={"enable_backward": False})
    def convert_weights(source: wp.array[wp.float64], weights: wp.array[scalar_type]):
        i = wp.tid()
        weights[i] = scalar_type(source[i])

    @cache.dynamic_kernel(suffix=scalar_type.__name__, kernel_options={"enable_backward": False})
    def fill_cells(
        raw_nodes: wp.array2d[int],
        offsets: wp.array[int],
        columns: wp.array[int],
        weights: wp.array[scalar_type],
        cell_nodes: wp.array2d[int],
        coefficients: wp.array3d[scalar_type],
    ):
        cell = wp.tid()
        for slot in range(cell_nodes.shape[1]):
            cell_nodes[cell, slot] = NULL_NODE_INDEX
            for raw_slot in range(coefficients.shape[2]):
                coefficients[cell, slot, raw_slot] = scalar_type(0.0)

        # Independent nodes sorted by index, then the weight of each raw slot
        count = int(0)
        for raw_slot in range(raw_nodes.shape[1]):
            raw_node = raw_nodes[cell, raw_slot]
            if raw_node >= 0:
                for entry in range(offsets[raw_node], offsets[raw_node + 1]):
                    count = _insert_sorted_key(cell_nodes, cell, count, columns[entry])
        for raw_slot in range(raw_nodes.shape[1]):
            raw_node = raw_nodes[cell, raw_slot]
            if raw_node >= 0:
                for entry in range(offsets[raw_node], offsets[raw_node + 1]):
                    slot = _find_sorted(cell_nodes, cell, count, columns[entry])
                    coefficients[cell, slot, raw_slot] = weights[entry]

    return convert_weights, fill_cells


def _raw_basis_kernels(raw_basis: ShapeBasisSpace):
    """Kernels evaluating raw node locations on cells and sides."""
    geometry = raw_basis.geometry
    coords_type = geometry.coords_type
    scalar_type = geometry.scalar_type
    raw_node_count = raw_basis.topology.MAX_NODES_PER_ELEMENT
    node_coords_in_element = raw_basis.node_coords_in_element

    @cache.dynamic_kernel(suffix=raw_basis.name, kernel_options={"enable_backward": False})
    def side_table(
        side_arg: geometry.SideArg,
        raw_nodes: wp.array2d[int],
        side_cells: wp.array2d[int],
        side_slots: wp.array2d[int],
    ):
        # List every raw node of both adjacent cells: functions that vanish on
        # the side can still have a nonzero gradient there
        side, entry = wp.tid()
        inner = geometry.side_inner_cell_index(side_arg, side)
        outer = geometry.side_outer_cell_index(side_arg, side)
        if outer == inner:
            outer = -1
        if entry == 0:
            side_cells[side, 0] = inner
            side_cells[side, 1] = outer
        cell = inner
        slot = entry
        if entry >= raw_node_count:
            cell = outer
            slot = entry - raw_node_count
        side_slots[side, entry] = -1
        if cell >= 0:
            if raw_nodes[cell, slot] >= 0:
                side_slots[side, entry] = slot

    @cache.dynamic_kernel(suffix=raw_basis.name, kernel_options={"enable_backward": False})
    def side_validity(
        side_arg: geometry.SideArg,
        topo_arg: raw_basis.topology.TopologyArg,
        basis_arg: raw_basis.BasisArg,
        side_cells: wp.array2d[int],
        side_slots: wp.array2d[int],
        side_valid: wp.array2d[int],
    ):
        # Whether the raw node of a side entry lies on the side
        side, entry = wp.tid()
        side_valid[side, entry] = 0
        slot = side_slots[side, entry]
        if slot < 0:
            return
        cell = side_cells[side, wp.where(entry < side_slots.shape[1] // 2, 0, 1)]
        coords = node_coords_in_element(geometry.side_to_cell_arg(side_arg), topo_arg, basis_arg, cell, slot)
        side_coords = geometry.side_from_cell_coords(side_arg, side, cell, coords)
        if side_coords[0] > -1.0e7:
            side_valid[side, entry] = 1

    @cache.dynamic_kernel(suffix=raw_basis.name, kernel_options={"enable_backward": False})
    def anchor_cell_coords(
        cell_arg: geometry.CellArg,
        topo_arg: raw_basis.topology.TopologyArg,
        basis_arg: raw_basis.BasisArg,
        raw_nodes: wp.array2d[int],
        offsets: wp.array[int],
        columns: wp.array[int],
        weights: wp.array[scalar_type],
        cell_nodes: wp.array2d[int],
        cell_counts: wp.array[int],
        anchors: wp.array[int],
        node_coords: wp.array2d[coords_type],
        copy_coords: wp.array2d[coords_type],
    ):
        # Coordinates of each independent node's anchor in the cells holding
        # that raw node, and of a copy of the node in the cells holding one:
        # the anchor if present, otherwise another raw node with the same unit
        # prolongation row, such as a periodic image.
        cell, slot = wp.tid()
        node_coords[cell, slot] = coords_type(OUTSIDE)
        copy_coords[cell, slot] = coords_type(OUTSIDE)
        if slot >= cell_counts[cell]:
            return
        node = cell_nodes[cell, slot]
        copy_slot = int(-1)
        for raw_slot in range(raw_node_count):
            raw_node = raw_nodes[cell, raw_slot]
            if raw_node == anchors[node]:
                node_coords[cell, slot] = node_coords_in_element(cell_arg, topo_arg, basis_arg, cell, raw_slot)
                copy_slot = raw_slot
                break
            if copy_slot < 0 and raw_node >= 0:
                if _is_unit_row(offsets, columns, weights, raw_node, node):
                    copy_slot = raw_slot
        if copy_slot >= 0:
            copy_coords[cell, slot] = node_coords_in_element(cell_arg, topo_arg, basis_arg, cell, copy_slot)

    @cache.dynamic_kernel(suffix=raw_basis.name, kernel_options={"enable_backward": False})
    def copy_side_coords(
        side_arg: geometry.SideArg,
        topo_arg: raw_basis.topology.TopologyArg,
        basis_arg: raw_basis.BasisArg,
        raw_nodes: wp.array2d[int],
        side_cells: wp.array2d[int],
        side_slots: wp.array2d[int],
        side_valid: wp.array2d[int],
        offsets: wp.array[int],
        columns: wp.array[int],
        weights: wp.array[scalar_type],
        side_nodes: wp.array2d[int],
        side_counts: wp.array[int],
        anchors: wp.array[int],
        copy_coords: wp.array2d[coords_type],
    ):
        # Side coordinates of a copy of each independent node lying on the
        # side, preferring its anchor, as for cells
        side, slot = wp.tid()
        copy_coords[side, slot] = coords_type(OUTSIDE)
        if slot >= side_counts[side]:
            return
        node = side_nodes[side, slot]
        copy_entry = int(-1)
        for entry in range(side_slots.shape[1]):
            if side_valid[side, entry] != 0:
                raw_node = _side_raw_node(raw_nodes, side_cells, side_slots, side, entry)
                if raw_node == anchors[node]:
                    copy_entry = entry
                    break
                if copy_entry < 0 and raw_node >= 0:
                    if _is_unit_row(offsets, columns, weights, raw_node, node):
                        copy_entry = entry
        if copy_entry >= 0:
            cell = side_cells[side, wp.where(copy_entry < side_slots.shape[1] // 2, 0, 1)]
            coords = node_coords_in_element(
                geometry.side_to_cell_arg(side_arg), topo_arg, basis_arg, cell, side_slots[side, copy_entry]
            )
            copy_coords[side, slot] = geometry.side_from_cell_coords(side_arg, side, cell, coords)

    @cache.dynamic_kernel(suffix=raw_basis.name, kernel_options={"enable_backward": False})
    def anchor_side_coords(
        side_arg: geometry.SideArg,
        topo_arg: raw_basis.topology.TopologyArg,
        basis_arg: raw_basis.BasisArg,
        raw_nodes: wp.array2d[int],
        side_cells: wp.array2d[int],
        side_slots: wp.array2d[int],
        side_valid: wp.array2d[int],
        anchors: wp.array[int],
        owner_side: wp.array[int],
        owner_side_coords: wp.array[coords_type],
    ):
        node = wp.tid()
        owner_side_coords[node] = coords_type(OUTSIDE)
        side = owner_side[node]
        if side == NULL_ELEMENT_INDEX:
            return
        for entry in range(side_slots.shape[1]):
            if side_valid[side, entry] != 0:
                if _side_raw_node(raw_nodes, side_cells, side_slots, side, entry) == anchors[node]:
                    cell = side_cells[side, wp.where(entry < side_slots.shape[1] // 2, 0, 1)]
                    coords = node_coords_in_element(
                        geometry.side_to_cell_arg(side_arg), topo_arg, basis_arg, cell, side_slots[side, entry]
                    )
                    owner_side_coords[node] = geometry.side_from_cell_coords(side_arg, side, cell, coords)
                    return

    return side_table, side_validity, anchor_cell_coords, anchor_side_coords, copy_side_coords


def _raise_build_errors(status, anchors):
    if status[_EMPTY_COLUMN] != _NO_KEY:
        raise ValueError(f"Prolongation column {int(status[_EMPTY_COLUMN])} has no nonzero entries")
    if status[_INVALID_ANCHOR] != _NO_KEY:
        column = int(status[_INVALID_ANCHOR])
        row = int(anchors[column : column + 1].numpy()[0])
        raise ValueError(
            f"Raw node {row} given as the anchor of independent node {column} does not have that node's unit row"
        )
    if status[_UNANCHORED_COLUMN] != _NO_KEY:
        raise ValueError(
            f"Independent node {int(status[_UNANCHORED_COLUMN])} has no anchor: no raw node has a prolongation "
            "row equal to its unit vector. Degrees of freedom without an anchor are not supported yet"
        )


def _build_constrained_basis(
    raw_basis: ShapeBasisSpace,
    offsets: wp.array,
    columns: wp.array,
    weights: wp.array,
    column_count: int,
    anchors: wp.array | None,
    max_nodes_per_element: int | None,
    raw_nodes: wp.array | None = None,
    check_columns: bool = True,
) -> "ConstrainedBasisSpace":
    """Build the constrained basis for a final prolongation ``P`` in CSR form.

    Args:
        raw_basis: Basis whose nodes are the rows of ``P``.
        offsets: Row offsets of ``P``, one row per raw node.
        columns: Sorted column indices of each row.
        weights: Entry weights, in the geometry scalar type.
        column_count: Number of independent nodes.
        anchors: Anchor raw node of each column, or ``None`` to detect them.
        max_nodes_per_element: Largest allowed cell node list, or ``None``.
        raw_nodes: Raw node indices of each cell, if already available.
        check_columns: Whether to check that every column is used and anchored.
    """
    geometry = raw_basis.geometry
    scalar_type = geometry.scalar_type
    coords_type = geometry.coords_type
    device = offsets.device
    cell_count = geometry.cell_count()
    side_count = geometry.side_count()
    raw_topology = raw_basis.topology
    raw_count = raw_topology.node_count()
    raw_node_count = raw_topology.MAX_NODES_PER_ELEMENT
    prolongation = (offsets, columns)

    if raw_nodes is None:
        raw_nodes = raw_topology.element_node_indices(
            out=wp.empty((cell_count, raw_node_count), dtype=int, device=device)
        )
    status = _new_build_status(device)
    side_table, side_validity, anchor_cell_coords, anchor_side_coords, copy_side_coords = _raw_basis_kernels(raw_basis)
    _convert_weights, fill_cells = _scalar_kernels(scalar_type)

    # Every column must be used, and anchored at a raw node with its unit row
    if check_columns:
        used = wp.zeros(column_count, dtype=int, device=device)
        wp.launch(_column_usage_kernel, dim=columns.shape[0], inputs=[columns, used], device=device)
        if anchors is None:
            anchors = wp.full(column_count, value=_NO_KEY, dtype=int, device=device)
            wp.launch(_detect_anchors_kernel, dim=raw_count, inputs=[*prolongation, weights, anchors], device=device)
        wp.launch(
            _check_anchors_kernel,
            dim=column_count,
            inputs=[*prolongation, weights, used, anchors, status],
            device=device,
        )
    anchor_of_raw = wp.full(raw_count, value=-1, dtype=int, device=device)
    wp.launch(_anchor_of_raw_kernel, dim=column_count, inputs=[anchors, anchor_of_raw], device=device)

    # Raw nodes whose rows make up each side's node list: all nodes of both
    # adjacent cells, as side gradients depend on all of them
    side_arg = geometry.side_arg_value(device)
    raw_topo_arg = raw_topology.topo_arg_value(device)
    raw_basis_arg = raw_basis.basis_arg_value(device)
    side_cells = wp.empty((side_count, 2), dtype=int, device=device)
    side_slots = wp.empty((side_count, 2 * raw_node_count), dtype=int, device=device)
    wp.launch(side_table, dim=side_slots.shape, inputs=[side_arg, raw_nodes, side_cells, side_slots], device=device)
    side_inputs = [raw_nodes, side_cells, side_slots]

    side_valid = wp.empty(side_slots.shape, dtype=int, device=device)
    wp.launch(
        side_validity,
        dim=side_slots.shape,
        inputs=[side_arg, raw_topo_arg, raw_basis_arg, side_cells, side_slots, side_valid],
        device=device,
    )

    # Cell and side node counts do not depend on each other
    cell_counts = wp.empty(cell_count, dtype=int, device=device)
    side_counts = wp.empty(side_count, dtype=int, device=device)
    wp.launch(_cell_count_kernel, dim=cell_count, inputs=[raw_nodes, *prolongation, cell_counts, status], device=device)
    wp.launch(
        _side_count_kernel, dim=side_count, inputs=[*side_inputs, *prolongation, side_counts, status], device=device
    )

    values = status.numpy()
    _raise_build_errors(values, anchors)
    cell_max = int(values[_CELL_MAX_COUNT])
    side_max = int(values[_SIDE_MAX_COUNT])
    if max_nodes_per_element is not None and cell_max > max_nodes_per_element:
        raise ValueError(
            f"A cell depends on {cell_max} independent nodes, more than max_nodes_per_element={max_nodes_per_element}. "
            "Wide node lists slow down assembly; raise the limit if they are intended"
        )

    cell_nodes = wp.empty((cell_count, cell_max), dtype=int, device=device)
    coefficients = wp.empty((cell_count, cell_max, raw_node_count), dtype=scalar_type, device=device)
    wp.launch(
        fill_cells,
        dim=cell_count,
        inputs=[raw_nodes, *prolongation, weights, cell_nodes, coefficients],
        device=device,
    )

    side_nodes = wp.empty((side_count, side_max), dtype=int, device=device)
    side_inner_slots = wp.empty_like(side_nodes)
    side_outer_slots = wp.empty_like(side_nodes)
    wp.launch(
        _side_fill_kernel,
        dim=side_count,
        inputs=[
            *side_inputs,
            *prolongation,
            cell_nodes,
            cell_counts,
            side_nodes,
            side_inner_slots,
            side_outer_slots,
        ],
        device=device,
    )

    # Owners are the first cell holding the anchor and the first side with
    # the anchor on it.
    owner_cell = wp.full(column_count, value=_NO_KEY, dtype=int, device=device)
    owner_cell_slot = wp.empty(column_count, dtype=int, device=device)
    owner_side = wp.full(column_count, value=_NO_KEY, dtype=int, device=device)
    owner_side_slot = wp.empty(column_count, dtype=int, device=device)
    wp.launch(_owner_cell_kernel, dim=raw_nodes.shape, inputs=[raw_nodes, anchor_of_raw, owner_cell], device=device)
    wp.launch(
        _owner_side_kernel,
        dim=side_slots.shape,
        inputs=[*side_inputs, side_valid, anchor_of_raw, owner_side],
        device=device,
    )
    wp.launch(
        _owner_slot_kernel,
        dim=column_count,
        inputs=[
            cell_nodes,
            cell_counts,
            side_nodes,
            side_counts,
            owner_cell,
            owner_cell_slot,
            owner_side,
            owner_side_slot,
        ],
        device=device,
    )

    node_coords = wp.empty((cell_count, cell_max), dtype=coords_type, device=device)
    copy_coords = wp.empty_like(node_coords)
    wp.launch(
        anchor_cell_coords,
        dim=node_coords.shape,
        inputs=[
            geometry.cell_arg_value(device),
            raw_topo_arg,
            raw_basis_arg,
            raw_nodes,
            *prolongation,
            weights,
            cell_nodes,
            cell_counts,
            anchors,
            node_coords,
            copy_coords,
        ],
        device=device,
    )
    side_copy_coords = wp.empty((side_count, side_max), dtype=coords_type, device=device)
    wp.launch(
        copy_side_coords,
        dim=side_copy_coords.shape,
        inputs=[
            side_arg,
            raw_topo_arg,
            raw_basis_arg,
            *side_inputs,
            side_valid,
            *prolongation,
            weights,
            side_nodes,
            side_counts,
            anchors,
            side_copy_coords,
        ],
        device=device,
    )
    owner_side_coords = wp.empty(column_count, dtype=coords_type, device=device)
    wp.launch(
        anchor_side_coords,
        dim=column_count,
        inputs=[
            side_arg,
            raw_topo_arg,
            raw_basis_arg,
            *side_inputs,
            side_valid,
            anchors,
            owner_side,
            owner_side_coords,
        ],
        device=device,
    )

    topology = ConstrainedSpaceTopology(
        raw_topology,
        cell_nodes,
        cell_counts,
        side_nodes,
        side_counts,
        side_inner_slots,
        side_outer_slots,
        column_count,
    )
    return ConstrainedBasisSpace(
        raw_basis,
        topology,
        coefficients,
        node_coords,
        copy_coords,
        side_copy_coords,
        anchors,
        offsets,
        columns,
        weights,
        owner_cell,
        owner_cell_slot,
        owner_side,
        owner_side_slot,
        owner_side_coords,
    )


def make_constrained_basis_space(
    base_basis: BasisSpace,
    prolongation: BsrMatrix,
    independent_raw_nodes: wp.array | None = None,
    max_nodes_per_element: int = _DEFAULT_MAX_NODES_PER_ELEMENT,
) -> ConstrainedBasisSpace:
    """Build a basis whose nodes are linear combinations of the nodes of ``base_basis``.

    ``prolongation`` is a sparse matrix ``P`` with one row per node of
    ``base_basis`` and one column per independent degree of freedom of the
    new basis: base nodal values are ``u_base = P @ u``. Each new basis
    function is the matching combination of base functions, so forms
    assemble directly on the independent degrees of freedom, and fields
    evaluate through ``P``. Use it for multi-point constraints such as
    periodic boundary conditions (see :func:`periodic_prolongation`) or
    hanging nodes (see :func:`hanging_node_prolongation`).

    Each independent node ``j`` needs an anchor: a base node whose row of
    ``P`` is exactly the unit vector ``e_j``. Anchors give the node its
    position and its ``at_node()`` samples. By default, the anchor is the
    smallest base node with that unit row. Rows need not sum to one.

    If ``base_basis`` is itself a constrained basis, ``P`` is composed with
    its prolongation and the result is built on its raw basis, so
    constrained spaces do not nest.

    Construction runs on the device of ``prolongation``; only sizes and error
    status are read back. The constrained basis must be rebuilt after the
    geometry changes. Nodal assembly is not available on it; use the raw basis
    and the ``reduce_dirichlet_projector()`` method of the result for strong
    boundary conditions.

    Args:
        base_basis: Scalar shape-function basis, such as one returned by
            :func:`make_polynomial_basis_space`, or a constrained basis.
        prolongation: Matrix ``P`` with ``1x1`` floating-point blocks and one
            row of blocks per node of ``base_basis``.
        independent_raw_nodes: Anchor of each column of ``P``, as node
            indices of ``base_basis``. ``None`` picks the smallest node with a
            unit row.
        max_nodes_per_element: Largest number of independent nodes allowed
            to act on one cell. Assembly kernels are sized by this number, so
            very wide constraints must be allowed explicitly.

    Returns:
        A constrained basis whose ``raw_basis`` is ``base_basis``, or the raw
        basis of ``base_basis`` if it is constrained.

    Raises:
        NotImplementedError: If the raw basis is not a scalar shape-function basis.
        TypeError: If ``prolongation`` is not a floating-point BSR matrix.
        ValueError: If ``prolongation`` does not have scalar blocks or one row per
            base node, refers to out-of-range columns, has an empty column or a
            column without an anchor, if ``independent_raw_nodes`` names a node
            without the matching unit row, or if a cell depends on more than
            ``max_nodes_per_element`` independent nodes.
    """
    constrained_base = isinstance(base_basis, ConstrainedBasisSpace)
    raw_basis = base_basis.raw_basis if constrained_base else base_basis
    if not isinstance(raw_basis, ShapeBasisSpace) or raw_basis.value != ShapeFunction.Value.Scalar:
        raise NotImplementedError(
            "Constrained basis spaces require a scalar shape-function basis, "
            "such as one returned by make_polynomial_basis_space()"
        )
    if not isinstance(prolongation, BsrMatrix):
        raise TypeError(f"Prolongation must be a BSR matrix, got {type(prolongation).__name__}")
    if prolongation.block_shape != (1, 1):
        raise ValueError(f"Prolongation must have scalar (1x1) blocks, got {prolongation.block_shape}")
    if prolongation.scalar_type not in float_types:
        raise TypeError(f"Prolongation must have a floating-point scalar type, got {prolongation.scalar_type}")
    base_count = base_basis.topology.node_count()
    if prolongation.nrow != base_count:
        raise ValueError(f"Prolongation must have one row per base node ({base_count}), got {prolongation.nrow}")
    column_count = prolongation.ncol
    if column_count < 1:
        raise ValueError("Prolongation must have at least one column")

    device = prolongation.device
    anchors = None
    if independent_raw_nodes is not None:
        if isinstance(independent_raw_nodes, wp.array):
            if independent_raw_nodes.dtype != wp.int32:
                raise TypeError("Independent raw nodes must be an array of int32")
            anchors = wp.clone(independent_raw_nodes, device=device)
        else:
            anchors = wp.array(independent_raw_nodes, dtype=int, device=device)
        if anchors.ndim != 1 or anchors.shape[0] != column_count:
            raise ValueError(f"Independent raw nodes must have one entry per prolongation column ({column_count})")

    offsets, columns, weights, nnz = _compact_prolongation(prolongation)
    if constrained_base:
        # Compose with the base prolongation once, in double precision, on the
        # construction device
        base_matrix = base_basis.prolongation_matrix(wp.float64)
        if base_matrix.device != device:
            base_matrix = _prolongation_matrix(
                _Prolongation(
                    base_matrix.offsets.to(device),
                    base_matrix.columns.to(device),
                    base_matrix.values.to(device),
                    None,
                    base_matrix.ncol,
                    base_matrix.nnz,
                ),
                base_matrix.nrow,
            )
        composed = bsr_mm(
            base_matrix,
            _prolongation_matrix(_Prolongation(offsets, columns, weights, None, column_count, nnz), base_count),
        )
        offsets, columns, weights, nnz = _compact_prolongation(composed)
        if anchors is not None:
            wp.launch(
                _gather_anchor_kernel,
                dim=column_count,
                inputs=[base_basis.independent_raw_nodes.to(device), anchors],
                device=device,
            )

    scalar_type = raw_basis.geometry.scalar_type
    convert_weights, _fill_cells = _scalar_kernels(scalar_type)
    scalar_weights = wp.empty(nnz, dtype=scalar_type, device=device)
    wp.launch(convert_weights, dim=nnz, inputs=[weights, scalar_weights], device=device)

    return _build_constrained_basis(
        raw_basis,
        offsets,
        columns,
        scalar_weights,
        column_count,
        anchors,
        max_nodes_per_element,
    )


def make_conforming_basis_space(raw_basis: BasisSpace) -> ConstrainedBasisSpace:
    """Eliminate the hanging nodes of a continuous Lagrange basis, such as on an :class:`AdaptiveNanogrid`.

    Nodes of a finer cell lying on a side shared with a coarser cell take the
    value of the coarser cell's interpolant, so that fields are continuous
    across cells of different sizes. This works for any element type and
    polynomial degree of an interpolatory basis; on geometries whose sides
    always match their neighbors' facets, the result has the raw basis's nodes.

    The returned basis enumerates only independent nodes. Its trace is built
    during this call and is ready for side restrictions and assembly.
    This is :func:`make_constrained_basis_space` applied to
    :func:`hanging_node_prolongation`, without re-validating the prolongation
    and without a limit on the cell node lists.
    """
    prolongation, raw_nodes = _hanging_node_constraints(raw_basis)
    return _build_constrained_basis(
        raw_basis,
        prolongation.offsets,
        prolongation.indices,
        prolongation.weights,
        prolongation.column_count,
        prolongation.independent_raw_nodes,
        max_nodes_per_element=None,
        raw_nodes=raw_nodes,
        check_columns=False,
    )
