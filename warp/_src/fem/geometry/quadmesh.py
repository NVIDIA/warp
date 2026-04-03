# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, ClassVar, Optional

import warp as wp
from warp._src.fem import cache
from warp._src.fem.cache import (
    TemporaryStore,
    borrow_temporary,
    borrow_temporary_like,
    cached_vec_type,
)
from warp._src.fem.types import (
    OUTSIDE,
    ElementIndex,
    make_coords,
)
from warp._src.fem.utils import compress_node_indices, host_read_at_index, masked_indices
from warp._src.types import type_scalar_type
from warp._src.utils import array_scan

from .closest_point import project_on_seg_at_origin
from .element import Element
from .geometry import Geometry

_wp_module_name_ = "warp.fem.geometry.quadmesh"


# Topology-only Arg structs (no position data, shared across precisions)
@wp.struct
class QuadmeshCellArg:
    """Arguments for cell topology device functions."""

    quad_vertex_indices: wp.array2d(dtype=int)
    quad_bvh: wp.uint64


@wp.struct
class QuadmeshSideArg:
    """Arguments for side topology device functions."""

    cell_arg: QuadmeshCellArg
    edge_vertex_indices: wp.array(dtype=wp.vec2i)
    edge_quad_indices: wp.array(dtype=wp.vec2i)


def _make_quadmesh_cell_arg(topo_type, pos_vec_type):
    """Generate a CellArg struct for the given position vector type."""

    @cache.dynamic_struct(suffix=(type_scalar_type(pos_vec_type), pos_vec_type._length_))
    class _CellArg:
        topology: topo_type
        positions: wp.array(dtype=pos_vec_type)

    return _CellArg


def _make_quadmesh_side_arg(topo_type, pos_vec_type):
    """Generate a SideArg struct for the given position vector type."""

    @cache.dynamic_struct(suffix=(type_scalar_type(pos_vec_type), pos_vec_type._length_))
    class _SideArg:
        topology: topo_type
        positions: wp.array(dtype=pos_vec_type)

    return _SideArg


class Quadmesh(Geometry):
    """Quadrilateral mesh geometry."""

    _dynamic_attribute_constructors: ClassVar = {
        "side_to_cell_arg": lambda obj: obj._make_quad_side_to_cell_arg(),
        **Geometry._dynamic_attribute_constructors,
    }

    def __init__(
        self,
        quad_vertex_indices: wp.array,
        positions: wp.array,
        build_bvh: bool = False,
        temporary_store: Optional[TemporaryStore] = None,
    ):
        """Construct a D-dimensional quadrilateral mesh.

        Args:
            quad_vertex_indices: warp array of shape (num_tris, 4) containing vertex indices for each quad, in counter-clockwise order
            positions: warp array of shape (num_vertices, D) containing the position of each vertex
            temporary_store: shared pool from which to allocate temporary arrays
        """

        self.quad_vertex_indices = quad_vertex_indices
        self.positions = positions

        # Infer scalar type and dimension from position array
        self._scalar_type = type_scalar_type(positions.dtype)
        self.dimension = positions.dtype._length_

        # Generate precision-appropriate Arg structs
        pos_vec = cached_vec_type(self.dimension, self._scalar_type)
        self.CellArg = _make_quadmesh_cell_arg(QuadmeshCellArg, pos_vec)
        self.SideArg = _make_quadmesh_side_arg(QuadmeshSideArg, pos_vec)

        self._edge_vertex_indices: wp.array = None
        self._edge_quad_indices: wp.array = None
        self._vertex_quad_offsets: wp.array = None
        self._vertex_quad_indices: wp.array = None
        self._build_topology(temporary_store)

        # Flip edges so that normals point away from inner cell
        if self.dimension == 2:
            orient_kernel = Quadmesh._orient_edges_2d
        else:
            orient_kernel = Quadmesh._orient_edges_3d
        wp.launch(
            kernel=orient_kernel,
            device=positions.device,
            dim=self.side_count(),
            inputs=[self._edge_vertex_indices, self._edge_quad_indices, self.quad_vertex_indices, self.positions],
        )

        # Process all dynamic attributes (Quadmesh primitives + Geometry dependents)
        cache.setup_dynamic_attributes(self)
        self.cell_closest_point = self._make_cell_closest_point()
        self.cell_coordinates = self._make_cell_coordinates()
        self.side_coordinates = self._make_side_coordinates(assume_linear=True)

        if build_bvh:
            self.build_bvh(self.positions.device)

    @property
    def scalar_type(self):
        return self._scalar_type

    def cell_count(self):
        return self.quad_vertex_indices.shape[0]

    def vertex_count(self):
        return self.positions.shape[0]

    def side_count(self):
        return self._edge_vertex_indices.shape[0]

    def boundary_side_count(self):
        return self._boundary_edge_indices.shape[0]

    def reference_cell(self) -> Element:
        return Element.SQUARE

    def reference_side(self) -> Element:
        return Element.LINE_SEGMENT

    @property
    def edge_quad_indices(self) -> wp.array:
        return self._edge_quad_indices

    @property
    def edge_vertex_indices(self) -> wp.array:
        return self._edge_vertex_indices

    @wp.struct
    class SideIndexArg:
        boundary_edge_indices: wp.array(dtype=int)

    def fill_cell_topo_arg(self, args: QuadmeshCellArg, device):
        args.quad_vertex_indices = self.quad_vertex_indices.to(device)
        args.quad_bvh = self.bvh_id(device)

    def fill_side_topo_arg(self, args: QuadmeshSideArg, device):
        self.fill_cell_topo_arg(args.cell_arg, device)
        args.edge_vertex_indices = self._edge_vertex_indices.to(device)
        args.edge_quad_indices = self._edge_quad_indices.to(device)

    def fill_cell_arg(self, args, device):
        self.fill_cell_topo_arg(args.topology, device)
        args.positions = self.positions.to(device)

    def fill_side_arg(self, args, device):
        self.fill_side_topo_arg(args.topology, device)
        args.positions = self.positions.to(device)

    def fill_side_index_arg(self, args: SideIndexArg, device):
        args.boundary_edge_indices = self._boundary_edge_indices.to(device)

    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int):
        return args.boundary_edge_indices[boundary_side_index]

    @wp.func
    def _edge_to_quad_coords(args: Any, side_index: ElementIndex, quad_index: ElementIndex, side_coords: Any):
        edge_vidx = args.edge_vertex_indices[side_index]
        quad_vidx = args.cell_arg.quad_vertex_indices[quad_index]

        vs = edge_vidx[0]
        ve = edge_vidx[1]
        s = side_coords[0]
        z = side_coords.dtype(0.0)
        o = side_coords.dtype(1.0)

        if vs == quad_vidx[0]:
            return wp.where(ve == quad_vidx[1], type(side_coords)(s, z, z), type(side_coords)(z, s, z))
        elif vs == quad_vidx[1]:
            return wp.where(ve == quad_vidx[2], type(side_coords)(o, s, z), type(side_coords)(o - s, z, z))
        elif vs == quad_vidx[2]:
            return wp.where(ve == quad_vidx[3], type(side_coords)(o - s, o, z), type(side_coords)(o, o - s, z))

        return wp.where(ve == quad_vidx[0], type(side_coords)(z, o - s, z), type(side_coords)(s, o, z))

    @wp.func
    def _quad_to_edge_coords(args: Any, side_index: ElementIndex, quad_index: ElementIndex, quad_coords: Any):
        edge_vidx = args.edge_vertex_indices[side_index]
        quad_vidx = args.cell_arg.quad_vertex_indices[quad_index]

        vs = edge_vidx[0]
        ve = edge_vidx[1]

        cx = quad_coords[0]
        cy = quad_coords[1]
        o = quad_coords.dtype(1.0)
        z = quad_coords.dtype(0.0)

        if vs == quad_vidx[0]:
            oc = wp.where(ve == quad_vidx[1], cy, cx)
            ec = wp.where(ve == quad_vidx[1], cx, cy)
        elif vs == quad_vidx[1]:
            oc = wp.where(ve == quad_vidx[2], o - cx, cy)
            ec = wp.where(ve == quad_vidx[2], cy, o - cx)
        elif vs == quad_vidx[2]:
            oc = wp.where(ve == quad_vidx[3], o - cy, o - cx)
            ec = wp.where(ve == quad_vidx[3], o - cx, o - cy)
        else:
            oc = wp.where(ve == quad_vidx[0], cx, o - cy)
            ec = wp.where(ve == quad_vidx[0], o - cy, cx)
        return wp.where(oc == z, type(quad_coords)(ec, z, z), type(quad_coords)(quad_coords.dtype(OUTSIDE)))

    def _make_quad_side_to_cell_arg(self):
        CellArgType = self.CellArg

        @cache.dynamic_func(suffix=self.name)
        def side_to_cell_arg(side_arg: self.SideArg):
            return CellArgType(side_arg.topology.cell_arg, side_arg.positions)

        return side_to_cell_arg

    # -- Topology building (precision-independent) --

    def _build_topology(self, temporary_store: TemporaryStore):
        device = self.quad_vertex_indices.device

        vertex_quad_offsets, vertex_quad_indices = compress_node_indices(
            self.vertex_count(), self.quad_vertex_indices, temporary_store=temporary_store
        )
        self._vertex_quad_offsets = vertex_quad_offsets.detach()
        self._vertex_quad_indices = vertex_quad_indices.detach()

        vertex_start_edge_count = borrow_temporary(temporary_store, dtype=int, device=device, shape=self.vertex_count())
        vertex_start_edge_count.zero_()
        vertex_start_edge_offsets = borrow_temporary_like(vertex_start_edge_count, temporary_store=temporary_store)

        vertex_edge_ends = borrow_temporary(temporary_store, dtype=int, device=device, shape=(4 * self.cell_count()))
        vertex_edge_quads = borrow_temporary(
            temporary_store, dtype=int, device=device, shape=(4 * self.cell_count(), 2)
        )

        wp.launch(
            kernel=Quadmesh._count_starting_edges_kernel,
            device=device,
            dim=self.cell_count(),
            inputs=[self.quad_vertex_indices, vertex_start_edge_count],
        )

        array_scan(in_array=vertex_start_edge_count, out_array=vertex_start_edge_offsets, inclusive=False)

        vertex_unique_edge_count = vertex_start_edge_count
        wp.launch(
            kernel=Quadmesh._count_unique_starting_edges_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                self._vertex_quad_offsets,
                self._vertex_quad_indices,
                self.quad_vertex_indices,
                vertex_start_edge_offsets,
                vertex_unique_edge_count,
                vertex_edge_ends,
                vertex_edge_quads,
            ],
        )

        vertex_unique_edge_offsets = borrow_temporary_like(vertex_start_edge_offsets, temporary_store=temporary_store)
        array_scan(in_array=vertex_start_edge_count, out_array=vertex_unique_edge_offsets, inclusive=False)

        edge_count = int(
            host_read_at_index(vertex_unique_edge_offsets, self.vertex_count() - 1, temporary_store=temporary_store)
        )

        self._edge_vertex_indices = wp.empty(shape=(edge_count,), dtype=wp.vec2i, device=device)
        self._edge_quad_indices = wp.empty(shape=(edge_count,), dtype=wp.vec2i, device=device)

        boundary_mask = borrow_temporary(temporary_store=temporary_store, shape=(edge_count,), dtype=int, device=device)

        wp.launch(
            kernel=Quadmesh._compress_edges_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                vertex_start_edge_offsets,
                vertex_unique_edge_offsets,
                vertex_unique_edge_count,
                vertex_edge_ends,
                vertex_edge_quads,
                self._edge_vertex_indices,
                self._edge_quad_indices,
                boundary_mask,
            ],
        )

        vertex_start_edge_offsets.release()
        vertex_unique_edge_offsets.release()
        vertex_unique_edge_count.release()
        vertex_edge_ends.release()
        vertex_edge_quads.release()

        boundary_edge_indices, _ = masked_indices(boundary_mask, temporary_store=temporary_store)
        self._boundary_edge_indices = boundary_edge_indices.detach()

        boundary_mask.release()

    # -- Topology kernels (precision-independent) --

    @wp.kernel
    def _count_starting_edges_kernel(
        quad_vertex_indices: wp.array2d(dtype=int), vertex_start_edge_count: wp.array(dtype=int)
    ):
        t = wp.tid()
        for k in range(4):
            v0 = quad_vertex_indices[t, k]
            v1 = quad_vertex_indices[t, (k + 1) % 4]

            if v0 < v1:
                wp.atomic_add(vertex_start_edge_count, v0, 1)
            else:
                wp.atomic_add(vertex_start_edge_count, v1, 1)

    @wp.func
    def _find(
        needle: int,
        values: wp.array(dtype=int),
        beg: int,
        end: int,
    ):
        for i in range(beg, end):
            if values[i] == needle:
                return i

        return -1

    @wp.kernel
    def _count_unique_starting_edges_kernel(
        vertex_quad_offsets: wp.array(dtype=int),
        vertex_quad_indices: wp.array(dtype=int),
        quad_vertex_indices: wp.array2d(dtype=int),
        vertex_start_edge_offsets: wp.array(dtype=int),
        vertex_start_edge_count: wp.array(dtype=int),
        edge_ends: wp.array(dtype=int),
        edge_quads: wp.array2d(dtype=int),
    ):
        v = wp.tid()

        edge_beg = vertex_start_edge_offsets[v]

        quad_beg = vertex_quad_offsets[v]
        quad_end = vertex_quad_offsets[v + 1]

        edge_cur = edge_beg

        for quad in range(quad_beg, quad_end):
            q = vertex_quad_indices[quad]

            for k in range(4):
                v0 = quad_vertex_indices[q, k]
                v1 = quad_vertex_indices[q, (k + 1) % 4]

                if v == wp.min(v0, v1):
                    other_v = wp.max(v0, v1)

                    seen_idx = Quadmesh._find(other_v, edge_ends, edge_beg, edge_cur)

                    if seen_idx == -1:
                        edge_ends[edge_cur] = other_v
                        edge_quads[edge_cur, 0] = q
                        edge_quads[edge_cur, 1] = q
                        edge_cur += 1
                    else:
                        edge_quads[seen_idx, 1] = q

        vertex_start_edge_count[v] = edge_cur - edge_beg

    @wp.kernel
    def _compress_edges_kernel(
        vertex_start_edge_offsets: wp.array(dtype=int),
        vertex_unique_edge_offsets: wp.array(dtype=int),
        vertex_unique_edge_count: wp.array(dtype=int),
        uncompressed_edge_ends: wp.array(dtype=int),
        uncompressed_edge_quads: wp.array2d(dtype=int),
        edge_vertex_indices: wp.array(dtype=wp.vec2i),
        edge_quad_indices: wp.array(dtype=wp.vec2i),
        boundary_mask: wp.array(dtype=int),
    ):
        v = wp.tid()

        start_beg = vertex_start_edge_offsets[v]
        unique_beg = vertex_unique_edge_offsets[v]
        unique_count = vertex_unique_edge_count[v]

        for e in range(unique_count):
            src_index = start_beg + e
            edge_index = unique_beg + e

            edge_vertex_indices[edge_index] = wp.vec2i(v, uncompressed_edge_ends[src_index])

            q0 = uncompressed_edge_quads[src_index, 0]
            q1 = uncompressed_edge_quads[src_index, 1]
            edge_quad_indices[edge_index] = wp.vec2i(q0, q1)
            if q0 == q1:
                boundary_mask[edge_index] = 1
            else:
                boundary_mask[edge_index] = 0

    # Edge orientation kernels — separate for 2D and 3D
    @wp.kernel
    def _orient_edges_2d(
        edge_vertex_indices: wp.array(dtype=wp.vec2i),
        edge_quad_indices: wp.array(dtype=wp.vec2i),
        quad_vertex_indices: wp.array2d(dtype=int),
        positions: wp.array(dtype=Any),
    ):
        e = wp.tid()

        quad = edge_quad_indices[e][0]
        quad_vidx = quad_vertex_indices[quad]
        edge_vidx = edge_vertex_indices[e]

        quad_centroid = (
            positions[quad_vidx[0]] + positions[quad_vidx[1]] + positions[quad_vidx[2]] + positions[quad_vidx[3]]
        ) / positions[quad_vidx[0]].dtype(4.0)

        v0 = positions[edge_vidx[0]]
        v1 = positions[edge_vidx[1]]

        edge_center = (v1 + v0) / v0.dtype(2.0)
        edge_vec = v1 - v0
        edge_normal = Geometry._element_normal(edge_vec)

        if wp.dot(quad_centroid - edge_center, edge_normal) > 0.0:
            edge_vertex_indices[e] = wp.vec2i(edge_vidx[1], edge_vidx[0])

    @wp.kernel
    def _orient_edges_3d(
        edge_vertex_indices: wp.array(dtype=wp.vec2i),
        edge_quad_indices: wp.array(dtype=wp.vec2i),
        quad_vertex_indices: wp.array2d(dtype=int),
        positions: wp.array(dtype=Any),
    ):
        e = wp.tid()

        quad = edge_quad_indices[e][0]
        quad_vidx = quad_vertex_indices[quad]
        edge_vidx = edge_vertex_indices[e]

        p0 = positions[quad_vidx[0]]
        p1 = positions[quad_vidx[1]]
        p2 = positions[quad_vidx[2]]
        p3 = positions[quad_vidx[3]]

        quad_centroid = (p0 + p1 + p2 + p3) / p0.dtype(4.0)
        quad_normal = wp.cross(p2 - p0, p3 - p1)

        v0 = positions[edge_vidx[0]]
        v1 = positions[edge_vidx[1]]

        edge_center = (v1 + v0) / v0.dtype(2.0)
        edge_vec = v1 - v0
        edge_normal = wp.cross(edge_vec, quad_normal)

        if wp.dot(quad_centroid - edge_center, edge_normal) > 0.0:
            edge_vertex_indices[e] = wp.vec2i(edge_vidx[1], edge_vidx[0])

    # -- Device functions --

    @wp.func
    def cell_bvh_id(cell_arg: Any):
        return cell_arg.topology.quad_bvh

    @wp.func
    def cell_bounds(cell_arg: Any, cell_index: ElementIndex):
        vidx = cell_arg.topology.quad_vertex_indices[cell_index]
        p0 = cell_arg.positions[vidx[0]]
        p1 = cell_arg.positions[vidx[1]]
        p2 = cell_arg.positions[vidx[2]]
        p3 = cell_arg.positions[vidx[3]]
        return wp.min(wp.min(p0, p1), wp.min(p2, p3)), wp.max(wp.max(p0, p1), wp.max(p2, p3))

    @wp.func
    def cell_position(args: Any, s: Any):
        quad_idx = args.topology.quad_vertex_indices[s.element_index]
        w_p = s.element_coords
        w_m = type(s.element_coords)(s.element_coords.dtype(1.0)) - s.element_coords
        return (
            w_m[0] * w_m[1] * args.positions[quad_idx[0]]
            + w_p[0] * w_m[1] * args.positions[quad_idx[1]]
            + w_p[0] * w_p[1] * args.positions[quad_idx[2]]
            + w_m[0] * w_p[1] * args.positions[quad_idx[3]]
        )

    @wp.func
    def cell_deformation_gradient(cell_arg: Any, s: Any):
        quad_idx = cell_arg.topology.quad_vertex_indices[s.element_index]
        w_p = s.element_coords
        w_m = type(s.element_coords)(s.element_coords.dtype(1.0)) - s.element_coords
        return (
            wp.outer(cell_arg.positions[quad_idx[0]], wp.vector(-w_m[1], -w_m[0], dtype=s.element_coords.dtype))
            + wp.outer(cell_arg.positions[quad_idx[1]], wp.vector(w_m[1], -w_p[0], dtype=s.element_coords.dtype))
            + wp.outer(cell_arg.positions[quad_idx[2]], wp.vector(w_p[1], w_p[0], dtype=s.element_coords.dtype))
            + wp.outer(cell_arg.positions[quad_idx[3]], wp.vector(-w_p[1], w_m[0], dtype=s.element_coords.dtype))
        )

    @wp.func
    def side_position(args: Any, s: Any):
        edge_idx = args.topology.edge_vertex_indices[s.element_index]
        return (s.element_coords.dtype(1.0) - s.element_coords[0]) * args.positions[edge_idx[0]] + s.element_coords[
            0
        ] * args.positions[edge_idx[1]]

    @wp.func
    def side_deformation_gradient(args: Any, s: Any):
        edge_idx = args.topology.edge_vertex_indices[s.element_index]
        v0 = args.positions[edge_idx[0]]
        v1 = args.positions[edge_idx[1]]
        return v1 - v0

    @wp.func
    def side_closest_point(args: Any, side_index: ElementIndex, pos: Any):
        edge_idx = args.topology.edge_vertex_indices[side_index]
        p0 = args.positions[edge_idx[0]]
        q = pos - p0
        e = args.positions[edge_idx[1]] - p0
        dist, t = project_on_seg_at_origin(q, e, wp.length_sq(e))
        return make_coords(t), dist

    @wp.func
    def side_inner_cell_index(arg: Any, side_index: ElementIndex):
        return arg.topology.edge_quad_indices[side_index][0]

    @wp.func
    def side_outer_cell_index(arg: Any, side_index: ElementIndex):
        return arg.topology.edge_quad_indices[side_index][1]

    @wp.func
    def side_inner_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        inner_cell_index = Quadmesh.side_inner_cell_index(args, side_index)
        return Quadmesh._edge_to_quad_coords(args.topology, side_index, inner_cell_index, side_coords)

    @wp.func
    def side_outer_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        outer_cell_index = Quadmesh.side_outer_cell_index(args, side_index)
        return Quadmesh._edge_to_quad_coords(args.topology, side_index, outer_cell_index, side_coords)

    @wp.func
    def side_from_cell_coords(args: Any, side_index: ElementIndex, quad_index: ElementIndex, quad_coords: Any):
        return Quadmesh._quad_to_edge_coords(args.topology, side_index, quad_index, quad_coords)


# Backward-compat aliases
class Quadmesh2D(Quadmesh):
    """Two-dimensional quadrilateral mesh."""

    pass


class Quadmesh3D(Quadmesh):
    """Three-dimensional quadrilateral mesh."""

    pass
