# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from warp.fem.cache import (
    TemporaryStore,
    borrow_temporary,
    borrow_temporary_like,
    cached_arg_value,
)
from warp.fem.types import OUTSIDE, Coords, ElementIndex, Sample

from .closest_point import project_on_seg_at_origin
from .element import LinearEdge, Square
from .geometry import Geometry


@wp.struct
class QuadmeshCellArg:
    quad_vertex_indices: wp.array2d(dtype=int)

    quad_bvh: wp.uint64


@wp.struct
class QuadmeshSideArg:
    cell_arg: QuadmeshCellArg
    edge_vertex_indices: wp.array(dtype=wp.vec2i)
    edge_quad_indices: wp.array(dtype=wp.vec2i)


class Quadmesh(Geometry):
    """Quadrilateral mesh geometry"""

    def __init__(
        self,
        quad_vertex_indices: wp.array,
        positions: wp.array,
        build_bvh: bool = False,
        temporary_store: Optional[TemporaryStore] = None,
    ):
        """
        Constructs a D-dimensional quadrilateral mesh.

        Args:
            quad_vertex_indices: warp array of shape (num_tris, 4) containing vertex indices for each quad, in counter-clockwise order
            positions: warp array of shape (num_vertices, D) containing the position of each vertex
            temporary_store: shared pool from which to allocate temporary arrays
        """

        self.quad_vertex_indices = quad_vertex_indices
        self.positions = positions

        self._edge_vertex_indices: wp.array = None
        self._edge_quad_indices: wp.array = None
        self._vertex_quad_offsets: wp.array = None
        self._vertex_quad_indices: wp.array = None
        self._build_topology(temporary_store)

        # Flip edges so that normals point away from inner cell
        wp.launch(
            kernel=self._orient_edges,
            device=positions.device,
            dim=self.side_count(),
            inputs=[self._edge_vertex_indices, self._edge_quad_indices, self.quad_vertex_indices, self.positions],
        )

        self._make_default_dependent_implementations()
        self.cell_closest_point = self._make_cell_closest_point()
        self.cell_coordinates = self._make_cell_coordinates()
        self.side_coordinates = self._make_side_coordinates(assume_linear=True)

        if build_bvh:
            self.build_bvh(self.positions.device)

    def cell_count(self):
        return self.quad_vertex_indices.shape[0]

    def vertex_count(self):
        return self.positions.shape[0]

    def side_count(self):
        return self._edge_vertex_indices.shape[0]

    def boundary_side_count(self):
        return self._boundary_edge_indices.shape[0]

    def reference_cell(self) -> Square:
        return Square()

    def reference_side(self) -> LinearEdge:
        return LinearEdge()

    @property
    def edge_quad_indices(self) -> wp.array:
        return self._edge_quad_indices

    @property
    def edge_vertex_indices(self) -> wp.array:
        return self._edge_vertex_indices

    @wp.struct
    class SideIndexArg:
        boundary_edge_indices: wp.array(dtype=int)

    @cached_arg_value
    def _cell_topo_arg_value(self, device):
        args = QuadmeshCellArg()

        args.quad_vertex_indices = self.quad_vertex_indices.to(device)

        return args

    @cached_arg_value
    def _side_topo_arg_value(self, device):
        args = QuadmeshSideArg()

        args.cell_arg = self._cell_topo_arg_value(device)
        args.edge_vertex_indices = self._edge_vertex_indices.to(device)
        args.edge_quad_indices = self._edge_quad_indices.to(device)

        return args

    def cell_arg_value(self, device):
        args = self.CellArg()

        args.topology = self._cell_topo_arg_value(device)
        args.positions = self.positions.to(device)

        args.topology.quad_bvh = self.bvh_id(device)

        return args

    def side_arg_value(self, device):
        args = self.SideArg()

        args.topology = self._side_topo_arg_value(device)
        args.positions = self.positions.to(device)

        args.topology.cell_arg.quad_bvh = self.bvh_id(device)

        return args

    @cached_arg_value
    def side_index_arg_value(self, device) -> SideIndexArg:
        args = self.SideIndexArg()

        args.boundary_edge_indices = self._boundary_edge_indices.to(device)

        return args

    @wp.func
    def _edge_to_quad_coords(
        args: QuadmeshSideArg, side_index: ElementIndex, quad_index: ElementIndex, side_coords: Coords
    ):
        edge_vidx = args.edge_vertex_indices[side_index]
        quad_vidx = args.cell_arg.quad_vertex_indices[quad_index]

        vs = edge_vidx[0]
        ve = edge_vidx[1]

        s = side_coords[0]

        if vs == quad_vidx[0]:
            return wp.where(ve == quad_vidx[1], Coords(s, 0.0, 0.0), Coords(0.0, s, 0.0))
        elif vs == quad_vidx[1]:
            return wp.where(ve == quad_vidx[2], Coords(1.0, s, 0.0), Coords(1.0 - s, 0.0, 0.0))
        elif vs == quad_vidx[2]:
            return wp.where(ve == quad_vidx[3], Coords(1.0 - s, 1.0, 0.0), Coords(1.0, 1.0 - s, 0.0))

        return wp.where(ve == quad_vidx[0], Coords(0.0, 1.0 - s, 0.0), Coords(s, 1.0, 0.0))

    @wp.func
    def _quad_to_edge_coords(
        args: QuadmeshSideArg,
        side_index: ElementIndex,
        quad_index: ElementIndex,
        quad_coords: Coords,
    ):
        edge_vidx = args.edge_vertex_indices[side_index]
        quad_vidx = args.cell_arg.quad_vertex_indices[quad_index]

        vs = edge_vidx[0]
        ve = edge_vidx[1]

        cx = quad_coords[0]
        cy = quad_coords[1]

        if vs == quad_vidx[0]:
            oc = wp.where(ve == quad_vidx[1], cy, cx)
            ec = wp.where(ve == quad_vidx[1], cx, cy)
        elif vs == quad_vidx[1]:
            oc = wp.where(ve == quad_vidx[2], 1.0 - cx, cy)
            ec = wp.where(ve == quad_vidx[2], cy, 1.0 - cx)
        elif vs == quad_vidx[2]:
            oc = wp.where(ve == quad_vidx[3], 1.0 - cy, 1.0 - cx)
            ec = wp.where(ve == quad_vidx[3], 1.0 - cx, 1.0 - cy)
        else:
            oc = wp.where(ve == quad_vidx[0], cx, 1.0 - cy)
            ec = wp.where(ve == quad_vidx[0], 1.0 - cy, cx)
        return wp.where(oc == 0.0, Coords(ec, 0.0, 0.0), Coords(OUTSIDE))

    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int):
        """Boundary side to side index"""

        return args.boundary_edge_indices[boundary_side_index]

    def _build_topology(self, temporary_store: TemporaryStore):
        from warp.fem.utils import compress_node_indices, host_read_at_index, masked_indices
        from warp.utils import array_scan

        device = self.quad_vertex_indices.device

        vertex_quad_offsets, vertex_quad_indices = compress_node_indices(
            self.vertex_count(), self.quad_vertex_indices, temporary_store=temporary_store
        )
        self._vertex_quad_offsets = vertex_quad_offsets.detach()
        self._vertex_quad_indices = vertex_quad_indices.detach()

        vertex_start_edge_count = borrow_temporary(temporary_store, dtype=int, device=device, shape=self.vertex_count())
        vertex_start_edge_count.array.zero_()
        vertex_start_edge_offsets = borrow_temporary_like(vertex_start_edge_count, temporary_store=temporary_store)

        vertex_edge_ends = borrow_temporary(temporary_store, dtype=int, device=device, shape=(4 * self.cell_count()))
        vertex_edge_quads = borrow_temporary(
            temporary_store, dtype=int, device=device, shape=(4 * self.cell_count(), 2)
        )

        # Count face edges starting at each vertex
        wp.launch(
            kernel=Quadmesh2D._count_starting_edges_kernel,
            device=device,
            dim=self.cell_count(),
            inputs=[self.quad_vertex_indices, vertex_start_edge_count.array],
        )

        array_scan(in_array=vertex_start_edge_count.array, out_array=vertex_start_edge_offsets.array, inclusive=False)

        # Count number of unique edges (deduplicate across faces)
        vertex_unique_edge_count = vertex_start_edge_count
        wp.launch(
            kernel=Quadmesh2D._count_unique_starting_edges_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                self._vertex_quad_offsets,
                self._vertex_quad_indices,
                self.quad_vertex_indices,
                vertex_start_edge_offsets.array,
                vertex_unique_edge_count.array,
                vertex_edge_ends.array,
                vertex_edge_quads.array,
            ],
        )

        vertex_unique_edge_offsets = borrow_temporary_like(vertex_start_edge_offsets, temporary_store=temporary_store)
        array_scan(in_array=vertex_start_edge_count.array, out_array=vertex_unique_edge_offsets.array, inclusive=False)

        # Get back edge count to host
        edge_count = int(
            host_read_at_index(
                vertex_unique_edge_offsets.array, self.vertex_count() - 1, temporary_store=temporary_store
            )
        )

        self._edge_vertex_indices = wp.empty(shape=(edge_count,), dtype=wp.vec2i, device=device)
        self._edge_quad_indices = wp.empty(shape=(edge_count,), dtype=wp.vec2i, device=device)

        boundary_mask = borrow_temporary(temporary_store=temporary_store, shape=(edge_count,), dtype=int, device=device)

        # Compress edge data
        wp.launch(
            kernel=Quadmesh2D._compress_edges_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                vertex_start_edge_offsets.array,
                vertex_unique_edge_offsets.array,
                vertex_unique_edge_count.array,
                vertex_edge_ends.array,
                vertex_edge_quads.array,
                self._edge_vertex_indices,
                self._edge_quad_indices,
                boundary_mask.array,
            ],
        )

        vertex_start_edge_offsets.release()
        vertex_unique_edge_offsets.release()
        vertex_unique_edge_count.release()
        vertex_edge_ends.release()
        vertex_edge_quads.release()

        boundary_edge_indices, _ = masked_indices(boundary_mask.array, temporary_store=temporary_store)
        self._boundary_edge_indices = boundary_edge_indices.detach()

        boundary_mask.release()

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

                    # Check if other_v has been seen
                    seen_idx = Quadmesh2D._find(other_v, edge_ends, edge_beg, edge_cur)

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

    @wp.func
    def cell_position(args: Any, s: Sample):
        quad_idx = args.topology.quad_vertex_indices[s.element_index]

        w_p = s.element_coords
        w_m = Coords(1.0) - s.element_coords

        # 0 : m m
        # 1 : p m
        # 2 : p p
        # 3 : m p

        return (
            w_m[0] * w_m[1] * args.positions[quad_idx[0]]
            + w_p[0] * w_m[1] * args.positions[quad_idx[1]]
            + w_p[0] * w_p[1] * args.positions[quad_idx[2]]
            + w_m[0] * w_p[1] * args.positions[quad_idx[3]]
        )

    @wp.func
    def cell_deformation_gradient(cell_arg: Any, s: Sample):
        """Deformation gradient at `coords`"""
        quad_idx = cell_arg.topology.quad_vertex_indices[s.element_index]

        w_p = s.element_coords
        w_m = Coords(1.0) - s.element_coords

        return (
            wp.outer(cell_arg.positions[quad_idx[0]], wp.vec2(-w_m[1], -w_m[0]))
            + wp.outer(cell_arg.positions[quad_idx[1]], wp.vec2(w_m[1], -w_p[0]))
            + wp.outer(cell_arg.positions[quad_idx[2]], wp.vec2(w_p[1], w_p[0]))
            + wp.outer(cell_arg.positions[quad_idx[3]], wp.vec2(-w_p[1], w_m[0]))
        )

    @wp.func
    def side_position(args: Any, s: Sample):
        edge_idx = args.topology.edge_vertex_indices[s.element_index]
        return (1.0 - s.element_coords[0]) * args.positions[edge_idx[0]] + s.element_coords[0] * args.positions[
            edge_idx[1]
        ]

    @wp.func
    def side_deformation_gradient(args: Any, s: Sample):
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

        dist, t = project_on_seg_at_origin(q, e, wp.lengh_sq(e))
        return Coords(t, 0.0, 0.0), dist

    @wp.func
    def side_inner_cell_index(arg: Any, side_index: ElementIndex):
        return arg.topology.edge_quad_indices[side_index][0]

    @wp.func
    def side_outer_cell_index(arg: Any, side_index: ElementIndex):
        return arg.topology.edge_quad_indices[side_index][1]

    @wp.func
    def side_inner_cell_coords(args: Any, side_index: ElementIndex, side_coords: Coords):
        inner_cell_index = Quadmesh3D.side_inner_cell_index(args, side_index)
        return Quadmesh._edge_to_quad_coords(args.topology, side_index, inner_cell_index, side_coords)

    @wp.func
    def side_outer_cell_coords(args: Any, side_index: ElementIndex, side_coords: Coords):
        outer_cell_index = Quadmesh3D.side_outer_cell_index(args, side_index)
        return Quadmesh._edge_to_quad_coords(args.topology, side_index, outer_cell_index, side_coords)

    @wp.func
    def side_from_cell_coords(
        args: Any,
        side_index: ElementIndex,
        quad_index: ElementIndex,
        quad_coords: Coords,
    ):
        return Quadmesh._quad_to_edge_coords(args.topology, side_index, quad_index, quad_coords)

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


@wp.struct
class Quadmesh2DCellArg:
    topology: QuadmeshCellArg
    positions: wp.array(dtype=wp.vec2)


@wp.struct
class Quadmesh2DSideArg:
    topology: QuadmeshSideArg
    positions: wp.array(dtype=wp.vec2)


class Quadmesh2D(Quadmesh):
    """Two-dimensional quadrilateral mesh"""

    dimension = 2
    CellArg = Quadmesh2DCellArg
    SideArg = Quadmesh2DSideArg

    @wp.func
    def side_to_cell_arg(side_arg: SideArg):
        return Quadmesh2DCellArg(side_arg.topology.cell_arg, side_arg.positions)

    @wp.kernel
    def _orient_edges(
        edge_vertex_indices: wp.array(dtype=wp.vec2i),
        edge_quad_indices: wp.array(dtype=wp.vec2i),
        quad_vertex_indices: wp.array2d(dtype=int),
        positions: wp.array(dtype=wp.vec2),
    ):
        e = wp.tid()

        tri = edge_quad_indices[e][0]

        quad_vidx = quad_vertex_indices[tri]
        edge_vidx = edge_vertex_indices[e]

        quad_centroid = (
            positions[quad_vidx[0]] + positions[quad_vidx[1]] + positions[quad_vidx[2]] + positions[quad_vidx[3]]
        ) / 4.0

        v0 = positions[edge_vidx[0]]
        v1 = positions[edge_vidx[1]]

        edge_center = 0.5 * (v1 + v0)
        edge_vec = v1 - v0
        edge_normal = Geometry._element_normal(edge_vec)

        # if edge normal points toward first triangle centroid, flip indices
        if wp.dot(quad_centroid - edge_center, edge_normal) > 0.0:
            edge_vertex_indices[e] = wp.vec2i(edge_vidx[1], edge_vidx[0])


@wp.struct
class Quadmesh3DCellArg:
    topology: QuadmeshCellArg
    positions: wp.array(dtype=wp.vec3)


@wp.struct
class Quadmesh3DSideArg:
    topology: QuadmeshSideArg
    positions: wp.array(dtype=wp.vec3)


class Quadmesh3D(Quadmesh):
    """Three-dimensional quadrilateral mesh"""

    dimension = 3
    CellArg = Quadmesh3DCellArg
    SideArg = Quadmesh3DSideArg

    @wp.func
    def side_to_cell_arg(side_arg: SideArg):
        return Quadmesh3DCellArg(side_arg.topology.cell_arg, side_arg.positions)

    @wp.kernel
    def _orient_edges(
        edge_vertex_indices: wp.array(dtype=wp.vec2i),
        edge_quad_indices: wp.array(dtype=wp.vec2i),
        quad_vertex_indices: wp.array2d(dtype=int),
        positions: wp.array(dtype=wp.vec3),
    ):
        e = wp.tid()

        tri = edge_quad_indices[e][0]

        quad_vidx = quad_vertex_indices[tri]
        edge_vidx = edge_vertex_indices[e]

        p0 = positions[quad_vidx[0]]
        p1 = positions[quad_vidx[1]]
        p2 = positions[quad_vidx[2]]
        p3 = positions[quad_vidx[3]]

        quad_centroid = (
            positions[quad_vidx[0]] + positions[quad_vidx[1]] + positions[quad_vidx[2]] + positions[quad_vidx[3]]
        ) / 4.0
        quad_normal = wp.cross(p2 - p0, p3 - p1)

        v0 = positions[edge_vidx[0]]
        v1 = positions[edge_vidx[1]]

        edge_center = 0.5 * (v1 + v0)
        edge_vec = v1 - v0
        edge_normal = wp.cross(edge_vec, quad_normal)

        # if edge normal points toward first triangle centroid, flip indices
        if wp.dot(quad_centroid - edge_center, edge_normal) > 0.0:
            edge_vertex_indices[e] = wp.vec2i(edge_vidx[1], edge_vidx[0])
