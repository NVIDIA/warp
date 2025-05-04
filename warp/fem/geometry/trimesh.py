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
from warp.fem.types import (
    OUTSIDE,
    Coords,
    ElementIndex,
    Sample,
)

from .closest_point import project_on_seg_at_origin, project_on_tri_at_origin
from .element import LinearEdge, Triangle
from .geometry import Geometry


@wp.struct
class TrimeshCellArg:
    tri_vertex_indices: wp.array2d(dtype=int)

    # for global cell lookup
    tri_bvh: wp.uint64


@wp.struct
class TrimeshSideArg:
    cell_arg: TrimeshCellArg
    edge_vertex_indices: wp.array(dtype=wp.vec2i)
    edge_tri_indices: wp.array(dtype=wp.vec2i)


class Trimesh(Geometry):
    """Triangular mesh geometry"""

    def __init__(
        self,
        tri_vertex_indices: wp.array,
        positions: wp.array,
        build_bvh: bool = False,
        temporary_store: Optional[TemporaryStore] = None,
    ):
        """
        Constructs a D-dimensional triangular mesh.

        Args:
            tri_vertex_indices: warp array of shape (num_tris, 3) containing vertex indices for each tri
            positions: warp array of shape (num_vertices, D) containing the position of each vertex
            temporary_store: shared pool from which to allocate temporary arrays
            build_bvh: Whether to also build the triangle BVH, which is necessary for the global `fem.lookup` operator to function without initial guess
        """

        self.tri_vertex_indices = tri_vertex_indices
        self.positions = positions

        self._edge_vertex_indices: wp.array = None
        self._edge_tri_indices: wp.array = None
        self._build_topology(temporary_store)

        # Flip edges so that normals point away from inner cell
        wp.launch(
            kernel=self._orient_edges,
            device=positions.device,
            dim=self.side_count(),
            inputs=[self._edge_vertex_indices, self._edge_tri_indices, self.tri_vertex_indices, self.positions],
        )

        self._make_default_dependent_implementations()
        self.cell_coordinates = self._make_cell_coordinates(assume_linear=True)
        self.side_coordinates = self._make_side_coordinates(assume_linear=True)

        if build_bvh:
            self.build_bvh(self.positions.device)

    def cell_count(self):
        return self.tri_vertex_indices.shape[0]

    def vertex_count(self):
        return self.positions.shape[0]

    def side_count(self):
        return self._edge_vertex_indices.shape[0]

    def boundary_side_count(self):
        return self._boundary_edge_indices.shape[0]

    def reference_cell(self) -> Triangle:
        return Triangle()

    def reference_side(self) -> LinearEdge:
        return LinearEdge()

    @property
    def edge_tri_indices(self) -> wp.array:
        return self._edge_tri_indices

    @property
    def edge_vertex_indices(self) -> wp.array:
        return self._edge_vertex_indices

    @wp.struct
    class SideIndexArg:
        boundary_edge_indices: wp.array(dtype=int)

    @cached_arg_value
    def _cell_topo_arg_value(self, device):
        args = TrimeshCellArg()

        args.tri_vertex_indices = self.tri_vertex_indices.to(device)

        return args

    @cached_arg_value
    def _side_topo_arg_value(self, device):
        args = TrimeshSideArg()

        args.cell_arg = self._cell_topo_arg_value(device)
        args.edge_vertex_indices = self._edge_vertex_indices.to(device)
        args.edge_tri_indices = self._edge_tri_indices.to(device)

        return args

    def cell_arg_value(self, device):
        args = self.CellArg()

        args.topology = self._cell_topo_arg_value(device)
        args.positions = self.positions.to(device)
        args.topology.tri_bvh = self.bvh_id(device)

        return args

    def side_arg_value(self, device):
        args = self.SideArg()

        args.topology = self._side_topo_arg_value(device)
        args.positions = self.positions.to(device)
        args.topology.cell_arg.tri_bvh = self.bvh_id(device)

        return args

    @cached_arg_value
    def side_index_arg_value(self, device) -> SideIndexArg:
        args = self.SideIndexArg()

        args.boundary_edge_indices = self._boundary_edge_indices.to(device)

        return args

    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int):
        """Boundary side to side index"""

        return args.boundary_edge_indices[boundary_side_index]

    @wp.func
    def _edge_to_tri_coords(
        args: TrimeshSideArg, side_index: ElementIndex, tri_index: ElementIndex, side_coords: Coords
    ):
        edge_vidx = args.edge_vertex_indices[side_index]
        tri_vidx = args.cell_arg.tri_vertex_indices[tri_index]

        v0 = tri_vidx[0]
        v1 = tri_vidx[1]

        cx = float(0.0)
        cy = float(0.0)
        cz = float(0.0)

        if edge_vidx[0] == v0:
            cx = 1.0 - side_coords[0]
        elif edge_vidx[0] == v1:
            cy = 1.0 - side_coords[0]
        else:
            cz = 1.0 - side_coords[0]

        if edge_vidx[1] == v0:
            cx = side_coords[0]
        elif edge_vidx[1] == v1:
            cy = side_coords[0]
        else:
            cz = side_coords[0]

        return Coords(cx, cy, cz)

    @wp.func
    def _tri_to_edge_coords(
        args: TrimeshSideArg,
        side_index: ElementIndex,
        tri_index: ElementIndex,
        tri_coords: Coords,
    ):
        edge_vidx = args.edge_vertex_indices[side_index]
        tri_vidx = args.cell_arg.tri_vertex_indices[tri_index]

        start = int(2)
        end = int(2)

        for k in range(2):
            v = tri_vidx[k]
            if edge_vidx[1] == v:
                end = k
            elif edge_vidx[0] == v:
                start = k

        return wp.where(tri_coords[start] + tri_coords[end] > 0.999, Coords(tri_coords[end], 0.0, 0.0), Coords(OUTSIDE))

    def _build_topology(self, temporary_store: TemporaryStore):
        from warp.fem.utils import compress_node_indices, host_read_at_index, masked_indices
        from warp.utils import array_scan

        device = self.tri_vertex_indices.device

        vertex_tri_offsets, vertex_tri_indices = compress_node_indices(
            self.vertex_count(), self.tri_vertex_indices, temporary_store=temporary_store
        )
        self._vertex_tri_offsets = vertex_tri_offsets.detach()
        self._vertex_tri_indices = vertex_tri_indices.detach()

        vertex_start_edge_count = borrow_temporary(temporary_store, dtype=int, device=device, shape=self.vertex_count())
        vertex_start_edge_count.array.zero_()
        vertex_start_edge_offsets = borrow_temporary_like(vertex_start_edge_count, temporary_store=temporary_store)

        vertex_edge_ends = borrow_temporary(temporary_store, dtype=int, device=device, shape=(3 * self.cell_count()))
        vertex_edge_tris = borrow_temporary(temporary_store, dtype=int, device=device, shape=(3 * self.cell_count(), 2))

        # Count face edges starting at each vertex
        wp.launch(
            kernel=Trimesh._count_starting_edges_kernel,
            device=device,
            dim=self.cell_count(),
            inputs=[self.tri_vertex_indices, vertex_start_edge_count.array],
        )

        array_scan(in_array=vertex_start_edge_count.array, out_array=vertex_start_edge_offsets.array, inclusive=False)

        # Count number of unique edges (deduplicate across faces)
        vertex_unique_edge_count = vertex_start_edge_count
        wp.launch(
            kernel=Trimesh._count_unique_starting_edges_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                self._vertex_tri_offsets,
                self._vertex_tri_indices,
                self.tri_vertex_indices,
                vertex_start_edge_offsets.array,
                vertex_unique_edge_count.array,
                vertex_edge_ends.array,
                vertex_edge_tris.array,
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
        self._edge_tri_indices = wp.empty(shape=(edge_count,), dtype=wp.vec2i, device=device)

        boundary_mask = borrow_temporary(temporary_store=temporary_store, shape=(edge_count,), dtype=int, device=device)

        # Compress edge data
        wp.launch(
            kernel=Trimesh._compress_edges_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                vertex_start_edge_offsets.array,
                vertex_unique_edge_offsets.array,
                vertex_unique_edge_count.array,
                vertex_edge_ends.array,
                vertex_edge_tris.array,
                self._edge_vertex_indices,
                self._edge_tri_indices,
                boundary_mask.array,
            ],
        )

        vertex_start_edge_offsets.release()
        vertex_unique_edge_offsets.release()
        vertex_unique_edge_count.release()
        vertex_edge_ends.release()
        vertex_edge_tris.release()

        boundary_edge_indices, _ = masked_indices(boundary_mask.array, temporary_store=temporary_store)
        self._boundary_edge_indices = boundary_edge_indices.detach()

        boundary_mask.release()

    @wp.kernel
    def _count_starting_edges_kernel(
        tri_vertex_indices: wp.array2d(dtype=int), vertex_start_edge_count: wp.array(dtype=int)
    ):
        t = wp.tid()
        for k in range(3):
            v0 = tri_vertex_indices[t, k]
            v1 = tri_vertex_indices[t, (k + 1) % 3]

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
        vertex_tri_offsets: wp.array(dtype=int),
        vertex_tri_indices: wp.array(dtype=int),
        tri_vertex_indices: wp.array2d(dtype=int),
        vertex_start_edge_offsets: wp.array(dtype=int),
        vertex_start_edge_count: wp.array(dtype=int),
        edge_ends: wp.array(dtype=int),
        edge_tris: wp.array2d(dtype=int),
    ):
        v = wp.tid()

        edge_beg = vertex_start_edge_offsets[v]

        tri_beg = vertex_tri_offsets[v]
        tri_end = vertex_tri_offsets[v + 1]

        edge_cur = edge_beg

        for tri in range(tri_beg, tri_end):
            t = vertex_tri_indices[tri]

            for k in range(3):
                v0 = tri_vertex_indices[t, k]
                v1 = tri_vertex_indices[t, (k + 1) % 3]

                if v == wp.min(v0, v1):
                    other_v = wp.max(v0, v1)

                    # Check if other_v has been seen
                    seen_idx = Trimesh._find(other_v, edge_ends, edge_beg, edge_cur)

                    if seen_idx == -1:
                        edge_ends[edge_cur] = other_v
                        edge_tris[edge_cur, 0] = t
                        edge_tris[edge_cur, 1] = t
                        edge_cur += 1
                    else:
                        edge_tris[seen_idx, 1] = t

        vertex_start_edge_count[v] = edge_cur - edge_beg

    @wp.kernel
    def _compress_edges_kernel(
        vertex_start_edge_offsets: wp.array(dtype=int),
        vertex_unique_edge_offsets: wp.array(dtype=int),
        vertex_unique_edge_count: wp.array(dtype=int),
        uncompressed_edge_ends: wp.array(dtype=int),
        uncompressed_edge_tris: wp.array2d(dtype=int),
        edge_vertex_indices: wp.array(dtype=wp.vec2i),
        edge_tri_indices: wp.array(dtype=wp.vec2i),
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

            t0 = uncompressed_edge_tris[src_index, 0]
            t1 = uncompressed_edge_tris[src_index, 1]
            edge_tri_indices[edge_index] = wp.vec2i(t0, t1)
            if t0 == t1:
                boundary_mask[edge_index] = 1
            else:
                boundary_mask[edge_index] = 0

    @wp.func
    def cell_bvh_id(cell_arg: Any):
        return cell_arg.topology.tri_bvh

    @wp.func
    def cell_bounds(cell_arg: Any, cell_index: ElementIndex):
        p0 = cell_arg.positions[cell_arg.topology.tri_vertex_indices[cell_index, 0]]
        p1 = cell_arg.positions[cell_arg.topology.tri_vertex_indices[cell_index, 1]]
        p2 = cell_arg.positions[cell_arg.topology.tri_vertex_indices[cell_index, 2]]

        return wp.min(wp.min(p0, p1), p2), wp.max(wp.max(p0, p1), p2)

    @wp.func
    def cell_position(args: Any, s: Sample):
        tri_idx = args.topology.tri_vertex_indices[s.element_index]
        return (
            s.element_coords[0] * args.positions[tri_idx[0]]
            + s.element_coords[1] * args.positions[tri_idx[1]]
            + s.element_coords[2] * args.positions[tri_idx[2]]
        )

    @wp.func
    def cell_deformation_gradient(args: Any, s: Sample):
        tri_idx = args.topology.tri_vertex_indices[s.element_index]
        p0 = args.positions[tri_idx[0]]
        p1 = args.positions[tri_idx[1]]
        p2 = args.positions[tri_idx[2]]
        return wp.matrix_from_cols(p1 - p0, p2 - p0)

    @wp.func
    def cell_closest_point(args: Any, tri_index: ElementIndex, pos: Any):
        vidx = args.topology.tri_vertex_indices[tri_index]
        p0 = args.positions[vidx[0]]

        q = pos - p0
        e1 = args.positions[vidx[1]] - p0
        e2 = args.positions[vidx[2]] - p0

        dist, coords = project_on_tri_at_origin(q, e1, e2)
        return coords, dist

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
        return arg.topology.edge_tri_indices[side_index][0]

    @wp.func
    def side_outer_cell_index(arg: Any, side_index: ElementIndex):
        return arg.topology.edge_tri_indices[side_index][1]

    @wp.func
    def side_inner_cell_coords(args: Any, side_index: ElementIndex, side_coords: Coords):
        inner_cell_index = Trimesh.side_inner_cell_index(args, side_index)
        return Trimesh._edge_to_tri_coords(args.topology, side_index, inner_cell_index, side_coords)

    @wp.func
    def side_outer_cell_coords(args: Any, side_index: ElementIndex, side_coords: Coords):
        outer_cell_index = Trimesh.side_outer_cell_index(args, side_index)
        return Trimesh._edge_to_tri_coords(args.topology, side_index, outer_cell_index, side_coords)

    @wp.func
    def side_from_cell_coords(
        args: Any,
        side_index: ElementIndex,
        tri_index: ElementIndex,
        tri_coords: Coords,
    ):
        return Trimesh._tri_to_edge_coords(args.topology, side_index, tri_index, tri_coords)


@wp.struct
class Trimesh2DCellArg:
    topology: TrimeshCellArg
    positions: wp.array(dtype=wp.vec2)


@wp.struct
class Trimesh2DSideArg:
    topology: TrimeshSideArg
    positions: wp.array(dtype=wp.vec2)


class Trimesh2D(Trimesh):
    """2D Triangular mesh geometry"""

    dimension = 2
    CellArg = Trimesh2DCellArg
    SideArg = Trimesh2DSideArg

    @wp.func
    def side_to_cell_arg(side_arg: SideArg):
        return Trimesh2DCellArg(side_arg.topology.cell_arg, side_arg.positions)

    @wp.kernel
    def _orient_edges(
        edge_vertex_indices: wp.array(dtype=wp.vec2i),
        edge_tri_indices: wp.array(dtype=wp.vec2i),
        tri_vertex_indices: wp.array2d(dtype=int),
        positions: wp.array(dtype=wp.vec2),
    ):
        e = wp.tid()

        tri = edge_tri_indices[e][0]

        tri_vidx = tri_vertex_indices[tri]
        edge_vidx = edge_vertex_indices[e]

        tri_centroid = (positions[tri_vidx[0]] + positions[tri_vidx[1]] + positions[tri_vidx[2]]) / 3.0

        v0 = positions[edge_vidx[0]]
        v1 = positions[edge_vidx[1]]

        edge_center = 0.5 * (v1 + v0)
        edge_vec = v1 - v0
        edge_normal = Geometry._element_normal(edge_vec)

        # if edge normal points toward first triangle centroid, flip indices
        if wp.dot(tri_centroid - edge_center, edge_normal) > 0.0:
            edge_vertex_indices[e] = wp.vec2i(edge_vidx[1], edge_vidx[0])


@wp.struct
class Trimesh3DCellArg:
    topology: TrimeshCellArg
    positions: wp.array(dtype=wp.vec3)


@wp.struct
class Trimesh3DSideArg:
    topology: TrimeshSideArg
    positions: wp.array(dtype=wp.vec3)


class Trimesh3D(Trimesh):
    """3D Triangular mesh geometry"""

    dimension = 3
    CellArg = Trimesh3DCellArg
    SideArg = Trimesh3DSideArg

    @wp.func
    def side_to_cell_arg(side_arg: SideArg):
        return Trimesh3DCellArg(side_arg.topology.cell_arg, side_arg.positions)

    @wp.kernel
    def _orient_edges(
        edge_vertex_indices: wp.array(dtype=wp.vec2i),
        edge_tri_indices: wp.array(dtype=wp.vec2i),
        tri_vertex_indices: wp.array2d(dtype=int),
        positions: wp.array(dtype=wp.vec3),
    ):
        e = wp.tid()

        tri = edge_tri_indices[e][0]

        tri_vidx = tri_vertex_indices[tri]
        edge_vidx = edge_vertex_indices[e]

        t0 = positions[tri_vidx[0]]
        t1 = positions[tri_vidx[1]]
        t2 = positions[tri_vidx[2]]

        tri_centroid = (t0 + t1 + t2) / 3.0
        tri_normal = wp.cross(t1 - t0, t2 - t0)

        v0 = positions[edge_vidx[0]]
        v1 = positions[edge_vidx[1]]

        edge_center = 0.5 * (v1 + v0)
        edge_vec = v1 - v0
        edge_normal = wp.cross(edge_vec, tri_normal)

        # if edge normal points toward first triangle centroid, flip indices
        if wp.dot(tri_centroid - edge_center, edge_normal) > 0.0:
            edge_vertex_indices[e] = wp.vec2i(edge_vidx[1], edge_vidx[0])
