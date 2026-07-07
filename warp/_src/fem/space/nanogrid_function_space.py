# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import warp as wp
from warp._src.fem import cache
from warp._src.fem.geometry import AdaptiveNanogrid, Nanogrid
from warp._src.fem.geometry.nanogrid import _build_node_grid
from warp._src.fem.types import ElementIndex

from .shape import CubeBSplineShapeFunctions, CubeShapeFunction
from .topology import SpaceTopology, forward_base_topology

_wp_module_name_ = "warp.fem.space.nanogrid_function_space"


@wp.struct
class NanogridTopologyArg:
    vertex_grid: wp.uint64
    face_grid: wp.uint64
    edge_grid: wp.uint64

    vertex_count: int
    edge_count: int
    face_count: int


class NanogridSpaceTopology(SpaceTopology):
    TopologyArg = NanogridTopologyArg

    def __init__(
        self,
        grid: Nanogrid | AdaptiveNanogrid,
        shape: CubeShapeFunction,
    ):
        self._shape = shape
        super().__init__(grid, shape.NODES_PER_ELEMENT)
        self._grid = grid

        self._vertex_grid = grid.vertex_grid.id

        need_edge_indices = shape.EDGE_NODE_COUNT > 0
        need_face_indices = shape.FACE_NODE_COUNT > 0

        if isinstance(grid, Nanogrid):
            if need_edge_indices:
                edge_grid, self._edge_count = grid._get_topology_edge_grid()
                self._edge_grid = edge_grid.id
            else:
                self._edge_grid = -1
                self._edge_count = 0
            if need_face_indices:
                face_grid, self._face_count = grid._get_topology_face_grid()
                self._face_grid = face_grid.id
            else:
                self._face_grid = -1
                self._face_count = 0
        else:
            self._edge_grid = grid.stacked_edge_grid.id if need_edge_indices else -1
            self._face_grid = grid.stacked_face_grid.id if need_face_indices else -1
            self._edge_count = grid.stacked_edge_count() if need_edge_indices else 0
            self._face_count = grid.stacked_face_count() if need_face_indices else 0

        self.element_node_index = self._make_element_node_index()

    def rebuild(self) -> None:
        """Refresh this topology after rebuilding its Nanogrid.

        The topology references grids that are refreshed by :meth:`warp.fem.Nanogrid.rebuild`, so no additional
        work is required.
        """

    @property
    def name(self):
        return f"{self.geometry.name}_{self._shape.name}"

    def fill_topo_arg(self, arg, device):
        arg.vertex_grid = self._vertex_grid
        arg.face_grid = self._face_grid
        arg.edge_grid = self._edge_grid

        arg.vertex_count = self._grid.vertex_count()
        arg.face_count = self._face_count
        arg.edge_count = self._edge_count

    def _make_element_node_index(self):
        element_node_index_generic = self._make_element_node_index_generic()

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            geo_arg: self._grid.CellArg,
            topo_arg: NanogridTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            ijk = geo_arg.cell_ijk[element_index]
            level = int(0)
            if wp.static(isinstance(self._grid, AdaptiveNanogrid)):
                level = int(geo_arg.cell_level[element_index])
            return element_node_index_generic(topo_arg, element_index, node_index_in_elt, ijk, level)

        return element_node_index

    def node_count(self) -> int:
        return (
            self._grid.vertex_count() * self._shape.VERTEX_NODE_COUNT
            + self._edge_count * self._shape.EDGE_NODE_COUNT
            + self._face_count * self._shape.FACE_NODE_COUNT
            + self._grid.cell_count() * self._shape.INTERIOR_NODE_COUNT
        )

    def _make_element_node_index_generic(self):
        VERTEX_NODE_COUNT = self._shape.VERTEX_NODE_COUNT
        EDGE_NODE_COUNT = self._shape.EDGE_NODE_COUNT
        FACE_NODE_COUNT = self._shape.FACE_NODE_COUNT
        INTERIOR_NODE_COUNT = self._shape.INTERIOR_NODE_COUNT

        @cache.dynamic_func(suffix=self.name)
        def element_node_index_generic(
            topo_arg: NanogridTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
            ijk: wp.vec3i,
            level: int,
        ):
            node_type, type_instance, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if wp.static(VERTEX_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.VERTEX:
                    n_ijk = _cell_vertex_coord(ijk, level, type_instance)
                    return (
                        wp.volume_lookup_index(topo_arg.vertex_grid, n_ijk[0], n_ijk[1], n_ijk[2]) * VERTEX_NODE_COUNT
                        + type_index
                    )

            offset = topo_arg.vertex_count * VERTEX_NODE_COUNT

            if wp.static(EDGE_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.EDGE:
                    axis = type_instance >> 2
                    node_offset = type_instance & 3

                    n_ijk = _cell_edge_coord(ijk, level, axis, node_offset)

                    edge_index = wp.volume_lookup_index(topo_arg.edge_grid, n_ijk[0], n_ijk[1], n_ijk[2])
                    return offset + EDGE_NODE_COUNT * edge_index + type_index

                offset += EDGE_NODE_COUNT * topo_arg.edge_count

            if wp.static(FACE_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.FACE:
                    axis = type_instance >> 1
                    node_offset = type_instance & 1

                    n_ijk = _cell_face_coord(ijk, level, axis, node_offset)

                    face_index = wp.volume_lookup_index(topo_arg.face_grid, n_ijk[0], n_ijk[1], n_ijk[2])
                    return offset + FACE_NODE_COUNT * face_index + type_index

                offset += FACE_NODE_COUNT * topo_arg.face_count

            return offset + INTERIOR_NODE_COUNT * element_index + type_index

        return element_node_index_generic


@wp.func
def _cell_vertex_coord(cell_ijk: wp.vec3i, cell_level: int, n: int):
    return cell_ijk + AdaptiveNanogrid.fine_ijk(wp.vec3i((n & 4) >> 2, (n & 2) >> 1, n & 1), cell_level)


@wp.func
def _cell_edge_coord(cell_ijk: wp.vec3i, cell_level: int, axis: int, offset: int):
    e_ijk = AdaptiveNanogrid.coarse_ijk(cell_ijk, cell_level)
    e_ijk[(axis + 1) % 3] += offset >> 1
    e_ijk[(axis + 2) % 3] += offset & 1
    return AdaptiveNanogrid.encode_axis_and_level(e_ijk, axis, cell_level)


@wp.func
def _cell_face_coord(cell_ijk: wp.vec3i, cell_level: int, axis: int, offset: int):
    f_ijk = AdaptiveNanogrid.coarse_ijk(cell_ijk, cell_level)
    f_ijk[axis] += offset
    return AdaptiveNanogrid.encode_axis_and_level(f_ijk, axis, cell_level)


class NanogridBSplineSpaceTopology(SpaceTopology):
    def __init__(
        self,
        grid: Nanogrid,
        shape: CubeBSplineShapeFunctions,
    ):
        self._shape = shape
        super().__init__(grid, shape.NODES_PER_ELEMENT)
        self._grid = grid

        if self._shape.PADDING == 0:
            self._padded_node_grid = grid.vertex_grid
            self._padded_node_count = grid.vertex_count()
            self._padded_node_candidates = None
            self._padded_node_candidate_mask = None
        elif grid._rebuildable:
            (
                self._padded_node_grid,
                self._padded_node_count,
                self._padded_node_candidates,
                self._padded_node_candidate_mask,
            ) = self._build_rebuildable_padded_node_grid(grid._cell_ijk, grid.cell_grid, self._shape.PADDING)
        else:
            self._padded_node_grid = self._build_padded_node_grid(grid._cell_ijk, grid.cell_grid, self._shape.PADDING)
            self._padded_node_count = self._padded_node_grid.get_voxel_count()
            self._padded_node_candidates = None
            self._padded_node_candidate_mask = None

        self.element_node_index = self._make_element_node_index()

    def rebuild(self) -> None:
        """Refresh this topology after rebuilding its Nanogrid.

        Reusing a B-spline topology after :meth:`warp.fem.Nanogrid.rebuild` is unsupported until this method has
        been called. The topology must have been constructed from a rebuildable Nanogrid.
        """

        if not self._grid._rebuildable:
            raise RuntimeError("B-spline topology was not constructed from a rebuildable Nanogrid")

        if self._padded_node_candidates is None:
            return

        wp.launch(
            _rebuildable_padded_node_indices,
            dim=self._padded_node_candidates.shape,
            inputs=[
                self._grid.cell_grid.id,
                self._grid._cell_ijk,
                self._shape.PADDING,
                self._padded_node_candidates,
                self._padded_node_candidate_mask,
            ],
            device=self._grid.cell_grid.device,
        )
        self._padded_node_grid.rebuild(
            self._padded_node_candidates.flatten(), point_mask=self._padded_node_candidate_mask
        )

    @wp.struct
    class TopologyArg:
        padded_node_grid: wp.uint64

    @property
    def name(self):
        return f"Nanogrid{self._shape.name}"

    def fill_topo_arg(self, arg, device):
        arg.padded_node_grid = self._padded_node_grid.id

    def node_count(self) -> int:
        return self._padded_node_count

    def _make_element_node_index(self):
        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            geo_arg: self._grid.CellArg,
            topo_arg: NanogridBSplineSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_i, node_j, node_k = self._shape._node_ijk(node_index_in_elt)

            cell = geo_arg.cell_ijk[element_index]
            node_x = cell[0] + node_i
            node_y = cell[1] + node_j
            node_z = cell[2] + node_k

            return wp.volume_lookup_index(topo_arg.padded_node_grid, node_x, node_y, node_z)

        return element_node_index

    @staticmethod
    def _build_padded_node_grid(
        cell_ijk: wp.array(dtype=wp.vec3i),
        grid: wp.Volume,
        padding: int,
    ):
        with wp.ScopedDevice(cell_ijk.device):
            for _ in range(padding):
                padded_voxels = wp.zeros((cell_ijk.shape[0], 3, 3, 3), dtype=wp.vec3i)
                wp.launch(_pad_voxels, padded_voxels.shape, (cell_ijk, padded_voxels))
                padded_volume = wp.Volume.allocate_by_voxels(
                    voxel_points=padded_voxels.flatten(),
                    voxel_size=1.0,  # arbitrary
                )
                cell_ijk = wp.array(dtype=wp.vec3i, shape=(padded_volume.get_voxel_count(),), device=cell_ijk.device)
                padded_volume.get_voxels(out=cell_ijk)

            return _build_node_grid(cell_ijk, grid, temporary_store=None)

    @staticmethod
    def _build_rebuildable_padded_node_grid(
        cell_ijk: wp.array(dtype=wp.vec3i),
        grid: wp.Volume,
        padding: int,
    ):
        nodes_per_dim = 2 * padding + 2
        candidates_per_cell = nodes_per_dim**3
        candidate_capacity = cell_ijk.shape[0] * candidates_per_cell
        candidates = wp.empty(
            shape=(cell_ijk.shape[0], nodes_per_dim, nodes_per_dim, nodes_per_dim),
            dtype=wp.vec3i,
            device=cell_ijk.device,
        )
        candidate_mask = wp.empty(candidate_capacity, dtype=wp.int32, device=cell_ijk.device)
        wp.launch(
            _rebuildable_padded_node_indices,
            dim=candidates.shape,
            inputs=[grid.id, cell_ijk, padding, candidates, candidate_mask],
            device=cell_ijk.device,
        )

        rebuild_info = grid.get_rebuild_info()
        max_leaf_nodes = min(candidate_capacity, rebuild_info.max_leaf_node_count * candidates_per_cell)
        max_lower_nodes = min(max_leaf_nodes, rebuild_info.max_lower_node_count * candidates_per_cell)
        max_upper_nodes = min(max_lower_nodes, rebuild_info.max_upper_node_count * candidates_per_cell)
        node_grid = wp.Volume.allocate_by_voxels(
            candidates.flatten(),
            voxel_size=grid.get_voxel_size(),
            device=cell_ijk.device,
            rebuildable=True,
            max_active_voxels=candidate_capacity,
            max_leaf_nodes=max_leaf_nodes,
            max_lower_nodes=max_lower_nodes,
            max_upper_nodes=max_upper_nodes,
            point_mask=candidate_mask,
        )
        return node_grid, candidate_capacity, candidates, candidate_mask


@wp.kernel
def _pad_voxels(voxel_ijk: wp.array(dtype=wp.vec3i), padded_ijk: wp.array4d(dtype=wp.vec3i)):
    pid, i, j, k = wp.tid()
    padded_ijk[pid, i, j, k] = voxel_ijk[pid] + wp.vec3i(i - 1, j - 1, k - 1)


@wp.kernel
def _rebuildable_padded_node_indices(
    cell_grid: wp.uint64,
    cell_ijk: wp.array(dtype=wp.vec3i),
    padding: int,
    node_ijk: wp.array4d(dtype=wp.vec3i),
    node_mask: wp.array(dtype=wp.int32),
):
    cell, i, j, k = wp.tid()
    node_ijk[cell, i, j, k] = cell_ijk[cell] + wp.vec3i(i - padding, j - padding, k - padding)

    nodes_per_dim = 2 * padding + 2
    node = ((cell * nodes_per_dim + i) * nodes_per_dim + j) * nodes_per_dim + k
    node_mask[node] = wp.where(cell < wp.volume_voxel_count(cell_grid), wp.int32(1), wp.int32(0))


def make_nanogrid_space_topology(grid: Nanogrid | AdaptiveNanogrid, shape: CubeShapeFunction):
    if isinstance(shape, CubeBSplineShapeFunctions):
        if isinstance(grid.base, AdaptiveNanogrid):
            raise ValueError(f"Adaptive Nanogrid does not support {shape.name}")
        else:
            return forward_base_topology(NanogridBSplineSpaceTopology, grid, shape)

    if isinstance(shape, CubeShapeFunction):
        return forward_base_topology(NanogridSpaceTopology, grid, shape)

    raise ValueError(f"Unsupported shape function {shape.name}")
