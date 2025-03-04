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

from typing import Optional

import numpy as np

import warp as wp
from warp.fem import cache, utils
from warp.fem.types import NULL_ELEMENT_INDEX, OUTSIDE, Coords, ElementIndex, Sample, make_free_sample

from .element import Cube, Square
from .geometry import Geometry
from .nanogrid import (
    Nanogrid,
    _add_axis_flag,
    _build_node_grid,
    _extract_axis_flag,
    _get_boundary_mask,
    _make_face_flags,
)

_FACE_LEVEL_BIT = wp.constant(wp.uint8(4))  # follows nanogrid.FACE_OUTER_OFFSET_BIT
_GRID_LEVEL_BIT = wp.constant(wp.int32(19))  # follows nanogrid.GRID_AXIS_FLAG


@wp.struct
class AdaptiveNanogridCellArg:
    # Utility device functions
    cell_grid: wp.uint64
    cell_ijk: wp.array(dtype=wp.vec3i)
    cell_level: wp.array(dtype=wp.uint8)
    inverse_transform: wp.mat33
    cell_volume: float
    level_count: int


@wp.struct
class AdaptiveNanogridSideArg:
    # Utility device functions
    cell_arg: AdaptiveNanogridCellArg
    face_ijk: wp.array(dtype=wp.vec3i)
    face_cell_indices: wp.array(dtype=wp.vec2i)
    face_flags: wp.array(dtype=wp.uint8)
    face_areas: wp.vec3


class AdaptiveNanogrid(Geometry):
    """Adaptive sparse grid"""

    dimension = 3

    def __init__(
        self,
        cell_grid: wp.Volume,
        cell_level: wp.array,
        level_count: int,
        temporary_store: cache.TemporaryStore,
    ):
        """
        Constructs an adaptive sparse grid geometry from an in-memory NanoVDB volume and a list of levels.

        It is not recommended to use this constructor directly; see the helper functions :func:`warp.fem.adaptive_nanogrid_from_field` and :func:`warp.fem.adaptive_nanogrid_from_hierarchy`

        Args:
            cell_grid: A warp volume (ideally backed by an index grid) whose voxels coordinates correspond to the lowest fine-resolution voxel of each cell.
              The cell's extent is then given by the `cell_level` array. For instance, a voxel at coordinates ``ijk`` and level ``0`` corresponds to a fine cell at the same coordinates,
              a voxel at coordinates ``2*ijk`` and level ``1`` corresponds to a cell spanning ``2^3`` voxels from ``2*ijk`` to ``2*ijk + (1,1,1)``, etc.
            cell_level: Refinement level for each voxel of the volume. Level 0 is the finest, level ``level_count-1`` is the coarsest.
            level_count: Number of levels in the grid
        """

        if level_count > 8:
            raise ValueError("Too many refinement levels, max 8 supported")

        self.level_count = level_count
        self._cell_grid = cell_grid
        self._cell_level = cell_level

        device = self._cell_grid.device
        self._cell_ijk = wp.array(dtype=wp.vec3i, shape=(cell_grid.get_voxel_count(),), device=device)
        self._cell_grid.get_voxels(out=self._cell_ijk)
        self._cell_grid_info = self._cell_grid.get_grid_info()

        self._node_grid = _build_node_grid(self._cell_ijk, self._cell_level, self._cell_grid, temporary_store)
        node_count = self._node_grid.get_voxel_count()
        self._node_ijk = wp.array(shape=(node_count,), dtype=wp.vec3i, device=device)
        self._node_grid.get_voxels(out=self._node_ijk)

        self._face_grid = None
        self._face_ijk = None

        self._stacked_edge_grid = None
        self._stacked_edge_count = 0
        self._stacked_face_grid = None
        self._stacked_face_count = 0

    @property
    def cell_grid(self) -> wp.Volume:
        return self._cell_grid

    @property
    def vertex_grid(self) -> wp.Volume:
        return self._node_grid

    @property
    def face_grid(self) -> wp.Volume:
        self._ensure_face_grid()
        return self._face_grid

    def cell_count(self):
        return self._cell_ijk.shape[0]

    def vertex_count(self):
        return self._node_ijk.shape[0]

    def side_count(self):
        self._ensure_face_grid()
        return self._face_ijk.shape[0]

    def boundary_side_count(self):
        self._ensure_face_grid()
        return self._boundary_face_indices.shape[0]

    @property
    def stacked_face_grid(self) -> wp.Volume:
        self._ensure_stacked_face_grid()
        return self._stacked_face_grid

    def stacked_face_count(self):
        self._ensure_stacked_face_grid()
        return self._stacked_face_count

    @property
    def stacked_edge_grid(self) -> wp.Volume:
        self._ensure_stacked_edge_grid()
        return self._stacked_edge_grid

    def stacked_edge_count(self):
        self._ensure_stacked_edge_grid()
        return self._stacked_edge_count

    def reference_cell(self) -> Cube:
        return Cube()

    def reference_side(self) -> Square:
        return Square()

    @property
    def transform(self):
        return np.array(self._cell_grid_info.transform_matrix).reshape(3, 3)

    CellArg = AdaptiveNanogridCellArg

    @cache.cached_arg_value
    def cell_arg_value(self, device) -> CellArg:
        args = self.CellArg()
        args.cell_grid = self._cell_grid.id
        args.cell_ijk = self._cell_ijk
        args.cell_level = self._cell_level

        transform = self.transform
        args.inverse_transform = wp.mat33f(np.linalg.inv(transform))
        args.cell_volume = abs(np.linalg.det(transform))
        args.level_count = self.level_count

        return args

    @wp.func
    def _cell_scale(args: CellArg, cell_index: int):
        return float(1 << int(args.cell_level[cell_index]))

    @wp.func
    def cell_position(args: CellArg, s: Sample):
        scale = AdaptiveNanogrid._cell_scale(args, s.element_index)
        uvw = wp.vec3(args.cell_ijk[s.element_index]) + s.element_coords * scale
        return wp.volume_index_to_world(args.cell_grid, uvw - wp.vec3(0.5))

    @wp.func
    def cell_deformation_gradient(args: CellArg, s: Sample):
        scale = AdaptiveNanogrid._cell_scale(args, s.element_index)
        return wp.inverse(args.inverse_transform) * scale

    @wp.func
    def cell_inverse_deformation_gradient(args: CellArg, s: Sample):
        scale = AdaptiveNanogrid._cell_scale(args, s.element_index)
        return args.inverse_transform / scale

    @wp.func
    def _make_sample(args: CellArg, cell_index: int, uvw: wp.vec3):
        ijk = args.cell_ijk[cell_index]
        return make_free_sample(cell_index, (uvw - wp.vec3(ijk)) / AdaptiveNanogrid._cell_scale(args, cell_index))

    @wp.func
    def cell_lookup(args: CellArg, pos: wp.vec3):
        uvw = wp.volume_world_to_index(args.cell_grid, pos) + wp.vec3(0.5)
        ijk = wp.vec3i(int(wp.floor(uvw[0])), int(wp.floor(uvw[1])), int(wp.floor(uvw[2])))
        cell_index = AdaptiveNanogrid.find_cell(args.cell_grid, ijk, args.level_count, args.cell_level)

        if cell_index == -1:
            coords = uvw - wp.vec3(ijk)

            if wp.min(coords) == 0.0 or wp.max(coords) == 1.0:
                il = wp.where(coords[0] > 0.5, 0, -1)
                jl = wp.where(coords[1] > 0.5, 0, -1)
                kl = wp.where(coords[2] > 0.5, 0, -1)

                for n in range(8):
                    ni = n >> 2
                    nj = (n & 2) >> 1
                    nk = n & 1
                    nijk = ijk + wp.vec3i(ni + il, nj + jl, nk + kl)

                    coords = uvw - wp.vec3(nijk)
                    if wp.min(coords) >= 0.0 and wp.max(coords) <= 1.0:
                        cell_index = AdaptiveNanogrid.find_cell(args.cell_grid, nijk, args.level_count, args.cell_level)
                        if cell_index != -1:
                            return AdaptiveNanogrid._make_sample(args, cell_index, uvw)

            return make_free_sample(NULL_ELEMENT_INDEX, Coords(OUTSIDE))

        return AdaptiveNanogrid._make_sample(args, cell_index, uvw)

    @wp.func
    def cell_lookup(args: CellArg, pos: wp.vec3, guess: Sample):
        s_global = AdaptiveNanogrid.cell_lookup(args, pos)

        if s_global.element_index != NULL_ELEMENT_INDEX:
            return s_global

        closest_voxel = int(NULL_ELEMENT_INDEX)
        closest_coords = Coords(OUTSIDE)
        closest_dist = float(1.0e8)

        # project to closest in stencil
        uvw = wp.volume_world_to_index(args.cell_grid, pos) + wp.vec3(0.5)
        cell_ijk = args.cell_ijk[guess.element_index]
        for ni in range(-1, 2):
            for nj in range(-1, 2):
                for nk in range(-1, 2):
                    nijk = cell_ijk + wp.vec3i(ni, nj, nk)
                    cell_idx = AdaptiveNanogrid.find_cell(args.cell_grid, nijk, args.level_count, args.cell_level)
                    if cell_idx != -1:
                        nijk = args.cell_ijk[cell_idx]
                        scale = AdaptiveNanogrid._cell_scale(args, cell_idx)
                        coords = (uvw - wp.vec3(nijk)) / scale
                        dist, proj_coords = Nanogrid._project_on_voxel_at_origin(coords)
                        dist *= scale
                        if dist <= closest_dist:
                            closest_dist = dist
                            closest_voxel = cell_idx
                            closest_coords = proj_coords

        return make_free_sample(closest_voxel, closest_coords)

    @wp.func
    def cell_measure(args: CellArg, s: Sample):
        scale = AdaptiveNanogrid._cell_scale(args, s.element_index)
        return args.cell_volume * scale * scale * scale

    @wp.func
    def cell_normal(args: CellArg, s: Sample):
        return wp.vec3(0.0)

    SideIndexArg = Nanogrid.SideIndexArg
    side_index_arg_value = Nanogrid.side_index_arg_value

    SideArg = AdaptiveNanogridSideArg

    @wp.func
    def side_to_cell_arg(side_arg: SideArg):
        return side_arg.cell_arg

    @cache.cached_arg_value
    def side_arg_value(self, device) -> SideArg:
        self._ensure_face_grid()

        args = self.SideArg()
        args.cell_arg = self.cell_arg_value(device)
        args.face_ijk = self._face_ijk.to(device)
        args.face_flags = self._face_flags.to(device)
        args.face_cell_indices = self._face_cell_indices.to(device)
        transform = self.transform
        args.face_areas = wp.vec3(
            tuple(np.linalg.norm(np.cross(transform[:, k - 2], transform[:, k - 1])) for k in range(3))
        )

        return args

    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int):
        return args.boundary_face_indices[boundary_side_index]

    @wp.func
    def _get_face_level(flags: wp.uint8):
        return wp.int32(flags >> _FACE_LEVEL_BIT)

    @wp.func
    def _get_face_scale(flags: wp.uint8):
        return float(1 << AdaptiveNanogrid._get_face_level(flags))

    @wp.func
    def side_position(args: SideArg, s: Sample):
        ijk = args.face_ijk[s.element_index]
        flags = args.face_flags[s.element_index]
        axis = Nanogrid._get_face_axis(flags)
        scale = AdaptiveNanogrid._get_face_scale(flags)

        uvw = wp.vec3(ijk) + scale * Nanogrid._side_to_cell_coords(axis, 0.0, s.element_coords)

        cell_grid = args.cell_arg.cell_grid
        return wp.volume_index_to_world(cell_grid, uvw - wp.vec3(0.5))

    @wp.func
    def side_deformation_gradient(args: SideArg, s: Sample):
        flags = args.face_flags[s.element_index]
        axis = Nanogrid._get_face_axis(flags)
        flip = Nanogrid._get_face_inner_offset(flags)
        scale = AdaptiveNanogrid._get_face_scale(flags)
        v1, v2 = Nanogrid._face_tangent_vecs(args.cell_arg.cell_grid, axis, flip)
        return wp.matrix_from_cols(v1, v2) * scale

    @wp.func
    def side_inner_inverse_deformation_gradient(args: SideArg, s: Sample):
        s_cell = make_free_sample(AdaptiveNanogrid.side_inner_cell_index(args, s.element_index), Coords())
        return AdaptiveNanogrid.cell_inverse_deformation_gradient(args.cell_arg, s_cell)

    @wp.func
    def side_outer_inverse_deformation_gradient(args: SideArg, s: Sample):
        s_cell = make_free_sample(AdaptiveNanogrid.side_outer_cell_index(args, s.element_index), Coords())
        return AdaptiveNanogrid.cell_inverse_deformation_gradient(args.cell_arg, s_cell)

    @wp.func
    def side_measure(args: SideArg, s: Sample):
        flags = args.face_flags[s.element_index]
        axis = Nanogrid._get_face_axis(flags)
        scale = AdaptiveNanogrid._get_face_scale(flags)
        return args.face_areas[axis] * scale * scale

    @wp.func
    def side_measure_ratio(args: SideArg, s: Sample):
        flags = args.face_flags[s.element_index]
        axis = Nanogrid._get_face_axis(flags)
        scale = AdaptiveNanogrid._get_face_scale(flags)
        return args.face_areas[axis] / (args.cell_arg.cell_volume * scale)

    @wp.func
    def side_normal(args: SideArg, s: Sample):
        flags = args.face_flags[s.element_index]
        axis = Nanogrid._get_face_axis(flags)
        flip = Nanogrid._get_face_inner_offset(flags)

        v1, v2 = Nanogrid._face_tangent_vecs(args.cell_arg.cell_grid, axis, flip)
        return wp.cross(v1, v2) / args.face_areas[axis]

    @wp.func
    def side_inner_cell_index(args: SideArg, side_index: ElementIndex):
        return args.face_cell_indices[side_index][0]

    @wp.func
    def side_outer_cell_index(args: SideArg, side_index: ElementIndex):
        return args.face_cell_indices[side_index][1]

    @wp.func
    def _coarse_cell_coords(
        fine_ijk: wp.vec3i,
        fine_level: int,
        fine_coords: Coords,
        coarse_ijk: wp.vec3i,
        coarse_level: int,
    ):
        return (wp.vec3f(fine_ijk - coarse_ijk) + fine_coords * float(1 << fine_level)) / float(1 << coarse_level)

    @wp.func
    def side_inner_cell_coords(args: SideArg, side_index: ElementIndex, side_coords: Coords):
        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)
        offset = Nanogrid._get_face_inner_offset(flags)

        same_level_cell_coords = Nanogrid._side_to_cell_coords(axis, 1.0 - float(offset), side_coords)
        same_level_cell_ijk = args.face_ijk[side_index]
        side_level = AdaptiveNanogrid._get_face_level(flags)
        same_level_cell_ijk[axis] += (offset - 1) << side_level

        cell_index = AdaptiveNanogrid.side_inner_cell_index(args, side_index)
        cell_level = int(args.cell_arg.cell_level[cell_index])
        cell_ijk = args.cell_arg.cell_ijk[cell_index]

        return AdaptiveNanogrid._coarse_cell_coords(
            same_level_cell_ijk, side_level, same_level_cell_coords, cell_ijk, cell_level
        )

    @wp.func
    def side_outer_cell_coords(args: SideArg, side_index: ElementIndex, side_coords: Coords):
        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)
        offset = Nanogrid._get_face_outer_offset(flags)

        same_level_cell_coords = Nanogrid._side_to_cell_coords(axis, float(offset), side_coords)
        same_level_cell_ijk = args.face_ijk[side_index]
        side_level = AdaptiveNanogrid._get_face_level(flags)
        same_level_cell_ijk[axis] -= offset << side_level

        cell_index = AdaptiveNanogrid.side_outer_cell_index(args, side_index)
        cell_level = int(args.cell_arg.cell_level[cell_index])
        cell_ijk = args.cell_arg.cell_ijk[cell_index]

        return AdaptiveNanogrid._coarse_cell_coords(
            same_level_cell_ijk, side_level, same_level_cell_coords, cell_ijk, cell_level
        )

    @wp.func
    def side_from_cell_coords(
        args: SideArg,
        side_index: ElementIndex,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)
        side_level = AdaptiveNanogrid._get_face_level(flags)
        cell_level = int(args.cell_arg.cell_level[element_index])

        cell_ijk = args.cell_arg.cell_ijk[element_index]
        side_ijk = args.face_ijk[side_index]

        same_level_cell_coords = AdaptiveNanogrid._coarse_cell_coords(
            cell_ijk, cell_level, element_coords, side_ijk, side_level
        )

        on_side = (
            same_level_cell_coords[axis] == 0.0
            and wp.min(same_level_cell_coords) >= 0.0
            and wp.max(same_level_cell_coords) <= 1.0
        )

        return wp.where(
            on_side,
            Coords(same_level_cell_coords[(axis + 1) % 3], same_level_cell_coords[(axis + 2) % 3], 0.0),
            Coords(OUTSIDE),
        )

    def _build_face_grid(self, temporary_store: Optional[cache.TemporaryStore] = None):
        device = self._cell_grid.device

        # Create a first grid with faces from cells
        cell_face_grid = _build_cell_face_grid(self._cell_ijk, self._cell_level, self._cell_grid, temporary_store)

        # Complete faces at resolution boundaries
        self._face_grid = _build_completed_face_grid(
            cell_face_grid, self._cell_grid, self.level_count, self._cell_level, temporary_store
        )

        face_count = self._face_grid.get_voxel_count()
        self._face_ijk = wp.array(shape=(face_count,), dtype=wp.vec3i, device=device)
        self._face_grid.get_voxels(out=self._face_ijk)

        # Finalize our faces, cache flags and neighbour indices
        self._face_cell_indices = wp.array(shape=(face_count,), dtype=wp.vec2i, device=device)
        self._face_flags = wp.array(shape=(face_count,), dtype=wp.uint8, device=device)
        boundary_face_mask = cache.borrow_temporary(temporary_store, shape=(face_count,), dtype=wp.int32, device=device)

        wp.launch(
            _build_face_indices_and_flags,
            dim=face_count,
            device=device,
            inputs=[
                self._cell_grid.id,
                self.level_count,
                self._cell_level,
                self._face_ijk,
                self._face_cell_indices,
                self._face_flags,
                boundary_face_mask.array,
            ],
        )
        boundary_face_indices, _ = utils.masked_indices(boundary_face_mask.array)
        self._boundary_face_indices = boundary_face_indices.detach()

    def _ensure_face_grid(self):
        if self._face_ijk is None:
            self._build_face_grid()

    def _ensure_stacked_edge_grid(self):
        if self._stacked_edge_grid is None:
            self._stacked_edge_grid = _build_stacked_edge_grid(
                self._cell_ijk, self._cell_level, self._cell_grid, temporary_store=None
            )
            self._stacked_edge_count = self._stacked_edge_grid.get_voxel_count()

    def _ensure_stacked_face_grid(self):
        if self._stacked_face_grid is None:
            self._stacked_face_grid = _build_stacked_face_grid(
                self._cell_ijk, self._cell_level, self._cell_grid, temporary_store=None
            )
            self._stacked_face_count = self._stacked_face_grid.get_voxel_count()

    @wp.func
    def coarse_ijk(ijk: wp.vec3i, level: int):
        # technically implementation-defined, but
        # right-shifting negative numbers 1-fills on all our platforms
        return wp.vec3i(ijk[0] >> level, ijk[1] >> level, ijk[2] >> level)

    @wp.func
    def fine_ijk(ijk: wp.vec3i, level: int):
        # Our coords cannot exceed 1<<21,so no worries about overwriting the sign bit
        return wp.vec3i(ijk[0] << level, ijk[1] << level, ijk[2] << level)

    @wp.func
    def encode_axis_and_level(ijk: wp.vec3i, axis: int, level: int):
        # Embed a 3-bit level in the voxel coordinates
        # by switching the _GRID_LEVEL_BIT for each axis

        for ax in range(3):
            coord = ijk[ax]
            level_flag = ((level >> ax) & 1) << _GRID_LEVEL_BIT
            ijk[ax] = wp.where(coord < 0, coord & ~level_flag, coord | level_flag)

        return _add_axis_flag(ijk, axis)

    @wp.func
    def find_cell(
        cell_grid: wp.uint64,
        ijk: wp.vec3i,
        level_count: int,
        cell_level: wp.array(dtype=wp.uint8),
    ):
        for l in range(level_count):
            mask = ~((1 << l) - 1)
            cell_index = wp.volume_lookup_index(cell_grid, ijk[0] & mask, ijk[1] & mask, ijk[2] & mask)
            if cell_index != -1:
                if int(cell_level[cell_index]) >= l:
                    return cell_index

        return -1


@wp.kernel
def _cell_node_indices(
    cell_ijk: wp.array(dtype=wp.vec3i),
    cell_level: wp.array(dtype=wp.uint8),
    node_ijk: wp.array2d(dtype=wp.vec3i),
):
    cell, n = wp.tid()
    level = int(cell_level[cell])
    offset = AdaptiveNanogrid.fine_ijk(wp.vec3i((n & 4) >> 2, (n & 2) >> 1, n & 1), level)
    node_ijk[cell, n] = cell_ijk[cell] + offset


@wp.kernel
def _cell_face_indices(
    cell_ijk: wp.array(dtype=wp.vec3i),
    cell_level: wp.array(dtype=wp.uint8),
    node_ijk: wp.array2d(dtype=wp.vec3i),
):
    cell = wp.tid()
    ijk = cell_ijk[cell]
    node_ijk[cell, 0] = _add_axis_flag(ijk, 0)
    node_ijk[cell, 1] = _add_axis_flag(ijk, 1)
    node_ijk[cell, 2] = _add_axis_flag(ijk, 2)

    offset = 1 << int(cell_level[cell])

    node_ijk[cell, 3] = _add_axis_flag(ijk + wp.vec3i(offset, 0, 0), 0)
    node_ijk[cell, 4] = _add_axis_flag(ijk + wp.vec3i(0, offset, 0), 1)
    node_ijk[cell, 5] = _add_axis_flag(ijk + wp.vec3i(0, 0, offset), 2)


@wp.kernel
def _cell_stacked_face_indices(
    cell_ijk: wp.array(dtype=wp.vec3i),
    cell_level: wp.array(dtype=wp.uint8),
    node_ijk: wp.array2d(dtype=wp.vec3i),
):
    cell = wp.tid()

    level = int(cell_level[cell])
    ijk = AdaptiveNanogrid.coarse_ijk(cell_ijk[cell], level)

    node_ijk[cell, 0] = AdaptiveNanogrid.encode_axis_and_level(ijk, 0, level)
    node_ijk[cell, 1] = AdaptiveNanogrid.encode_axis_and_level(ijk, 1, level)
    node_ijk[cell, 2] = AdaptiveNanogrid.encode_axis_and_level(ijk, 2, level)

    node_ijk[cell, 3] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(1, 0, 0), 0, level)
    node_ijk[cell, 4] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(0, 1, 0), 1, level)
    node_ijk[cell, 5] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(0, 0, 1), 2, level)


@wp.kernel
def _cell_stacked_edge_indices(
    cell_ijk: wp.array(dtype=wp.vec3i),
    cell_level: wp.array(dtype=wp.uint8),
    edge_ijk: wp.array2d(dtype=wp.vec3i),
):
    cell = wp.tid()
    level = int(cell_level[cell])
    ijk = AdaptiveNanogrid.coarse_ijk(cell_ijk[cell], level)

    edge_ijk[cell, 0] = AdaptiveNanogrid.encode_axis_and_level(ijk, 0, level)
    edge_ijk[cell, 1] = AdaptiveNanogrid.encode_axis_and_level(ijk, 1, level)
    edge_ijk[cell, 2] = AdaptiveNanogrid.encode_axis_and_level(ijk, 2, level)

    edge_ijk[cell, 3] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(0, 1, 0), 0, level)
    edge_ijk[cell, 4] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(0, 0, 1), 1, level)
    edge_ijk[cell, 5] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(1, 0, 0), 2, level)

    edge_ijk[cell, 6] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(0, 1, 1), 0, level)
    edge_ijk[cell, 7] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(1, 0, 1), 1, level)
    edge_ijk[cell, 8] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(1, 1, 0), 2, level)

    edge_ijk[cell, 9] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(0, 0, 1), 0, level)
    edge_ijk[cell, 10] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(1, 0, 0), 1, level)
    edge_ijk[cell, 11] = AdaptiveNanogrid.encode_axis_and_level(ijk + wp.vec3i(0, 1, 0), 2, level)


def _build_node_grid(cell_ijk, cell_level, cell_grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]

    cell_nodes = cache.borrow_temporary(temporary_store, shape=(cell_count, 8), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(
        _cell_node_indices,
        dim=cell_nodes.array.shape,
        inputs=[cell_ijk, cell_level, cell_nodes.array],
        device=cell_ijk.device,
    )
    node_grid = wp.Volume.allocate_by_voxels(
        cell_nodes.array.flatten(), voxel_size=cell_grid.get_voxel_size()[0], device=cell_ijk.device
    )

    return node_grid


def _build_cell_face_grid(cell_ijk, cell_level, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]

    cell_faces = cache.borrow_temporary(temporary_store, shape=(cell_count, 6), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(
        _cell_face_indices, dim=cell_count, inputs=[cell_ijk, cell_level, cell_faces.array], device=cell_ijk.device
    )
    face_grid = wp.Volume.allocate_by_voxels(
        cell_faces.array.flatten(), voxel_size=grid.get_voxel_size()[0], device=cell_ijk.device
    )

    return face_grid


def _build_completed_face_grid(
    cell_face_grid: wp.Volume,
    cell_grid: wp.Volume,
    level_count: int,
    cell_level: wp.array,
    temporary_store: cache.TemporaryStore,
):
    device = cell_grid.device

    cell_face_count = cell_face_grid.get_voxel_count()
    cell_face_ijk = cache.borrow_temporary(temporary_store, shape=(cell_face_count,), dtype=wp.vec3i, device=device)

    additional_face_count = cache.borrow_temporary(temporary_store, shape=1, dtype=int, device=device)

    # Count the number of supplemental faces we need to add at resolution boundaries
    cell_face_grid.get_voxels(out=cell_face_ijk.array)
    additional_face_count.array.zero_()
    wp.launch(
        _count_multires_faces,
        dim=cell_face_count,
        device=device,
        inputs=[
            cell_grid.id,
            level_count,
            cell_level,
            cell_face_ijk.array,
            additional_face_count.array,
        ],
    )

    # Cat these new faces with the original ones
    cat_face_count = cell_face_count + int(additional_face_count.array.numpy()[0])
    cat_face_ijk = cache.borrow_temporary(temporary_store, shape=(cat_face_count,), dtype=wp.vec3i, device=device)
    wp.copy(src=cell_face_ijk.array, dest=cat_face_ijk.array, dest_offset=cat_face_count - cell_face_count)

    wp.launch(
        _fill_multires_faces,
        dim=cell_face_count,
        device=device,
        inputs=[
            cell_grid.id,
            level_count,
            cell_level,
            cell_face_ijk.array,
            additional_face_count.array,
            cat_face_ijk.array,
        ],
    )

    # Now recreate a new grid with all those faces
    face_grid = wp.Volume.allocate_by_voxels(
        cat_face_ijk.array.flatten(), voxel_size=cell_face_grid.get_voxel_size(), device=device
    )

    return face_grid


def _build_stacked_face_grid(cell_ijk, cell_level, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]

    cell_faces = cache.borrow_temporary(temporary_store, shape=(cell_count, 6), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(
        _cell_stacked_face_indices,
        dim=cell_count,
        inputs=[cell_ijk, cell_level, cell_faces.array],
        device=cell_ijk.device,
    )
    face_grid = wp.Volume.allocate_by_voxels(
        cell_faces.array.flatten(), voxel_size=grid.get_voxel_size()[0], device=cell_ijk.device
    )

    return face_grid


def _build_stacked_edge_grid(cell_ijk, cell_level, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]

    cell_edges = cache.borrow_temporary(temporary_store, shape=(cell_count, 12), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(
        _cell_stacked_edge_indices,
        dim=cell_count,
        inputs=[cell_ijk, cell_level, cell_edges.array],
        device=cell_ijk.device,
    )
    edge_grid = wp.Volume.allocate_by_voxels(
        cell_edges.array.flatten(), voxel_size=grid.get_voxel_size()[0], device=cell_ijk.device
    )

    return edge_grid


@wp.func
def _find_face_neighbours(
    cell_grid: wp.uint64,
    ijk: wp.vec3i,
    axis: int,
    level_count: int,
    cell_level: wp.array(dtype=wp.uint8),
):
    ijk_minus = ijk
    ijk_minus[axis] -= 1

    plus_cell_index = AdaptiveNanogrid.find_cell(cell_grid, ijk, level_count, cell_level)
    minus_cell_index = AdaptiveNanogrid.find_cell(cell_grid, ijk_minus, level_count, cell_level)
    return plus_cell_index, minus_cell_index


@wp.kernel
def _count_multires_faces(
    cell_grid: wp.uint64,
    level_count: int,
    cell_level: wp.array(dtype=wp.uint8),
    face_ijk: wp.array(dtype=wp.vec3i),
    count: wp.array(dtype=int),
):
    face = wp.tid()

    axis, ijk = _extract_axis_flag(face_ijk[face])

    plus_cell_index, minus_cell_index = _find_face_neighbours(cell_grid, ijk, axis, level_count, cell_level)

    if plus_cell_index == -1 or minus_cell_index == -1:
        return

    plus_level = int(cell_level[plus_cell_index])
    minus_level = int(cell_level[minus_cell_index])
    level_diff = wp.abs(plus_level - minus_level)

    if level_diff != 0:
        fine_face_count = 1 << (2 * level_diff)
        wp.atomic_add(count, 0, fine_face_count)


@wp.kernel
def _fill_multires_faces(
    cell_grid: wp.uint64,
    level_count: int,
    cell_level: wp.array(dtype=wp.uint8),
    face_ijk: wp.array(dtype=wp.vec3i),
    count: wp.array(dtype=int),
    added_ijk: wp.array(dtype=wp.vec3i),
):
    face = wp.tid()

    axis, ijk = _extract_axis_flag(face_ijk[face])
    plus_cell_index, minus_cell_index = _find_face_neighbours(cell_grid, ijk, axis, level_count, cell_level)

    if plus_cell_index == -1 or minus_cell_index == -1:
        return

    plus_level = int(cell_level[plus_cell_index])
    minus_level = int(cell_level[minus_cell_index])
    level_diff = wp.abs(plus_level - minus_level)

    if level_diff != 0:
        fine_face_count = 1 << (2 * level_diff)
        side_mask = (1 << level_diff) - 1

        fine_level = min(plus_level, minus_level)
        base_level = max(plus_level, minus_level)

        base_mask = ~((1 << base_level) - 1)
        base_ijk = wp.vec3i(ijk[0] & base_mask, ijk[1] & base_mask, ijk[2] & base_mask)

        offset = wp.atomic_sub(count, 0, fine_face_count) - fine_face_count
        for f in range(fine_face_count):
            f_ijk = base_ijk
            f_ijk[(axis + 1) % 3] |= (f & side_mask) << fine_level
            f_ijk[(axis + 2) % 3] |= (f >> level_diff) << fine_level
            added_ijk[offset + f] = _add_axis_flag(f_ijk, axis)


@wp.kernel
def _build_face_indices_and_flags(
    cell_grid: wp.uint64,
    level_count: int,
    cell_level: wp.array(dtype=wp.uint8),
    face_ijk: wp.array(dtype=wp.vec3i),
    face_cell_indices: wp.array(dtype=wp.vec2i),
    face_flags: wp.array(dtype=wp.uint8),
    boundary_face_mask: wp.array(dtype=int),
):
    face = wp.tid()

    axis, ijk = _extract_axis_flag(face_ijk[face])

    plus_cell_index, minus_cell_index = _find_face_neighbours(cell_grid, ijk, axis, level_count, cell_level)

    inner_cell = wp.where(minus_cell_index == -1, plus_cell_index, minus_cell_index)
    outer_cell = wp.where(plus_cell_index == -1, minus_cell_index, plus_cell_index)

    face_level = wp.min(cell_level[inner_cell], cell_level[outer_cell])

    face_ijk[face] = ijk
    flags = _make_face_flags(axis, plus_cell_index, minus_cell_index) | (face_level << _FACE_LEVEL_BIT)
    face_flags[face] = flags
    boundary_face_mask[face] = _get_boundary_mask(flags)

    face_cell_indices[face] = wp.vec2i(inner_cell, outer_cell)
