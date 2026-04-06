# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, ClassVar

import warp as wp
from warp._src.fem import cache, utils
from warp._src.fem.cache import cached_vec_type
from warp._src.fem.types import OUTSIDE, ElementIndex, make_coords, make_free_sample

from .geometry import Geometry
from .nanogrid import NanogridBase

_wp_module_name_ = "warp.fem.geometry.adaptive_nanogrid"

_FACE_LEVEL_BIT = wp.constant(wp.uint8(4))  # follows nanogrid.FACE_OUTER_OFFSET_BIT
_GRID_LEVEL_BIT = wp.constant(wp.int32(19))  # follows nanogrid.GRID_AXIS_FLAG


def _make_adaptive_cell_arg(scalar_type):
    mat33_type = cache.cached_mat_type((3, 3), scalar_type)

    @cache.dynamic_struct(suffix=scalar_type)
    class AdaptiveNanogridCellArg:
        cell_grid: wp.uint64
        cell_ijk: wp.array(dtype=wp.vec3i)
        cell_level: wp.array(dtype=wp.uint8)
        inverse_transform: mat33_type
        cell_volume: scalar_type
        level_count: int

    return AdaptiveNanogridCellArg


def _make_adaptive_side_arg(cell_arg_type, scalar_type):
    vec3_type = cached_vec_type(3, scalar_type)

    @cache.dynamic_struct(suffix=scalar_type)
    class AdaptiveNanogridSideArg:
        cell_arg: cell_arg_type
        face_ijk: wp.array(dtype=wp.vec3i)
        face_cell_indices: wp.array(dtype=wp.vec2i)
        face_flags: wp.array(dtype=wp.uint8)
        face_areas: vec3_type

    return AdaptiveNanogridSideArg


class AdaptiveNanogrid(NanogridBase):
    """Adaptive sparse grid."""

    dimension = 3

    _dynamic_attribute_constructors: ClassVar = {
        # Functions that capture face_tangent_vecs (itself dynamic)
        "side_deformation_gradient": lambda obj: obj._make_side_deformation_gradient(),
        "side_normal": lambda obj: obj._make_side_normal(),
        **Geometry._dynamic_attribute_constructors,
    }

    def __init__(
        self,
        cell_grid: wp.Volume,
        cell_level: wp.array,
        level_count: int,
        temporary_store: cache.TemporaryStore,
        scalar_type: type = wp.float32,
    ):
        """Construct an adaptive sparse grid geometry from an in-memory NanoVDB volume and a list of levels.

        It is not recommended to use this constructor directly; see the helper functions
        :func:`warp.fem.adaptive_nanogrid_from_field` and :func:`warp.fem.adaptive_nanogrid_from_hierarchy`.

        Args:
            cell_grid: A warp volume (ideally backed by an index grid) whose voxels coordinates correspond to the lowest fine-resolution voxel of each cell.
              The cell's extent is then given by the ``cell_level`` array. For instance, a voxel at coordinates ``ijk`` and level ``0`` corresponds to a fine cell at the same coordinates,
              a voxel at coordinates ``2*ijk`` and level ``1`` corresponds to a cell spanning ``2^3`` voxels from ``2*ijk`` to ``2*ijk + (1,1,1)``, etc.
            cell_level: Refinement level for each voxel of the volume. Level 0 is the finest, level ``level_count-1`` is the coarsest.
            level_count: Number of levels in the grid
            scalar_type: Scalar type for coordinate and weight computations (``wp.float32`` or ``wp.float64``)
        """

        if level_count > 8:
            raise ValueError("Too many refinement levels, max 8 supported")

        self.level_count = level_count
        self._cell_level = cell_level

        device = cell_grid.device
        cell_ijk = wp.array(dtype=wp.vec3i, shape=(cell_grid.get_voxel_count(),), device=device)
        cell_grid.get_voxels(out=cell_ijk)

        node_grid = _build_node_grid(cell_ijk, cell_level, cell_grid, temporary_store)
        node_count = node_grid.get_voxel_count()
        node_ijk = wp.array(shape=(node_count,), dtype=wp.vec3i, device=device)
        node_grid.get_voxels(out=node_ijk)

        super().__init__(cell_grid, cell_ijk, node_grid, node_ijk, scalar_type=scalar_type)

        self._stacked_edge_grid = None
        self._stacked_edge_count = 0
        self._stacked_face_grid = None
        self._stacked_face_count = 0

        self.CellArg = _make_adaptive_cell_arg(scalar_type)
        self.SideArg = _make_adaptive_side_arg(self.CellArg, scalar_type)

        cache.setup_dynamic_attributes(self)

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

    def fill_cell_arg(self, arg, device):
        arg.cell_grid = self._cell_grid.id
        arg.cell_ijk = self._cell_ijk
        arg.cell_level = self._cell_level
        arg.inverse_transform = self._inverse_transform
        arg.cell_volume = self._cell_volume
        arg.level_count = self.level_count

    def fill_side_arg(self, arg, device):
        self._ensure_face_grid()
        self.fill_cell_arg(arg.cell_arg, device)
        arg.face_ijk = self._face_ijk.to(device)
        arg.face_flags = self._face_flags.to(device)
        arg.face_cell_indices = self._face_cell_indices.to(device)
        arg.face_areas = self._face_areas

    def supports_cell_lookup(self, device):
        return True

    # -- Dynamic function constructors --

    @wp.func
    def cell_position(args: Any, s: Any):
        scale = s.element_coords.dtype(1 << int(args.cell_level[s.element_index]))
        uvw = (
            type(s.element_coords)(args.cell_ijk[s.element_index])
            + s.element_coords * scale
            - type(s.element_coords)(s.element_coords.dtype(0.5))
        )
        return wp.volume_index_to_world(args.cell_grid, uvw)

    @wp.func
    def cell_deformation_gradient(args: Any, s: Any):
        scale = s.element_coords.dtype(1 << int(args.cell_level[s.element_index]))
        return wp.inverse(args.inverse_transform) * scale

    @wp.func
    def cell_inverse_deformation_gradient(args: Any, s: Any):
        scale = s.element_coords.dtype(1 << int(args.cell_level[s.element_index]))
        return args.inverse_transform / scale

    @wp.func
    def cell_measure(args: Any, s: Any):
        scale = s.element_coords.dtype(1 << int(args.cell_level[s.element_index]))
        return args.cell_volume * scale * scale * scale

    @wp.func
    def cell_normal(args: Any, s: Any):
        return type(s.element_coords)(s.element_coords.dtype(0.0))

    @wp.func
    def _lookup_cell_index(args: Any, i: int, j: int, k: int):
        return AdaptiveNanogrid.find_cell(args.cell_grid, wp.vec3i(i, j, k), args.level_count, args.cell_level)

    @wp.func
    def _cell_coordinates_local(args: Any, cell_index: int, uvw: Any):
        ijk = type(uvw)(args.cell_ijk[cell_index])
        rel_pos = uvw - ijk
        scale = uvw.dtype(1 << int(args.cell_level[cell_index]))
        return rel_pos / scale

    @wp.func
    def _cell_closest_point_local(args: Any, cell_index: int, uvw: Any):
        ijk = type(uvw)(args.cell_ijk[cell_index])
        rel_pos = uvw - ijk
        scale = uvw.dtype(1 << int(args.cell_level[cell_index]))
        coords = wp.min(wp.max(rel_pos / scale, type(uvw)(uvw.dtype(0.0))), type(uvw)(uvw.dtype(1.0)))
        return wp.length_sq(wp.volume_index_to_world_dir(args.cell_grid, coords * scale - rel_pos)), coords

    @wp.func
    def cell_coordinates(args: Any, cell_index: int, pos: Any):
        uvw = wp.volume_world_to_index(args.cell_grid, pos) + type(pos)(pos.dtype(0.5))
        return AdaptiveNanogrid._cell_coordinates_local(args, cell_index, uvw)

    @wp.func
    def cell_closest_point(args: Any, cell_index: int, pos: Any):
        uvw = wp.volume_world_to_index(args.cell_grid, pos) + type(pos)(pos.dtype(0.5))
        dist, coords = AdaptiveNanogrid._cell_closest_point_local(args, cell_index, uvw)
        return coords, dist

    @wp.func
    def side_position(args: Any, s: Any):
        ijk = args.face_ijk[s.element_index]
        flags = args.face_flags[s.element_index]
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)
        scale = s.element_coords.dtype(1 << _get_face_level(flags))

        uvw = type(s.element_coords)(ijk) + scale * NanogridBase._side_to_cell_coords(
            axis, flip, s.element_coords.dtype(0.0), s.element_coords
        )
        return wp.volume_index_to_world(
            args.cell_arg.cell_grid, uvw - type(s.element_coords)(s.element_coords.dtype(0.5))
        )

    def _make_side_deformation_gradient(self):
        SampleType = self.sample_type
        scalar = self._scalar_type
        face_tangent_vecs = self._make_face_tangent_vecs()

        @cache.dynamic_func(suffix=self.name)
        def side_deformation_gradient(args: self.SideArg, s: SampleType):
            flags = args.face_flags[s.element_index]
            axis = NanogridBase._get_face_axis(flags)
            flip = NanogridBase._get_face_inner_offset(flags)
            scale = scalar(1 << _get_face_level(flags))
            v1, v2 = face_tangent_vecs(args.cell_arg.cell_grid, axis, flip)
            return wp.matrix_from_cols(v1, v2) * scale

        return side_deformation_gradient

    @wp.func
    def side_inner_inverse_deformation_gradient(args: Any, s: Any):
        s_cell = make_free_sample(
            AdaptiveNanogrid.side_inner_cell_index(args, s.element_index), type(s.element_coords)()
        )
        return AdaptiveNanogrid.cell_inverse_deformation_gradient(args.cell_arg, s_cell)

    @wp.func
    def side_outer_inverse_deformation_gradient(args: Any, s: Any):
        s_cell = make_free_sample(
            AdaptiveNanogrid.side_outer_cell_index(args, s.element_index), type(s.element_coords)()
        )
        return AdaptiveNanogrid.cell_inverse_deformation_gradient(args.cell_arg, s_cell)

    @wp.func
    def side_measure(args: Any, s: Any):
        flags = args.face_flags[s.element_index]
        axis = NanogridBase._get_face_axis(flags)
        scale = s.element_coords.dtype(1 << _get_face_level(flags))
        return args.face_areas[axis] * scale * scale

    @wp.func
    def side_measure_ratio(args: Any, s: Any):
        flags = args.face_flags[s.element_index]
        axis = NanogridBase._get_face_axis(flags)
        scale = s.element_coords.dtype(1 << _get_face_level(flags))
        return args.face_areas[axis] / (args.cell_arg.cell_volume * scale)

    def _make_side_normal(self):
        SampleType = self.sample_type
        face_tangent_vecs = self._make_face_tangent_vecs()

        @cache.dynamic_func(suffix=self.name)
        def side_normal(args: self.SideArg, s: SampleType):
            flags = args.face_flags[s.element_index]
            axis = NanogridBase._get_face_axis(flags)
            flip = NanogridBase._get_face_inner_offset(flags)
            v1, v2 = face_tangent_vecs(args.cell_arg.cell_grid, axis, flip)
            return wp.cross(v1, v2) / args.face_areas[axis]

        return side_normal

    @wp.func
    def side_inner_cell_index(args: Any, side_index: ElementIndex):
        return args.face_cell_indices[side_index][0]

    @wp.func
    def side_outer_cell_index(args: Any, side_index: ElementIndex):
        return args.face_cell_indices[side_index][1]

    @wp.func
    def side_inner_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        flags = args.face_flags[side_index]
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)
        offset = NanogridBase._get_face_inner_offset(flags)

        same_level_cell_coords = NanogridBase._side_to_cell_coords(
            axis, flip, side_coords.dtype(1.0) - side_coords.dtype(offset), side_coords
        )
        same_level_cell_ijk = args.face_ijk[side_index]
        side_level = _get_face_level(flags)
        same_level_cell_ijk[axis] += (offset - 1) << side_level

        cell_index = AdaptiveNanogrid.side_inner_cell_index(args, side_index)
        cell_level = int(args.cell_arg.cell_level[cell_index])
        cell_ijk = args.cell_arg.cell_ijk[cell_index]

        return _coarse_cell_coords(same_level_cell_ijk, side_level, same_level_cell_coords, cell_ijk, cell_level)

    @wp.func
    def side_outer_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        flags = args.face_flags[side_index]
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)
        offset = NanogridBase._get_face_outer_offset(flags)

        same_level_cell_coords = NanogridBase._side_to_cell_coords(axis, flip, side_coords.dtype(offset), side_coords)
        same_level_cell_ijk = args.face_ijk[side_index]
        side_level = _get_face_level(flags)
        same_level_cell_ijk[axis] -= offset << side_level

        cell_index = AdaptiveNanogrid.side_outer_cell_index(args, side_index)
        cell_level = int(args.cell_arg.cell_level[cell_index])
        cell_ijk = args.cell_arg.cell_ijk[cell_index]

        return _coarse_cell_coords(same_level_cell_ijk, side_level, same_level_cell_coords, cell_ijk, cell_level)

    @wp.func
    def side_from_cell_coords(args: Any, side_index: ElementIndex, element_index: ElementIndex, element_coords: Any):
        flags = args.face_flags[side_index]
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)
        side_level = _get_face_level(flags)
        cell_level = int(args.cell_arg.cell_level[element_index])

        cell_ijk = args.cell_arg.cell_ijk[element_index]
        side_ijk = args.face_ijk[side_index]

        same_level_cell_coords = _coarse_cell_coords(cell_ijk, cell_level, element_coords, side_ijk, side_level)

        on_side = (
            same_level_cell_coords[axis] == element_coords.dtype(0.0)
            and wp.min(same_level_cell_coords) >= element_coords.dtype(0.0)
            and wp.max(same_level_cell_coords) <= element_coords.dtype(1.0)
        )

        return wp.where(
            on_side,
            NanogridBase._cell_to_side_coords(axis, flip, same_level_cell_coords),
            type(element_coords)(element_coords.dtype(OUTSIDE)),
        )

    @wp.func
    def side_to_cell_arg(side_arg: Any):
        return side_arg.cell_arg

    @wp.func
    def side_coordinates(args: Any, side_index: int, pos: Any):
        ijk = args.face_ijk[side_index]
        fine_cell_coords = (
            wp.volume_world_to_index(args.cell_arg.cell_grid, pos) + type(pos)(pos.dtype(0.5)) - type(pos)(ijk)
        )

        flags = args.face_flags[side_index]
        side_level = _get_face_level(flags)
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)

        return NanogridBase._cell_to_side_coords(axis, flip, fine_cell_coords / pos.dtype(1 << side_level))

    @wp.func
    def side_closest_point(args: Any, side_index: int, pos: Any):
        coords = AdaptiveNanogrid.side_coordinates(args, side_index, pos)
        z = pos.dtype(0.0)
        o = pos.dtype(1.0)
        proj_coords = make_coords(wp.clamp(coords[0], z, o), wp.clamp(coords[1], z, o))

        flags = args.face_flags[side_index]
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)
        side_level = _get_face_level(flags)
        cell_coord_offset = NanogridBase._side_to_cell_coords(axis, flip, z, coords - proj_coords) * coords.dtype(
            1 << side_level
        )

        return proj_coords, wp.length_sq(wp.volume_index_to_world_dir(args.cell_arg.cell_grid, cell_coord_offset))

    # -- Topology building --

    def _build_face_grid(self, temporary_store: cache.TemporaryStore | None = None):
        device = self._cell_grid.device

        cell_face_grid = _build_cell_face_grid(self._cell_ijk, self._cell_level, self._cell_grid, temporary_store)

        self._face_grid = _build_completed_face_grid(
            cell_face_grid, self._cell_grid, self.level_count, self._cell_level, temporary_store
        )

        face_count = self._face_grid.get_voxel_count()
        self._face_ijk = wp.array(shape=(face_count,), dtype=wp.vec3i, device=device)
        self._face_grid.get_voxels(out=self._face_ijk)

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
                boundary_face_mask,
            ],
        )
        boundary_face_indices, _ = utils.masked_indices(boundary_face_mask)
        boundary_face_mask.release()
        self._boundary_face_indices = boundary_face_indices.detach()

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

    # -- Static helper functions (integer-only or used by kernels) --

    @wp.func
    def coarse_ijk(ijk: wp.vec3i, level: int):
        return wp.vec3i(ijk[0] >> level, ijk[1] >> level, ijk[2] >> level)

    @wp.func
    def fine_ijk(ijk: wp.vec3i, level: int):
        return wp.vec3i(ijk[0] << level, ijk[1] << level, ijk[2] << level)

    @wp.func
    def encode_axis_and_level(ijk: wp.vec3i, axis: int, level: int):
        for ax in range(3):
            coord = ijk[ax]
            level_flag = ((level >> ax) & 1) << _GRID_LEVEL_BIT
            ijk[ax] = wp.where(coord < 0, coord & ~level_flag, coord | level_flag)

        return NanogridBase._add_axis_flag(ijk, axis)

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


# -- Module-level helper functions used in dynamic closures --


@wp.func
def _get_face_level(flags: wp.uint8):
    return wp.int32(flags >> _FACE_LEVEL_BIT)


@wp.func
def _coarse_cell_coords(
    fine_ijk: wp.vec3i,
    fine_level: int,
    fine_coords: Any,
    coarse_ijk: wp.vec3i,
    coarse_level: int,
):
    return (
        type(fine_coords)(fine_ijk - coarse_ijk) + fine_coords * fine_coords.dtype(1 << fine_level)
    ) / fine_coords.dtype(1 << coarse_level)


# -- Topology-building kernels (precision-independent) --


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
    node_ijk[cell, 0] = NanogridBase._add_axis_flag(ijk, 0)
    node_ijk[cell, 1] = NanogridBase._add_axis_flag(ijk, 1)
    node_ijk[cell, 2] = NanogridBase._add_axis_flag(ijk, 2)

    offset = 1 << int(cell_level[cell])

    node_ijk[cell, 3] = NanogridBase._add_axis_flag(ijk + wp.vec3i(offset, 0, 0), 0)
    node_ijk[cell, 4] = NanogridBase._add_axis_flag(ijk + wp.vec3i(0, offset, 0), 1)
    node_ijk[cell, 5] = NanogridBase._add_axis_flag(ijk + wp.vec3i(0, 0, offset), 2)


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
        _cell_node_indices, dim=cell_nodes.shape, inputs=[cell_ijk, cell_level, cell_nodes], device=cell_ijk.device
    )
    node_grid = wp.Volume.allocate_by_voxels(
        cell_nodes.flatten(), voxel_size=cell_grid.get_voxel_size()[0], device=cell_ijk.device
    )
    cell_nodes.release()
    return node_grid


def _build_cell_face_grid(cell_ijk, cell_level, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]
    cell_faces = cache.borrow_temporary(temporary_store, shape=(cell_count, 6), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(_cell_face_indices, dim=cell_count, inputs=[cell_ijk, cell_level, cell_faces], device=cell_ijk.device)
    face_grid = wp.Volume.allocate_by_voxels(
        cell_faces.flatten(), voxel_size=grid.get_voxel_size()[0], device=cell_ijk.device
    )
    cell_faces.release()
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

    cell_face_grid.get_voxels(out=cell_face_ijk)
    additional_face_count.zero_()
    wp.launch(
        _count_multires_faces,
        dim=cell_face_count,
        device=device,
        inputs=[cell_grid.id, level_count, cell_level, cell_face_ijk, additional_face_count],
    )

    cat_face_count = cell_face_count + int(additional_face_count.numpy()[0])
    cat_face_ijk = cache.borrow_temporary(temporary_store, shape=(cat_face_count,), dtype=wp.vec3i, device=device)
    wp.copy(src=cell_face_ijk, dest=cat_face_ijk, dest_offset=cat_face_count - cell_face_count)

    wp.launch(
        _fill_multires_faces,
        dim=cell_face_count,
        device=device,
        inputs=[cell_grid.id, level_count, cell_level, cell_face_ijk, additional_face_count, cat_face_ijk],
    )
    cell_face_ijk.release()
    additional_face_count.release()

    face_grid = wp.Volume.allocate_by_voxels(
        cat_face_ijk.flatten(), voxel_size=cell_face_grid.get_voxel_size(), device=device
    )
    cat_face_ijk.release()
    return face_grid


def _build_stacked_face_grid(cell_ijk, cell_level, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]
    cell_faces = cache.borrow_temporary(temporary_store, shape=(cell_count, 6), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(
        _cell_stacked_face_indices, dim=cell_count, inputs=[cell_ijk, cell_level, cell_faces], device=cell_ijk.device
    )
    face_grid = wp.Volume.allocate_by_voxels(
        cell_faces.flatten(), voxel_size=grid.get_voxel_size()[0], device=cell_ijk.device
    )
    cell_faces.release()
    return face_grid


def _build_stacked_edge_grid(cell_ijk, cell_level, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]
    cell_edges = cache.borrow_temporary(temporary_store, shape=(cell_count, 12), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(
        _cell_stacked_edge_indices, dim=cell_count, inputs=[cell_ijk, cell_level, cell_edges], device=cell_ijk.device
    )
    edge_grid = wp.Volume.allocate_by_voxels(
        cell_edges.flatten(), voxel_size=grid.get_voxel_size()[0], device=cell_ijk.device
    )
    cell_edges.release()
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
    axis, ijk = NanogridBase._extract_axis_flag(face_ijk[face])
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
    axis, ijk = NanogridBase._extract_axis_flag(face_ijk[face])
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
            added_ijk[offset + f] = NanogridBase._add_axis_flag(f_ijk, axis)


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
    axis, ijk = NanogridBase._extract_axis_flag(face_ijk[face])
    plus_cell_index, minus_cell_index = _find_face_neighbours(cell_grid, ijk, axis, level_count, cell_level)
    inner_cell = wp.where(minus_cell_index == -1, plus_cell_index, minus_cell_index)
    outer_cell = wp.where(plus_cell_index == -1, minus_cell_index, plus_cell_index)
    face_level = wp.min(cell_level[inner_cell], cell_level[outer_cell])
    face_ijk[face] = ijk
    flags = NanogridBase._make_face_flags(axis, plus_cell_index, minus_cell_index) | (face_level << _FACE_LEVEL_BIT)
    face_flags[face] = flags
    boundary_face_mask[face] = NanogridBase._get_boundary_mask(flags)
    face_cell_indices[face] = wp.vec2i(inner_cell, outer_cell)
