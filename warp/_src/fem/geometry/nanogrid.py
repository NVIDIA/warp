# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from functools import cached_property
from typing import Any, ClassVar

import numpy as np

import warp as wp
from warp._src.fem import cache, utils
from warp._src.fem.cache import cached_vec_type
from warp._src.fem.types import NULL_ELEMENT_INDEX, OUTSIDE, ElementIndex, Sample, make_coords, make_free_sample
from warp._src.logger import log_warning

from .element import Element
from .geometry import Geometry, _array_load

_wp_module_name_ = "warp.fem.geometry.nanogrid"


class NanogridBase(Geometry):
    """Base class for regular and adaptive Nanogrid."""

    dimension = 3

    # Flag used for building edge/face grids to disambiguiate axis within the grid
    GRID_AXIS_FLAG = wp.constant(wp.int32(1 << 20))

    FACE_AXIS_MASK = wp.constant(wp.uint8((1 << 2) - 1))
    FACE_INNER_OFFSET_BIT = wp.constant(wp.uint8(2))
    FACE_OUTER_OFFSET_BIT = wp.constant(wp.uint8(3))

    def __init__(
        self,
        cell_grid: wp.Volume,
        cell_ijk: "wp.array(dtype=wp.vec3i)",
        node_grid: wp.Volume,
        node_ijk: "wp.array(dtype=wp.vec3i)",
        scalar_type: type = wp.float32,
        cell_env: wp.array | None = None,
        env_offsets: wp.array | None = None,
    ):
        self._cell_grid = cell_grid
        self._cell_ijk = cell_ijk
        self._node_grid = node_grid
        self._node_ijk = node_ijk
        self._scalar_type = scalar_type
        device = cell_grid.device

        if env_offsets is None:
            env_offsets = wp.array(np.zeros((1, 3), dtype=np.int32), dtype=wp.vec3i, device=device)
        elif env_offsets.device != device:
            env_offsets = env_offsets.to(device)

        self._env_offsets = env_offsets
        self._env_count = env_offsets.shape[0]

        if cell_env is None:
            cell_env = wp.zeros(shape=cell_ijk.shape, dtype=int, device=device)
        elif cell_env.device != device:
            cell_env = cell_env.to(device)

        if cell_env.shape[0] != cell_ijk.shape[0]:
            raise ValueError("Cell environment array must have one entry per Nanogrid cell")

        self._cell_env = cell_env

        self._face_grid = None
        self._face_ijk = None
        self._face_env = None
        self._boundary_face_indices = None

        self._cell_grid_info = cell_grid.get_grid_info()
        self._init_transform()

    @property
    def scalar_type(self):
        return self._scalar_type

    def reference_cell(self) -> Element:
        return Element.CUBE

    def reference_side(self) -> Element:
        return Element.SQUARE

    def _init_transform(self):
        scalar = self._scalar_type
        mat33_type = cache.cached_mat_type((3, 3), scalar)
        vec3_type = cached_vec_type(3, scalar)

        transform_np = np.array(self.transform, dtype=np.float64).reshape(3, 3)

        diag = np.diag(transform_np)
        if np.max(np.abs(transform_np - np.diag(diag))) < 1.0e-6 * np.max(diag):
            inv_diag = 1.0 / diag
            self._inverse_transform = mat33_type(inv_diag[0], 0.0, 0.0, 0.0, inv_diag[1], 0.0, 0.0, 0.0, inv_diag[2])
            self._cell_volume = scalar(float(diag[0] * diag[1] * diag[2]))
            self._face_areas = vec3_type(float(diag[1] * diag[2]), float(diag[2] * diag[0]), float(diag[0] * diag[1]))
        else:
            inv_np = np.linalg.inv(transform_np)
            self._inverse_transform = mat33_type(*inv_np.flatten().tolist())
            self._cell_volume = scalar(float(abs(np.linalg.det(transform_np))))
            self._face_areas = vec3_type(
                *[float(np.linalg.norm(np.cross(transform_np[:, k - 2], transform_np[:, k - 1]))) for k in range(3)]
            )

    @property
    def transform(self):
        """Transform matrix mapping index to world space."""
        return self._cell_grid_info.transform_matrix

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

    @property
    def cell_env(self) -> wp.array:
        return self._cell_env

    @property
    def env_offsets(self) -> wp.array:
        return self._env_offsets

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

    def environment_count(self):
        return self._env_count

    @wp.struct
    class SideIndexArg:
        boundary_face_indices: wp.array(dtype=int)

    def fill_side_index_arg(self, arg: SideIndexArg, device):
        self._ensure_face_grid()
        arg.boundary_face_indices = self._boundary_face_indices.to(device)

    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int):
        return args.boundary_face_indices[boundary_side_index]

    def make_filtered_cell_lookup(grid_geo, filter_func: wp.Function = None):
        suffix = f"{grid_geo.name}{filter_func.key if filter_func is not None else ''}"
        scalar = grid_geo._scalar_type
        vec3_type = cached_vec_type(3, scalar)
        CoordsType = grid_geo.coords_type

        if grid_geo.environment_count() <= 1:

            @cache.dynamic_func(suffix=suffix)
            def cell_lookup_default(
                args: grid_geo.CellArg, pos: vec3_type, max_dist: float, filter_data: Any, filter_target: Any
            ):
                grid = args.cell_grid

                uvw = wp.volume_world_to_index(grid, pos) + vec3_type(scalar(0.5))
                i, j, k = int(wp.floor(uvw[0])), int(wp.floor(uvw[1])), int(wp.floor(uvw[2]))
                cell_index = grid_geo._lookup_cell_index(args, i, j, k)

                if cell_index != -1:
                    coords = grid_geo._cell_coordinates_local(args, cell_index, uvw)
                    if wp.static(filter_func is None):
                        return make_free_sample(cell_index, coords)
                    else:
                        if filter_func(filter_data, cell_index) == filter_target:
                            return make_free_sample(cell_index, coords)

                cell_size = vec3_type(
                    wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(1.0), scalar(0.0), scalar(0.0)))),
                    wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(0.0), scalar(1.0), scalar(0.0)))),
                    wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(0.0), scalar(0.0), scalar(1.0)))),
                )

                offset = scalar(0.5)
                min_cell_size = wp.min(cell_size)
                max_offset = wp.ceil(max_dist / min_cell_size)
                scales = wp.cw_div(vec3_type(min_cell_size), vec3_type(cell_size))

                closest_cell = NULL_ELEMENT_INDEX
                closest_coords = CoordsType()

                while closest_cell == NULL_ELEMENT_INDEX:
                    uvw_min = wp.vec3i(uvw - offset * scales)
                    uvw_max = wp.vec3i(uvw + offset * scales) + wp.vec3i(1)

                    closest_dist = min_cell_size * min_cell_size * scalar(offset * offset)

                    for i in range(uvw_min[0], uvw_max[0]):
                        for j in range(uvw_min[1], uvw_max[1]):
                            for k in range(uvw_min[2], uvw_max[2]):
                                cell_index = grid_geo._lookup_cell_index(args, i, j, k)
                                if cell_index == -1:
                                    continue

                                if wp.static(filter_func is not None):
                                    if filter_func(filter_data, cell_index) != filter_target:
                                        continue
                                dist, coords = grid_geo._cell_closest_point_local(args, cell_index, uvw)

                                if dist <= closest_dist:
                                    closest_dist = dist
                                    closest_coords = coords
                                    closest_cell = cell_index

                    if offset >= max_offset:
                        break
                    offset = wp.min(scalar(3.0) * offset, max_offset)

                return make_free_sample(closest_cell, closest_coords)

        @cache.dynamic_func(suffix=(suffix, "env"))
        def cell_lookup_env(
            args: grid_geo.CellArg,
            pos: vec3_type,
            max_dist: float,
            filter_data: Any,
            filter_target: Any,
            env_index: int,
        ):
            grid = args.cell_grid

            uvw = wp.volume_world_to_index(grid, pos) + vec3_type(scalar(0.5))
            env_offset = args.env_offsets[env_index]
            packed_uvw = uvw + vec3_type(env_offset)
            i, j, k = int(wp.floor(packed_uvw[0])), int(wp.floor(packed_uvw[1])), int(wp.floor(packed_uvw[2]))
            cell_index = grid_geo._lookup_cell_index(args, i, j, k)

            if cell_index != -1:
                if args.cell_env[cell_index] == env_index:
                    coords = grid_geo._cell_coordinates_local(args, cell_index, uvw)
                    if wp.static(filter_func is None):
                        return make_free_sample(cell_index, coords)
                    else:
                        if filter_func(filter_data, cell_index) == filter_target:
                            return make_free_sample(cell_index, coords)

            cell_size = vec3_type(
                wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(1.0), scalar(0.0), scalar(0.0)))),
                wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(0.0), scalar(1.0), scalar(0.0)))),
                wp.length(wp.volume_index_to_world_dir(grid, vec3_type(scalar(0.0), scalar(0.0), scalar(1.0)))),
            )

            offset = scalar(0.5)
            min_cell_size = wp.min(cell_size)
            max_offset = wp.ceil(max_dist / min_cell_size)
            scales = wp.cw_div(vec3_type(min_cell_size), vec3_type(cell_size))

            closest_cell = NULL_ELEMENT_INDEX
            closest_coords = CoordsType()

            while closest_cell == NULL_ELEMENT_INDEX:
                uvw_min = wp.vec3i(uvw - offset * scales) + env_offset
                uvw_max = wp.vec3i(uvw + offset * scales) + wp.vec3i(1) + env_offset

                closest_dist = min_cell_size * min_cell_size * scalar(offset * offset)

                for i in range(uvw_min[0], uvw_max[0]):
                    for j in range(uvw_min[1], uvw_max[1]):
                        for k in range(uvw_min[2], uvw_max[2]):
                            cell_index = grid_geo._lookup_cell_index(args, i, j, k)
                            if cell_index == -1:
                                continue

                            if args.cell_env[cell_index] != env_index:
                                continue

                            if wp.static(filter_func is not None):
                                if filter_func(filter_data, cell_index) != filter_target:
                                    continue
                            dist, coords = grid_geo._cell_closest_point_local(args, cell_index, uvw)

                            if dist <= closest_dist:
                                closest_dist = dist
                                closest_coords = coords
                                closest_cell = cell_index

                if offset >= max_offset:
                    break
                offset = wp.min(scalar(3.0) * offset, max_offset)

            return make_free_sample(closest_cell, closest_coords)

        if grid_geo.environment_count() <= 1:

            @cache.dynamic_func(suffix=suffix, allow_overloads=True)
            def cell_lookup(
                args: grid_geo.CellArg, pos: vec3_type, max_dist: float, filter_data: Any, filter_target: Any
            ):
                return cell_lookup_default(args, pos, max_dist, filter_data, filter_target)

        @cache.dynamic_func(suffix=suffix, allow_overloads=True)
        def cell_lookup(
            args: grid_geo.CellArg,
            pos: vec3_type,
            max_dist: float,
            filter_data: Any,
            filter_target: Any,
            env_index: int,
        ):
            return cell_lookup_env(args, pos, max_dist, filter_data, filter_target, env_index)

        return cell_lookup

    @cached_property
    def cell_lookup(self) -> wp.Function:
        """Device function for looking up the closest cell to a position."""
        unfiltered_cell_lookup = self.make_filtered_cell_lookup(filter_func=None)

        null_filter_data = 0
        null_filter_target = 0

        pos_type = cache.cached_vec_type(self.dimension, dtype=self.scalar_type)
        SampleType = self.sample_type
        lookup_suffix = (self.name, self.environment_count() > 1)

        if self.environment_count() <= 1:

            @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
            def cell_lookup(args: self.CellArg, pos: pos_type, max_dist: float):
                return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target)

        @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
        def cell_lookup(args: self.CellArg, pos: pos_type, max_dist: float, env_index: int):
            return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target, env_index)

        @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
        def cell_lookup(args: self.CellArg, pos: pos_type, guess: SampleType):
            guess_pos = self.cell_position(args, guess)
            max_dist = wp.length(guess_pos - pos)
            if wp.static(self.environment_count() <= 1):
                return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target)

            env_index = self.cell_environment_index(args, guess.element_index)
            return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target, env_index)

        if self.environment_count() <= 1:

            @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
            def cell_lookup(args: self.CellArg, pos: pos_type):
                max_dist = 0.0
                return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target)

        @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
        def cell_lookup(args: self.CellArg, pos: pos_type, env_index: int):
            max_dist = 0.0
            return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target, env_index)

        filtered_cell_lookup = self.make_filtered_cell_lookup(filter_func=_array_load)

        if self.environment_count() <= 1:

            @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
            def cell_lookup(
                args: self.CellArg,
                pos: pos_type,
                max_dist: float,
                filter_array: wp.array(dtype=Any),
                filter_target: Any,
            ):
                return filtered_cell_lookup(args, pos, max_dist, filter_array, filter_target)

        @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
        def cell_lookup(
            args: self.CellArg,
            pos: pos_type,
            max_dist: float,
            filter_array: wp.array(dtype=Any),
            filter_target: Any,
            env_index: int,
        ):
            return filtered_cell_lookup(args, pos, max_dist, filter_array, filter_target, env_index)

        if self.environment_count() <= 1:

            @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
            def cell_lookup(args: self.CellArg, pos: pos_type, filter_array: wp.array(dtype=Any), filter_target: Any):
                max_dist = 0.0
                return filtered_cell_lookup(args, pos, max_dist, filter_array, filter_target)

        @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
        def cell_lookup(
            args: self.CellArg,
            pos: pos_type,
            filter_array: wp.array(dtype=Any),
            filter_target: Any,
            env_index: int,
        ):
            max_dist = 0.0
            return filtered_cell_lookup(args, pos, max_dist, filter_array, filter_target, env_index)

        return cell_lookup

    def _ensure_face_grid(self):
        if self._face_ijk is None:
            self._build_face_grid()

    # -- Static integer-only helpers --

    @wp.func
    def _add_axis_flag(ijk: wp.vec3i, axis: int):
        coord = ijk[axis]
        ijk[axis] = wp.where(coord < 0, coord & (~NanogridBase.GRID_AXIS_FLAG), coord | NanogridBase.GRID_AXIS_FLAG)
        return ijk

    @wp.func
    def _extract_axis_flag(ijk: wp.vec3i):
        for ax in range(3):
            coord = ijk[ax]
            if coord < 0:
                if (ijk[ax] & NanogridBase.GRID_AXIS_FLAG) == 0:
                    ijk[ax] = ijk[ax] | NanogridBase.GRID_AXIS_FLAG
                    return ax, ijk
            else:
                if (ijk[ax] & NanogridBase.GRID_AXIS_FLAG) != 0:
                    ijk[ax] = ijk[ax] & (~NanogridBase.GRID_AXIS_FLAG)
                    return ax, ijk
        return -1, ijk

    @wp.func
    def _make_face_flags(axis: int, plus_cell_index: int, minus_cell_index: int):
        plus_boundary = wp.uint8(wp.where(plus_cell_index == -1, 1, 0)) << NanogridBase.FACE_OUTER_OFFSET_BIT
        minus_boundary = wp.uint8(wp.where(minus_cell_index == -1, 1, 0)) << NanogridBase.FACE_INNER_OFFSET_BIT
        return wp.uint8(axis) | plus_boundary | minus_boundary

    @wp.func
    def _get_boundary_mask(flags: wp.uint8):
        return int((flags >> NanogridBase.FACE_OUTER_OFFSET_BIT) | (flags >> NanogridBase.FACE_INNER_OFFSET_BIT)) & 1

    @wp.func
    def _get_face_axis(flags: wp.uint8):
        return wp.int32(flags & NanogridBase.FACE_AXIS_MASK)

    @wp.func
    def _get_face_inner_offset(flags: wp.uint8):
        return wp.int32(flags >> NanogridBase.FACE_INNER_OFFSET_BIT) & 1

    @wp.func
    def _get_face_outer_offset(flags: wp.uint8):
        return wp.int32(flags >> NanogridBase.FACE_OUTER_OFFSET_BIT) & 1

    @wp.func
    def cell_environment_index(args: Any, cell_index: ElementIndex):
        return args.cell_env[cell_index]

    @wp.func
    def cell_environment_index(args: Any, s: Sample):
        return args.cell_env[s.element_index]

    @wp.func
    def side_environment_index(args: Any, side_index: ElementIndex):
        return args.face_env[side_index]

    @wp.func
    def side_environment_index(args: Any, s: Sample):
        return args.face_env[s.element_index]

    @wp.func
    def _local_cell_ijk(args: Any, cell_index: ElementIndex):
        env_index = args.cell_env[cell_index]
        return args.cell_ijk[cell_index] - args.env_offsets[env_index]

    @wp.func
    def _local_face_ijk(args: Any, side_index: ElementIndex):
        env_index = args.face_env[side_index]
        return args.face_ijk[side_index] - args.cell_arg.env_offsets[env_index]

    def _make_face_tangent_vecs(self):
        scalar = self._scalar_type
        vec3_type = cached_vec_type(3, scalar)

        @cache.dynamic_func(suffix=self.name)
        def face_tangent_vecs(cell_grid: wp.uint64, axis: int, flip: int):
            u_axis = vec3_type(scalar(0.0))
            v_axis = vec3_type(scalar(0.0))
            u_axis[(axis + 1 + flip) % 3] = scalar(1.0)
            v_axis[(axis + 2 - flip) % 3] = scalar(1.0)
            return wp.volume_index_to_world_dir(cell_grid, u_axis), wp.volume_index_to_world_dir(cell_grid, v_axis)

        return face_tangent_vecs

    @wp.func
    def _side_to_cell_coords(axis: int, flip: int, inner: Any, side_coords: Any):
        uvw = type(side_coords)()
        uvw[axis] = inner
        uvw[(axis + 1 + flip) % 3] = side_coords[0]
        uvw[(axis + 2 - flip) % 3] = side_coords[1]
        return uvw

    @wp.func
    def _cell_to_side_coords(axis: int, flip: int, cell_coords: Any):
        return type(cell_coords)(
            cell_coords[(axis + 1 + flip) % 3], cell_coords[(axis + 2 - flip) % 3], cell_coords.dtype(0.0)
        )


def _make_nanogrid_cell_arg(scalar_type):
    mat33_type = cache.cached_mat_type((3, 3), scalar_type)

    @cache.dynamic_struct(suffix=scalar_type)
    class NanogridCellArg:
        cell_grid: wp.uint64
        cell_ijk: wp.array(dtype=wp.vec3i)
        cell_env: wp.array(dtype=int)
        env_offsets: wp.array(dtype=wp.vec3i)
        inverse_transform: mat33_type
        cell_volume: scalar_type

    return NanogridCellArg


def _make_nanogrid_side_arg(cell_arg_type, scalar_type):
    vec3_type = cached_vec_type(3, scalar_type)

    @cache.dynamic_struct(suffix=scalar_type)
    class NanogridSideArg:
        cell_arg: cell_arg_type
        face_ijk: wp.array(dtype=wp.vec3i)
        face_env: wp.array(dtype=int)
        face_flags: wp.array(dtype=wp.uint8)
        face_areas: vec3_type

    return NanogridSideArg


class Nanogrid(NanogridBase):
    """Sparse grid geometry."""

    _dynamic_attribute_constructors: ClassVar = {
        # Functions that capture face_tangent_vecs (itself dynamic)
        "side_deformation_gradient": lambda obj: obj._make_side_deformation_gradient(),
        "side_normal": lambda obj: obj._make_side_normal(),
        **Geometry._dynamic_attribute_constructors,
    }

    @classmethod
    def from_environment_voxels(
        cls,
        points: wp.array | Sequence[wp.array] | None = None,
        point_envs: wp.array | Sequence[Sequence[int]] | None = None,
        env_count: int | None = None,
        env_offsets: wp.array | Sequence[Sequence[int]] | None = None,
        *,
        point_mask: wp.array | None = None,
        guard_cells: int = 3,
        voxel_size: int | float | Sequence[float] | None = 1.0,
        translation=(0.0, 0.0, 0.0),
        transform=None,
        temporary_store: cache.TemporaryStore | None = None,
        scalar_type: type = wp.float32,
        device=None,
        rebuildable: bool = False,
        max_active_voxels: int | None = None,
        max_leaf_nodes: int | None = None,
        max_lower_nodes: int | None = None,
        max_upper_nodes: int | None = None,
        status: wp.array | None = None,
        cell_ijks: Sequence[wp.array] | None = None,
    ):
        """Construct a sparse grid geometry from environment-tagged active voxel points.

        The active voxel points are interpreted in local FEM space.
        They are packed into a single NanoVDB index grid by adding an environment
        offset, while FEM positions subtract the offset so environments may remain
        colocated in world space.

        Args:
            points: Flat ``wp.vec3i`` or ``wp.vec3f`` array of active voxel points.
                Deprecated: a sequence of per-environment ``wp.vec3i`` arrays is also accepted for compatibility.
            point_envs: Flat ``int32`` array with one environment index per point.
            env_count: Number of environments represented by ``point_envs``.
            env_offsets: Optional packed-grid offsets, one ``wp.vec3i`` per environment.
                If omitted, offsets are generated along the x axis with at least one
                guard region between consecutive environments. Custom offsets are an
                advanced override for callers that need deterministic packed NanoVDB
                coordinates, for example to match an externally built volume. They
                must still keep active cells from different environments from sharing
                packed-grid faces.
            point_mask: Optional ``int32`` array with one entry per point. Points with a zero mask value are ignored.
            guard_cells: Number of empty packed cells between generated environment
                tiles. The default isolates padded B-spline node grids up to degree 3.
            voxel_size: Voxel size for the packed NanoVDB volume. Ignored if ``transform`` is provided.
            translation: Translation between packed index and world spaces.
            transform: Linear transform between packed index and world spaces.
            temporary_store: Shared pool from which to allocate temporary arrays.
            scalar_type: Scalar type for grid coordinates (``wp.float32`` or ``wp.float64``).
            device: CUDA device on which to build the packed volume.
            rebuildable: Whether to allocate persistent capacity for rebuilds.
            max_active_voxels: Maximum number of active voxels for rebuilds. Defaults to the packed cell count.
            max_leaf_nodes: Maximum number of NanoVDB leaf nodes for rebuilds. Defaults to ``max_active_voxels``.
            max_lower_nodes: Maximum number of lower internal nodes for rebuilds. Defaults to ``max_leaf_nodes``.
            max_upper_nodes: Maximum number of upper internal nodes for rebuilds. Defaults to ``max_lower_nodes``.
            status: Optional one-element ``uint32`` array receiving rebuild status flags.
            cell_ijks: Deprecated keyword alias for the old per-environment ``points`` sequence form.
        """

        if cell_ijks is not None:
            if points is not None:
                raise TypeError("points and cell_ijks cannot both be provided")
            points = cell_ijks
        if points is None:
            raise TypeError("points is required")

        if not isinstance(points, wp.array):
            points, point_envs, env_count, env_offsets = _environment_voxels_from_legacy_sequence(
                points, point_envs, env_count, env_offsets, device
            )

        grid, cell_env, env_offsets = _make_environment_cell_grid(
            points,
            point_envs,
            env_count,
            env_offsets=env_offsets,
            point_mask=point_mask,
            voxel_size=voxel_size,
            translation=translation,
            transform=transform,
            temporary_store=temporary_store,
            device=device,
            guard_cells=guard_cells,
            rebuildable=rebuildable,
            max_active_voxels=max_active_voxels,
            max_leaf_nodes=max_leaf_nodes,
            max_lower_nodes=max_lower_nodes,
            max_upper_nodes=max_upper_nodes,
            status=status,
        )
        return cls(
            grid,
            temporary_store=temporary_store,
            scalar_type=scalar_type,
            cell_env=cell_env,
            env_offsets=env_offsets,
            rebuildable=rebuildable,
        )

    def __init__(
        self,
        grid: wp.Volume,
        temporary_store: cache.TemporaryStore | None = None,
        scalar_type: type = wp.float32,
        cell_env: wp.array | None = None,
        env_offsets: wp.array | None = None,
        rebuildable: bool = False,
    ):
        """Construct a sparse grid geometry from an in-memory NanoVDB volume.

        Args:
            grid: The NanoVDB volume. Any type is accepted, but for indexing efficiency an index grid is recommended.
                If ``grid`` is an ``"on"`` index grid, cells will be created for active voxels only, otherwise cells will
                be created for all leaf voxels.
            temporary_store: shared pool from which to allocate temporary arrays
            scalar_type: Scalar type for grid coordinates (``wp.float32`` or ``wp.float64``)
            rebuildable: Whether to retain capacity-sized topology buffers that can be refreshed with
                :meth:`rebuild_topology_from_cells`.
        """

        self._cell_grid = grid
        self._cell_grid_info = grid.get_grid_info()

        grid_rebuildable = grid.is_rebuildable
        if rebuildable and not grid_rebuildable:
            raise RuntimeError("Rebuildable Nanogrids require a rebuildable Volume")

        device = self._cell_grid.device
        voxel_buffer_count = grid.get_voxel_count()
        cell_ijk = wp.array(dtype=wp.vec3i, shape=(voxel_buffer_count,), device=device)
        grid.get_voxels(out=cell_ijk)
        cell_count = grid.get_rebuild_info().max_voxel_count if rebuildable else grid.get_active_stats().voxel_count
        cell_ijk = cell_ijk[:cell_count]

        node_candidates = None
        node_candidate_mask = None
        if rebuildable:
            node_grid, node_candidates, node_candidate_mask = _build_rebuildable_node_grid(cell_ijk, grid)
        else:
            node_grid = _build_node_grid(cell_ijk, grid, temporary_store)
        node_count = node_grid.get_voxel_count()
        node_ijk = wp.array(shape=(node_count,), dtype=wp.vec3i, device=device)
        node_grid.get_voxels(out=node_ijk)

        super().__init__(
            grid,
            cell_ijk,
            node_grid,
            node_ijk,
            scalar_type=scalar_type,
            cell_env=cell_env,
            env_offsets=env_offsets,
        )

        self._rebuildable = rebuildable
        self._node_candidates = node_candidates
        self._node_candidate_mask = node_candidate_mask

        self._edge_count = 0
        self._edge_grid = None

        # Dynamic Arg structs
        self.CellArg = _make_nanogrid_cell_arg(scalar_type)
        self.SideArg = _make_nanogrid_side_arg(self.CellArg, scalar_type)

        cache.setup_dynamic_attributes(self)

    def rebuild(
        self,
        points: wp.array,
        point_envs: wp.array | None = None,
        *,
        status: wp.array | None = None,
        point_mask: wp.array | None = None,
    ) -> wp.array | None:
        """Rebuild the cell grid and refresh Nanogrid topology buffers.

        Args:
            points: Active voxel points. When ``point_envs`` is provided, points are interpreted in local
                environment space and packed through ``env_offsets`` before rebuilding the cell grid.
            point_envs: Optional ``int32`` array with one environment index per point. Required for
                multi-environment Nanogrids.
            status: Optional one-element ``uint32`` array receiving rebuild status flags.
            point_mask: Optional ``int32`` array with one entry per point. Points with a zero mask value are ignored.

        Returns:
            The status array passed to ``status``, or ``None`` if no status array was provided.
        """

        if not self._rebuildable:
            raise RuntimeError("Nanogrid was not constructed in rebuildable mode")

        points = _nanogrid_rebuild_points_array(points, self._cell_grid.device)
        point_count = points.shape[0]
        point_envs = _nanogrid_optional_int_array(point_envs, point_count, self._cell_grid.device, "point_envs")
        point_mask = _nanogrid_optional_int_array(point_mask, point_count, self._cell_grid.device, "point_mask")

        if point_envs is None and self.environment_count() > 1:
            raise ValueError("point_envs is required when rebuilding a multi-environment Nanogrid")

        packed_points = None
        rebuild_points = points
        if point_envs is not None:
            _, inverse_transform, translation_vec = _environment_transform_args(
                None, self._cell_grid_info.translation, self._cell_grid_info.transform_matrix
            )
            packed_points = _pack_environment_points(
                points,
                point_envs,
                point_mask,
                self._env_offsets,
                inverse_transform,
                translation_vec,
                temporary_store=None,
                device=self._cell_grid.device,
            )
            rebuild_points = packed_points

        status = self._cell_grid.rebuild(rebuild_points, status=status, point_mask=point_mask)
        if point_envs is not None:
            _fill_cell_env_from_points(self._cell_grid, rebuild_points, point_envs, point_mask, self._cell_env)
        if packed_points is not None:
            packed_points.release()

        self.rebuild_topology_from_cells()
        return status

    def rebuild_topology_from_cells(self):
        """Refresh Nanogrid topology buffers from the current cell grid."""

        if not self._rebuildable:
            raise RuntimeError("Nanogrid was not constructed in rebuildable mode")

        self._refresh_rebuildable_topology()

    @property
    def edge_grid(self) -> wp.Volume:
        self._ensure_edge_grid()
        return self._edge_grid

    def edge_count(self):
        self._ensure_edge_grid()
        return self._edge_count

    def fill_cell_arg(self, arg, device):
        arg.cell_grid = self._cell_grid.id
        arg.cell_ijk = self._cell_ijk
        arg.cell_env = self._cell_env
        arg.env_offsets = self._env_offsets
        arg.inverse_transform = self._inverse_transform
        arg.cell_volume = self._cell_volume

    def fill_side_arg(self, arg, device):
        self._ensure_face_grid()
        self.fill_cell_arg(arg.cell_arg, device)
        arg.face_ijk = self._face_ijk.to(device)
        arg.face_env = self._face_env.to(device)
        arg.face_flags = self._face_flags.to(device)
        arg.face_areas = self._face_areas

    def supports_cell_lookup(self, device):
        return True

    # -- Dynamic function constructors --

    @wp.func
    def cell_position(args: Any, s: Any):
        uvw = (
            type(s.element_coords)(NanogridBase._local_cell_ijk(args, s.element_index))
            + s.element_coords
            - type(s.element_coords)(s.element_coords.dtype(0.5))
        )
        return wp.volume_index_to_world(args.cell_grid, uvw)

    @wp.func
    def cell_deformation_gradient(args: Any, s: Any):
        return wp.inverse(args.inverse_transform)

    @wp.func
    def cell_inverse_deformation_gradient(args: Any, s: Any):
        return args.inverse_transform

    @wp.func
    def cell_measure(args: Any, s: Any):
        return args.cell_volume

    @wp.func
    def cell_normal(args: Any, s: Any):
        return type(s.element_coords)(s.element_coords.dtype(0.0))

    @wp.func
    def _lookup_cell_index(args: Any, i: int, j: int, k: int):
        return wp.volume_lookup_index(args.cell_grid, i, j, k)

    @wp.func
    def _cell_coordinates_local(args: Any, cell_index: int, uvw: Any):
        ijk = type(uvw)(NanogridBase._local_cell_ijk(args, cell_index))
        return uvw - ijk

    @wp.func
    def _cell_closest_point_local(args: Any, cell_index: int, uvw: Any):
        ijk = type(uvw)(NanogridBase._local_cell_ijk(args, cell_index))
        rel_pos = uvw - ijk
        coords = wp.min(wp.max(rel_pos, type(uvw)(uvw.dtype(0.0))), type(uvw)(uvw.dtype(1.0)))
        return wp.length_sq(wp.volume_index_to_world_dir(args.cell_grid, coords - rel_pos)), coords

    @wp.func
    def cell_coordinates(args: Any, cell_index: int, pos: Any):
        uvw = wp.volume_world_to_index(args.cell_grid, pos) + type(pos)(pos.dtype(0.5))
        return Nanogrid._cell_coordinates_local(args, cell_index, uvw)

    @wp.func
    def cell_closest_point(args: Any, cell_index: int, pos: Any):
        uvw = wp.volume_world_to_index(args.cell_grid, pos) + type(pos)(pos.dtype(0.5))
        dist, coords = Nanogrid._cell_closest_point_local(args, cell_index, uvw)
        return coords, dist

    @wp.func
    def side_position(args: Any, s: Any):
        ijk = NanogridBase._local_face_ijk(args, s.element_index)
        flags = args.face_flags[s.element_index]
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)

        uvw = type(s.element_coords)(ijk) + NanogridBase._side_to_cell_coords(
            axis, flip, s.element_coords.dtype(0.0), s.element_coords
        )
        return wp.volume_index_to_world(
            args.cell_arg.cell_grid, uvw - type(s.element_coords)(s.element_coords.dtype(0.5))
        )

    def _make_side_deformation_gradient(self):
        SampleType = self.sample_type
        face_tangent_vecs = self._make_face_tangent_vecs()

        @cache.dynamic_func(suffix=self.name)
        def side_deformation_gradient(args: self.SideArg, s: SampleType):
            flags = args.face_flags[s.element_index]
            axis = NanogridBase._get_face_axis(flags)
            flip = NanogridBase._get_face_inner_offset(flags)
            v1, v2 = face_tangent_vecs(args.cell_arg.cell_grid, axis, flip)
            return wp.matrix_from_cols(v1, v2)

        return side_deformation_gradient

    @wp.func
    def side_inner_inverse_deformation_gradient(args: Any, s: Any):
        return Nanogrid.cell_inverse_deformation_gradient(args.cell_arg, s)

    @wp.func
    def side_outer_inverse_deformation_gradient(args: Any, s: Any):
        return Nanogrid.cell_inverse_deformation_gradient(args.cell_arg, s)

    @wp.func
    def side_measure(args: Any, s: Any):
        axis = NanogridBase._get_face_axis(args.face_flags[s.element_index])
        return args.face_areas[axis]

    @wp.func
    def side_measure_ratio(args: Any, s: Any):
        axis = NanogridBase._get_face_axis(args.face_flags[s.element_index])
        return args.face_areas[axis] / args.cell_arg.cell_volume

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
        ijk = args.face_ijk[side_index]
        flags = args.face_flags[side_index]
        axis = NanogridBase._get_face_axis(flags)
        offset = NanogridBase._get_face_inner_offset(flags)
        ijk[axis] += offset - 1
        return wp.volume_lookup_index(args.cell_arg.cell_grid, ijk[0], ijk[1], ijk[2])

    @wp.func
    def side_outer_cell_index(args: Any, side_index: ElementIndex):
        ijk = args.face_ijk[side_index]
        flags = args.face_flags[side_index]
        axis = NanogridBase._get_face_axis(flags)
        offset = NanogridBase._get_face_outer_offset(flags)
        ijk[axis] -= offset
        return wp.volume_lookup_index(args.cell_arg.cell_grid, ijk[0], ijk[1], ijk[2])

    @wp.func
    def side_inner_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        flags = args.face_flags[side_index]
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)
        offset = side_coords.dtype(NanogridBase._get_face_inner_offset(flags))
        return NanogridBase._side_to_cell_coords(axis, flip, side_coords.dtype(1.0) - offset, side_coords)

    @wp.func
    def side_outer_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        flags = args.face_flags[side_index]
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)
        offset = side_coords.dtype(NanogridBase._get_face_outer_offset(flags))
        return NanogridBase._side_to_cell_coords(axis, flip, offset, side_coords)

    @wp.func
    def side_from_cell_coords(args: Any, side_index: ElementIndex, element_index: ElementIndex, element_coords: Any):
        flags = args.face_flags[side_index]
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)

        cell_ijk = args.cell_arg.cell_ijk[element_index]
        side_ijk = args.face_ijk[side_index]

        same_env = NanogridBase.side_environment_index(args, side_index) == NanogridBase.cell_environment_index(
            args.cell_arg, element_index
        )
        on_side = same_env and element_coords.dtype(side_ijk[axis] - cell_ijk[axis]) == element_coords[axis]
        return wp.where(
            on_side,
            NanogridBase._cell_to_side_coords(axis, flip, element_coords),
            type(element_coords)(element_coords.dtype(OUTSIDE)),
        )

    @wp.func
    def side_to_cell_arg(side_arg: Any):
        return side_arg.cell_arg

    @wp.func
    def side_coordinates(args: Any, side_index: int, pos: Any):
        ijk = NanogridBase._local_face_ijk(args, side_index)
        cell_coords = (
            wp.volume_world_to_index(args.cell_arg.cell_grid, pos) + type(pos)(pos.dtype(0.5)) - type(pos)(ijk)
        )

        flags = args.face_flags[side_index]
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)
        return NanogridBase._cell_to_side_coords(axis, flip, cell_coords)

    @wp.func
    def side_closest_point(args: Any, side_index: int, pos: Any):
        coords = Nanogrid.side_coordinates(args, side_index, pos)
        z = pos.dtype(0.0)
        o = pos.dtype(1.0)
        proj_coords = make_coords(wp.clamp(coords[0], z, o), wp.clamp(coords[1], z, o))

        flags = args.face_flags[side_index]
        axis = NanogridBase._get_face_axis(flags)
        flip = NanogridBase._get_face_inner_offset(flags)
        cell_coord_offset = NanogridBase._side_to_cell_coords(axis, flip, z, coords - proj_coords)

        return proj_coords, wp.length_sq(wp.volume_index_to_world_dir(args.cell_arg.cell_grid, cell_coord_offset))

    def _build_face_grid(self, temporary_store: cache.TemporaryStore | None = None):
        device = self._cell_grid.device
        self._face_grid = _build_face_grid(self._cell_ijk, self._cell_grid, temporary_store)
        face_count = self._face_grid.get_voxel_count()
        self._face_ijk = wp.array(shape=(face_count,), dtype=wp.vec3i, device=device)
        self._face_grid.get_voxels(out=self._face_ijk)

        self._face_env = wp.array(shape=(face_count,), dtype=int, device=device)
        self._face_flags = wp.array(shape=(face_count,), dtype=wp.uint8, device=device)
        boundary_face_mask = cache.borrow_temporary(temporary_store, shape=(face_count,), dtype=wp.int32, device=device)

        wp.launch(
            _build_face_flags,
            dim=face_count,
            device=device,
            inputs=[
                self._cell_grid.id,
                self._cell_env,
                self._face_ijk,
                self._face_env,
                self._face_flags,
                boundary_face_mask,
            ],
        )
        boundary_face_indices, _ = utils.masked_indices(boundary_face_mask)
        boundary_face_mask.release()
        self._boundary_face_indices = boundary_face_indices.detach()

    def _build_edge_grid(self, temporary_store: cache.TemporaryStore | None = None):
        self._edge_grid = _build_edge_grid(self._cell_ijk, self._cell_grid, temporary_store)
        self._edge_count = self._edge_grid.get_voxel_count()

    def _ensure_edge_grid(self):
        if self._edge_grid is None:
            self._build_edge_grid()

    def _refresh_rebuildable_topology(self):
        self._cell_grid.get_voxels(out=self._cell_ijk)

        _fill_rebuildable_node_candidates(
            self._cell_grid, self._cell_ijk, self._node_candidates, self._node_candidate_mask
        )
        self._node_grid.rebuild(self._node_candidates.flatten(), point_mask=self._node_candidate_mask)
        self._node_grid.get_voxels(out=self._node_ijk)

        self._face_grid = None
        self._face_ijk = None
        self._face_env = None
        self._face_flags = None
        self._boundary_face_indices = None
        self._edge_grid = None
        self._edge_count = 0

        self.cell_arg_value.invalidate(self)
        self.side_arg_value.invalidate(self)
        self.side_index_arg_value.invalidate(self)


def _environment_voxels_from_legacy_sequence(
    cell_ijks: Sequence[wp.array],
    legacy_env_offsets: wp.array | Sequence[Sequence[int]] | None,
    env_count: int | None,
    env_offsets: wp.array | Sequence[Sequence[int]] | None,
    device,
):
    log_warning(
        "The sequence form Nanogrid.from_environment_voxels(cell_ijks, env_offsets=...) is deprecated; "
        "pass flat points, point_envs, and env_count instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    if env_count is not None:
        raise TypeError("env_count is not accepted with the deprecated cell_ijks sequence form")
    if env_offsets is not None and legacy_env_offsets is not None:
        raise TypeError("env_offsets was provided both positionally and by keyword")
    if env_offsets is None:
        env_offsets = legacy_env_offsets

    cell_ijks = tuple(cell_ijks)
    if not cell_ijks:
        raise ValueError("At least one environment cell array is required")

    if device is None:
        if not isinstance(cell_ijks[0], wp.array):
            raise ValueError("Environment 0 cell coordinates must be a 1D wp.vec3i array")
        device = cell_ijks[0].device
    else:
        device = wp.get_device(device)

    normalized = []
    point_count = 0
    for env_index, env_cell_ijk in enumerate(cell_ijks):
        if not isinstance(env_cell_ijk, wp.array) or env_cell_ijk.dtype != wp.vec3i or env_cell_ijk.ndim != 1:
            raise ValueError(f"Environment {env_index} cell coordinates must be a 1D wp.vec3i array")
        normalized_cell_ijk = env_cell_ijk
        if env_cell_ijk.device != device:
            normalized_cell_ijk = env_cell_ijk.to(device)
        normalized.append(normalized_cell_ijk)
        point_count += normalized_cell_ijk.shape[0]

    points = wp.empty(shape=point_count, dtype=wp.vec3i, device=device)
    point_envs = wp.empty(shape=point_count, dtype=wp.int32, device=device)

    point_offset = 0
    for env_index, env_cell_ijk in enumerate(normalized):
        env_point_count = env_cell_ijk.shape[0]
        if env_point_count:
            wp.copy(points, env_cell_ijk, dest_offset=point_offset, count=env_point_count)
            point_envs[point_offset : point_offset + env_point_count].fill_(env_index)
        point_offset += env_point_count

    return points, point_envs, len(normalized), env_offsets


def _normalize_environment_voxels(points: wp.array, point_envs: wp.array, env_count: int, device):
    if device is None:
        device = points.device
    else:
        device = wp.get_device(device)

    if not device.is_cuda:
        raise RuntimeError("NanoVDB environment volumes can only be built on CUDA devices")

    points = _nanogrid_rebuild_points_array(points, device)
    point_envs = _nanogrid_optional_int_array(point_envs, points.shape[0], device, "point_envs")
    if point_envs is None:
        raise ValueError("point_envs is required when constructing an environment Nanogrid")

    env_count = int(env_count)
    if env_count <= 0:
        raise ValueError("env_count must be positive")

    return points, point_envs, env_count, device


def _environment_transform_args(voxel_size, translation, transform):
    if transform is None:
        if voxel_size is None:
            raise ValueError("voxel_size must be provided when transform is None")
        if np.isscalar(voxel_size):
            transform_np = np.diag([float(voxel_size)] * 3)
        else:
            transform_np = np.diag(np.array(voxel_size, dtype=np.float32).reshape(3))
    else:
        transform_np = np.array(transform, dtype=np.float32).reshape(3, 3)

    translation_np = np.array(translation, dtype=np.float32).reshape(3)
    inverse_np = np.linalg.inv(transform_np)

    return (
        wp.mat33f(*transform_np.astype(np.float32).flatten()),
        wp.mat33f(*inverse_np.astype(np.float32).flatten()),
        wp.vec3f(*translation_np),
    )


def _normalize_flat_environment_offsets(
    env_offsets: wp.array | Sequence[Sequence[int]] | None,
    env_count: int,
    device,
    *,
    cell_counts: wp.array,
    min_x: wp.array,
    max_x: wp.array,
    guard_cells: int,
    alignment: int,
    temporary_store: cache.TemporaryStore | None,
):
    if env_offsets is not None:
        if isinstance(env_offsets, wp.array):
            if env_offsets.dtype != wp.vec3i or env_offsets.ndim != 1:
                raise ValueError("Environment offsets must be a 1D wp.vec3i array")
            if env_offsets.device != device:
                env_offsets = env_offsets.to(device)
        else:
            env_offsets = wp.array(env_offsets, dtype=wp.vec3i, device=device)

        if env_offsets.shape[0] != env_count:
            raise ValueError("Environment offsets must have one entry per environment")
        if alignment > 1:
            env_offsets_np = env_offsets.numpy()
            if np.any(env_offsets_np % alignment != 0):
                raise ValueError(
                    f"Adaptive Nanogrid environment offsets must be aligned to {alignment} fine-grid voxels"
                )
        return env_offsets

    spans = cache.borrow_temporary(temporary_store, shape=env_count, dtype=int, device=device)
    starts = cache.borrow_temporary(temporary_store, shape=env_count, dtype=int, device=device)

    wp.launch(
        _compute_environment_spans,
        dim=env_count,
        inputs=[cell_counts, min_x, max_x, guard_cells, alignment, spans],
        device=device,
    )
    utils.array_scan(spans, starts, inclusive=False)

    env_offsets = wp.empty(shape=env_count, dtype=wp.vec3i, device=device)
    wp.launch(
        _compute_environment_offsets_from_starts,
        dim=env_count,
        inputs=[cell_counts, min_x, starts, alignment, env_offsets],
        device=device,
    )

    spans.release()
    starts.release()
    return env_offsets


def _make_environment_cell_grid(
    points: wp.array,
    point_envs: wp.array,
    env_count: int,
    env_offsets: wp.array | Sequence[Sequence[int]] | None,
    *,
    point_mask: wp.array | None = None,
    voxel_size: int | float | Sequence[float] | None = 1.0,
    translation=(0.0, 0.0, 0.0),
    transform=None,
    temporary_store: cache.TemporaryStore | None,
    device=None,
    guard_cells: int = 3,
    rebuildable: bool = False,
    max_active_voxels: int | None = None,
    max_leaf_nodes: int | None = None,
    max_lower_nodes: int | None = None,
    max_upper_nodes: int | None = None,
    status: wp.array | None = None,
):
    points, point_envs, env_count, device = _normalize_environment_voxels(points, point_envs, env_count, device)
    point_count = points.shape[0]
    point_mask = _nanogrid_optional_int_array(point_mask, point_count, device, "point_mask")
    _, inverse_transform, translation_vec = _environment_transform_args(voxel_size, translation, transform)

    cell_counts = cache.borrow_temporary(temporary_store, shape=env_count, dtype=int, device=device)
    min_x = cache.borrow_temporary(temporary_store, shape=env_count, dtype=int, device=device)
    max_x = cache.borrow_temporary(temporary_store, shape=env_count, dtype=int, device=device)

    wp.launch(_initialize_environment_bounds, dim=env_count, inputs=[cell_counts, min_x, max_x], device=device)
    if wp.types.types_equal(points.dtype, wp.vec3i):
        wp.launch(
            _accumulate_environment_bounds_ijk,
            dim=point_count,
            device=device,
            inputs=[points, point_envs, point_mask, cell_counts, min_x, max_x],
        )
    else:
        wp.launch(
            _accumulate_environment_bounds_world,
            dim=point_count,
            device=device,
            inputs=[points, point_envs, point_mask, inverse_transform, translation_vec, cell_counts, min_x, max_x],
        )

    env_offsets = _normalize_flat_environment_offsets(
        env_offsets,
        env_count,
        device,
        cell_counts=cell_counts,
        min_x=min_x,
        max_x=max_x,
        guard_cells=guard_cells,
        alignment=1,
        temporary_store=temporary_store,
    )

    packed_ijks = _pack_environment_points(
        points,
        point_envs,
        point_mask,
        env_offsets,
        inverse_transform,
        translation_vec,
        temporary_store=temporary_store,
        device=device,
    )

    grid = wp.Volume.allocate_by_voxels(
        packed_ijks,
        voxel_size=voxel_size,
        translation=translation,
        transform=transform,
        device=device,
        rebuildable=rebuildable,
        max_active_voxels=max_active_voxels,
        max_leaf_nodes=max_leaf_nodes,
        max_lower_nodes=max_lower_nodes,
        max_upper_nodes=max_upper_nodes,
        status=status,
        point_mask=point_mask,
    )
    cell_env = wp.empty(shape=(grid.get_voxel_count(),), dtype=int, device=device)
    _fill_cell_env_from_points(grid, packed_ijks, point_envs, point_mask, cell_env)

    packed_ijks.release()
    cell_counts.release()
    min_x.release()
    max_x.release()
    return grid, cell_env, env_offsets


def _pack_environment_points(
    points: wp.array,
    point_envs: wp.array,
    point_mask: wp.array | None,
    env_offsets: wp.array,
    inverse_transform: wp.mat33f,
    translation_vec: wp.vec3f,
    *,
    temporary_store: cache.TemporaryStore | None,
    device,
):
    packed_ijks = cache.borrow_temporary(temporary_store, shape=points.shape[0], dtype=wp.vec3i, device=device)
    if wp.types.types_equal(points.dtype, wp.vec3i):
        wp.launch(
            _pack_environment_voxels_ijk,
            dim=points.shape[0],
            device=device,
            inputs=[points, point_envs, point_mask, env_offsets, packed_ijks],
        )
    else:
        wp.launch(
            _pack_environment_voxels_world,
            dim=points.shape[0],
            device=device,
            inputs=[points, point_envs, point_mask, inverse_transform, translation_vec, env_offsets, packed_ijks],
        )
    return packed_ijks


@wp.kernel
def _initialize_environment_bounds(
    cell_counts: wp.array(dtype=int), min_x: wp.array(dtype=int), max_x: wp.array(dtype=int)
):
    env = wp.tid()
    cell_counts[env] = 0
    min_x[env] = 2147483647
    max_x[env] = -2147483648


@wp.func
def _align_environment_offset(x: int, alignment: int):
    return ((x + alignment - 1) // alignment) * alignment


@wp.func
def _world_point_cell_ijk(point: wp.vec3f, inverse_transform: wp.mat33f, translation: wp.vec3f):
    uvw = wp.mul(inverse_transform, point - translation) + wp.vec3f(0.5)
    return wp.vec3i(int(wp.floor(uvw[0])), int(wp.floor(uvw[1])), int(wp.floor(uvw[2])))


@wp.kernel
def _accumulate_environment_bounds_ijk(
    points: wp.array(dtype=wp.vec3i),
    point_envs: wp.array(dtype=int),
    point_mask: wp.array(dtype=wp.int32),
    cell_counts: wp.array(dtype=int),
    min_x: wp.array(dtype=int),
    max_x: wp.array(dtype=int),
):
    point = wp.tid()
    if point_mask:
        if point_mask[point] == 0:
            return

    env = point_envs[point]
    cell = points[point]
    wp.atomic_add(cell_counts, env, 1)
    wp.atomic_min(min_x, env, cell[0])
    wp.atomic_max(max_x, env, cell[0])


@wp.kernel
def _accumulate_environment_bounds_world(
    points: wp.array(dtype=wp.vec3f),
    point_envs: wp.array(dtype=int),
    point_mask: wp.array(dtype=wp.int32),
    inverse_transform: wp.mat33f,
    translation: wp.vec3f,
    cell_counts: wp.array(dtype=int),
    min_x: wp.array(dtype=int),
    max_x: wp.array(dtype=int),
):
    point = wp.tid()
    if point_mask:
        if point_mask[point] == 0:
            return

    env = point_envs[point]
    cell = _world_point_cell_ijk(points[point], inverse_transform, translation)
    wp.atomic_add(cell_counts, env, 1)
    wp.atomic_min(min_x, env, cell[0])
    wp.atomic_max(max_x, env, cell[0])


@wp.kernel
def _compute_environment_spans(
    cell_counts: wp.array(dtype=int),
    min_x: wp.array(dtype=int),
    max_x: wp.array(dtype=int),
    guard_cells: int,
    alignment: int,
    spans: wp.array(dtype=int),
):
    env = wp.tid()
    span = wp.where(cell_counts[env] == 0, guard_cells, max_x[env] - min_x[env] + 1 + guard_cells)
    spans[env] = _align_environment_offset(span, alignment)


@wp.kernel
def _compute_environment_offsets_from_starts(
    cell_counts: wp.array(dtype=int),
    min_x: wp.array(dtype=int),
    starts: wp.array(dtype=int),
    alignment: int,
    env_offsets: wp.array(dtype=wp.vec3i),
):
    env = wp.tid()
    start = _align_environment_offset(starts[env], alignment)
    offset_x = wp.where(cell_counts[env] == 0, start, _align_environment_offset(start - min_x[env], alignment))
    env_offsets[env] = wp.vec3i(offset_x, 0, 0)


@wp.kernel
def _pack_environment_voxels_ijk(
    points: wp.array(dtype=wp.vec3i),
    point_envs: wp.array(dtype=int),
    point_mask: wp.array(dtype=wp.int32),
    env_offsets: wp.array(dtype=wp.vec3i),
    packed_ijk: wp.array(dtype=wp.vec3i),
):
    point = wp.tid()
    if point_mask:
        if point_mask[point] == 0:
            packed_ijk[point] = wp.vec3i(0)
            return

    packed_ijk[point] = points[point] + env_offsets[point_envs[point]]


@wp.kernel
def _pack_environment_voxels_world(
    points: wp.array(dtype=wp.vec3f),
    point_envs: wp.array(dtype=int),
    point_mask: wp.array(dtype=wp.int32),
    inverse_transform: wp.mat33f,
    translation: wp.vec3f,
    env_offsets: wp.array(dtype=wp.vec3i),
    packed_ijk: wp.array(dtype=wp.vec3i),
):
    point = wp.tid()
    if point_mask:
        if point_mask[point] == 0:
            packed_ijk[point] = wp.vec3i(0)
            return

    packed_ijk[point] = (
        _world_point_cell_ijk(points[point], inverse_transform, translation) + env_offsets[point_envs[point]]
    )


def _nanogrid_rebuild_points_array(points: wp.array, device) -> wp.array:
    if not isinstance(points, wp.array) or not points.is_contiguous:
        raise RuntimeError(
            "points must be contiguous and either a 1D Warp array of vec3f or vec3i or a 2D n-by-3 array of int32 or float32."
        )
    if points.device != device:
        points = points.to(device)

    if points.ndim == 1:
        if wp.types.types_equal(points.dtype, wp.vec3i) or wp.types.types_equal(points.dtype, wp.vec3f):
            return points
    elif points.ndim == 2 and points.shape[1] == 3:
        if points.dtype == wp.int32:
            return points.view(wp.vec3i)
        if points.dtype == wp.float32:
            return points.view(wp.vec3f)

    raise RuntimeError(
        "points must be contiguous and either a 1D Warp array of vec3f or vec3i or a 2D n-by-3 array of int32 or float32."
    )


def _nanogrid_optional_int_array(values: wp.array | None, point_count: int, device, name: str) -> wp.array | None:
    if values is None:
        return None
    if not isinstance(values, wp.array) or values.dtype != wp.int32 or values.ndim != 1 or not values.is_contiguous:
        raise RuntimeError(f"{name} must be a contiguous 1D Warp array with dtype int32")
    if values.shape[0] < point_count:
        raise RuntimeError(f"{name} must have at least {point_count} entries")
    if values.device != device:
        values = values.to(device)
    return values


def _fill_cell_env_from_points(
    cell_grid: wp.Volume,
    points: wp.array,
    point_envs: wp.array,
    point_mask: wp.array | None,
    cell_env: wp.array,
):
    if wp.types.types_equal(points.dtype, wp.vec3i):
        kernel = _fill_cell_env_from_ijk_points_masked
    else:
        kernel = _fill_cell_env_from_world_points_masked

    inputs = [cell_grid.id, points, point_envs, point_mask, cell_env]
    wp.launch(kernel, dim=points.shape[0], inputs=inputs, device=cell_grid.device)


@wp.kernel
def _fill_cell_env_from_ijk_points_masked(
    cell_grid: wp.uint64,
    points: wp.array(dtype=wp.vec3i),
    point_envs: wp.array(dtype=int),
    point_mask: wp.array(dtype=wp.int32),
    cell_env: wp.array(dtype=int),
):
    point = wp.tid()

    if point_mask:
        if point_mask[point] == 0:
            return

    ijk = points[point]
    cell_index = wp.volume_lookup_index(cell_grid, ijk[0], ijk[1], ijk[2])
    if cell_index != -1:
        cell_env[cell_index] = point_envs[point]


@wp.kernel
def _fill_cell_env_from_world_points_masked(
    cell_grid: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    point_envs: wp.array(dtype=int),
    point_mask: wp.array(dtype=wp.int32),
    cell_env: wp.array(dtype=int),
):
    point = wp.tid()

    if point_mask:
        if point_mask[point] == 0:
            return

    uvw = wp.volume_world_to_index(cell_grid, points[point]) + wp.vec3f(0.5)
    ijk = wp.vec3i(int(wp.floor(uvw[0])), int(wp.floor(uvw[1])), int(wp.floor(uvw[2])))
    cell_index = wp.volume_lookup_index(cell_grid, ijk[0], ijk[1], ijk[2])
    if cell_index != -1:
        cell_env[cell_index] = point_envs[point]


# -- Topology-building kernels (precision-independent) --


@wp.kernel
def _cell_node_indices(cell_ijk: wp.array(dtype=wp.vec3i), node_ijk: wp.array2d(dtype=wp.vec3i)):
    cell, n = wp.tid()
    node_ijk[cell, n] = cell_ijk[cell] + wp.vec3i((n & 4) >> 2, (n & 2) >> 1, n & 1)


@wp.kernel
def _rebuildable_cell_node_indices(
    cell_grid: wp.uint64,
    cell_ijk: wp.array(dtype=wp.vec3i),
    node_ijk: wp.array2d(dtype=wp.vec3i),
    node_mask: wp.array(dtype=wp.int32),
):
    cell, n = wp.tid()
    node_ijk[cell, n] = cell_ijk[cell] + wp.vec3i((n & 4) >> 2, (n & 2) >> 1, n & 1)
    node_mask[cell * 8 + n] = wp.where(cell < wp.volume_voxel_count(cell_grid), wp.int32(1), wp.int32(0))


@wp.kernel
def _cell_face_indices(cell_ijk: wp.array(dtype=wp.vec3i), node_ijk: wp.array2d(dtype=wp.vec3i)):
    cell = wp.tid()
    ijk = cell_ijk[cell]
    node_ijk[cell, 0] = NanogridBase._add_axis_flag(ijk, 0)
    node_ijk[cell, 1] = NanogridBase._add_axis_flag(ijk, 1)
    node_ijk[cell, 2] = NanogridBase._add_axis_flag(ijk, 2)
    node_ijk[cell, 3] = NanogridBase._add_axis_flag(ijk + wp.vec3i(1, 0, 0), 0)
    node_ijk[cell, 4] = NanogridBase._add_axis_flag(ijk + wp.vec3i(0, 1, 0), 1)
    node_ijk[cell, 5] = NanogridBase._add_axis_flag(ijk + wp.vec3i(0, 0, 1), 2)


@wp.kernel
def _cell_edge_indices(cell_ijk: wp.array(dtype=wp.vec3i), edge_ijk: wp.array2d(dtype=wp.vec3i)):
    cell = wp.tid()
    ijk = cell_ijk[cell]
    edge_ijk[cell, 0] = NanogridBase._add_axis_flag(ijk, 0)
    edge_ijk[cell, 1] = NanogridBase._add_axis_flag(ijk, 1)
    edge_ijk[cell, 2] = NanogridBase._add_axis_flag(ijk, 2)
    edge_ijk[cell, 3] = NanogridBase._add_axis_flag(ijk + wp.vec3i(0, 1, 0), 0)
    edge_ijk[cell, 4] = NanogridBase._add_axis_flag(ijk + wp.vec3i(0, 0, 1), 1)
    edge_ijk[cell, 5] = NanogridBase._add_axis_flag(ijk + wp.vec3i(1, 0, 0), 2)
    edge_ijk[cell, 6] = NanogridBase._add_axis_flag(ijk + wp.vec3i(0, 1, 1), 0)
    edge_ijk[cell, 7] = NanogridBase._add_axis_flag(ijk + wp.vec3i(1, 0, 1), 1)
    edge_ijk[cell, 8] = NanogridBase._add_axis_flag(ijk + wp.vec3i(1, 1, 0), 2)
    edge_ijk[cell, 9] = NanogridBase._add_axis_flag(ijk + wp.vec3i(0, 0, 1), 0)
    edge_ijk[cell, 10] = NanogridBase._add_axis_flag(ijk + wp.vec3i(1, 0, 0), 1)
    edge_ijk[cell, 11] = NanogridBase._add_axis_flag(ijk + wp.vec3i(0, 1, 0), 2)


def _build_node_grid(cell_ijk, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]
    cell_nodes = cache.borrow_temporary(temporary_store, shape=(cell_count, 8), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(_cell_node_indices, dim=cell_nodes.shape, inputs=[cell_ijk, cell_nodes], device=cell_ijk.device)
    node_grid = wp.Volume.allocate_by_voxels(
        cell_nodes.flatten(), voxel_size=grid.get_voxel_size(), device=cell_ijk.device
    )
    cell_nodes.release()
    return node_grid


def _fill_rebuildable_node_candidates(
    cell_grid: wp.Volume,
    cell_ijk: wp.array,
    node_candidates: wp.array2d,
    node_candidate_mask: wp.array,
):
    wp.launch(
        _rebuildable_cell_node_indices,
        dim=node_candidates.shape,
        inputs=[cell_grid.id, cell_ijk, node_candidates, node_candidate_mask],
        device=cell_ijk.device,
    )


def _build_rebuildable_node_grid(cell_ijk, grid: wp.Volume):
    node_capacity = cell_ijk.shape[0] * 8
    node_candidates = wp.empty(shape=(cell_ijk.shape[0], 8), dtype=wp.vec3i, device=cell_ijk.device)
    node_candidate_mask = wp.empty(shape=(node_capacity,), dtype=wp.int32, device=cell_ijk.device)
    _fill_rebuildable_node_candidates(grid, cell_ijk, node_candidates, node_candidate_mask)

    rebuild_info = grid.get_rebuild_info()
    max_leaf_nodes = min(node_capacity, rebuild_info.max_leaf_node_count * 8)
    max_lower_nodes = min(max_leaf_nodes, rebuild_info.max_lower_node_count * 8)
    max_upper_nodes = min(max_lower_nodes, rebuild_info.max_upper_node_count * 8)

    node_grid = wp.Volume.allocate_by_voxels(
        node_candidates.flatten(),
        voxel_size=grid.get_voxel_size(),
        device=cell_ijk.device,
        rebuildable=True,
        max_active_voxels=node_capacity,
        max_leaf_nodes=max_leaf_nodes,
        max_lower_nodes=max_lower_nodes,
        max_upper_nodes=max_upper_nodes,
        point_mask=node_candidate_mask,
    )

    return node_grid, node_candidates, node_candidate_mask


def _build_face_grid(cell_ijk, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]
    cell_faces = cache.borrow_temporary(temporary_store, shape=(cell_count, 6), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(_cell_face_indices, dim=cell_count, inputs=[cell_ijk, cell_faces], device=cell_ijk.device)
    face_grid = wp.Volume.allocate_by_voxels(
        cell_faces.flatten(), voxel_size=grid.get_voxel_size(), device=cell_ijk.device
    )
    cell_faces.release()
    return face_grid


def _build_edge_grid(cell_ijk, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]
    cell_edges = cache.borrow_temporary(temporary_store, shape=(cell_count, 12), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(_cell_edge_indices, dim=cell_count, inputs=[cell_ijk, cell_edges], device=cell_ijk.device)
    edge_grid = wp.Volume.allocate_by_voxels(
        cell_edges.flatten(), voxel_size=grid.get_voxel_size(), device=cell_ijk.device
    )
    cell_edges.release()
    return edge_grid


@wp.kernel
def _build_face_flags(
    cell_grid: wp.uint64,
    cell_env: wp.array(dtype=int),
    face_ijk: wp.array(dtype=wp.vec3i),
    face_env: wp.array(dtype=int),
    face_flags: wp.array(dtype=wp.uint8),
    boundary_face_mask: wp.array(dtype=int),
):
    face = wp.tid()
    axis, ijk = NanogridBase._extract_axis_flag(face_ijk[face])
    ijk_minus = ijk
    ijk_minus[axis] -= 1
    plus_cell_index = wp.volume_lookup_index(cell_grid, ijk[0], ijk[1], ijk[2])
    minus_cell_index = wp.volume_lookup_index(cell_grid, ijk_minus[0], ijk_minus[1], ijk_minus[2])
    face_ijk[face] = ijk
    flags = NanogridBase._make_face_flags(axis, plus_cell_index, minus_cell_index)
    env_cell_index = wp.where(plus_cell_index == -1, minus_cell_index, plus_cell_index)
    face_env[face] = cell_env[env_cell_index]
    face_flags[face] = flags
    boundary_face_mask[face] = NanogridBase._get_boundary_mask(flags)
