# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from typing import Any, ClassVar

import warp as wp
from warp._src.fem import cache
from warp._src.fem.cache import cached_arg_value, cached_vec_type, dynamic_func
from warp._src.fem.types import NULL_ELEMENT_INDEX, OUTSIDE, ElementIndex, Sample, make_coords, make_free_sample

from .closest_point import project_on_box_at_origin_2d
from .element import Element
from .geometry import Geometry, _array_load


def _make_grid2d_cell_arg(scalar_type):
    vec2_type = cached_vec_type(2, scalar_type)

    @cache.dynamic_struct(suffix=scalar_type)
    class Grid2DCellArg:
        res: wp.vec2i
        cell_size: vec2_type
        origin: vec2_type

    return Grid2DCellArg


def _make_grid2d_side_arg(cell_arg_type, scalar_type):
    @cache.dynamic_struct(suffix=scalar_type)
    class Grid2DSideArg:
        cell_count: int
        axis_offsets: wp.vec2i
        cell_arg: cell_arg_type

    return Grid2DSideArg


class Grid2D(Geometry):
    """Two-dimensional regular grid geometry."""

    dimension = 2

    ALT_AXIS = 0
    LONG_AXIS = 1

    _dynamic_attribute_constructors: ClassVar = {
        **Geometry._dynamic_attribute_constructors,
    }

    def __init__(
        self,
        res: wp.vec2i,
        bounds_lo: wp.vec2 | None = None,
        bounds_hi: wp.vec2 | None = None,
        scalar_type: type = wp.float32,
        env_count: int = 1,
    ):
        """Construct a dense 2D grid.

        Args:
            res: Resolution of the grid along each dimension
            bounds_lo: Position of the lower bound of the axis-aligned grid
            bounds_hi: Position of the upper bound of the axis-aligned grid
            scalar_type: Scalar type for grid coordinates (``wp.float32`` or ``wp.float64``)
            env_count: Number of colocated, topologically independent environments
        """

        if env_count < 1:
            raise ValueError("Environment count must be at least one")

        self._scalar_type = scalar_type
        vec2_type = cached_vec_type(2, scalar_type)

        if bounds_lo is None:
            bounds_lo = vec2_type(0.0)
        elif scalar_type != wp.float32:
            bounds_lo = vec2_type(float(bounds_lo[0]), float(bounds_lo[1]))

        if bounds_hi is None:
            bounds_hi = vec2_type(1.0)
        elif scalar_type != wp.float32:
            bounds_hi = vec2_type(float(bounds_hi[0]), float(bounds_hi[1]))

        self.bounds_lo = bounds_lo
        self.bounds_hi = bounds_hi

        self._res = res
        self._env_count = env_count

        # Dynamic Arg structs
        self.CellArg = _make_grid2d_cell_arg(scalar_type)
        self.SideArg = _make_grid2d_side_arg(self.CellArg, scalar_type)
        self.SideIndexArg = self.SideArg

        cache.setup_dynamic_attributes(self)

    @property
    def scalar_type(self):
        return self._scalar_type

    @property
    def name(self) -> str:
        return super().name

    @cached_property
    def extents(self):
        """Extent of the grid along each axis."""
        vec2_type = cached_vec_type(2, self._scalar_type)
        return vec2_type(
            float(self.bounds_hi[0]) - float(self.bounds_lo[0]),
            float(self.bounds_hi[1]) - float(self.bounds_lo[1]),
        )

    @cached_property
    def cell_size(self):
        """Size of a cell along each axis."""
        vec2_type = cached_vec_type(2, self._scalar_type)
        ex = self.extents
        return vec2_type(
            float(ex[0]) / float(self.res[0]),
            float(ex[1]) / float(self.res[1]),
        )

    def cell_count(self):
        return self._env_count * self._cell_count_per_environment()

    def vertex_count(self):
        return self._env_count * self._vertex_count_per_environment()

    def side_count(self):
        return self._env_count * self._side_count_per_environment()

    def boundary_side_count(self):
        return self._env_count * self._boundary_side_count_per_environment()

    def environment_count(self):
        return self._env_count

    def _cell_count_per_environment(self):
        return self.res[0] * self.res[1]

    def _vertex_count_per_environment(self):
        return (self.res[0] + 1) * (self.res[1] + 1)

    def _side_count_per_environment(self):
        return 2 * self._cell_count_per_environment() + self.res[0] + self.res[1]

    def _boundary_side_count_per_environment(self):
        return 2 * (self.res[0] + self.res[1])

    def reference_cell(self) -> Element:
        return Element.SQUARE

    def reference_side(self) -> Element:
        return Element.LINE_SEGMENT

    @property
    def res(self):
        return self._res

    @property
    def origin(self):
        return self.bounds_lo

    @cached_property
    def strides(self):
        return wp.vec2i(self.res[1], 1)

    Cell = wp.vec2i

    # -- Static integer-only helpers --

    @wp.func
    def _to_2d_index(x_stride: int, index: int):
        x = index // x_stride
        y = index - x_stride * x
        return wp.vec2i(x, y)

    @wp.func
    def _from_2d_index(x_stride: int, index: wp.vec2i):
        return x_stride * index[0] + index[1]

    @wp.func
    def cell_count_per_environment(res: wp.vec2i):
        return res[0] * res[1]

    @wp.func
    def vertex_count_per_environment(res: wp.vec2i):
        return (res[0] + 1) * (res[1] + 1)

    @wp.func
    def side_count_per_environment(res: wp.vec2i):
        return 2 * Grid2D.cell_count_per_environment(res) + res[0] + res[1]

    @wp.func
    def boundary_side_count_per_environment(res: wp.vec2i):
        return 2 * (res[0] + res[1])

    @wp.func
    def cell_env_index(res: wp.vec2i, cell_index: ElementIndex):
        if cell_index == NULL_ELEMENT_INDEX:
            return NULL_ELEMENT_INDEX

        cell_count = Grid2D.cell_count_per_environment(res)
        return cell_index // cell_count

    @wp.func
    def side_env_index(res: wp.vec2i, side_index: ElementIndex):
        if side_index == NULL_ELEMENT_INDEX:
            return NULL_ELEMENT_INDEX

        side_count = Grid2D.side_count_per_environment(res)
        return side_index // side_count

    @wp.func
    def local_cell_index(res: wp.vec2i, cell_index: ElementIndex):
        if cell_index == NULL_ELEMENT_INDEX:
            return NULL_ELEMENT_INDEX

        cell_count = Grid2D.cell_count_per_environment(res)
        return cell_index - cell_count * (cell_index // cell_count)

    @wp.func
    def local_side_index_from_res(res: wp.vec2i, side_index: ElementIndex):
        if side_index == NULL_ELEMENT_INDEX:
            return NULL_ELEMENT_INDEX

        side_count = Grid2D.side_count_per_environment(res)
        return side_index - side_count * (side_index // side_count)

    @wp.func
    def cell_environment_index(args: Any, cell_index: ElementIndex):
        return Grid2D.cell_env_index(args.res, cell_index)

    @wp.func
    def cell_environment_index(args: Any, s: Sample):
        return Grid2D.cell_env_index(args.res, s.element_index)

    @wp.func
    def side_environment_index(arg: Any, side_index: ElementIndex):
        res = arg.cell_arg.res
        return Grid2D.side_env_index(res, side_index)

    @wp.func
    def side_environment_index(arg: Any, s: Sample):
        return Grid2D.side_environment_index(arg, s.element_index)

    @wp.func
    def local_side_index(arg: Any, side_index: ElementIndex):
        res = arg.cell_arg.res
        return Grid2D.local_side_index_from_res(res, side_index)

    @wp.func
    def cell_index(res: wp.vec2i, cell: wp.vec2i):
        return Grid2D._from_2d_index(res[1], cell)

    @wp.func
    def get_cell(res: wp.vec2i, cell_index: ElementIndex):
        return Grid2D._to_2d_index(res[1], cell_index)

    @wp.struct
    class Side:
        axis: int
        origin: wp.vec2i

    @wp.func
    def orient(axis: int, vec: Any):
        return wp.where(axis == 0, vec, type(vec)(vec[1], vec[0]))

    @wp.func
    def orient(axis: int, coord: int):
        return wp.where(axis == 0, coord, 1 - coord)

    @wp.func
    def is_flipped(side: Side):
        return (side.axis == 0) == (side.origin[Grid2D.ALT_AXIS] == 0)

    # -- Fill arg --

    def fill_cell_arg(self, args, device):
        args.res = self.res
        args.cell_size = self.cell_size
        args.origin = self.bounds_lo

    @cached_arg_value
    def side_arg_value(self, device):
        args = self.SideArg()
        args.axis_offsets = wp.vec2i(0, self.res[1])
        args.cell_count = self._cell_count_per_environment()
        args.cell_arg = self.cell_arg_value(device)
        return args

    def side_index_arg_value(self, device):
        return self.side_arg_value(device)

    def supports_cell_lookup(self, device):
        return True

    # -- Dynamic function constructors --

    @wp.func
    def _get_side_from_grid(
        res: wp.vec2i,
        cell_count: int,
        axis_offsets: wp.vec2i,
        side_index: ElementIndex,
    ):
        local_side_index = Grid2D.local_side_index_from_res(res, side_index)
        if local_side_index < 2 * cell_count:
            axis = local_side_index // cell_count
            cell_index = local_side_index - axis * cell_count
            origin = Grid2D.orient(axis, Grid2D.get_cell(res, cell_index))
            return Grid2D.Side(axis, origin)

        axis_side_index = local_side_index - 2 * cell_count
        axis = wp.where(axis_side_index < axis_offsets[1], 0, 1)

        altitude = res[Grid2D.orient(axis, 0)]
        longitude = axis_side_index - axis_offsets[axis]

        return Grid2D.Side(axis, wp.vec2i(altitude, longitude))

    @wp.func
    def _get_side(arg: Any, side_index: ElementIndex):
        res = arg.cell_arg.res
        return Grid2D._get_side_from_grid(res, arg.cell_count, arg.axis_offsets, side_index)

    @wp.func
    def _side_index_from_grid(res: wp.vec2i, cell_count: int, axis_offsets: wp.vec2i, side: Any):
        alt_axis = Grid2D.orient(side.axis, 0)
        if side.origin[0] == res[alt_axis]:
            longitude = side.origin[1]
            return 2 * cell_count + axis_offsets[side.axis] + longitude

        cell_index = Grid2D.cell_index(res, Grid2D.orient(side.axis, side.origin))
        return side.axis * cell_count + cell_index

    @wp.func
    def _side_index(arg: Any, side: Any):
        res = arg.cell_arg.res
        return Grid2D._side_index_from_grid(res, arg.cell_count, arg.axis_offsets, side)

    @wp.func
    def boundary_side_index(args: Any, boundary_side_index: int):
        res = args.cell_arg.res
        axis_offsets = args.axis_offsets
        boundary_side_count = Grid2D.boundary_side_count_per_environment(res)
        side_count = Grid2D.side_count_per_environment(res)
        env_index = boundary_side_index // boundary_side_count
        boundary_side_index = boundary_side_index - env_index * boundary_side_count

        axis_side_index = boundary_side_index // 2
        border = boundary_side_index - 2 * axis_side_index

        if axis_side_index < axis_offsets[1]:
            axis = 0
        else:
            axis = 1

        longitude = axis_side_index - axis_offsets[axis]
        altitude = border * res[axis]

        side = Grid2D.Side(axis, wp.vec2i(altitude, longitude))
        return env_index * side_count + Grid2D._side_index_from_grid(res, args.cell_count, axis_offsets, side)

    @wp.func
    def cell_position(args: Any, s: Any):
        cell = Grid2D.get_cell(args.res, Grid2D.local_cell_index(args.res, s.element_index))
        return (
            type(args.cell_size)(
                (args.cell_size.dtype(cell[0]) + s.element_coords[0]) * args.cell_size[0],
                (args.cell_size.dtype(cell[1]) + s.element_coords[1]) * args.cell_size[1],
            )
            + args.origin
        )

    @wp.func
    def cell_deformation_gradient(args: Any, s: Any):
        return wp.diag(args.cell_size)

    @wp.func
    def cell_inverse_deformation_gradient(args: Any, s: Any):
        return wp.diag(wp.cw_div(type(args.cell_size)(args.cell_size.dtype(1.0)), args.cell_size))

    @wp.func
    def cell_coordinates(args: Any, cell_index: int, pos: Any):
        uvw = wp.cw_div(pos - args.origin, args.cell_size)
        ij = Grid2D.get_cell(args.res, Grid2D.local_cell_index(args.res, cell_index))
        return make_coords(uvw[0] - pos.dtype(ij[0]), uvw[1] - pos.dtype(ij[1]))

    @wp.func
    def cell_closest_point(args: Any, cell_index: int, pos: Any):
        local_cell_index = Grid2D.local_cell_index(args.res, cell_index)
        ij_world = wp.cw_mul(type(pos)(Grid2D.get_cell(args.res, local_cell_index)), args.cell_size) + args.origin
        dist_sq, coords = project_on_box_at_origin_2d(pos - ij_world, args.cell_size)
        return coords, dist_sq

    @wp.func
    def cell_measure(args: Any, s: Any):
        return args.cell_size[0] * args.cell_size[1]

    @wp.func
    def cell_normal(args: Any, s: Any):
        return type(args.cell_size)(args.cell_size.dtype(0.0))

    @wp.func
    def side_position(args: Any, s: Any):
        res = args.cell_arg.res
        cell_count = args.cell_count
        axis_offsets = args.axis_offsets
        cell_size = args.cell_arg.cell_size
        origin = args.cell_arg.origin
        side = Grid2D._get_side_from_grid(res, cell_count, axis_offsets, s.element_index)

        flip = Grid2D.is_flipped(side)
        coord = wp.where(flip, cell_size.dtype(1.0) - s.element_coords[0], s.element_coords[0])

        local_pos = type(cell_size)(side.origin) + type(cell_size)(cell_size.dtype(0.0), coord)
        pos = origin + wp.cw_mul(Grid2D.orient(side.axis, local_pos), cell_size)

        return pos

    @wp.func
    def side_deformation_gradient(args: Any, s: Any):
        res = args.cell_arg.res
        cell_count = args.cell_count
        axis_offsets = args.axis_offsets
        cell_size = args.cell_arg.cell_size
        side = Grid2D._get_side_from_grid(res, cell_count, axis_offsets, s.element_index)

        flip = Grid2D.is_flipped(side)
        sign = wp.where(flip, cell_size.dtype(-1.0), cell_size.dtype(1.0))

        return wp.cw_mul(
            Grid2D.orient(side.axis, type(cell_size)(cell_size.dtype(0.0), sign)),
            cell_size,
        )

    @wp.func
    def side_inner_inverse_deformation_gradient(args: Any, s: Any):
        return Grid2D.cell_inverse_deformation_gradient(args.cell_arg, s)

    @wp.func
    def side_outer_inverse_deformation_gradient(args: Any, s: Any):
        return Grid2D.cell_inverse_deformation_gradient(args.cell_arg, s)

    @wp.func
    def side_measure(args: Any, s: Any):
        res = args.cell_arg.res
        side = Grid2D._get_side_from_grid(res, args.cell_count, args.axis_offsets, s.element_index)
        long_axis = Grid2D.orient(side.axis, Grid2D.LONG_AXIS)
        return args.cell_arg.cell_size[long_axis]

    @wp.func
    def side_measure_ratio(args: Any, s: Any):
        res = args.cell_arg.res
        side = Grid2D._get_side_from_grid(res, args.cell_count, args.axis_offsets, s.element_index)
        alt_axis = Grid2D.orient(side.axis, Grid2D.ALT_AXIS)
        return args.cell_arg.cell_size.dtype(1.0) / args.cell_arg.cell_size[alt_axis]

    @wp.func
    def side_normal(args: Any, s: Any):
        res = args.cell_arg.res
        side = Grid2D._get_side_from_grid(res, args.cell_count, args.axis_offsets, s.element_index)
        flip = side.origin[Grid2D.ALT_AXIS] == 0
        sign = wp.where(flip, args.cell_arg.cell_size.dtype(-1.0), args.cell_arg.cell_size.dtype(1.0))
        local_n = type(args.cell_arg.cell_size)(sign, args.cell_arg.cell_size.dtype(0.0))
        return Grid2D.orient(side.axis, local_n)

    @wp.func
    def side_inner_cell_index(arg: Any, side_index: ElementIndex):
        if side_index == NULL_ELEMENT_INDEX:
            return NULL_ELEMENT_INDEX

        res = arg.cell_arg.res
        cell_count = arg.cell_count
        side = Grid2D._get_side_from_grid(res, cell_count, arg.axis_offsets, side_index)
        inner_alt = wp.where(side.origin[Grid2D.ALT_AXIS] == 0, 0, side.origin[Grid2D.ALT_AXIS] - 1)
        inner_origin = wp.vec2i(inner_alt, side.origin[1])
        cell = Grid2D.orient(side.axis, inner_origin)
        env_index = Grid2D.side_env_index(res, side_index)
        return env_index * cell_count + Grid2D.cell_index(res, cell)

    @wp.func
    def side_outer_cell_index(arg: Any, side_index: ElementIndex):
        if side_index == NULL_ELEMENT_INDEX:
            return NULL_ELEMENT_INDEX

        res = arg.cell_arg.res
        cell_count = arg.cell_count
        side = Grid2D._get_side_from_grid(res, cell_count, arg.axis_offsets, side_index)
        alt_axis = Grid2D.orient(side.axis, 0)
        outer_alt = wp.where(
            side.origin[Grid2D.ALT_AXIS] == res[alt_axis],
            res[alt_axis] - 1,
            side.origin[0],
        )
        outer_origin = wp.vec2i(outer_alt, side.origin[Grid2D.LONG_AXIS])
        cell = Grid2D.orient(side.axis, outer_origin)
        env_index = Grid2D.side_env_index(res, side_index)
        return env_index * cell_count + Grid2D.cell_index(res, cell)

    @wp.func
    def side_inner_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        res = args.cell_arg.res
        cell_size = args.cell_arg.cell_size
        side = Grid2D._get_side_from_grid(res, args.cell_count, args.axis_offsets, side_index)
        inner_alt = wp.where(side.origin[Grid2D.ALT_AXIS] == 0, side_coords.dtype(0.0), side_coords.dtype(1.0))
        flip = Grid2D.is_flipped(side)
        side_coord = wp.where(flip, side_coords.dtype(1.0) - side_coords[0], side_coords[0])
        coords = Grid2D.orient(side.axis, type(cell_size)(inner_alt, side_coord))
        return make_coords(coords[0], coords[1])

    @wp.func
    def side_outer_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        res = args.cell_arg.res
        side = Grid2D._get_side_from_grid(res, args.cell_count, args.axis_offsets, side_index)
        alt_axis = Grid2D.orient(side.axis, Grid2D.ALT_AXIS)
        outer_alt = wp.where(
            side.origin[Grid2D.ALT_AXIS] == res[alt_axis], side_coords.dtype(1.0), side_coords.dtype(0.0)
        )
        flip = Grid2D.is_flipped(side)
        side_coord = wp.where(flip, side_coords.dtype(1.0) - side_coords[0], side_coords[0])
        coords = Grid2D.orient(side.axis, type(args.cell_arg.cell_size)(outer_alt, side_coord))
        return make_coords(coords[0], coords[1])

    @wp.func
    def side_from_cell_coords(
        args: Any,
        side_index: ElementIndex,
        element_index: ElementIndex,
        element_coords: Any,
    ):
        if side_index == NULL_ELEMENT_INDEX or element_index == NULL_ELEMENT_INDEX:
            return type(element_coords)(element_coords.dtype(OUTSIDE))

        res = args.cell_arg.res
        side = Grid2D._get_side_from_grid(res, args.cell_count, args.axis_offsets, side_index)
        if Grid2D.side_env_index(res, side_index) != Grid2D.cell_env_index(res, element_index):
            return type(element_coords)(element_coords.dtype(OUTSIDE))

        cell = Grid2D.get_cell(res, Grid2D.local_cell_index(res, element_index))

        if element_coords.dtype(side.origin[Grid2D.ALT_AXIS] - cell[side.axis]) == element_coords[side.axis]:
            long_axis = Grid2D.orient(side.axis, Grid2D.LONG_AXIS)
            axis_coord = element_coords[long_axis]
            flip = Grid2D.is_flipped(side)
            side_coord = wp.where(flip, element_coords.dtype(1.0) - axis_coord, axis_coord)
            return make_coords(side_coord)

        return type(element_coords)(element_coords.dtype(OUTSIDE))

    @wp.func
    def side_to_cell_arg(side_arg: Any):
        return side_arg.cell_arg

    @wp.func
    def side_coordinates(args: Any, side_index: int, pos: Any):
        cell_arg = args.cell_arg
        side = Grid2D._get_side(args, side_index)
        long_axis = Grid2D.orient(side.axis, Grid2D.LONG_AXIS)
        flip = Grid2D.is_flipped(side)

        long_loc = (pos[long_axis] - cell_arg.origin[long_axis]) / cell_arg.cell_size[long_axis] - pos.dtype(
            side.origin[1]
        )
        coord = wp.where(flip, pos.dtype(1.0) - long_loc, long_loc)
        return make_coords(coord)

    @wp.func
    def side_closest_point(args: Any, side_index: int, pos: Any):
        coord = Grid2D.side_coordinates(args, side_index, pos)[0]

        cell_arg = args.cell_arg
        side = Grid2D._get_side(args, side_index)
        long_axis = Grid2D.orient(side.axis, Grid2D.LONG_AXIS)
        proj_coord = wp.clamp(coord, pos.dtype(0.0), pos.dtype(1.0))
        dist = (coord - proj_coord) * cell_arg.cell_size[long_axis]
        return make_coords(proj_coord), dist * dist

    def make_filtered_cell_lookup(self, filter_func: wp.Function = None):
        """Create a filtered cell lookup function.

        Args:
            filter_func: Optional device predicate to filter candidate cells.
        """
        suffix = f"{self.name}{filter_func.key if filter_func is not None else ''}"
        scalar = self._scalar_type
        vec2_type = cached_vec_type(2, scalar)
        CoordsType = self.coords_type

        @dynamic_func(suffix=(suffix, "env"))
        def cell_lookup_env(
            args: self.CellArg,
            pos: vec2_type,
            max_dist: scalar,
            filter_data: Any,
            filter_target: Any,
            env_index: int,
        ):
            cell_size = args.cell_size
            res = args.res
            cell_offset = env_index * Grid2D.cell_count_per_environment(res)

            loc_pos = wp.cw_div(pos - args.origin, cell_size)
            x = wp.clamp(loc_pos[0], scalar(0.0), scalar(res[0]))
            y = wp.clamp(loc_pos[1], scalar(0.0), scalar(res[1]))

            x_cell = wp.min(wp.floor(x), scalar(res[0] - 1))
            y_cell = wp.min(wp.floor(y), scalar(res[1] - 1))

            coords = CoordsType(x - x_cell, y - y_cell, scalar(0.0))
            cell_index = cell_offset + Grid2D.cell_index(res, Grid2D.Cell(int(x_cell), int(y_cell)))

            if wp.static(filter_func is None):
                return make_free_sample(cell_index, coords)
            else:
                if filter_func(filter_data, cell_index) == filter_target:
                    return make_free_sample(cell_index, coords)

                offset = scalar(0.5)
                min_cell_size = wp.min(cell_size)
                max_offset = wp.ceil(max_dist / min_cell_size)

                scales = wp.cw_div(vec2_type(min_cell_size), cell_size)

                closest_cell = NULL_ELEMENT_INDEX
                closest_coords = CoordsType()

                while closest_cell == NULL_ELEMENT_INDEX:
                    i_min = wp.max(0, int(wp.floor(x - offset * scales[0])))
                    i_max = wp.min(res[0], int(wp.floor(x + offset * scales[0])) + 1)
                    j_min = wp.max(0, int(wp.floor(y - offset * scales[1])))
                    j_max = wp.min(res[1], int(wp.floor(y + offset * scales[1])) + 1)

                    closest_dist = min_cell_size * min_cell_size * scalar(offset * offset)

                    for i in range(i_min, i_max):
                        for j in range(j_min, j_max):
                            ij = Grid2D.Cell(i, j)
                            cell_index = cell_offset + Grid2D.cell_index(res, ij)
                            if filter_func(filter_data, cell_index) == filter_target:
                                rel_pos = wp.cw_mul(loc_pos - vec2_type(ij), cell_size)
                                dist, coords = project_on_box_at_origin_2d(rel_pos, cell_size)

                                if dist <= closest_dist:
                                    closest_dist = dist
                                    closest_coords = coords
                                    closest_cell = cell_index

                    if offset >= max_offset:
                        break
                    offset = wp.min(scalar(3.0) * offset, max_offset)

                return make_free_sample(closest_cell, closest_coords)

        if self.environment_count() <= 1:

            @dynamic_func(suffix=suffix, allow_overloads=True)
            def cell_lookup(
                args: self.CellArg, pos: vec2_type, max_dist: self._scalar_type, filter_data: Any, filter_target: Any
            ):
                return cell_lookup_env(args, pos, max_dist, filter_data, filter_target, 0)

        @dynamic_func(suffix=suffix, allow_overloads=True)
        def cell_lookup(
            args: self.CellArg,
            pos: vec2_type,
            max_dist: self._scalar_type,
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
            def cell_lookup(args: self.CellArg, pos: pos_type, max_dist: self._scalar_type):
                return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target)

        @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
        def cell_lookup(args: self.CellArg, pos: pos_type, max_dist: self._scalar_type, env_index: int):
            return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target, env_index)

        @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
        def cell_lookup(args: self.CellArg, pos: pos_type, guess: SampleType):
            guess_pos = self.cell_position(args, guess)
            max_dist = wp.length(guess_pos - pos)
            if wp.static(self.environment_count() <= 1):
                return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target)

            env_index = Grid2D.cell_environment_index(args, guess.element_index)
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
                max_dist: self._scalar_type,
                filter_array: wp.array(dtype=Any),
                filter_target: Any,
            ):
                return filtered_cell_lookup(args, pos, max_dist, filter_array, filter_target)

        @cache.dynamic_func(suffix=lookup_suffix, allow_overloads=True)
        def cell_lookup(
            args: self.CellArg,
            pos: pos_type,
            max_dist: self._scalar_type,
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
