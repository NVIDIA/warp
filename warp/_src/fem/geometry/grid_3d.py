# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from typing import Any, ClassVar

import warp as wp
from warp._src.fem import cache
from warp._src.fem.cache import cached_arg_value, cached_vec_type, dynamic_func
from warp._src.fem.types import NULL_ELEMENT_INDEX, OUTSIDE, ElementIndex, make_coords, make_free_sample

from .closest_point import project_on_box_at_origin, project_on_box_at_origin_2d
from .element import Element
from .geometry import Geometry

_wp_module_name_ = "warp.fem.geometry.grid_3d"


def _make_grid3d_cell_arg(scalar_type):
    vec3_type = cached_vec_type(3, scalar_type)

    @cache.dynamic_struct(suffix=scalar_type)
    class Grid3DCellArg:
        res: wp.vec3i
        cell_size: vec3_type
        origin: vec3_type

    return Grid3DCellArg


def _make_grid3d_side_arg(cell_arg_type, scalar_type):
    @cache.dynamic_struct(suffix=scalar_type)
    class Grid3DSideArg:
        cell_count: int
        axis_offsets: wp.vec3i
        cell_arg: cell_arg_type

    return Grid3DSideArg


class Grid3D(Geometry):
    """Three-dimensional regular grid geometry."""

    dimension = 3

    _dynamic_attribute_constructors: ClassVar = {
        **Geometry._dynamic_attribute_constructors,
    }

    def __init__(
        self,
        res: wp.vec3i,
        bounds_lo: wp.vec3 | None = None,
        bounds_hi: wp.vec3 | None = None,
        scalar_type: type = wp.float32,
    ):
        """Construct a dense 3D grid.

        Args:
            res: Resolution of the grid along each dimension
            bounds_lo: Position of the lower bound of the axis-aligned grid
            bounds_hi: Position of the upper bound of the axis-aligned grid
            scalar_type: Scalar type for grid coordinates (``wp.float32`` or ``wp.float64``)
        """

        self._scalar_type = scalar_type
        vec3_type = cached_vec_type(3, scalar_type)

        if bounds_lo is None:
            bounds_lo = vec3_type(0.0)
        elif scalar_type != wp.float32:
            bounds_lo = vec3_type(float(bounds_lo[0]), float(bounds_lo[1]), float(bounds_lo[2]))

        if bounds_hi is None:
            bounds_hi = vec3_type(1.0)
        elif scalar_type != wp.float32:
            bounds_hi = vec3_type(float(bounds_hi[0]), float(bounds_hi[1]), float(bounds_hi[2]))

        self.bounds_lo = bounds_lo
        self.bounds_hi = bounds_hi
        self._res = res

        # Dynamic Arg structs
        self.CellArg = _make_grid3d_cell_arg(scalar_type)
        self.SideArg = _make_grid3d_side_arg(self.CellArg, scalar_type)
        self.SideIndexArg = self.SideArg

        cache.setup_dynamic_attributes(self)

    @property
    def scalar_type(self):
        return self._scalar_type

    @cached_property
    def extents(self):
        vec3_type = cached_vec_type(3, self._scalar_type)
        return vec3_type(
            float(self.bounds_hi[0]) - float(self.bounds_lo[0]),
            float(self.bounds_hi[1]) - float(self.bounds_lo[1]),
            float(self.bounds_hi[2]) - float(self.bounds_lo[2]),
        )

    @cached_property
    def cell_size(self):
        vec3_type = cached_vec_type(3, self._scalar_type)
        ex = self.extents
        return vec3_type(
            float(ex[0]) / float(self.res[0]),
            float(ex[1]) / float(self.res[1]),
            float(ex[2]) / float(self.res[2]),
        )

    def cell_count(self):
        return self.res[0] * self.res[1] * self.res[2]

    def vertex_count(self):
        return (self.res[0] + 1) * (self.res[1] + 1) * (self.res[2] + 1)

    def side_count(self):
        return (
            (self.res[0] + 1) * self.res[1] * self.res[2]
            + self.res[0] * (self.res[1] + 1) * self.res[2]
            + self.res[0] * self.res[1] * (self.res[2] + 1)
        )

    def edge_count(self):
        return (
            (self.res[0] + 1) * (self.res[1] + 1) * self.res[2]
            + self.res[0] * (self.res[1] + 1) * (self.res[2] + 1)
            + (self.res[0] + 1) * self.res[1] * (self.res[2] + 1)
        )

    def boundary_side_count(self):
        return 2 * (self.res[1] * self.res[2] + self.res[0] * self.res[2] + self.res[0] * self.res[1])

    def reference_cell(self) -> Element:
        return Element.CUBE

    def reference_side(self) -> Element:
        return Element.SQUARE

    @property
    def res(self):
        return self._res

    @property
    def origin(self):
        return self.bounds_lo

    @cached_property
    def strides(self):
        return wp.vec3i(self.res[1] * self.res[2], self.res[2], 1)

    Cell = wp.vec3i

    # -- Static integer-only helpers --

    @wp.func
    def _to_3d_index(strides: wp.vec2i, index: int):
        x = index // strides[0]
        y = (index - strides[0] * x) // strides[1]
        z = index - strides[0] * x - strides[1] * y
        return wp.vec3i(x, y, z)

    @wp.func
    def _from_3d_index(strides: wp.vec2i, index: wp.vec3i):
        return strides[0] * index[0] + strides[1] * index[1] + index[2]

    @wp.func
    def cell_index(res: wp.vec3i, cell: wp.vec3i):
        strides = wp.vec2i(res[1] * res[2], res[2])
        return Grid3D._from_3d_index(strides, cell)

    @wp.func
    def get_cell(res: wp.vec3i, cell_index: ElementIndex):
        strides = wp.vec2i(res[1] * res[2], res[2])
        return Grid3D._to_3d_index(strides, cell_index)

    @wp.struct
    class Side:
        axis: int
        origin: wp.vec3i

    # Default (fp32) SideArg
    @wp.func
    def _world_to_local(axis: int, vec: Any):
        return type(vec)(vec[axis], vec[(axis + 1) % 3], vec[(axis + 2) % 3])

    @wp.func
    def _local_to_world(axis: int, vec: Any):
        return type(vec)(vec[(2 * axis) % 3], vec[(2 * axis + 1) % 3], vec[(2 * axis + 2) % 3])

    @wp.func
    def _local_to_world_axis(axis: int, loc_index: Any):
        return (axis + loc_index) % 3

    # -- Fill arg --

    def fill_cell_arg(self, args, device):
        args.res = self.res
        args.origin = self.bounds_lo
        args.cell_size = self.cell_size

    @cached_arg_value
    def side_arg_value(self, device):
        args = self.SideArg()
        axis_dims = wp.vec3i(
            self.res[1] * self.res[2],
            self.res[2] * self.res[0],
            self.res[0] * self.res[1],
        )
        args.axis_offsets = wp.vec3i(0, axis_dims[0], axis_dims[0] + axis_dims[1])
        args.cell_count = self.cell_count()
        args.cell_arg = self.cell_arg_value(device)
        return args

    def side_index_arg_value(self, device):
        return self.side_arg_value(device)

    def supports_cell_lookup(self, device):
        return True

    # -- Dynamic function constructors --

    @wp.func
    def _get_side(arg: Any, side_index: ElementIndex):
        res = arg.cell_arg.res

        if side_index < 3 * arg.cell_count:
            axis = side_index // arg.cell_count
            cell_index = side_index - axis * arg.cell_count
            origin_loc = Grid3D._world_to_local(axis, Grid3D.get_cell(res, cell_index))
            return Grid3D.Side(axis, origin_loc)

        axis_offsets = arg.axis_offsets
        axis_side_index = side_index - 3 * arg.cell_count
        if axis_side_index < axis_offsets[1]:
            axis = 0
        elif axis_side_index < axis_offsets[2]:
            axis = 1
        else:
            axis = 2

        altitude = res[Grid3D._local_to_world_axis(axis, 0)]
        lat_long = axis_side_index - axis_offsets[axis]
        latitude_res = res[Grid3D._local_to_world_axis(axis, 2)]
        longitude = lat_long // latitude_res
        latitude = lat_long - longitude * latitude_res

        return Grid3D.Side(axis, wp.vec3i(altitude, longitude, latitude))

    @wp.func
    def _side_index(arg: Any, side: Any):
        alt_axis = Grid3D._local_to_world_axis(side.axis, 0)
        if side.origin[0] == arg.cell_arg.res[alt_axis]:
            longitude = side.origin[1]
            latitude = side.origin[2]
            latitude_res = arg.cell_arg.res[Grid3D._local_to_world_axis(side.axis, 2)]
            lat_long = latitude_res * longitude + latitude
            return 3 * arg.cell_count + arg.axis_offsets[side.axis] + lat_long

        cell_index = Grid3D.cell_index(arg.cell_arg.res, Grid3D._local_to_world(side.axis, side.origin))
        return side.axis * arg.cell_count + cell_index

    @wp.func
    def boundary_side_index(args: Any, boundary_side_index: int):
        axis_side_index = boundary_side_index // 2
        border = boundary_side_index - 2 * axis_side_index

        if axis_side_index < args.axis_offsets[1]:
            axis = 0
        elif axis_side_index < args.axis_offsets[2]:
            axis = 1
        else:
            axis = 2

        lat_long = axis_side_index - args.axis_offsets[axis]
        latitude_res = args.cell_arg.res[Grid3D._local_to_world_axis(axis, 2)]
        longitude = lat_long // latitude_res
        latitude = lat_long - longitude * latitude_res
        altitude = border * args.cell_arg.res[axis]

        side = Grid3D.Side(axis, wp.vec3i(altitude, longitude, latitude))
        return Grid3D._side_index(args, side)

    @wp.func
    def cell_position(args: Any, s: Any):
        cell = Grid3D.get_cell(args.res, s.element_index)
        return (
            type(args.cell_size)(
                (args.cell_size.dtype(cell[0]) + s.element_coords[0]) * args.cell_size[0],
                (args.cell_size.dtype(cell[1]) + s.element_coords[1]) * args.cell_size[1],
                (args.cell_size.dtype(cell[2]) + s.element_coords[2]) * args.cell_size[2],
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
        ijk = Grid3D.get_cell(args.res, cell_index)
        return uvw - type(pos)(ijk)

    @wp.func
    def cell_closest_point(args: Any, cell_index: int, pos: Any):
        ijk_world = wp.cw_mul(type(pos)(Grid3D.get_cell(args.res, cell_index)), args.cell_size) + args.origin
        dist_sq, coords = project_on_box_at_origin(pos - ijk_world, args.cell_size)
        return coords, dist_sq

    @wp.func
    def cell_measure(args: Any, s: Any):
        return args.cell_size[0] * args.cell_size[1] * args.cell_size[2]

    @wp.func
    def cell_normal(args: Any, s: Any):
        return type(args.cell_size)(args.cell_size.dtype(0.0))

    @wp.func
    def side_position(args: Any, s: Any):
        side = Grid3D._get_side(args, s.element_index)
        coord0 = wp.where(
            side.origin[0] == 0, args.cell_arg.cell_size.dtype(1.0) - s.element_coords[0], s.element_coords[0]
        )

        local_pos = type(args.cell_arg.cell_size)(
            args.cell_arg.cell_size.dtype(side.origin[0]),
            args.cell_arg.cell_size.dtype(side.origin[1]) + coord0,
            args.cell_arg.cell_size.dtype(side.origin[2]) + s.element_coords[1],
        )
        return args.cell_arg.origin + wp.cw_mul(Grid3D._local_to_world(side.axis, local_pos), args.cell_arg.cell_size)

    @wp.func
    def side_deformation_gradient(args: Any, s: Any):
        side = Grid3D._get_side(args, s.element_index)
        sign = wp.where(side.origin[0] == 0, args.cell_arg.cell_size.dtype(-1.0), args.cell_arg.cell_size.dtype(1.0))
        return wp.matrix_from_cols(
            wp.cw_mul(
                Grid3D._local_to_world(
                    side.axis,
                    type(args.cell_arg.cell_size)(
                        args.cell_arg.cell_size.dtype(0.0), sign, args.cell_arg.cell_size.dtype(0.0)
                    ),
                ),
                args.cell_arg.cell_size,
            ),
            wp.cw_mul(
                Grid3D._local_to_world(
                    side.axis,
                    type(args.cell_arg.cell_size)(
                        args.cell_arg.cell_size.dtype(0.0),
                        args.cell_arg.cell_size.dtype(0.0),
                        args.cell_arg.cell_size.dtype(1.0),
                    ),
                ),
                args.cell_arg.cell_size,
            ),
        )

    @wp.func
    def side_inner_inverse_deformation_gradient(args: Any, s: Any):
        return Grid3D.cell_inverse_deformation_gradient(args.cell_arg, s)

    @wp.func
    def side_outer_inverse_deformation_gradient(args: Any, s: Any):
        return Grid3D.cell_inverse_deformation_gradient(args.cell_arg, s)

    @wp.func
    def side_measure(args: Any, s: Any):
        side = Grid3D._get_side(args, s.element_index)
        long_axis = Grid3D._local_to_world_axis(side.axis, 1)
        lat_axis = Grid3D._local_to_world_axis(side.axis, 2)
        return args.cell_arg.cell_size[long_axis] * args.cell_arg.cell_size[lat_axis]

    @wp.func
    def side_measure_ratio(args: Any, s: Any):
        side = Grid3D._get_side(args, s.element_index)
        alt_axis = Grid3D._local_to_world_axis(side.axis, 0)
        return args.cell_arg.cell_size.dtype(1.0) / args.cell_arg.cell_size[alt_axis]

    @wp.func
    def side_normal(args: Any, s: Any):
        side = Grid3D._get_side(args, s.element_index)
        sign = wp.where(side.origin[0] == 0, args.cell_arg.cell_size.dtype(-1.0), args.cell_arg.cell_size.dtype(1.0))
        local_n = type(args.cell_arg.cell_size)(
            sign, args.cell_arg.cell_size.dtype(0.0), args.cell_arg.cell_size.dtype(0.0)
        )
        return Grid3D._local_to_world(side.axis, local_n)

    @wp.func
    def side_inner_cell_index(arg: Any, side_index: ElementIndex):
        side = Grid3D._get_side(arg, side_index)
        inner_alt = wp.where(side.origin[0] == 0, 0, side.origin[0] - 1)
        inner_origin = wp.vec3i(inner_alt, side.origin[1], side.origin[2])
        cell = Grid3D._local_to_world(side.axis, inner_origin)
        return Grid3D.cell_index(arg.cell_arg.res, cell)

    @wp.func
    def side_outer_cell_index(arg: Any, side_index: ElementIndex):
        side = Grid3D._get_side(arg, side_index)
        alt_axis = Grid3D._local_to_world_axis(side.axis, 0)
        outer_alt = wp.where(
            side.origin[0] == arg.cell_arg.res[alt_axis], arg.cell_arg.res[alt_axis] - 1, side.origin[0]
        )
        outer_origin = wp.vec3i(outer_alt, side.origin[1], side.origin[2])
        cell = Grid3D._local_to_world(side.axis, outer_origin)
        return Grid3D.cell_index(arg.cell_arg.res, cell)

    @wp.func
    def side_inner_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        side = Grid3D._get_side(args, side_index)
        inner_alt = wp.where(side.origin[0] == 0, side_coords.dtype(0.0), side_coords.dtype(1.0))
        side_coord0 = wp.where(side.origin[0] == 0, side_coords.dtype(1.0) - side_coords[0], side_coords[0])
        return Grid3D._local_to_world(side.axis, type(args.cell_arg.cell_size)(inner_alt, side_coord0, side_coords[1]))

    @wp.func
    def side_outer_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        side = Grid3D._get_side(args, side_index)
        alt_axis = Grid3D._local_to_world_axis(side.axis, 0)
        outer_alt = wp.where(
            side.origin[0] == args.cell_arg.res[alt_axis], side_coords.dtype(1.0), side_coords.dtype(0.0)
        )
        side_coord0 = wp.where(side.origin[0] == 0, side_coords.dtype(1.0) - side_coords[0], side_coords[0])
        return Grid3D._local_to_world(side.axis, type(args.cell_arg.cell_size)(outer_alt, side_coord0, side_coords[1]))

    @wp.func
    def side_from_cell_coords(
        args: Any,
        side_index: ElementIndex,
        element_index: ElementIndex,
        element_coords: Any,
    ):
        side = Grid3D._get_side(args, side_index)
        cell = Grid3D.get_cell(args.cell_arg.res, element_index)

        if element_coords.dtype(side.origin[0] - cell[side.axis]) == element_coords[side.axis]:
            long_axis = Grid3D._local_to_world_axis(side.axis, 1)
            lat_axis = Grid3D._local_to_world_axis(side.axis, 2)
            long_coord = element_coords[long_axis]
            long_coord = wp.where(side.origin[0] == 0, element_coords.dtype(1.0) - long_coord, long_coord)
            return make_coords(long_coord, element_coords[lat_axis])

        return type(element_coords)(element_coords.dtype(OUTSIDE))

    @wp.func
    def side_to_cell_arg(side_arg: Any):
        return side_arg.cell_arg

    @wp.func
    def side_coordinates(args: Any, side_index: int, pos: Any):
        cell_arg = args.cell_arg
        side = Grid3D._get_side(args, side_index)

        pos_loc = Grid3D._world_to_local(side.axis, wp.cw_div(pos - cell_arg.origin, cell_arg.cell_size)) - type(pos)(
            side.origin
        )

        coord0 = wp.where(side.origin[0] == 0, pos.dtype(1.0) - pos_loc[1], pos_loc[1])
        return make_coords(coord0, pos_loc[2])

    @wp.func
    def side_closest_point(args: Any, side_index: int, pos: Any):
        coord = Grid3D.side_coordinates(args, side_index, pos)

        cell_arg = args.cell_arg
        side = Grid3D._get_side(args, side_index)

        loc_cell_size = Grid3D._world_to_local(side.axis, cell_arg.cell_size)
        long_lat_sizes = wp.vector(loc_cell_size[1], loc_cell_size[2], dtype=pos.dtype)
        dist, proj_coord = project_on_box_at_origin_2d(wp.vector(coord[0], coord[1], dtype=pos.dtype), long_lat_sizes)
        return proj_coord, dist

    def make_filtered_cell_lookup(self, filter_func: wp.Function = None):
        """Create a filtered cell lookup function.

        Args:
            filter_func: Optional device predicate to filter candidate cells.
        """
        suffix = f"{self.name}{filter_func.key if filter_func is not None else ''}"
        scalar = self._scalar_type
        vec3_type = cached_vec_type(3, scalar)
        CoordsType = self.coords_type

        @dynamic_func(suffix=suffix)
        def cell_lookup(args: self.CellArg, pos: vec3_type, max_dist: float, filter_data: Any, filter_target: Any):
            cell_size = args.cell_size
            res = args.res

            loc_pos = wp.cw_div(pos - args.origin, cell_size)
            x = wp.clamp(loc_pos[0], scalar(0.0), scalar(res[0]))
            y = wp.clamp(loc_pos[1], scalar(0.0), scalar(res[1]))
            z = wp.clamp(loc_pos[2], scalar(0.0), scalar(res[2]))

            x_cell = wp.min(wp.floor(x), scalar(res[0]) - scalar(1.0))
            y_cell = wp.min(wp.floor(y), scalar(res[1]) - scalar(1.0))
            z_cell = wp.min(wp.floor(z), scalar(res[2]) - scalar(1.0))

            coords = CoordsType(x - x_cell, y - y_cell, z - z_cell)
            cell_index = Grid3D.cell_index(res, Grid3D.Cell(int(x_cell), int(y_cell), int(z_cell)))

            if wp.static(filter_func is None):
                return make_free_sample(cell_index, coords)
            else:
                if filter_func(filter_data, cell_index) == filter_target:
                    return make_free_sample(cell_index, coords)

                offset = scalar(0.5)
                min_cell_size = wp.min(cell_size)
                max_offset = wp.ceil(max_dist / min_cell_size)
                scales = wp.cw_div(vec3_type(min_cell_size), cell_size)

                closest_cell = NULL_ELEMENT_INDEX
                closest_coords = CoordsType()

                while closest_cell == NULL_ELEMENT_INDEX:
                    i_min = wp.max(0, int(wp.floor(x - offset * scales[0])))
                    i_max = wp.min(res[0], int(wp.floor(x + offset * scales[0])) + 1)
                    j_min = wp.max(0, int(wp.floor(y - offset * scales[1])))
                    j_max = wp.min(res[1], int(wp.floor(y + offset * scales[1])) + 1)
                    k_min = wp.max(0, int(wp.floor(z - offset * scales[2])))
                    k_max = wp.min(res[2], int(wp.floor(z + offset * scales[2])) + 1)

                    closest_dist = min_cell_size * min_cell_size * scalar(offset * offset)

                    for i in range(i_min, i_max):
                        for j in range(j_min, j_max):
                            for k in range(k_min, k_max):
                                ijk = Grid3D.Cell(i, j, k)
                                cell_index = Grid3D.cell_index(res, ijk)
                                if filter_func(filter_data, cell_index) == filter_target:
                                    rel_pos = wp.cw_mul(loc_pos - vec3_type(ijk), cell_size)
                                    dist, coords = project_on_box_at_origin(rel_pos, cell_size)

                                    if dist <= closest_dist:
                                        closest_dist = dist
                                        closest_coords = coords
                                        closest_cell = cell_index

                    if offset >= max_offset:
                        break
                    offset = wp.min(scalar(3.0) * offset, max_offset)

                return make_free_sample(closest_cell, closest_coords)

        return cell_lookup
