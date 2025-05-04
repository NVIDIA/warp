# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from warp.fem.cache import cached_arg_value, dynamic_func
from warp.fem.types import NULL_ELEMENT_INDEX, OUTSIDE, Coords, ElementIndex, Sample, make_free_sample

from .closest_point import project_on_box_at_origin
from .element import LinearEdge, Square
from .geometry import Geometry


@wp.struct
class Grid2DCellArg:
    res: wp.vec2i
    cell_size: wp.vec2
    origin: wp.vec2


class Grid2D(Geometry):
    """Two-dimensional regular grid geometry"""

    dimension = 2

    ALT_AXIS = 0
    LONG_AXIS = 1

    def __init__(self, res: wp.vec2i, bounds_lo: Optional[wp.vec2] = None, bounds_hi: Optional[wp.vec2] = None):
        """Constructs a dense 2D grid

        Args:
            res: Resolution of the grid along each dimension
            bounds_lo: Position of the lower bound of the axis-aligned grid
            bounds_hi: Position of the upper bound of the axis-aligned grid
        """

        if bounds_lo is None:
            bounds_lo = wp.vec2(0.0)

        if bounds_hi is None:
            bounds_hi = wp.vec2(1.0)

        self.bounds_lo = bounds_lo
        self.bounds_hi = bounds_hi

        self._res = res

    @property
    def extents(self) -> wp.vec3:
        # Avoid using native sub due to higher over of calling builtins from Python
        return wp.vec2(
            self.bounds_hi[0] - self.bounds_lo[0],
            self.bounds_hi[1] - self.bounds_lo[1],
        )

    @property
    def cell_size(self) -> wp.vec2:
        ex = self.extents
        return wp.vec2(
            ex[0] / self.res[0],
            ex[1] / self.res[1],
        )

    def cell_count(self):
        return self.res[0] * self.res[1]

    def vertex_count(self):
        return (self.res[0] + 1) * (self.res[1] + 1)

    def side_count(self):
        return 2 * self.cell_count() + self.res[0] + self.res[1]

    def boundary_side_count(self):
        return 2 * (self.res[0] + self.res[1])

    def reference_cell(self) -> Square:
        return Square()

    def reference_side(self) -> LinearEdge:
        return LinearEdge()

    @property
    def res(self):
        return self._res

    @property
    def origin(self):
        return self.bounds_lo

    @property
    def strides(self):
        return wp.vec2i(self.res[1], 1)

    # Utility device functions
    CellArg = Grid2DCellArg
    Cell = wp.vec2i

    @wp.func
    def _to_2d_index(x_stride: int, index: int):
        x = index // x_stride
        y = index - x_stride * x
        return wp.vec2i(x, y)

    @wp.func
    def _from_2d_index(x_stride: int, index: wp.vec2i):
        return x_stride * index[0] + index[1]

    @wp.func
    def cell_index(res: wp.vec2i, cell: Cell):
        return Grid2D._from_2d_index(res[1], cell)

    @wp.func
    def get_cell(res: wp.vec2i, cell_index: ElementIndex):
        return Grid2D._to_2d_index(res[1], cell_index)

    @wp.struct
    class Side:
        axis: int  # normal; 0: horizontal, 1: vertical
        origin: wp.vec2i  # index of vertex at corner (0,0)

    @wp.struct
    class SideArg:
        cell_count: int
        axis_offsets: wp.vec2i
        cell_arg: Grid2DCellArg

    SideIndexArg = SideArg

    @wp.func
    def orient(axis: int, vec: Any):
        return wp.where(axis == 0, vec, type(vec)(vec[1], vec[0]))

    @wp.func
    def orient(axis: int, coord: int):
        return wp.where(axis == 0, coord, 1 - coord)

    @wp.func
    def is_flipped(side: Side):
        # Flip such that the boundary is CCW
        return (side.axis == 0) == (side.origin[Grid2D.ALT_AXIS] == 0)

    @wp.func
    def side_index(arg: SideArg, side: Side):
        alt_axis = Grid2D.orient(side.axis, 0)
        if side.origin[0] == arg.cell_arg.res[alt_axis]:
            # Upper-boundary side
            longitude = side.origin[1]
            return 2 * arg.cell_count + arg.axis_offsets[side.axis] + longitude

        cell_index = Grid2D.cell_index(arg.cell_arg.res, Grid2D.orient(side.axis, side.origin))
        return side.axis * arg.cell_count + cell_index

    @wp.func
    def get_side(arg: SideArg, side_index: ElementIndex):
        if side_index < 2 * arg.cell_count:
            axis = side_index // arg.cell_count
            cell_index = side_index - axis * arg.cell_count
            origin = Grid2D.orient(axis, Grid2D.get_cell(arg.cell_arg.res, cell_index))
            return Grid2D.Side(axis, origin)

        axis_side_index = side_index - 2 * arg.cell_count
        axis = wp.where(axis_side_index < arg.axis_offsets[1], 0, 1)

        altitude = arg.cell_arg.res[Grid2D.orient(axis, 0)]
        longitude = axis_side_index - arg.axis_offsets[axis]

        origin_loc = wp.vec2i(altitude, longitude)
        return Grid2D.Side(axis, origin_loc)

    # Geometry device interface

    @cached_arg_value
    def cell_arg_value(self, device) -> CellArg:
        args = self.CellArg()
        args.res = self.res
        args.cell_size = self.cell_size
        args.origin = self.bounds_lo
        return args

    @wp.func
    def cell_position(args: CellArg, s: Sample):
        cell = Grid2D.get_cell(args.res, s.element_index)
        return (
            wp.vec2(
                (float(cell[0]) + s.element_coords[0]) * args.cell_size[0],
                (float(cell[1]) + s.element_coords[1]) * args.cell_size[1],
            )
            + args.origin
        )

    @wp.func
    def cell_deformation_gradient(args: CellArg, s: Sample):
        return wp.diag(args.cell_size)

    @wp.func
    def cell_inverse_deformation_gradient(args: CellArg, s: Sample):
        return wp.diag(wp.cw_div(wp.vec2(1.0), args.cell_size))

    @wp.func
    def cell_coordinates(args: Grid2DCellArg, cell_index: int, pos: wp.vec2):
        uvw = wp.cw_div(pos - args.origin, args.cell_size)
        ij = Grid2D.get_cell(args.res, cell_index)
        return Coords(uvw[0] - float(ij[0]), uvw[1] - float(ij[1]), 0.0)

    @wp.func
    def cell_closest_point(args: Grid2DCellArg, cell_index: int, pos: wp.vec2):
        ij_world = wp.cw_mul(wp.vec2(Grid2D.get_cell(args.res, cell_index)), args.cell_size) + args.origin
        dist_sq, coords = project_on_box_at_origin(pos - ij_world, args.cell_size)
        return coords, dist_sq

    def supports_cell_lookup(self, device):
        return True

    def make_filtered_cell_lookup(self, filter_func: wp.Function = None):
        suffix = f"{self.name}{filter_func.func.__qualname__ if filter_func is not None else ''}"

        @dynamic_func(suffix=suffix)
        def cell_lookup(args: self.CellArg, pos: wp.vec2, max_dist: float, filter_data: Any, filter_target: Any):
            cell_size = args.cell_size
            res = args.res

            # Start at closest point on grid
            loc_pos = wp.cw_div(pos - args.origin, cell_size)
            x = wp.clamp(loc_pos[0], 0.0, float(res[0]))
            y = wp.clamp(loc_pos[1], 0.0, float(res[1]))

            x_cell = wp.min(wp.floor(x), float(res[0] - 1))
            y_cell = wp.min(wp.floor(y), float(res[1] - 1))

            coords = Coords(x - x_cell, y - y_cell, 0.0)
            cell_index = Grid2D.cell_index(res, Grid2D.Cell(int(x_cell), int(y_cell)))

            if wp.static(filter_func is None):
                return make_free_sample(cell_index, coords)
            else:
                if filter_func(filter_data, cell_index) == filter_target:
                    return make_free_sample(cell_index, coords)

                offset = float(0.5)
                min_cell_size = wp.min(cell_size)
                max_offset = wp.ceil(max_dist / min_cell_size)

                scales = wp.cw_div(wp.vec2(min_cell_size), cell_size)

                closest_cell = NULL_ELEMENT_INDEX
                closest_coords = Coords()

                # Iterate over increasingly larger neighborhoods
                while closest_cell == NULL_ELEMENT_INDEX:
                    i_min = wp.max(0, int(wp.floor(x - offset * scales[0])))
                    i_max = wp.min(res[0], int(wp.floor(x + offset * scales[0])) + 1)
                    j_min = wp.max(0, int(wp.floor(y - offset * scales[1])))
                    j_max = wp.min(res[1], int(wp.floor(y + offset * scales[1])) + 1)

                    closest_dist = min_cell_size * min_cell_size * float(offset * offset)

                    for i in range(i_min, i_max):
                        for j in range(j_min, j_max):
                            ij = Grid2D.Cell(i, j)
                            cell_index = Grid2D.cell_index(res, ij)
                            if filter_func(filter_data, cell_index) == filter_target:
                                rel_pos = wp.cw_mul(loc_pos - wp.vec2(ij), cell_size)
                                dist, coords = project_on_box_at_origin(rel_pos, cell_size)

                                if dist <= closest_dist:
                                    closest_dist = dist
                                    closest_coords = coords
                                    closest_cell = cell_index

                    if offset >= max_offset:
                        break
                    offset = wp.min(3.0 * offset, max_offset)

                return make_free_sample(closest_cell, closest_coords)

        return cell_lookup

    @wp.func
    def cell_measure(args: CellArg, s: Sample):
        return args.cell_size[0] * args.cell_size[1]

    @wp.func
    def cell_normal(args: CellArg, s: Sample):
        return wp.vec2(0.0)

    @cached_arg_value
    def side_arg_value(self, device) -> SideArg:
        args = self.SideArg()

        args.axis_offsets = wp.vec2i(
            0,
            self.res[1],
        )
        args.cell_count = self.cell_count()
        args.cell_arg = self.cell_arg_value(device)
        return args

    def side_index_arg_value(self, device) -> SideIndexArg:
        return self.side_arg_value(device)

    @wp.func
    def boundary_side_index(args: SideArg, boundary_side_index: int):
        """Boundary side to side index"""

        axis_side_index = boundary_side_index // 2
        border = boundary_side_index - 2 * axis_side_index

        if axis_side_index < args.axis_offsets[1]:
            axis = 0
        else:
            axis = 1

        longitude = axis_side_index - args.axis_offsets[axis]
        altitude = border * args.cell_arg.res[axis]

        side = Grid2D.Side(axis, wp.vec2i(altitude, longitude))
        return Grid2D.side_index(args, side)

    @wp.func
    def side_position(args: SideArg, s: Sample):
        side = Grid2D.get_side(args, s.element_index)

        flip = Grid2D.is_flipped(side)
        coord = wp.where(flip, 1.0 - s.element_coords[0], s.element_coords[0])

        local_pos = wp.vec2(side.origin) + wp.vec2(0.0, coord)
        pos = args.cell_arg.origin + wp.cw_mul(Grid2D.orient(side.axis, local_pos), args.cell_arg.cell_size)

        return pos

    @wp.func
    def side_deformation_gradient(args: SideArg, s: Sample):
        side = Grid2D.get_side(args, s.element_index)

        flip = Grid2D.is_flipped(side)
        sign = wp.where(flip, -1.0, 1.0)

        return wp.cw_mul(Grid2D.orient(side.axis, wp.vec2(0.0, sign)), args.cell_arg.cell_size)

    @wp.func
    def side_inner_inverse_deformation_gradient(args: SideArg, s: Sample):
        return Grid2D.cell_inverse_deformation_gradient(args.cell_arg, s)

    @wp.func
    def side_outer_inverse_deformation_gradient(args: SideArg, s: Sample):
        return Grid2D.cell_inverse_deformation_gradient(args.cell_arg, s)

    @wp.func
    def side_measure(args: SideArg, s: Sample):
        side = Grid2D.get_side(args, s.element_index)
        long_axis = Grid2D.orient(side.axis, Grid2D.LONG_AXIS)
        return args.cell_arg.cell_size[long_axis]

    @wp.func
    def side_measure_ratio(args: SideArg, s: Sample):
        side = Grid2D.get_side(args, s.element_index)
        alt_axis = Grid2D.orient(side.axis, Grid2D.ALT_AXIS)
        return 1.0 / args.cell_arg.cell_size[alt_axis]

    @wp.func
    def side_normal(args: SideArg, s: Sample):
        side = Grid2D.get_side(args, s.element_index)

        # intentionally not using is_flipped to account for normql sign switch with orient(axis=1)
        flip = side.origin[Grid2D.ALT_AXIS] == 0
        sign = wp.where(flip, -1.0, 1.0)

        local_n = wp.vec2(sign, 0.0)
        return Grid2D.orient(side.axis, local_n)

    @wp.func
    def side_inner_cell_index(arg: SideArg, side_index: ElementIndex):
        side = Grid2D.get_side(arg, side_index)

        inner_alt = wp.where(side.origin[Grid2D.ALT_AXIS] == 0, 0, side.origin[Grid2D.ALT_AXIS] - 1)

        inner_origin = wp.vec2i(inner_alt, side.origin[1])

        cell = Grid2D.orient(side.axis, inner_origin)
        return Grid2D.cell_index(arg.cell_arg.res, cell)

    @wp.func
    def side_outer_cell_index(arg: SideArg, side_index: ElementIndex):
        side = Grid2D.get_side(arg, side_index)

        alt_axis = Grid2D.orient(side.axis, 0)
        outer_alt = wp.where(
            side.origin[Grid2D.ALT_AXIS] == arg.cell_arg.res[alt_axis], arg.cell_arg.res[alt_axis] - 1, side.origin[0]
        )

        outer_origin = wp.vec2i(outer_alt, side.origin[Grid2D.LONG_AXIS])

        cell = Grid2D.orient(side.axis, outer_origin)
        return Grid2D.cell_index(arg.cell_arg.res, cell)

    @wp.func
    def side_inner_cell_coords(args: SideArg, side_index: ElementIndex, side_coords: Coords):
        side = Grid2D.get_side(args, side_index)

        inner_alt = wp.where(side.origin[Grid2D.ALT_AXIS] == 0, 0.0, 1.0)

        flip = Grid2D.is_flipped(side)
        side_coord = wp.where(flip, 1.0 - side_coords[0], side_coords[0])

        coords = Grid2D.orient(side.axis, wp.vec2(inner_alt, side_coord))
        return Coords(coords[0], coords[1], 0.0)

    @wp.func
    def side_outer_cell_coords(args: SideArg, side_index: ElementIndex, side_coords: Coords):
        side = Grid2D.get_side(args, side_index)

        alt_axis = Grid2D.orient(side.axis, Grid2D.ALT_AXIS)
        outer_alt = wp.where(side.origin[Grid2D.ALT_AXIS] == args.cell_arg.res[alt_axis], 1.0, 0.0)

        flip = Grid2D.is_flipped(side)
        side_coord = wp.where(flip, 1.0 - side_coords[0], side_coords[0])

        coords = Grid2D.orient(side.axis, wp.vec2(outer_alt, side_coord))
        return Coords(coords[0], coords[1], 0.0)

    @wp.func
    def side_from_cell_coords(
        args: SideArg,
        side_index: ElementIndex,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        side = Grid2D.get_side(args, side_index)
        cell = Grid2D.get_cell(args.cell_arg.res, element_index)

        if float(side.origin[Grid2D.ALT_AXIS] - cell[side.axis]) == element_coords[side.axis]:
            long_axis = Grid2D.orient(side.axis, Grid2D.LONG_AXIS)
            axis_coord = element_coords[long_axis]
            flip = Grid2D.is_flipped(side)
            side_coord = wp.where(flip, 1.0 - axis_coord, axis_coord)
            return Coords(side_coord, 0.0, 0.0)

        return Coords(OUTSIDE)

    @wp.func
    def side_to_cell_arg(side_arg: SideArg):
        return side_arg.cell_arg

    @wp.func
    def side_coordinates(args: SideArg, side_index: int, pos: wp.vec2):
        cell_arg = args.cell_arg
        side = Grid2D.get_side(args, side_index)
        long_axis = Grid2D.orient(side.axis, Grid2D.LONG_AXIS)
        flip = Grid2D.is_flipped(side)

        long_loc = (pos[long_axis] - cell_arg.origin[long_axis]) / cell_arg.cell_size[long_axis] - float(side.origin[1])
        coord = wp.where(flip, 1.0 - long_loc, long_loc)

        return Coords(coord, 0.0, 0.0)

    @wp.func
    def side_closest_point(args: SideArg, side_index: int, pos: wp.vec2):
        coord = Grid2D.side_coordinates(args, side_index, pos)

        cell_arg = args.cell_arg
        side = Grid2D.get_side(args, side_index)
        long_axis = Grid2D.orient(side.axis, Grid2D.LONG_AXIS)
        proj_coord = wp.clamp(coord, 0.0, 1.0)
        dist = (coord - proj_coord) * cell_arg.cell_size[long_axis]
        return Coords(proj_coord, 0.0, 0.0), dist * dist
