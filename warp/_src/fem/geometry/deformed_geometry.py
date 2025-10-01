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

from typing import ClassVar

import warp as wp
from warp.fem import cache
from warp.fem.polynomial import Polynomial
from warp.fem.types import Coords, ElementIndex, Sample, make_free_sample

from .geometry import Geometry

_mat32 = wp.mat(shape=(3, 2), dtype=float)


class DeformedGeometry(Geometry):
    _dynamic_attribute_constructors_phase_1: ClassVar = {
        "CellArg": lambda obj: obj._make_cell_arg(),
        "SideArg": lambda obj: obj._make_side_arg(),
        "cell_position": lambda obj: obj._make_cell_position(),
        "cell_deformation_gradient": lambda obj: obj._make_cell_deformation_gradient(),
        "side_to_cell_arg": lambda obj: obj._make_side_to_cell_arg(),
        "side_position": lambda obj: obj._make_side_position(),
        "side_deformation_gradient": lambda obj: obj._make_side_deformation_gradient(),
        "side_inner_cell_index": lambda obj: obj._make_side_inner_cell_index(),
        "side_outer_cell_index": lambda obj: obj._make_side_outer_cell_index(),
        "side_inner_cell_coords": lambda obj: obj._make_side_inner_cell_coords(),
        "side_outer_cell_coords": lambda obj: obj._make_side_outer_cell_coords(),
        "side_from_cell_coords": lambda obj: obj._make_side_from_cell_coords(),
        "cell_bvh_id": lambda obj: obj._make_cell_bvh_id(),
        "cell_bounds": lambda obj: obj._make_cell_bounds(),
    }

    _dynamic_attribute_constructors_phase_2: ClassVar = {
        "cell_closest_point": lambda obj: obj._make_cell_closest_point(),
        "side_closest_point": lambda obj: obj._make_side_closest_point(),
        "cell_coordinates": lambda obj: obj._make_cell_coordinates(),
        "side_coordinates": lambda obj: obj._make_side_coordinates(),
    }

    def __init__(self, field: "wp.fem.field.GeometryField", relative: bool = True, build_bvh: bool = False):
        """Constructs a Deformed Geometry from a displacement or absolute position field defined over a base geometry.
        The deformation field does not need to be isoparameteric.

        See also: :meth:`warp.fem.DiscreteField.make_deformed_geometry`
        """

        from warp.fem.field import DiscreteField, GeometryField

        if isinstance(field, DiscreteField):
            if not wp.types.type_is_vector(field.dtype) or wp.types.type_size(field.dtype) != field.geometry.dimension:
                raise ValueError(
                    "Invalid value type for position field, must be vector-valued with same dimension as underlying geometry"
                )
        if field.eval_grad_inner is None:
            raise ValueError("Gradient evaluation is not supported on the passed field")

        self._relative = relative

        self.field: GeometryField = field
        self.field_trace = field.trace()
        self.dimension = self.base.dimension

        self.SideIndexArg = self.base.SideIndexArg
        self.cell_count = self.base.cell_count
        self.vertex_count = self.base.vertex_count
        self.side_count = self.base.side_count
        self.boundary_side_count = self.base.boundary_side_count
        self.reference_cell = self.base.reference_cell
        self.reference_side = self.base.reference_side

        self.side_index_arg_value = self.base.side_index_arg_value
        self.fill_side_index_arg = self.base.fill_side_index_arg
        self.boundary_side_index = self.base.boundary_side_index

        cache.setup_dynamic_attributes(self, constructors=self._dynamic_attribute_constructors_phase_1)

        self._make_default_dependent_implementations()

        cache.setup_dynamic_attributes(self, constructors=self._dynamic_attribute_constructors_phase_2)

        if build_bvh:
            self.build_bvh(self.field.dof_values.device)

    @property
    def name(self) -> str:
        return f"DefGeo_{self.field.name}_{'rel' if self._relative else 'abs'}"

    @property
    def base(self) -> Geometry:
        return self.field.geometry.base

    # Geometry device interface

    def _make_cell_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class CellArg:
            base_arg: self.base.CellArg
            field_arg: self.field.EvalArg
            cell_bvh: wp.uint64

        return CellArg

    def cell_arg_value(self, device) -> "DeformedGeometry.CellArg":
        args = self.CellArg()
        self.fill_cell_arg(args, device)
        return args

    def fill_cell_arg(self, args: "DeformedGeometry.CellArg", device):
        self.base.fill_cell_arg(args.base_arg, device)
        self.field.fill_eval_arg(args.field_arg, device)
        args.cell_bvh = self.bvh_id(device)

    def _make_cell_position(self):
        @cache.dynamic_func(suffix=self.name)
        def cell_position_absolute(cell_arg: self.CellArg, s: Sample):
            field_arg = self.field.ElementEvalArg(cell_arg.base_arg, cell_arg.field_arg)
            return self.field.eval_inner(field_arg, s)

        @cache.dynamic_func(suffix=self.name)
        def cell_position(cell_arg: self.CellArg, s: Sample):
            field_arg = self.field.ElementEvalArg(cell_arg.base_arg, cell_arg.field_arg)
            return self.field.eval_inner(field_arg, s) + self.base.cell_position(cell_arg.base_arg, s)

        return cell_position if self._relative else cell_position_absolute

    def _make_cell_deformation_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def cell_deformation_gradient_absolute(cell_arg: self.CellArg, s: Sample):
            field_arg = self.field.ElementEvalArg(cell_arg.base_arg, cell_arg.field_arg)
            return self.field.eval_reference_grad_inner(field_arg, s)

        @cache.dynamic_func(suffix=self.name)
        def cell_deformation_gradient(cell_arg: self.CellArg, s: Sample):
            field_arg = self.field.ElementEvalArg(cell_arg.base_arg, cell_arg.field_arg)
            return self.field.eval_reference_grad_inner(field_arg, s) + self.base.cell_deformation_gradient(
                cell_arg.base_arg, s
            )

        return cell_deformation_gradient if self._relative else cell_deformation_gradient_absolute

    def _make_side_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class SideArg:
            base_arg: self.base.SideArg
            trace_arg: self.field_trace.EvalArg
            field_arg: self.field.EvalArg
            cell_bvh: wp.uint64

        return SideArg

    def side_arg_value(self, device) -> "DeformedGeometry.SideArg":
        args = self.SideArg()
        self.fill_side_arg(args, device)
        return args

    def fill_side_arg(self, args: "DeformedGeometry.SideArg", device):
        self.base.fill_side_arg(args.base_arg, device)
        self.field.fill_eval_arg(args.field_arg, device)
        self.field_trace.fill_eval_arg(args.trace_arg, device)
        args.cell_bvh = self.bvh_id(device)

    def _make_side_position(self):
        @cache.dynamic_func(suffix=self.name)
        def side_position_absolute(args: self.SideArg, s: Sample):
            trace_arg = self.field_trace.ElementEvalArg(args.base_arg, args.trace_arg)
            return self.field_trace.eval_inner(trace_arg, s)

        @cache.dynamic_func(suffix=self.name)
        def side_position(args: self.SideArg, s: Sample):
            trace_arg = self.field_trace.ElementEvalArg(args.base_arg, args.trace_arg)
            return self.field_trace.eval_inner(trace_arg, s) + self.base.side_position(args.base_arg, s)

        return side_position if self._relative else side_position_absolute

    def _make_side_deformation_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def side_deformation_gradient_absolute(args: self.SideArg, s: Sample):
            base_def_grad = self.base.side_deformation_gradient(args.base_arg, s)
            trace_arg = self.field_trace.ElementEvalArg(args.base_arg, args.trace_arg)

            Du = self.field_trace.eval_grad_inner(trace_arg, s)
            return Du * base_def_grad

        @cache.dynamic_func(suffix=self.name)
        def side_deformation_gradient(args: self.SideArg, s: Sample):
            base_def_grad = self.base.side_deformation_gradient(args.base_arg, s)
            trace_arg = self.field_trace.ElementEvalArg(args.base_arg, args.trace_arg)

            Du = self.field_trace.eval_grad_inner(trace_arg, s)
            return base_def_grad + Du * base_def_grad

        return side_deformation_gradient if self._relative else side_deformation_gradient_absolute

    def _make_side_inner_cell_index(self):
        @cache.dynamic_func(suffix=self.name)
        def side_inner_cell_index(args: self.SideArg, side_index: ElementIndex):
            return self.base.side_inner_cell_index(args.base_arg, side_index)

        return side_inner_cell_index

    def _make_side_outer_cell_index(self):
        @cache.dynamic_func(suffix=self.name)
        def side_outer_cell_index(args: self.SideArg, side_index: ElementIndex):
            return self.base.side_outer_cell_index(args.base_arg, side_index)

        return side_outer_cell_index

    def _make_side_inner_cell_coords(self):
        @cache.dynamic_func(suffix=self.name)
        def side_inner_cell_coords(args: self.SideArg, side_index: ElementIndex, side_coords: Coords):
            return self.base.side_inner_cell_coords(args.base_arg, side_index, side_coords)

        return side_inner_cell_coords

    def _make_side_outer_cell_coords(self):
        @cache.dynamic_func(suffix=self.name)
        def side_outer_cell_coords(args: self.SideArg, side_index: ElementIndex, side_coords: Coords):
            return self.base.side_outer_cell_coords(args.base_arg, side_index, side_coords)

        return side_outer_cell_coords

    def _make_side_from_cell_coords(self):
        @cache.dynamic_func(suffix=self.name)
        def side_from_cell_coords(
            args: self.SideArg,
            side_index: ElementIndex,
            cell_index: ElementIndex,
            cell_coords: Coords,
        ):
            return self.base.side_from_cell_coords(args.base_arg, side_index, cell_index, cell_coords)

        return side_from_cell_coords

    def _make_side_to_cell_arg(self):
        @cache.dynamic_func(suffix=self.name)
        def side_to_cell_arg(side_arg: self.SideArg):
            return self.CellArg(self.base.side_to_cell_arg(side_arg.base_arg), side_arg.field_arg, side_arg.cell_bvh)

        return side_to_cell_arg

    def _make_cell_bvh_id(self):
        @cache.dynamic_func(suffix=self.name)
        def cell_bvh_id(cell_arg: self.CellArg):
            return cell_arg.cell_bvh

        return cell_bvh_id

    def _make_cell_bounds(self):
        points, _weights = self.reference_cell().instantiate_quadrature(
            order=self.field.degree, family=Polynomial.LOBATTO_GAUSS_LEGENDRE
        )

        points = cache.cached_mat_type((len(points), 3), dtype=float)(points)
        point_count = len(points)

        @cache.dynamic_func(suffix=self.name)
        def cell_bounds(cell_arg: self.CellArg, cell_index: ElementIndex):
            lower = wp.vec3(1.0e8)
            upper = wp.vec3(-1.0e8)
            for k in range(point_count):
                pos = self.cell_position(cell_arg, make_free_sample(cell_index, points[k]))
                lower = wp.min(lower, pos)
                upper = wp.max(upper, pos)

            # pad the BBox to account for potential overflows
            pad = 0.25 * (upper - lower)

            return lower - pad, upper + pad

        return cell_bounds
