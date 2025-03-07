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

import warp as wp
from warp.fem import cache
from warp.fem.types import Coords, ElementIndex, Sample

from .geometry import Geometry

_mat32 = wp.mat(shape=(3, 2), dtype=float)


class DeformedGeometry(Geometry):
    def __init__(self, field: "wp.fem.field.GeometryField", relative: bool = True):
        """Constructs a Deformed Geometry from a displacement or absolute position field defined over a base geometry.
        The deformation field does not need to be isoparameteric.

        See also: :meth:`warp.fem.DiscreteField.make_deformed_geometry`
        """

        from warp.fem.field import DiscreteField, GeometryField

        if isinstance(field, DiscreteField):
            if (
                not wp.types.type_is_vector(field.dtype)
                or wp.types.type_length(field.dtype) != field.geometry.dimension
            ):
                raise ValueError(
                    "Invalid value type for position field, must be vector-valued with same dimension as underlying geometry"
                )
        if field.eval_grad_inner is None:
            raise ValueError("Gradient evaluation is not supported on the passed field")

        self._relative = relative

        self.field: GeometryField = field
        self.dimension = self.base.dimension

        self.CellArg = self.field.ElementEvalArg

        self.field_trace = field.trace()
        self.SideArg = self._make_side_arg()
        self.SideIndexArg = self.base.SideIndexArg

        self.cell_count = self.base.cell_count
        self.vertex_count = self.base.vertex_count
        self.side_count = self.base.side_count
        self.boundary_side_count = self.base.boundary_side_count
        self.reference_cell = self.base.reference_cell
        self.reference_side = self.base.reference_side

        self.side_index_arg_value = self.base.side_index_arg_value

        self.cell_position = self._make_cell_position()
        self.cell_deformation_gradient = self._make_cell_deformation_gradient()

        self.boundary_side_index = self.base.boundary_side_index

        self.side_to_cell_arg = self._make_side_to_cell_arg()
        self.side_position = self._make_side_position()
        self.side_deformation_gradient = self._make_side_deformation_gradient()
        self.side_inner_cell_index = self._make_side_inner_cell_index()
        self.side_outer_cell_index = self._make_side_outer_cell_index()
        self.side_inner_cell_coords = self._make_side_inner_cell_coords()
        self.side_outer_cell_coords = self._make_side_outer_cell_coords()
        self.side_from_cell_coords = self._make_side_from_cell_coords()

        self._make_default_dependent_implementations()

    @property
    def name(self) -> str:
        return f"DefGeo_{self.field.name}_{'rel' if self._relative else 'abs'}"

    @property
    def base(self) -> Geometry:
        return self.field.geometry.base

    # Geometry device interface

    @cache.cached_arg_value
    def cell_arg_value(self, device) -> "DeformedGeometry.CellArg":
        args = self.CellArg()

        args.elt_arg = self.base.cell_arg_value(device)
        args.eval_arg = self.field.eval_arg_value(device)

        return args

    def _make_cell_position(self):
        @cache.dynamic_func(suffix=self.name)
        def cell_position_absolute(cell_arg: self.CellArg, s: Sample):
            return self.field.eval_inner(cell_arg, s)

        @cache.dynamic_func(suffix=self.name)
        def cell_position(cell_arg: self.CellArg, s: Sample):
            return self.field.eval_inner(cell_arg, s) + self.base.cell_position(cell_arg.elt_arg, s)

        return cell_position if self._relative else cell_position_absolute

    def _make_cell_deformation_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def cell_deformation_gradient_absolute(cell_arg: self.CellArg, s: Sample):
            return self.field.eval_reference_grad_inner(cell_arg, s)

        @cache.dynamic_func(suffix=self.name)
        def cell_deformation_gradient(cell_arg: self.CellArg, s: Sample):
            return self.field.eval_reference_grad_inner(cell_arg, s) + self.base.cell_deformation_gradient(
                cell_arg.elt_arg, s
            )

        return cell_deformation_gradient if self._relative else cell_deformation_gradient_absolute

    def _make_side_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class SideArg:
            base_arg: self.base.SideArg
            trace_arg: self.field_trace.EvalArg
            field_arg: self.field.EvalArg

        return SideArg

    @cache.cached_arg_value
    def side_arg_value(self, device) -> "DeformedGeometry.SideArg":
        args = self.SideArg()

        args.base_arg = self.base.side_arg_value(device)
        args.field_arg = self.field.eval_arg_value(device)
        args.trace_arg = self.field_trace.eval_arg_value(device)

        return args

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
            return self.CellArg(self.base.side_to_cell_arg(side_arg.base_arg), side_arg.field_arg)

        return side_to_cell_arg
