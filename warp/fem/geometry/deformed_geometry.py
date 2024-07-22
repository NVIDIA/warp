from typing import Any

import warp as wp
from warp.fem import cache
from warp.fem.types import Coords, ElementIndex, Sample, make_free_sample

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
        self.base = self.field.geometry
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
        self.cell_inverse_deformation_gradient = self._make_cell_inverse_deformation_gradient()
        self.cell_measure = self._make_cell_measure()

        self.boundary_side_index = self.base.boundary_side_index

        self.side_to_cell_arg = self._make_side_to_cell_arg()
        self.side_position = self._make_side_position()
        self.side_deformation_gradient = self._make_side_deformation_gradient()
        self.side_inner_cell_index = self._make_side_inner_cell_index()
        self.side_outer_cell_index = self._make_side_outer_cell_index()
        self.side_inner_cell_coords = self._make_side_inner_cell_coords()
        self.side_outer_cell_coords = self._make_side_outer_cell_coords()
        self.side_from_cell_coords = self._make_side_from_cell_coords()
        self.side_inner_inverse_deformation_gradient = self._make_side_inner_inverse_deformation_gradient()
        self.side_outer_inverse_deformation_gradient = self._make_side_outer_inverse_deformation_gradient()
        self.side_measure = self._make_side_measure()
        self.side_measure_ratio = self._make_side_measure_ratio()
        self.side_normal = self._make_side_normal()

    @property
    def name(self):
        return f"DefGeo_{self.field.name}_{'rel' if self._relative else 'abs'}"

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

    def _make_cell_inverse_deformation_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def cell_inverse_deformation_gradient(cell_arg: self.CellArg, s: Sample):
            return wp.inverse(self.cell_deformation_gradient(cell_arg, s))

        return cell_inverse_deformation_gradient

    def _make_cell_measure(self):
        REF_MEASURE = wp.constant(self.reference_cell().measure())

        @cache.dynamic_func(suffix=self.name)
        def cell_measure(args: self.CellArg, s: Sample):
            return wp.abs(wp.determinant(self.cell_deformation_gradient(args, s))) * REF_MEASURE

        return cell_measure

    @wp.func
    def cell_normal(args: Any, s: Sample):
        return wp.vec2(0.0)

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

    def _make_side_inner_inverse_deformation_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def side_inner_inverse_deformation_gradient(args: self.SideArg, s: Sample):
            cell_index = self.side_inner_cell_index(args, s.element_index)
            cell_coords = self.side_inner_cell_coords(args, s.element_index, s.element_coords)
            cell_arg = self.side_to_cell_arg(args)
            return self.cell_inverse_deformation_gradient(cell_arg, make_free_sample(cell_index, cell_coords))

    def _make_side_outer_inverse_deformation_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def side_outer_inverse_deformation_gradient(args: self.SideArg, s: Sample):
            cell_index = self.side_outer_cell_index(args, s.element_index)
            cell_coords = self.side_outer_cell_coords(args, s.element_index, s.element_coords)
            cell_arg = self.side_to_cell_arg(args)
            return self.cell_inverse_deformation_gradient(cell_arg, make_free_sample(cell_index, cell_coords))

    @wp.func
    def _side_measure(F: wp.vec2):
        return wp.length(F)

    @wp.func
    def _side_measure(F: _mat32):
        Fcross = wp.vec3(
            F[1, 0] * F[2, 1] - F[2, 0] * F[1, 1],
            F[2, 0] * F[0, 1] - F[0, 0] * F[2, 1],
            F[0, 0] * F[1, 1] - F[1, 0] * F[0, 1],
        )
        return wp.length(Fcross)

    @wp.func
    def _side_normal(F: wp.vec2):
        return wp.normalize(wp.vec2(-F[1], F[0]))

    @wp.func
    def _side_normal(F: _mat32):
        Fcross = wp.vec3(
            F[1, 0] * F[2, 1] - F[2, 0] * F[1, 1],
            F[2, 0] * F[0, 1] - F[0, 0] * F[2, 1],
            F[0, 0] * F[1, 1] - F[1, 0] * F[0, 1],
        )
        return wp.normalize(Fcross)

    def _make_side_measure(self):
        REF_MEASURE = wp.constant(self.reference_side().measure())

        @cache.dynamic_func(suffix=self.name)
        def side_measure(args: self.SideArg, s: Sample):
            F = self.side_deformation_gradient(args, s)
            return DeformedGeometry._side_measure(F) * REF_MEASURE

        return side_measure

    def _make_side_measure_ratio(self):
        @cache.dynamic_func(suffix=self.name)
        def side_measure_ratio(args: self.SideArg, s: Sample):
            inner = self.side_inner_cell_index(args, s.element_index)
            outer = self.side_outer_cell_index(args, s.element_index)
            inner_coords = self.side_inner_cell_coords(args, s.element_index, s.element_coords)
            outer_coords = self.side_outer_cell_coords(args, s.element_index, s.element_coords)
            cell_arg = self.side_to_cell_arg(args)
            return self.side_measure(args, s) / wp.min(
                self.cell_measure(cell_arg, make_free_sample(inner, inner_coords)),
                self.cell_measure(cell_arg, make_free_sample(outer, outer_coords)),
            )

        return side_measure_ratio

    def _make_side_normal(self):
        @cache.dynamic_func(suffix=self.name)
        def side_normal(args: self.SideArg, s: Sample):
            F = self.side_deformation_gradient(args, s)
            return DeformedGeometry._side_normal(F)

        return side_normal

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
