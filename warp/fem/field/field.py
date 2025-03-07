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

from typing import Any, Dict, Optional, Set

import warp as wp
from warp.fem import cache
from warp.fem.domain import GeometryDomain, Sides
from warp.fem.geometry import DeformedGeometry, Geometry
from warp.fem.operator import Operator, integrand
from warp.fem.space import FunctionSpace, SpacePartition
from warp.fem.types import NULL_ELEMENT_INDEX, ElementKind, Sample


class FieldLike:
    """Base class for integrable fields"""

    EvalArg: wp.codegen.Struct
    """Structure containing field-level arguments passed to device functions for field evaluation"""

    ElementEvalArg: wp.codegen.Struct
    """Structure combining geometry-level and field-level arguments passed to device functions for field evaluation"""

    def eval_arg_value(self, device) -> "EvalArg":  # noqa: F821
        """Value of the field-level arguments to be passed to device functions"""
        raise NotImplementedError

    @property
    def degree(self) -> int:
        """Polynomial degree of the field, used to estimate necessary quadrature order"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def __str__(self) -> str:
        return self.name

    def eval_arg_value(self, device):
        """Value of arguments to be passed to device functions"""
        raise NotImplementedError

    def gradient_valid(self) -> bool:
        """Whether the gradient operator is implemented for this field."""
        return False

    def divergence_valid(self) -> bool:
        """Whether the divergence operator is implemented for this field."""
        return False

    @staticmethod
    def eval_inner(args: "ElementEvalArg", s: Sample):  # noqa: F821
        """Device function evaluating the inner field value at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_grad_inner(args: "ElementEvalArg", s: Sample):  # noqa: F821
        """Device function evaluating the inner field gradient at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_div_inner(args: "ElementEvalArg", s: Sample):  # noqa: F821
        """Device function evaluating the inner field divergence at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_outer(args: "ElementEvalArg", s: Sample):  # noqa: F821
        """Device function evaluating the outer field value at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_grad_outer(args: "ElementEvalArg", s: Sample):  # noqa: F821
        """Device function evaluating the outer field gradient at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_div_outer(args: "ElementEvalArg", s: Sample):  # noqa: F821
        """Device function evaluating the outer field divergence at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_degree(args: "ElementEvalArg"):  # noqa: F821
        """Polynomial degree of the field is applicable, or hint for determination of interpolation order"""
        raise NotImplementedError

    def notify_operator_usage(self, ops: Set[Operator]):
        """Makes the Domain aware that the operators `ops` will be applied"""
        pass


class GeometryField(FieldLike):
    """Base class for fields defined over a geometry"""

    @property
    def geometry(self) -> Geometry:
        """Geometry over which the field is expressed"""
        raise NotImplementedError

    @property
    def element_kind(self) -> ElementKind:
        """Kind of element over which the field is expressed"""
        raise NotImplementedError

    @staticmethod
    def eval_reference_grad_inner(args: "ElementEvalArg", s: Sample):  # noqa: F821
        """Device function evaluating the inner field gradient with respect to reference element coordinates at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_reference_grad_outer(args: "ElementEvalArg", s: Sample):  # noqa: F821
        """Device function evaluating the outer field gradient with respect to reference element coordinates at a sample point"""
        raise NotImplementedError

    def trace(self) -> FieldLike:
        """Trace of this field over lower-dimensional elements"""
        raise NotImplementedError

    def make_deformed_geometry(self, relative=True) -> Geometry:
        """Returns a deformed version of the underlying geometry, with positions displaced according to this field's values.

        Args:
            relative: If ``True``, the field is interpreted as a relative displacement over the original geometry.
              If ``False``, the field values are interpreted as absolute positions.

        """
        return DeformedGeometry(self, relative=relative)


class SpaceField(GeometryField):
    """Base class for fields defined over a function space"""

    def __init__(self, space: FunctionSpace, space_partition: SpacePartition):
        self._space = space
        self._space_partition = space_partition

        self.gradient_valid = self.space.gradient_valid
        self.divergence_valid = self.space.divergence_valid

    @property
    def geometry(self) -> Geometry:
        return self._space.geometry

    @property
    def element_kind(self) -> ElementKind:
        return self._space.element_kind

    @property
    def space(self) -> FunctionSpace:
        return self._space

    @property
    def space_partition(self) -> SpacePartition:
        return self._space_partition

    @property
    def degree(self) -> int:
        return self.space.degree

    @property
    def dtype(self) -> type:
        return self.space.dtype

    @property
    def dof_dtype(self) -> type:
        return self.space.dof_dtype

    @property
    def gradient_dtype(self):
        """Return type of the gradient operator. Assumes self.gradient_valid()"""
        if wp.types.type_is_vector(self.dtype):
            return cache.cached_mat_type(
                shape=(wp.types.type_length(self.dtype), self.geometry.dimension),
                dtype=wp.types.type_scalar_type(self.dtype),
            )
        return cache.cached_vec_type(length=self.geometry.dimension, dtype=wp.types.type_scalar_type(self.dtype))

    @property
    def divergence_dtype(self):
        """Return type of the divergence operator. Assumes self.gradient_valid()"""
        if wp.types.type_is_vector(self.dtype):
            return wp.types.type_scalar_type(self.dtype)
        return cache.cached_vec_type(length=self.dtype._shape_[1], dtype=wp.types.type_scalar_type(self.dtype))

    def _make_eval_degree(self):
        ORDER = self.space.ORDER

        @cache.dynamic_func(suffix=self.name)
        def degree(args: self.ElementEvalArg):
            return ORDER

        return degree


class DiscreteField(SpaceField):
    """Explicitly-valued field defined over a partition of a discrete function space"""

    @property
    def dof_values(self) -> wp.array:
        """Array of degrees of freedom values"""
        raise NotImplementedError

    @dof_values.setter
    def dof_values(self, values: wp.array):
        """Sets degrees of freedom values from an array"""
        raise NotImplementedError

    @staticmethod
    def set_node_value(args: "FieldLike.EvalArg", node_index: int, value: Any):
        """Device function setting the value at given node"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return f"{self.__class__.__qualname__}_{self.space.name}_{self.space_partition.name}"


class ImplicitField(GeometryField):
    """Field defined from an arbitrary function over a domain.
    Does not support autodiff yet, so if gradient/divergence evaluation is required corresponding functions must be provided.

    Args:
        domain: Domain over which the field is defined
        func: Warp function evaluating the field at a given position. Must accept at least one argument, with the first argument being the evaluation position (``wp.vec2`` or ``wp.vec3``).
        values: Optional dictionary of additional argument values to be passed to the evaluation function.
        grad_func: Optional gradient evaluation function; must take same arguments as `func`
        div_func: Optional divergence evaluation function; must take same arguments as `func`
        degree: Optional hint for automatic determination of quadrature orders when integrating this field
    """

    def __init__(
        self,
        domain: GeometryDomain,
        func: wp.Function,
        values: Optional[Dict[str, Any]] = None,
        grad_func: Optional[wp.Function] = None,
        div_func: Optional[wp.Function] = None,
        degree=0,
    ):
        self.domain = domain
        self._degree = degree

        if not isinstance(func, wp.Function):
            raise ValueError("Implicit field function must be a warp Function (decorated with `wp.func`)")

        self._func = func
        self._grad_func = grad_func
        self._div_func = div_func

        argspec = integrand(func.func).argspec
        arg_types = argspec.annotations

        pos_arg_type = arg_types.pop(argspec.args[0]) if arg_types else None
        if not pos_arg_type or not wp.types.types_equal(
            pos_arg_type, wp.vec(length=domain.geometry.dimension, dtype=float), match_generic=True
        ):
            raise ValueError(
                f"Implicit field function '{func.func.__name__}' must accept a position as its first argument"
            )

        self.EvalArg = cache.get_argument_struct(arg_types)
        self.values = values

        self.ElementEvalArg = self._make_element_eval_arg()
        self.eval_degree = self._make_eval_degree()

        self.eval_inner = self._make_eval_func(func)
        self.eval_grad_inner = self._make_eval_func(grad_func)
        self.eval_div_inner = self._make_eval_func(div_func)
        self.eval_reference_grad_inner = self._make_eval_reference_grad()

        self.eval_outer = self.eval_inner
        self.eval_grad_outer = self.eval_grad_inner
        self.eval_div_outer = self.eval_div_inner
        self.eval_reference_grad_outer = self.eval_reference_grad_inner

    @property
    def values(self):
        return self._func_arg

    @values.setter
    def values(self, v):
        self._func_arg = cache.populate_argument_struct(self.EvalArg, v, self._func.func.__name__)

    @property
    def geometry(self) -> Geometry:
        return self.domain.geometry

    @property
    def element_kind(self) -> ElementKind:
        return self.domain.element_kind

    def eval_arg_value(self, device):
        return self._func_arg

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def name(self) -> str:
        return f"Implicit_{self.domain.name}_{self.degree}_{self.EvalArg.key}"

    def _make_eval_func(self, func):
        if func is None:
            return None

        @cache.dynamic_func(
            suffix=f"{self.name}_{func.key}",
            code_transformers=[cache.ExpandStarredArgumentStruct({"args.eval_arg": self.EvalArg})],
        )
        def eval_inner(args: self.ElementEvalArg, s: Sample):
            pos = self.domain.element_position(args.elt_arg, s)
            return func(pos, *args.eval_arg)

        return eval_inner

    def _make_eval_reference_grad(self):
        if self.eval_grad_inner is None:
            return None

        @cache.dynamic_func(suffix=f"{self.eval_grad_inner.key}")
        def eval_reference_grad_inner(args: self.ElementEvalArg, s: Sample):
            return self.eval_grad_inner(args, s) * self.domain.element_deformation_gradient(args.elt_arg, s)

        return eval_reference_grad_inner

    def _make_element_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class ElementEvalArg:
            elt_arg: self.domain.ElementArg
            eval_arg: self.EvalArg

        return ElementEvalArg

    def _make_eval_degree(self):
        ORDER = wp.constant(self._degree)

        @cache.dynamic_func(suffix=self.name)
        def degree(args: self.ElementEvalArg):
            return ORDER

        return degree

    def trace(self):
        if self.element_kind == ElementKind.SIDE:
            raise RuntimeError("Trace only available for field defined on cell elements")

        return ImplicitField(
            domain=Sides(self.domain.geometry_partition),
            func=self._func,
            values={name: getattr(self.values, name) for name in self.EvalArg.vars},
            grad_func=self._grad_func,
            div_func=self._div_func,
            degree=self._degree,
        )


class UniformField(GeometryField):
    """Field defined as a constant value over a domain.

    Args:
        domain: Domain over which the field is defined
        value: Uniform value over the domain
    """

    def __init__(self, domain: GeometryDomain, value: Any):
        self.domain = domain

        if not wp.types.is_value(value):
            raise ValueError("value must be a Warp scalar, vector or matrix")

        self.dtype = wp.types.type_to_warp(type(value))
        self._value = self.dtype(value)

        scalar_type = wp.types.type_scalar_type(self.dtype)
        if wp.types.type_is_vector(self.dtype):
            grad_type = wp.mat(shape=(wp.types.type_length(self.dtype), self.geometry.dimension), dtype=scalar_type)
            div_type = scalar_type
        elif wp.types.type_is_matrix(self.dtype):
            grad_type = None
            div_type = wp.vec(length=(wp.types.type_length(self.dtype) // self.geometry.dimension), dtype=scalar_type)
        else:
            div_type = None
            grad_type = wp.vec(length=self.geometry.dimension, dtype=scalar_type)

        self.EvalArg = self._make_eval_arg()
        self.ElementEvalArg = self._make_element_eval_arg()
        self.eval_degree = self._make_eval_degree()

        self.eval_inner = self._make_eval_inner()
        self.eval_grad_inner = self._make_eval_zero(grad_type)
        self.eval_div_inner = self._make_eval_zero(div_type)
        self.eval_reference_grad_inner = self.eval_grad_inner

        self.eval_outer = self.eval_inner
        self.eval_grad_outer = self.eval_grad_inner
        self.eval_div_outer = self.eval_div_inner
        self.eval_reference_grad_outer = self.eval_reference_grad_inner

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        value_type = wp.types.type_to_warp(type(v))
        assert wp.types.types_equal(value_type, self.dtype)
        self._value = self.dtype(v)

    @property
    def geometry(self) -> Geometry:
        return self.domain.geometry

    @property
    def element_kind(self) -> ElementKind:
        return self.domain.element_kind

    def eval_arg_value(self, device):
        arg = self.EvalArg()
        arg.value = self.value
        return arg

    @property
    def degree(self) -> int:
        return 0

    @property
    def name(self) -> str:
        return f"Uniform{self.domain.name}_{wp.types.get_type_code(self.dtype)}"

    def _make_eval_inner(self):
        @cache.dynamic_func(suffix=self.name)
        def eval_inner(args: self.ElementEvalArg, s: Sample):
            return args.eval_arg.value

        return eval_inner

    def _make_eval_zero(self, dtype):
        if dtype is None:
            return None

        scalar_type = wp.types.type_scalar_type(dtype)

        @cache.dynamic_func(suffix=f"{self.name}_{wp.types.get_type_code(dtype)}")
        def eval_zero(args: self.ElementEvalArg, s: Sample):
            return dtype(scalar_type(0.0))

        return eval_zero

    def _make_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class EvalArg:
            value: self.dtype

        return EvalArg

    def _make_element_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class ElementEvalArg:
            elt_arg: self.domain.ElementArg
            eval_arg: self.EvalArg

        return ElementEvalArg

    def _make_eval_degree(self):
        @cache.dynamic_func(suffix=self.name)
        def degree(args: self.ElementEvalArg):
            return 0

        return degree

    def trace(self):
        if self.element_kind == ElementKind.SIDE:
            raise RuntimeError("Trace only available for field defined on cell elements")

        return UniformField(domain=Sides(self.domain.geometry_partition), value=self.value)


class NonconformingField(GeometryField):
    """Field defined as the map of a DiscreteField over a non-conforming geometry.

    Args:
        domain: The new domain over which the nonconforming field will be evaluated
        field: Nonconforming discrete field
        background: Uniform value or domain-conforming field determining the value outside of the geometry of definition of `field`
    """

    _LOOKUP_EPS = wp.constant(1.0e-6)

    def __init__(self, domain: GeometryDomain, field: DiscreteField, background: Any = 0.0):
        self.domain = domain

        self.field = field
        self.dtype = field.dtype

        if not isinstance(background, GeometryField):
            background = UniformField(domain, self.dtype(background))
        elif background.geometry != domain.geometry or background.element_kind != domain.element_kind:
            raise ValueError("Background field must be conforming to the domain")
        self.background = background

        self.EvalArg = self._make_eval_arg()
        self.ElementEvalArg = self._make_element_eval_arg()
        self.eval_degree = self._make_eval_degree()

        self.eval_inner = self._make_nonconforming_eval("eval_inner")
        self.eval_grad_inner = self._make_nonconforming_eval("eval_grad_inner")
        self.eval_div_inner = self._make_nonconforming_eval("eval_div_inner")
        self.eval_reference_grad_inner = self._make_eval_reference_grad()

        # Nonconforming evaluation is position based, does not handle discontinuous fields
        self.eval_outer = self.eval_inner
        self.eval_grad_outer = self.eval_grad_inner
        self.eval_div_outer = self.eval_div_inner
        self.eval_reference_grad_outer = self.eval_reference_grad_inner

    @property
    def geometry(self) -> Geometry:
        return self.domain.geometry

    @property
    def element_kind(self) -> ElementKind:
        return self.domain.element_kind

    @cache.cached_arg_value
    def eval_arg_value(self, device):
        arg = self.EvalArg()
        arg.field_cell_eval_arg = self.field.ElementEvalArg()
        arg.field_cell_eval_arg.elt_arg = self.field.geometry.cell_arg_value(device)
        arg.field_cell_eval_arg.eval_arg = self.field.eval_arg_value(device)
        arg.background_arg = self.background.eval_arg_value(device)
        return arg

    @property
    def degree(self) -> int:
        return self.field.degree

    @property
    def name(self) -> str:
        return f"{self.domain.name}_{self.field.name}_{self.background.name}"

    def _make_nonconforming_eval(self, eval_func_name):
        field_eval = getattr(self.field, eval_func_name)
        bg_eval = getattr(self.background, eval_func_name)

        if field_eval is None or bg_eval is None:
            return None

        @cache.dynamic_func(suffix=f"{eval_func_name}_{self.name}")
        def eval_nc(args: self.ElementEvalArg, s: Sample):
            pos = self.domain.element_position(args.elt_arg, s)
            cell_arg = args.eval_arg.field_cell_eval_arg.elt_arg
            nonconforming_s = self.field.geometry.cell_lookup(cell_arg, pos)
            if (
                nonconforming_s.element_index == NULL_ELEMENT_INDEX
                or wp.length_sq(pos - self.field.geometry.cell_position(cell_arg, nonconforming_s))
                > NonconformingField._LOOKUP_EPS
            ):
                return bg_eval(self.background.ElementEvalArg(args.elt_arg, args.eval_arg.background_arg), s)
            return field_eval(
                self.field.ElementEvalArg(cell_arg, args.eval_arg.field_cell_eval_arg.eval_arg), nonconforming_s
            )

        return eval_nc

    def _make_eval_reference_grad(self):
        if self.eval_grad_inner is None:
            return None

        @cache.dynamic_func(suffix=f"{self.eval_grad_inner.key}")
        def eval_reference_grad_inner(args: self.ElementEvalArg, s: Sample):
            return self.eval_grad_inner(args, s) * self.domain.element_deformation_gradient(args.elt_arg, s)

        return eval_reference_grad_inner

    def _make_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class EvalArg:
            field_cell_eval_arg: self.field.ElementEvalArg
            background_arg: self.background.EvalArg

        return EvalArg

    def _make_element_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class ElementEvalArg:
            elt_arg: self.domain.ElementArg
            eval_arg: self.EvalArg

        return ElementEvalArg

    def _make_eval_degree(self):
        @cache.dynamic_func(suffix=self.name)
        def degree(args: self.ElementEvalArg):
            return self.field.eval_degree(args.eval_arg.field_cell_eval_arg)

        return degree

    def trace(self):
        if self.element_kind == ElementKind.SIDE:
            raise RuntimeError("Trace only available for field defined on cell elements")

        return NonconformingField(
            domain=Sides(self.domain.geometry_partition), field=self.field, background=self.background.trace()
        )
