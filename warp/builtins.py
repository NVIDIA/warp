# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import builtins
import functools
import math
from typing import Any, Callable, Mapping, Sequence

import warp.build
import warp.context
from warp.codegen import Reference, Var, get_arg_value, strip_reference
from warp.types import *

from .context import add_builtin


def seq_check_equal(seq_1, seq_2):
    if not isinstance(seq_1, Sequence) or not isinstance(seq_2, Sequence):
        return False

    if len(seq_1) != len(seq_2):
        return False

    return all(x == y for x, y in zip(seq_1, seq_2))


def sametypes(arg_types: Mapping[str, Any]):
    arg_types_iter = iter(arg_types.values())
    arg_type_0 = next(arg_types_iter)
    return all(types_equal(arg_type_0, t) for t in arg_types_iter)


def sametypes_create_value_func(default: TypeVar):
    def fn(arg_types, arg_values):
        if arg_types is None:
            return default

        if not sametypes(arg_types):
            raise RuntimeError(f"Input types must be the same, got {[type_repr(t) for t in arg_types.values()]}")

        arg_type_0 = next(iter(arg_types.values()))
        return arg_type_0

    return fn


def extract_tuple(arg, as_constant=False):
    if isinstance(arg, Var):
        if isinstance(arg.type, warp.types.tuple_t):
            out = arg.type.values
        else:
            out = (arg,)
    elif isinstance(arg, warp.types.tuple_t):
        out = arg.values
    elif not isinstance(arg, Sequence):
        out = (arg,)
    else:
        out = arg

    if as_constant:
        return tuple(x.constant if isinstance(x, Var) else x for x in out)

    return out


def static_len_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return int

    length = warp.types.type_length(arg_types["a"])
    return Var(None, type=int, constant=length)


# ---------------------------------
# Scalar Math

add_builtin(
    "min",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Return the minimum of two scalars.",
    group="Scalar Math",
)

add_builtin(
    "max",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Return the maximum of two scalars.",
    group="Scalar Math",
)

add_builtin(
    "clamp",
    input_types={"x": Scalar, "low": Scalar, "high": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Clamp the value of ``x`` to the range [low, high].",
    group="Scalar Math",
)

add_builtin(
    "abs",
    input_types={"x": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Return the absolute value of ``x``.",
    group="Scalar Math",
)
add_builtin(
    "sign",
    input_types={"x": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Return -1 if ``x`` < 0, return 1 otherwise.",
    group="Scalar Math",
)

add_builtin(
    "step",
    input_types={"x": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Return 1.0 if ``x`` < 0.0, return 0.0 otherwise.",
    group="Scalar Math",
)
add_builtin(
    "nonzero",
    input_types={"x": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Return 1.0 if ``x`` is not equal to zero, return 0.0 otherwise.",
    group="Scalar Math",
)

add_builtin(
    "sin",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the sine of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "cos",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the cosine of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "acos",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return arccos of ``x`` in radians. Inputs are automatically clamped to [-1.0, 1.0].",
    group="Scalar Math",
)
add_builtin(
    "asin",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return arcsin of ``x`` in radians. Inputs are automatically clamped to [-1.0, 1.0].",
    group="Scalar Math",
)
add_builtin(
    "sqrt",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the square root of ``x``, where ``x`` is positive.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "cbrt",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the cube root of ``x``.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "tan",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the tangent of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "atan",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the arctangent of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "atan2",
    input_types={"y": Float, "x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the 2-argument arctangent, atan2, of the point ``(x, y)`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "sinh",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the sinh of ``x``.",
    group="Scalar Math",
)
add_builtin(
    "cosh",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the cosh of ``x``.",
    group="Scalar Math",
)
add_builtin(
    "tanh",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the tanh of ``x``.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "degrees",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Convert ``x`` from radians into degrees.",
    group="Scalar Math",
)
add_builtin(
    "radians",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Convert ``x`` from degrees into radians.",
    group="Scalar Math",
)

add_builtin(
    "log",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the natural logarithm (base-e) of ``x``, where ``x`` is positive.",
    group="Scalar Math",
)
add_builtin(
    "log2",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the binary logarithm (base-2) of ``x``, where ``x`` is positive.",
    group="Scalar Math",
)
add_builtin(
    "log10",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the common logarithm (base-10) of ``x``, where ``x`` is positive.",
    group="Scalar Math",
)
add_builtin(
    "exp",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the value of the exponential function :math:`e^x`.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "pow",
    input_types={"x": Float, "y": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Return the result of ``x`` raised to power of ``y``.",
    group="Scalar Math",
    require_original_output_arg=True,
)

add_builtin(
    "round",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Return the nearest integer value to ``x``, rounding halfway cases away from zero.

    This is the most intuitive form of rounding in the colloquial sense, but can be slower than other options like :func:`warp.rint()`.
    Differs from :func:`numpy.round()`, which behaves the same way as :func:`numpy.rint()`.""",
)

add_builtin(
    "rint",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Return the nearest integer value to ``x``, rounding halfway cases to nearest even integer.

    It is generally faster than :func:`warp.round()`. Equivalent to :func:`numpy.rint()`.""",
)

add_builtin(
    "trunc",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Return the nearest integer that is closer to zero than ``x``.

    In other words, it discards the fractional part of ``x``.
    It is similar to casting ``float(int(a))``, but preserves the negative sign when ``x`` is in the range [-0.0, -1.0).
    Equivalent to :func:`numpy.trunc()` and :func:`numpy.fix()`.""",
)

add_builtin(
    "floor",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Return the largest integer that is less than or equal to ``x``.""",
)

add_builtin(
    "ceil",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Return the smallest integer that is greater than or equal to ``x``.""",
)

add_builtin(
    "frac",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Retrieve the fractional part of ``x``.

    In other words, it discards the integer part of ``x`` and is equivalent to ``x - trunc(x)``.""",
)

add_builtin(
    "isfinite",
    input_types={"a": Scalar},
    value_type=builtins.bool,
    group="Scalar Math",
    doc="""Return ``True`` if ``a`` is a finite number, otherwise return ``False``.""",
)
add_builtin(
    "isfinite",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Return ``True`` if all elements of the vector ``a`` are finite, otherwise return ``False``.",
)
add_builtin(
    "isfinite",
    input_types={"a": quaternion(dtype=Scalar)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Return ``True`` if all elements of the quaternion ``a`` are finite, otherwise return ``False``.",
)
add_builtin(
    "isfinite",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Return ``True`` if all elements of the matrix ``a`` are finite, otherwise return ``False``.",
)

add_builtin(
    "isnan",
    input_types={"a": Scalar},
    value_type=builtins.bool,
    doc="Return ``True`` if ``a`` is NaN, otherwise return ``False``.",
    group="Scalar Math",
)
add_builtin(
    "isnan",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Return ``True`` if any element of the vector ``a`` is NaN, otherwise return ``False``.",
)
add_builtin(
    "isnan",
    input_types={"a": quaternion(dtype=Scalar)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Return ``True`` if any element of the quaternion ``a`` is NaN, otherwise return ``False``.",
)
add_builtin(
    "isnan",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Return ``True`` if any element of the matrix ``a`` is NaN, otherwise return ``False``.",
)

add_builtin(
    "isinf",
    input_types={"a": Scalar},
    value_type=builtins.bool,
    group="Scalar Math",
    doc="""Return ``True`` if ``a`` is positive or negative infinity, otherwise return ``False``.""",
)
add_builtin(
    "isinf",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Return ``True`` if any element of the vector ``a`` is positive or negative infinity, otherwise return ``False``.",
)
add_builtin(
    "isinf",
    input_types={"a": quaternion(dtype=Scalar)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Return ``True`` if any element of the quaternion ``a`` is positive or negative infinity, otherwise return ``False``.",
)
add_builtin(
    "isinf",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Return ``True`` if any element of the matrix ``a`` is positive or negative infinity, otherwise return ``False``.",
)


def scalar_infer_type(arg_types: Mapping[str, type] | tuple[type, ...] | None):
    if arg_types is None:
        return Scalar

    if isinstance(arg_types, Mapping):
        arg_types = tuple(arg_types.values())

    scalar_types = set()
    for t in arg_types:
        t_val = strip_reference(t)
        if hasattr(t_val, "_wp_scalar_type_"):
            scalar_types.add(t_val._wp_scalar_type_)
        elif t_val in scalar_and_bool_types:
            scalar_types.add(t_val)

    if len(scalar_types) > 1:
        raise RuntimeError(
            f"Couldn't figure out return type as arguments have multiple precisions: {list(scalar_types)}"
        )
    return next(iter(scalar_types))


def scalar_sametypes_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Scalar

    if not sametypes(arg_types):
        raise RuntimeError(f"Input types must be exactly the same, got {[type_repr(t) for t in arg_types.values()]}")

    return scalar_infer_type(arg_types)


def float_infer_type(arg_types: Mapping[str, type]):
    if arg_types is None:
        return Float

    return scalar_infer_type(arg_types)


def float_sametypes_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Float

    return scalar_sametypes_value_func(arg_types, arg_values)


# ---------------------------------
# Vector Math

add_builtin(
    "dot",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=scalar_sametypes_value_func,
    group="Vector Math",
    doc="Compute the dot product between two vectors.",
)
add_builtin(
    "ddot",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=scalar_sametypes_value_func,
    group="Vector Math",
    doc="Compute the double dot product between two matrices.",
)

add_builtin(
    "min",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="Return the element-wise minimum of two vectors.",
    group="Vector Math",
)
add_builtin(
    "max",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="Return the element-wise maximum of two vectors.",
    group="Vector Math",
)

add_builtin(
    "min",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_func=scalar_sametypes_value_func,
    doc="Return the minimum element of a vector ``a``.",
    group="Vector Math",
)
add_builtin(
    "max",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_func=scalar_sametypes_value_func,
    doc="Return the maximum element of a vector ``a``.",
    group="Vector Math",
)

add_builtin(
    "argmin",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_func=lambda arg_types, arg_values: warp.uint32,
    doc="Return the index of the minimum element of a vector ``a``.",
    group="Vector Math",
    missing_grad=True,
)
add_builtin(
    "argmax",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_func=lambda arg_types, arg_values: warp.uint32,
    doc="Return the index of the maximum element of a vector ``a``.",
    group="Vector Math",
    missing_grad=True,
)

add_builtin(
    "abs",
    input_types={"x": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="Return the absolute values of the elements of ``x``.",
    group="Vector Math",
)

add_builtin(
    "sign",
    input_types={"x": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(Scalar),
    doc="Return -1 for the negative elements of ``x``, and 1 otherwise.",
    group="Vector Math",
)


def outer_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return matrix(shape=(Any, Any), dtype=Scalar)

    arg_type_values = tuple(arg_types.values())
    scalarType = scalar_infer_type(arg_type_values)
    vectorLengths = tuple(t._length_ for t in arg_type_values)
    return matrix(shape=(vectorLengths), dtype=scalarType)


add_builtin(
    "outer",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    value_func=outer_value_func,
    group="Vector Math",
    doc="Compute the outer product ``a*b^T`` for two vectors.",
)

add_builtin(
    "cross",
    input_types={"a": vector(length=3, dtype=Scalar), "b": vector(length=3, dtype=Scalar)},
    value_func=sametypes_create_value_func(vector(length=3, dtype=Scalar)),
    group="Vector Math",
    doc="Compute the cross product of two 3D vectors.",
)
add_builtin(
    "skew",
    input_types={"vec": vector(length=3, dtype=Scalar)},
    value_func=lambda arg_types, arg_values: matrix(shape=(3, 3), dtype=Scalar)
    if arg_types is None
    else matrix(shape=(3, 3), dtype=arg_types["vec"]._wp_scalar_type_),
    group="Vector Math",
    doc="Compute the skew-symmetric 3x3 matrix for a 3D vector ``vec``.",
)

add_builtin(
    "length",
    input_types={"a": vector(length=Any, dtype=Float)},
    value_func=float_sametypes_value_func,
    group="Vector Math",
    doc="Compute the length of a floating-point vector ``a``.",
    require_original_output_arg=True,
)
add_builtin(
    "length",
    input_types={"a": quaternion(dtype=Float)},
    value_func=float_sametypes_value_func,
    group="Vector Math",
    doc="Compute the length of a quaternion ``a``.",
    require_original_output_arg=True,
)
add_builtin(
    "length_sq",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_func=scalar_sametypes_value_func,
    group="Vector Math",
    doc="Compute the squared length of a vector ``a``.",
)
add_builtin(
    "length_sq",
    input_types={"a": quaternion(dtype=Scalar)},
    value_func=scalar_sametypes_value_func,
    group="Vector Math",
    doc="Compute the squared length of a quaternion ``a``.",
)
add_builtin(
    "normalize",
    input_types={"a": vector(length=Any, dtype=Float)},
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Float)),
    group="Vector Math",
    doc="Compute the normalized value of ``a``. If ``length(a)`` is 0 then the zero vector is returned.",
    require_original_output_arg=True,
)
add_builtin(
    "normalize",
    input_types={"a": quaternion(dtype=Float)},
    value_func=sametypes_create_value_func(quaternion(dtype=Float)),
    group="Vector Math",
    doc="Compute the normalized value of ``a``. If ``length(a)`` is 0, then the zero quaternion is returned.",
)

add_builtin(
    "transpose",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=lambda arg_types, arg_values: matrix(shape=(Any, Any), dtype=Scalar)
    if arg_types is None
    else matrix(shape=(arg_types["a"]._shape_[1], arg_types["a"]._shape_[0]), dtype=arg_types["a"]._wp_scalar_type_),
    group="Vector Math",
    doc="Return the transpose of the matrix ``a``.",
)


def inverse_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return matrix(shape=(Any, Any), dtype=Float)

    return arg_types["a"]


add_builtin(
    "inverse",
    input_types={"a": matrix(shape=(2, 2), dtype=Float)},
    value_func=inverse_value_func,
    group="Vector Math",
    doc="Return the inverse of a 2x2 matrix ``a``.",
    require_original_output_arg=True,
)

add_builtin(
    "inverse",
    input_types={"a": matrix(shape=(3, 3), dtype=Float)},
    value_func=inverse_value_func,
    group="Vector Math",
    doc="Return the inverse of a 3x3 matrix ``a``.",
    require_original_output_arg=True,
)

add_builtin(
    "inverse",
    input_types={"a": matrix(shape=(4, 4), dtype=Float)},
    value_func=inverse_value_func,
    group="Vector Math",
    doc="Return the inverse of a 4x4 matrix ``a``.",
    require_original_output_arg=True,
)


def determinant_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Float

    return arg_types["a"]._wp_scalar_type_


add_builtin(
    "determinant",
    input_types={"a": matrix(shape=(2, 2), dtype=Float)},
    value_func=determinant_value_func,
    group="Vector Math",
    doc="Return the determinant of a 2x2 matrix ``a``.",
)

add_builtin(
    "determinant",
    input_types={"a": matrix(shape=(3, 3), dtype=Float)},
    value_func=determinant_value_func,
    group="Vector Math",
    doc="Return the determinant of a 3x3 matrix ``a``.",
)

add_builtin(
    "determinant",
    input_types={"a": matrix(shape=(4, 4), dtype=Float)},
    value_func=determinant_value_func,
    group="Vector Math",
    doc="Return the determinant of a 4x4 matrix ``a``.",
)


def trace_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Scalar

    if arg_types["a"]._shape_[0] != arg_types["a"]._shape_[1]:
        raise RuntimeError(f"Matrix shape is {arg_types['a']._shape_}. Cannot find the trace of non square matrices")
    return arg_types["a"]._wp_scalar_type_


add_builtin(
    "trace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=trace_value_func,
    group="Vector Math",
    doc="Return the trace of the matrix ``a``.",
)


def diag_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return matrix(shape=(Any, Any), dtype=Scalar)

    return matrix(shape=(arg_types["vec"]._length_, arg_types["vec"]._length_), dtype=arg_types["vec"]._wp_scalar_type_)


add_builtin(
    "diag",
    input_types={"vec": vector(length=Any, dtype=Scalar)},
    value_func=diag_value_func,
    group="Vector Math",
    doc="Returns a matrix with the components of the vector ``vec`` on the diagonal.",
)


def get_diag_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return vector(length=(Any), dtype=Scalar)

    if arg_types["mat"]._shape_[0] != arg_types["mat"]._shape_[1]:
        raise RuntimeError(
            f"Matrix shape is {arg_types['mat']._shape_}; get_diag is only available for square matrices."
        )
    return vector(length=arg_types["mat"]._shape_[0], dtype=arg_types["mat"]._wp_scalar_type_)


add_builtin(
    "get_diag",
    input_types={"mat": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=get_diag_value_func,
    group="Vector Math",
    doc="Returns a vector containing the diagonal elements of the square matrix ``mat``.",
)

add_builtin(
    "cw_mul",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    group="Vector Math",
    doc="Component-wise multiplication of two vectors.",
)
add_builtin(
    "cw_div",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    group="Vector Math",
    doc="Component-wise division of two vectors.",
    require_original_output_arg=True,
)

add_builtin(
    "cw_mul",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    group="Vector Math",
    doc="Component-wise multiplication of two matrices.",
)
add_builtin(
    "cw_div",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    group="Vector Math",
    doc="Component-wise division of two matrices.",
    require_original_output_arg=True,
)


# scalar type constructors between all storage / compute types
scalar_types_all = [*scalar_types, bool, int, float]
for t in scalar_types_all:
    for u in scalar_types_all:
        add_builtin(
            t.__name__,
            input_types={"a": u},
            value_type=t,
            doc="",
            hidden=True,
            group="Scalar Math",
            export=False,
            namespace="wp::" if t is not bool else "",
        )


def vector_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return vector(length=Any, dtype=Scalar)

    length = arg_values.get("length", None)
    dtype = arg_values.get("dtype", None)

    variadic_arg_types = arg_types.get("args", ())
    variadic_arg_count = len(variadic_arg_types)
    if variadic_arg_count == 0:
        # Zero-initialization, e.g.: `wp.vecXX()`, `wp.vector(length=2, dtype=wp.float16)`.
        if length is None:
            raise RuntimeError("the `length` argument must be specified when zero-initializing a vector")

        if dtype is None:
            dtype = float32
    elif variadic_arg_count == 1:
        value_type = strip_reference(variadic_arg_types[0])
        if type_is_vector(value_type):
            # Copy constructor, e.g.: `wp.vecXX(other_vec)`, `wp.vector(other_vec)`.
            if length is None:
                length = value_type._length_
            elif value_type._length_ != length:
                raise RuntimeError(
                    f"incompatible vector of length {length} given when copy constructing "
                    f"a vector of length {value_type._length_}"
                )

            if dtype is None:
                dtype = value_type._wp_scalar_type_
        else:
            # Initialization by filling a value, e.g.: `wp.vecXX(123)`,
            # `wp.vector(123, length=2)`.
            if length is None:
                raise RuntimeError("the `length` argument must be specified when filling a vector with a value")

            if dtype is None:
                dtype = value_type
            elif not warp.types.scalars_equal(value_type, dtype):
                raise RuntimeError(
                    f"the value used to fill this vector is expected to be of the type `{dtype.__name__}`"
                )
    else:
        # Initializing by value, e.g.: `wp.vec2(1, 2)`, `wp.vector(1, 2, length=2)`.
        if length is None:
            length = variadic_arg_count
        elif length != variadic_arg_count:
            raise RuntimeError(
                f"incompatible number of values given ({variadic_arg_count}) "
                f"when constructing a vector of length {length}"
            )

        try:
            value_type = scalar_infer_type(variadic_arg_types)
        except RuntimeError:
            raise RuntimeError("all values given when constructing a vector must have the same type") from None

        if dtype is None:
            dtype = value_type
        elif not warp.types.scalars_equal(value_type, dtype):
            raise RuntimeError(
                f"all values used to initialize this vector are expected to be of the type `{dtype.__name__}`"
            )

    if length is None:
        raise RuntimeError("could not infer the `length` argument when calling the `wp.vector()` function")

    if dtype is None:
        raise RuntimeError("could not infer the `dtype` argument when calling the `wp.vector()` function")

    return vector(length=length, dtype=dtype)


def vector_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    length = return_type._length_
    dtype = return_type._wp_scalar_type_

    variadic_args = args.get("args", ())

    func_args = variadic_args
    template_args = (length, dtype)
    return (func_args, template_args)


add_builtin(
    "vector",
    input_types={"*args": Scalar, "length": int, "dtype": Scalar},
    defaults={"length": None, "dtype": None},
    variadic=True,
    initializer_list_func=lambda arg_types, arg_values: len(arg_types.get("args", ())) > 4,
    value_func=vector_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k not in ("length", "dtype")},
    dispatch_func=vector_dispatch_func,
    native_func="vec_t",
    doc="Construct a vector of given length and dtype.",
    group="Vector Math",
    export=False,
)


def matrix_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return matrix(shape=(Any, Any), dtype=Scalar)

    shape = arg_values.get("shape", None)
    dtype = arg_values.get("dtype", None)

    variadic_arg_types = arg_types.get("args", ())
    variadic_arg_count = len(variadic_arg_types)
    if variadic_arg_count == 0:
        # Zero-initialization, e.g.: `wp.matXX()`, `wp.matrix(shape=(2, 2), dtype=wp.float16)`.
        if shape is None:
            raise RuntimeError("the `shape` argument must be specified when zero-initializing a matrix")

        if dtype is None:
            dtype = float32
    elif variadic_arg_count == 1:
        value_type = strip_reference(variadic_arg_types[0])
        if type_is_matrix(value_type):
            # Copy constructor, e.g.: `wp.matXX(other_mat)`, `wp.matrix(other_mat)`.
            if shape is None:
                shape = value_type._shape_
            elif not seq_check_equal(value_type._shape_, shape):
                raise RuntimeError(
                    f"incompatible matrix of shape {tuple(shape)} given when copy constructing "
                    f"a matrix of shape {tuple(value_type._shape_)}"
                )

            if dtype is None:
                dtype = value_type._wp_scalar_type_
        else:
            # Initialization by filling a value, e.g.: `wp.matXX(123)`,
            # `wp.matrix(123, shape=(2, 2))`.
            if shape is None:
                raise RuntimeError("the `shape` argument must be specified when filling a matrix with a value")

            if dtype is None:
                dtype = value_type
            elif not warp.types.scalars_equal(value_type, dtype):
                raise RuntimeError(
                    f"the value used to fill this matrix is expected to be of the type `{dtype.__name__}`"
                )
    else:
        # Initializing by value, e.g.: `wp.mat22(1, 2, 3, 4)`, `wp.matrix(1, 2, 3, 4, shape=(2, 2))`.
        if shape is None:
            raise RuntimeError("the `shape` argument must be specified when initializing a matrix by value")

        if all(type_is_vector(x) for x in variadic_arg_types):
            warp.utils.warn(
                "the built-in `wp.matrix()` won't support taking column vectors as input "
                "in the future. Use `wp.matrix_from_rows()` or `wp.matrix_from_cols()` instead.",
                DeprecationWarning,
            )

            if shape[1] != variadic_arg_count:
                raise RuntimeError(
                    f"incompatible number of column vectors given ({variadic_arg_count}) "
                    f"when constructing a matrix of shape {tuple(shape)}"
                )

            if any(x._length_ != shape[0] for x in variadic_arg_types):
                raise RuntimeError(
                    f"incompatible column vector lengths given when constructing a matrix of shape {tuple(shape)}"
                )
        elif shape[0] * shape[1] != variadic_arg_count:
            raise RuntimeError(
                f"incompatible number of values given ({variadic_arg_count}) "
                f"when constructing a matrix of shape {tuple(shape)}"
            )

        try:
            value_type = scalar_infer_type(variadic_arg_types)
        except RuntimeError:
            raise RuntimeError("all values given when constructing a matrix must have the same type") from None

        if dtype is None:
            dtype = value_type
        elif not warp.types.scalars_equal(value_type, dtype):
            raise RuntimeError(
                f"all values used to initialize this matrix are expected to be of the type `{dtype.__name__}`"
            )

    if shape is None:
        raise RuntimeError("could not infer the `shape` argument when calling the `wp.matrix()` function")

    if dtype is None:
        raise RuntimeError("could not infer the `dtype` argument when calling the `wp.matrix()` function")

    return matrix(shape=shape, dtype=dtype)


def matrix_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    shape = return_type._shape_
    dtype = return_type._wp_scalar_type_

    variadic_args = args.get("args", ())

    func_args = variadic_args
    template_args = (*shape, dtype)
    return (func_args, template_args)


# only use initializer list if matrix size < 5x5, or for scalar construction
def matrix_initializer_list_func(args, return_type):
    shape = return_type._shape_

    variadic_args = args.get("args", ())
    variadic_arg_count = len(variadic_args)
    return not (
        variadic_arg_count <= 1  # zero/fill initialization
        or (shape[0] == shape[1] and shape[1] < 5)  # value construction for small matrices
    )


add_builtin(
    "matrix",
    input_types={"*args": Scalar, "shape": Tuple[int, int], "dtype": Scalar},
    defaults={"shape": None, "dtype": None},
    variadic=True,
    value_func=matrix_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k not in ("shape", "dtype")},
    dispatch_func=matrix_dispatch_func,
    initializer_list_func=matrix_initializer_list_func,
    native_func="mat_t",
    doc="Construct a matrix. If the positional ``arg_types`` are not given, then matrix will be zero-initialized.",
    group="Vector Math",
    export=False,
)


def matrix_from_vecs_create_value_func(cols: bool):
    def fn(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
        if arg_types is None:
            return matrix(shape=(Any, Any), dtype=Scalar)

        variadic_arg_types = arg_types.get("args", ())
        variadic_arg_count = len(variadic_arg_types)

        if not all(type_is_vector(x) for x in variadic_arg_types):
            raise RuntimeError("all arguments are expected to be vectors")

        length = variadic_arg_types[0]._length_
        if any(x._length_ != length for x in variadic_arg_types):
            raise RuntimeError("all vectors are expected to have the same length")

        dtype = variadic_arg_types[0]._wp_scalar_type_
        if any(x._wp_scalar_type_ != dtype for x in variadic_arg_types):
            raise RuntimeError("all vectors are expected to have the same dtype")

        shape = (length, variadic_arg_count) if cols else (variadic_arg_count, length)
        return matrix(shape=shape, dtype=dtype)

    return fn


def matrix_from_vecs_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    shape = return_type._shape_
    dtype = return_type._wp_scalar_type_

    variadic_args = args.get("args", ())

    func_args = variadic_args

    if shape in ((2, 2), (3, 3), (4, 4)):
        # Template specializations exist for these shapes, don't pass them
        # as template parameters.
        template_args = (dtype,)
    else:
        template_args = (*shape, dtype)

    return (func_args, template_args)


def matrix_from_vecs_initializer_list_func(args, return_type):
    shape = return_type._shape_

    return shape[0] != shape[1] or shape[0] > 4


add_builtin(
    "matrix_from_cols",
    input_types={"*args": vector(length=Any, dtype=Scalar)},
    variadic=True,
    value_func=matrix_from_vecs_create_value_func(cols=True),
    dispatch_func=matrix_from_vecs_dispatch_func,
    initializer_list_func=matrix_from_vecs_initializer_list_func,
    native_func="matrix_from_cols",
    doc="Construct a matrix from column vectors.",
    group="Vector Math",
    export=False,
)

add_builtin(
    "matrix_from_rows",
    input_types={"*args": vector(length=Any, dtype=Scalar)},
    variadic=True,
    value_func=matrix_from_vecs_create_value_func(cols=False),
    dispatch_func=matrix_from_vecs_dispatch_func,
    initializer_list_func=matrix_from_vecs_initializer_list_func,
    native_func="matrix_from_rows",
    doc="Construct a matrix from row vectors.",
    group="Vector Math",
    export=False,
)


def identity_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return matrix(shape=(Any, Any), dtype=Scalar)

    n = arg_values["n"]
    dtype = arg_values["dtype"]

    if n is None:
        raise RuntimeError("'n' must be a constant when calling identity()")

    return matrix(shape=(n, n), dtype=dtype)


def identity_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    shape = return_type._shape_
    dtype = return_type._wp_scalar_type_

    func_args = ()
    template_args = (shape[0], dtype)
    return (func_args, template_args)


add_builtin(
    "identity",
    input_types={"n": int, "dtype": Scalar},
    value_func=identity_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=identity_dispatch_func,
    doc="Create an identity matrix with shape=(n,n) with the type given by ``dtype``.",
    group="Vector Math",
    export=False,
)


def matrix_transform_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    warp.utils.warn(
        "the built-in `wp.matrix()` function to construct a 4x4 matrix from a 3D position, quaternion, "
        "and 3D scale vector will be deprecated in favor of `wp.transform_compose()`.",
        DeprecationWarning,
    )
    if arg_types is None:
        return matrix(shape=(4, 4), dtype=Float)

    dtype = arg_values.get("dtype", None)

    value_arg_types = tuple(v for k, v in arg_types.items() if k != "dtype")
    try:
        value_type = scalar_infer_type(value_arg_types)
    except RuntimeError:
        raise RuntimeError(
            "all values given when constructing a transformation matrix must have the same type"
        ) from None

    if dtype is None:
        dtype = value_type
    elif not warp.types.scalars_equal(value_type, dtype):
        raise RuntimeError(
            f"all values used to initialize this transformation matrix are expected to be of the type `{dtype.__name__}`"
        )

    return matrix(shape=(4, 4), dtype=dtype)


def matrix_transform_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = return_type._wp_scalar_type_

    func_args = tuple(v for k, v in args.items() if k != "dtype")
    template_args = (4, 4, dtype)
    return (func_args, template_args)


add_builtin(
    "matrix",
    input_types={
        "pos": vector(length=3, dtype=Float),
        "rot": quaternion(dtype=Float),
        "scale": vector(length=3, dtype=Float),
        "dtype": Float,
    },
    defaults={"dtype": None},
    value_func=matrix_transform_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=matrix_transform_dispatch_func,
    native_func="mat_t",
    doc="""Construct a 4x4 transformation matrix that applies the transformations as
    Translation(pos)*Rotation(rot)*Scaling(scale) when applied to column vectors, i.e.: y = (TRS)*x

    .. warning::
       This function has been deprecated in favor of :func:`warp.math.transform_compose()`.""",
    group="Vector Math",
    export=False,
)


def svd3_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return (
            matrix(shape=(3, 3), dtype=Float),
            vector(length=3, dtype=Float),
            matrix(shape=(3, 3), dtype=Float),
        )

    dtype = arg_types["A"]._wp_scalar_type_
    return (
        matrix(shape=(3, 3), dtype=dtype),
        vector(length=3, dtype=dtype),
        matrix(shape=(3, 3), dtype=dtype),
    )


add_builtin(
    "svd3",
    input_types={"A": matrix(shape=(3, 3), dtype=Float)},
    value_func=svd3_value_func,
    group="Vector Math",
    doc="""Compute the SVD of a 3x3 matrix ``A``. The singular values are returned in ``sigma``,
    while the left and right basis vectors are returned in ``U`` and ``V``.""",
)

add_builtin(
    "svd3",
    input_types={
        "A": matrix(shape=(3, 3), dtype=Float),
        "U": matrix(shape=(3, 3), dtype=Float),
        "sigma": vector(length=3, dtype=Float),
        "V": matrix(shape=(3, 3), dtype=Float),
    },
    value_type=None,
    group="Vector Math",
    export=False,
    doc="""Compute the SVD of a 3x3 matrix ``A``. The singular values are returned in ``sigma``,
    while the left and right basis vectors are returned in ``U`` and ``V``.""",
)


def svd2_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return (
            matrix(shape=(2, 2), dtype=Float),
            vector(length=2, dtype=Float),
            matrix(shape=(2, 2), dtype=Float),
        )

    dtype = arg_types["A"]._wp_scalar_type_
    return (
        matrix(shape=(2, 2), dtype=dtype),
        vector(length=2, dtype=dtype),
        matrix(shape=(2, 2), dtype=dtype),
    )


add_builtin(
    "svd2",
    input_types={"A": matrix(shape=(2, 2), dtype=Float)},
    value_func=svd2_value_func,
    group="Vector Math",
    doc="""Compute the SVD of a 2x2 matrix ``A``. The singular values are returned in ``sigma``,
    while the left and right basis vectors are returned in ``U`` and ``V``.""",
)

add_builtin(
    "svd2",
    input_types={
        "A": matrix(shape=(2, 2), dtype=Float),
        "U": matrix(shape=(2, 2), dtype=Float),
        "sigma": vector(length=2, dtype=Float),
        "V": matrix(shape=(2, 2), dtype=Float),
    },
    value_type=None,
    group="Vector Math",
    export=False,
    doc="""Compute the SVD of a 2x2 matrix ``A``. The singular values are returned in ``sigma``,
    while the left and right basis vectors are returned in ``U`` and ``V``.""",
)


def qr3_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return (
            matrix(shape=(3, 3), dtype=Float),
            matrix(shape=(3, 3), dtype=Float),
        )

    dtype = arg_types["A"]._wp_scalar_type_
    return (
        matrix(shape=(3, 3), dtype=dtype),
        matrix(shape=(3, 3), dtype=dtype),
    )


add_builtin(
    "qr3",
    input_types={"A": matrix(shape=(3, 3), dtype=Float)},
    value_func=qr3_value_func,
    group="Vector Math",
    doc="""Compute the QR decomposition of a 3x3 matrix ``A``. The orthogonal matrix is returned in ``Q``,
    while the upper triangular matrix is returned in ``R``.""",
)

add_builtin(
    "qr3",
    input_types={
        "A": matrix(shape=(3, 3), dtype=Float),
        "Q": matrix(shape=(3, 3), dtype=Float),
        "R": matrix(shape=(3, 3), dtype=Float),
    },
    value_type=None,
    group="Vector Math",
    export=False,
    doc="""Compute the QR decomposition of a 3x3 matrix ``A``. The orthogonal matrix is returned in ``Q``,
    while the upper triangular matrix is returned in ``R``.""",
)


def eig3_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return (matrix(shape=(3, 3), dtype=Float), vector(length=3, dtype=Float))

    dtype = arg_types["A"]._wp_scalar_type_
    return (
        matrix(shape=(3, 3), dtype=dtype),
        vector(length=3, dtype=dtype),
    )


add_builtin(
    "eig3",
    input_types={"A": matrix(shape=(3, 3), dtype=Float)},
    value_func=eig3_value_func,
    group="Vector Math",
    doc="""Compute the eigendecomposition of a 3x3 matrix ``A``. The eigenvectors are returned as the columns of ``Q``,
    while the corresponding eigenvalues are returned in ``d``.""",
)

add_builtin(
    "eig3",
    input_types={
        "A": matrix(shape=(3, 3), dtype=Float),
        "Q": matrix(shape=(3, 3), dtype=Float),
        "d": vector(length=3, dtype=Float),
    },
    value_type=None,
    group="Vector Math",
    export=False,
    doc="""Compute the eigendecomposition of a 3x3 matrix ``A``. The eigenvectors are returned as the columns of ``Q``,
    while the corresponding eigenvalues are returned in ``d``.""",
)

# ---------------------------------
# Quaternion Math


def quaternion_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return quaternion(dtype=Float)

    dtype = arg_values.get("dtype", None)

    variadic_arg_types = tuple(v for k, v in arg_types.items() if k != "dtype")
    variadic_arg_count = len(variadic_arg_types)

    if variadic_arg_count == 0:
        # Zero-initialization, e.g.: `wp.quat()`, `wp.quaternion(dtype=wp.float16)`.
        if dtype is None:
            dtype = float32
        elif dtype not in float_types:
            raise RuntimeError(
                f"a float type is expected when zero-initializing a quaternion but got `{type(dtype).__name__}` instead"
            )
    elif variadic_arg_count == 1:
        if type_is_quaternion(variadic_arg_types[0]):
            # Copy constructor, e.g.: `wp.quat(other_vec)`, `wp.quaternion(other_vec)`.
            in_quat = variadic_arg_types[0]
            if dtype is None:
                dtype = in_quat._wp_scalar_type_
    else:
        try:
            value_type = scalar_infer_type(variadic_arg_types)
        except RuntimeError:
            raise RuntimeError("all values given when constructing a quaternion must have the same type") from None

        if dtype is None:
            dtype = value_type
        elif not warp.types.scalars_equal(value_type, dtype):
            raise RuntimeError(
                f"all values used to initialize this quaternion are expected to be of the type `{dtype.__name__}`"
            )

    if dtype is None:
        raise RuntimeError("could not infer the `dtype` argument when calling the `wp.quaternion()` function")

    return quaternion(dtype=dtype)


def quaternion_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = return_type._wp_scalar_type_

    variadic_args = tuple(v for k, v in args.items() if k != "dtype")

    func_args = variadic_args
    template_args = (dtype,)
    return (func_args, template_args)


add_builtin(
    "quaternion",
    input_types={"dtype": Float},
    defaults={"dtype": None},
    value_func=quaternion_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=quaternion_dispatch_func,
    native_func="quat_t",
    group="Quaternion Math",
    doc="""Construct a zero-initialized quaternion. Quaternions are laid out as
    [ix, iy, iz, r], where ix, iy, iz are the imaginary part, and r the real part.""",
    export=False,
)
add_builtin(
    "quaternion",
    input_types={"x": Float, "y": Float, "z": Float, "w": Float, "dtype": Scalar},
    defaults={"dtype": None},
    value_func=quaternion_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=quaternion_dispatch_func,
    native_func="quat_t",
    group="Quaternion Math",
    doc="Create a quaternion using the supplied components (type inferred from component type).",
    export=False,
)
add_builtin(
    "quaternion",
    input_types={"ijk": vector(length=3, dtype=Float), "real": Float, "dtype": Float},
    defaults={"dtype": None},
    value_func=quaternion_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=quaternion_dispatch_func,
    native_func="quat_t",
    group="Quaternion Math",
    doc="Create a quaternion using the supplied vector/scalar (type inferred from scalar type).",
    export=False,
)

add_builtin(
    "quaternion",
    input_types={"quat": quaternion(dtype=Float), "dtype": Float},
    defaults={"dtype": None},
    value_func=quaternion_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=quaternion_dispatch_func,
    native_func="quat_t",
    group="Quaternion Math",
    doc="Construct a quaternion of type dtype from another quaternion of a different dtype.",
    export=False,
)


def quat_identity_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        # return quaternion(dtype=Float)
        return quatf

    dtype = arg_types.get("dtype", float32)
    return quaternion(dtype=dtype)


def quat_identity_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = return_type._wp_scalar_type_

    func_args = ()
    template_args = (dtype,)
    return (func_args, template_args)


add_builtin(
    "quat_identity",
    input_types={"dtype": Float},
    defaults={"dtype": None},
    value_func=quat_identity_value_func,
    export_func=lambda input_types: {},
    dispatch_func=quat_identity_dispatch_func,
    group="Quaternion Math",
    doc="Construct an identity quaternion with zero imaginary part and real part of 1.0",
    export=True,
)

add_builtin(
    "quat_from_axis_angle",
    input_types={"axis": vector(length=3, dtype=Float), "angle": Float},
    value_func=lambda arg_types, arg_values: quaternion(dtype=float_infer_type(arg_types)),
    group="Quaternion Math",
    doc="Construct a quaternion representing a rotation of angle radians around the given axis.",
)


def quat_to_axis_angle_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return (vector(length=3, dtype=Float), Float)

    dtype = arg_types["quat"]._wp_scalar_type_
    return (vector(length=3, dtype=dtype), dtype)


add_builtin(
    "quat_to_axis_angle",
    input_types={"quat": quaternion(dtype=Float)},
    value_func=quat_to_axis_angle_value_func,
    group="Quaternion Math",
    doc="Extract the rotation axis and angle radians a quaternion represents.",
)

add_builtin(
    "quat_to_axis_angle",
    input_types={"quat": quaternion(dtype=Float), "axis": vector(length=3, dtype=Float), "angle": Float},
    value_type=None,
    group="Quaternion Math",
    doc="Extract the rotation axis and angle radians a quaternion represents.",
    export=False,
)


add_builtin(
    "quat_from_matrix",
    input_types={"mat": matrix(shape=(3, 3), dtype=Float)},
    value_func=lambda arg_types, arg_values: quaternion(dtype=float_infer_type(arg_types)),
    group="Quaternion Math",
    doc="""Construct a quaternion from a 3x3 matrix.

    If the matrix is not a pure rotation, but for example includes scaling or skewing, the result is undefined.""",
)
add_builtin(
    "quat_from_matrix",
    input_types={"mat": matrix(shape=(4, 4), dtype=Float)},
    value_func=lambda arg_types, arg_values: quaternion(dtype=float_infer_type(arg_types)),
    group="Quaternion Math",
    doc="""Construct a quaternion from a 4x4 matrix.

    If the top-left 3x3 block of the matrix is not a pure rotation, but for example includes scaling or skewing, the result is undefined.""",
)
add_builtin(
    "quat_rpy",
    input_types={"roll": Float, "pitch": Float, "yaw": Float},
    value_func=lambda arg_types, arg_values: quaternion(dtype=float_infer_type(arg_types)),
    group="Quaternion Math",
    doc="Construct a quaternion representing a combined roll (z), pitch (x), yaw rotations (y) in radians.",
)
add_builtin(
    "quat_inverse",
    input_types={"quat": quaternion(dtype=Float)},
    value_func=lambda arg_types, arg_values: quaternion(dtype=float_infer_type(arg_types)),
    group="Quaternion Math",
    doc="Compute quaternion conjugate.",
)
add_builtin(
    "quat_rotate",
    input_types={"quat": quaternion(dtype=Float), "vec": vector(length=3, dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=float_infer_type(arg_types)),
    group="Quaternion Math",
    doc="Rotate a vector by a quaternion.",
)
add_builtin(
    "quat_rotate_inv",
    input_types={"quat": quaternion(dtype=Float), "vec": vector(length=3, dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=float_infer_type(arg_types)),
    group="Quaternion Math",
    doc="Rotate a vector by the inverse of a quaternion.",
)
add_builtin(
    "quat_slerp",
    input_types={"a": quaternion(dtype=Float), "b": quaternion(dtype=Float), "t": Float},
    value_func=lambda arg_types, arg_values: quaternion(dtype=float_infer_type(arg_types)),
    group="Quaternion Math",
    doc="Linearly interpolate between two quaternions.",
    require_original_output_arg=True,
)
add_builtin(
    "quat_to_matrix",
    input_types={"quat": quaternion(dtype=Float)},
    value_func=lambda arg_types, arg_values: matrix(shape=(3, 3), dtype=float_infer_type(arg_types)),
    group="Quaternion Math",
    doc="Convert a quaternion to a 3x3 rotation matrix.",
)

add_builtin(
    "dot",
    input_types={"a": quaternion(dtype=Float), "b": quaternion(dtype=Float)},
    value_func=float_sametypes_value_func,
    group="Quaternion Math",
    doc="Compute the dot product between two quaternions.",
)
# ---------------------------------
# Transformations


def transformation_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return transformation(dtype=Float)

    dtype = arg_values.get("dtype", None)

    variadic_arg_types = arg_types.get("args", ())
    variadic_arg_count = len(variadic_arg_types)
    if variadic_arg_count == 0:
        # Zero-initialization, e.g.: `wp.transform()`, `wp.transformation(dtype=wp.float16)`.
        if dtype is None:
            dtype = float32
    elif variadic_arg_count == 1:
        # Initialization by filling a value, e.g.: `wp.transform(123)`,
        # `wp.transformation(123)`.
        value_type = strip_reference(variadic_arg_types[0])
        if dtype is None:
            dtype = value_type
        elif not warp.types.scalars_equal(value_type, dtype):
            raise RuntimeError(
                f"the value used to fill this transform is expected to be of the type `{dtype.__name__}`"
            )
    elif variadic_arg_count == 7:
        # Initializing by value, e.g.: `wp.transform(1, 2, 3, 4, 5, 6, 7)`.
        try:
            value_type = scalar_infer_type(variadic_arg_types)
        except RuntimeError:
            raise RuntimeError("all values given when constructing a transform must have the same type") from None

        if dtype is None:
            dtype = value_type
        elif not warp.types.scalars_equal(value_type, dtype):
            raise RuntimeError(
                f"all values used to initialize this transform are expected to be of the type `{dtype.__name__}`"
            )

    if dtype is None:
        raise RuntimeError("could not infer the `dtype` argument when calling the `wp.transform()` function")

    return transformation(dtype=dtype)


def transformation_pq_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return transformation(dtype=Float)

    try:
        value_type = float_infer_type(arg_types)
    except RuntimeError:
        raise RuntimeError(
            "all values given when constructing a transformation matrix must have the same type"
        ) from None

    dtype = arg_values.get("dtype", None)
    if dtype is None:
        dtype = value_type
    elif not warp.types.scalars_equal(value_type, dtype):
        raise RuntimeError(
            f"all values used to initialize this transformation matrix are expected to be of the type `{dtype.__name__}`"
        )

    return transformation(dtype=dtype)


def transformation_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = return_type._wp_scalar_type_

    variadic_args = tuple(v for k, v in args.items() if k != "dtype")

    func_args = variadic_args
    template_args = (dtype,)
    return (func_args, template_args)


add_builtin(
    "transformation",
    input_types={"p": vector(length=3, dtype=Float), "q": quaternion(dtype=Float), "dtype": Float},
    defaults={"dtype": None},
    value_func=transformation_pq_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=transformation_dispatch_func,
    native_func="transform_t",
    group="Transformations",
    doc="Construct a rigid-body transformation with translation part ``p`` and rotation ``q``.",
    export=False,
)


add_builtin(
    "transformation",
    input_types={"*args": Float, "dtype": Float},
    defaults={"dtype": None},
    variadic=True,
    initializer_list_func=lambda arg_types, arg_values: len(arg_types.get("args", ())) > 1,
    value_func=transformation_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k not in ("dtype")},
    dispatch_func=transformation_dispatch_func,
    native_func="transform_t",
    doc="Construct a spatial transform vector of given dtype.",
    group="Spatial Math",
    export=False,
)


def transform_identity_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        # return transformation(dtype=Float)
        return transformf

    dtype = arg_types.get("dtype", float32)
    return transformation(dtype=dtype)


def transform_identity_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = return_type._wp_scalar_type_

    func_args = ()
    template_args = (dtype,)
    return (func_args, template_args)


add_builtin(
    "transform_identity",
    input_types={"dtype": Float},
    defaults={"dtype": None},
    value_func=transform_identity_value_func,
    export_func=lambda input_types: {},
    dispatch_func=transform_identity_dispatch_func,
    group="Transformations",
    doc="Construct an identity transform with zero translation and identity rotation.",
    export=True,
)

add_builtin(
    "transform_get_translation",
    input_types={"xform": transformation(dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=float_infer_type(arg_types)),
    group="Transformations",
    doc="Return the translational part of a transform ``xform``.",
)
add_builtin(
    "transform_get_rotation",
    input_types={"xform": transformation(dtype=Float)},
    value_func=lambda arg_types, arg_values: quaternion(dtype=float_infer_type(arg_types)),
    group="Transformations",
    doc="Return the rotational part of a transform ``xform``.",
)
add_builtin(
    "transform_set_translation",
    input_types={"xform": transformation(dtype=Float), "p": vector(length=3, dtype=Float)},
    value_type=None,
    group="Transformations",
    doc="Set the translational part of a transform ``xform``.",
)
add_builtin(
    "transform_set_rotation",
    input_types={"xform": transformation(dtype=Float), "q": quaternion(dtype=Float)},
    value_type=None,
    group="Transformations",
    doc="Set the rotational part of a transform ``xform``.",
)
# performs a copy internally if wp.config.enable_vector_component_overwrites is True
add_builtin(
    "transform_set_translation_copy",
    input_types={"xform": transformation(dtype=Float), "p": vector(length=3, dtype=Float)},
    value_type=transformation(dtype=Float),
    group="Transformations",
    doc="Set the translational part of a transform ``xform``.",
    hidden=True,
    export=False,
)
# performs a copy internally if wp.config.enable_vector_component_overwrites is True
add_builtin(
    "transform_set_rotation_copy",
    input_types={"xform": transformation(dtype=Float), "q": quaternion(dtype=Float)},
    value_type=transformation(dtype=Float),
    group="Transformations",
    doc="Set the rotational part of a transform ``xform``.",
    hidden=True,
    export=False,
)
add_builtin(
    "transform_multiply",
    input_types={"a": transformation(dtype=Float), "b": transformation(dtype=Float)},
    value_func=lambda arg_types, arg_values: transformation(dtype=float_infer_type(arg_types)),
    group="Transformations",
    doc="Multiply two rigid body transformations together.",
)
add_builtin(
    "transform_point",
    input_types={"xform": transformation(dtype=Float), "point": vector(length=3, dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=float_infer_type(arg_types)),
    group="Transformations",
    doc="Apply the transform to a point ``point`` treating the homogeneous coordinate as w=1 (translation and rotation).",
)
add_builtin(
    "transform_point",
    input_types={"mat": matrix(shape=(4, 4), dtype=Float), "point": vector(length=3, dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=float_infer_type(arg_types)),
    group="Vector Math",
    doc="""Apply the transform to a point ``point`` treating the homogeneous coordinate as w=1.

    The transformation is applied treating ``point`` as a column vector, e.g.: ``y = mat*point``.

    This is in contrast to some libraries, notably USD, which applies transforms to row vectors, ``y^T = point^T*mat^T``.
    If the transform is coming from a library that uses row-vectors, then users should transpose the transformation
    matrix before calling this method.""",
)
add_builtin(
    "transform_vector",
    input_types={"xform": transformation(dtype=Float), "vec": vector(length=3, dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=float_infer_type(arg_types)),
    group="Transformations",
    doc="Apply the transform to a vector ``vec`` treating the homogeneous coordinate as w=0 (rotation only).",
)
add_builtin(
    "transform_vector",
    input_types={"mat": matrix(shape=(4, 4), dtype=Float), "vec": vector(length=3, dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=float_infer_type(arg_types)),
    group="Vector Math",
    doc="""Apply the transform to a vector ``vec`` treating the homogeneous coordinate as w=0.

    The transformation is applied treating ``vec`` as a column vector, e.g.: ``y = mat*vec``.

    This is in contrast to some libraries, notably USD, which applies transforms to row vectors, ``y^T = vec^T*mat^T``.
    If the transform is coming from a library that uses row-vectors, then users should transpose the transformation
    matrix before calling this method.""",
)
add_builtin(
    "transform_inverse",
    input_types={"xform": transformation(dtype=Float)},
    value_func=sametypes_create_value_func(transformation(dtype=Float)),
    group="Transformations",
    doc="Compute the inverse of the transformation ``xform``.",
)
# ---------------------------------
# Spatial Math


def spatial_vector_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return vector(length=6, dtype=Float)

    dtype = arg_values.get("dtype", None)

    variadic_arg_types = tuple(v for k, v in arg_types.items() if k != "dtype")
    variadic_arg_count = len(variadic_arg_types)
    if variadic_arg_count == 0:
        if dtype is None:
            dtype = float32
    elif variadic_arg_count == 2:
        if any(not type_is_vector(x) for x in variadic_arg_types) or any(x._length_ != 3 for x in variadic_arg_types):
            raise RuntimeError("arguments `w` and `v` are expected to be vectors of length 3")
    elif variadic_arg_count != 6:
        raise RuntimeError("2 vectors or 6 scalar values are expected when constructing a spatial vector")

    if variadic_arg_count:
        try:
            value_type = float_infer_type(variadic_arg_types)
        except RuntimeError:
            raise RuntimeError("all values given when constructing a spatial vector must have the same type") from None

        if dtype is None:
            dtype = value_type
        elif not warp.types.scalars_equal(value_type, dtype):
            raise RuntimeError(
                f"all values used to initialize this spatial vector are expected to be of the type `{dtype.__name__}`"
            )

    return vector(length=6, dtype=dtype)


def spatial_vector_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    length = return_type._length_
    dtype = return_type._wp_scalar_type_

    variadic_args = tuple(v for k, v in args.items() if k != "dtype")

    func_args = variadic_args
    template_args = (length, dtype)
    return (func_args, template_args)


add_builtin(
    "spatial_vector",
    input_types={"dtype": Float},
    defaults={"dtype": None},
    value_func=spatial_vector_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=spatial_vector_dispatch_func,
    native_func="vec_t",
    group="Spatial Math",
    doc="Zero-initialize a 6D screw vector.",
    export=False,
)


add_builtin(
    "spatial_vector",
    input_types={"w": vector(length=3, dtype=Float), "v": vector(length=3, dtype=Float), "dtype": Float},
    defaults={"dtype": None},
    value_func=spatial_vector_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=spatial_vector_dispatch_func,
    native_func="vec_t",
    group="Spatial Math",
    doc="Construct a 6D screw vector from two 3D vectors.",
    export=False,
)

add_builtin(
    "spatial_vector",
    input_types={"wx": Float, "wy": Float, "wz": Float, "vx": Float, "vy": Float, "vz": Float, "dtype": Float},
    defaults={"dtype": None},
    initializer_list_func=lambda arg_types, arg_values: True,
    value_func=spatial_vector_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=spatial_vector_dispatch_func,
    native_func="vec_t",
    group="Spatial Math",
    doc="Construct a 6D screw vector from six values.",
    export=False,
)


add_builtin(
    "spatial_adjoint",
    input_types={"r": matrix(shape=(3, 3), dtype=Float), "s": matrix(shape=(3, 3), dtype=Float)},
    value_func=lambda arg_types, arg_values: matrix(shape=(6, 6), dtype=float_infer_type(arg_types)),
    group="Spatial Math",
    doc="Construct a 6x6 spatial inertial matrix from two 3x3 diagonal blocks.",
    export=False,
)
add_builtin(
    "spatial_dot",
    input_types={"a": vector(length=6, dtype=Float), "b": vector(length=6, dtype=Float)},
    value_func=float_sametypes_value_func,
    group="Spatial Math",
    doc="Compute the dot product of two 6D screw vectors.",
)
add_builtin(
    "spatial_cross",
    input_types={"a": vector(length=6, dtype=Float), "b": vector(length=6, dtype=Float)},
    value_func=sametypes_create_value_func(vector(length=6, dtype=Float)),
    group="Spatial Math",
    doc="Compute the cross product of two 6D screw vectors.",
)
add_builtin(
    "spatial_cross_dual",
    input_types={"a": vector(length=6, dtype=Float), "b": vector(length=6, dtype=Float)},
    value_func=sametypes_create_value_func(vector(length=6, dtype=Float)),
    group="Spatial Math",
    doc="Compute the dual cross product of two 6D screw vectors.",
)

add_builtin(
    "spatial_top",
    input_types={"svec": vector(length=6, dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=Float)
    if arg_types is None
    else vector(length=3, dtype=arg_types["svec"]._wp_scalar_type_),
    group="Spatial Math",
    doc="Return the top (first) part of a 6D screw vector.",
)
add_builtin(
    "spatial_bottom",
    input_types={"svec": vector(length=6, dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=Float)
    if arg_types is None
    else vector(length=3, dtype=arg_types["svec"]._wp_scalar_type_),
    group="Spatial Math",
    doc="Return the bottom (second) part of a 6D screw vector.",
)

add_builtin(
    "spatial_jacobian",
    input_types={
        "S": array(dtype=vector(length=6, dtype=Float)),
        "joint_parents": array(dtype=int),
        "joint_qd_start": array(dtype=int),
        "joint_start": int,
        "joint_count": int,
        "J_start": int,
        "J_out": array(dtype=Float),
    },
    value_type=None,
    doc="",
    group="Spatial Math",
)

add_builtin(
    "spatial_mass",
    input_types={
        "I_s": array(dtype=matrix(shape=(6, 6), dtype=Float)),
        "joint_start": int,
        "joint_count": int,
        "M_start": int,
        "M": array(dtype=Float),
    },
    value_type=None,
    doc="",
    group="Spatial Math",
)

# ------------------
# Tile-based primitives


def tile_zeros_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, ...])

    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    if "dtype" not in arg_values:
        raise TypeError("tile_zeros() missing required keyword argument 'dtype'")

    if "storage" not in arg_values:
        raise TypeError("tile_zeros() missing required keyword argument 'storage'")

    if arg_values["storage"] not in {"shared", "register"}:
        raise ValueError(f"Invalid value for 'storage': {arg_values['storage']!r}. Expected 'shared' or 'register'.")

    dtype = arg_values["dtype"]

    return tile(dtype=dtype, shape=shape, storage=arg_values["storage"])


def tile_zeros_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    dtype = arg_values["dtype"]

    template_args = []
    template_args.append(dtype)
    template_args.extend(shape)

    return ([], template_args)


add_builtin(
    "tile_zeros",
    input_types={"shape": Tuple[int, ...], "dtype": Any, "storage": str},
    defaults={"storage": "register", "dtype": float},
    value_func=tile_zeros_value_func,
    dispatch_func=tile_zeros_dispatch_func,
    variadic=False,
    missing_grad=True,
    doc="""Allocate a tile of zero-initialized items.

    :param shape: Shape of the output tile
    :param dtype: Data type of output tile's elements (default float)
    :param storage: The storage location for the tile: ``"register"`` for registers
      (default) or ``"shared"`` for shared memory.
    :returns: A zero-initialized tile with shape and data type as specified""",
    group="Tile Primitives",
    export=False,
)

# overload for scalar shape
add_builtin(
    "tile_zeros",
    input_types={"shape": int, "dtype": Any, "storage": str},
    defaults={"storage": "register", "dtype": float},
    value_func=tile_zeros_value_func,
    dispatch_func=tile_zeros_dispatch_func,
    variadic=False,
    missing_grad=True,
    hidden=True,
    group="Tile Primitives",
    export=False,
)


def tile_ones_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, ...])

    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    if "dtype" not in arg_values:
        raise TypeError("tile_ones() missing required keyword argument 'dtype'")

    if "storage" not in arg_values:
        raise TypeError("tile_ones() missing required keyword argument 'storage'")

    if arg_values["storage"] not in {"shared", "register"}:
        raise ValueError(f"Invalid value for 'storage': {arg_values['storage']!r}. Expected 'shared' or 'register'.")

    dtype = arg_values["dtype"]

    return tile(dtype=dtype, shape=shape, storage=arg_values["storage"])


def tile_ones_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    dtype = arg_values["dtype"]

    template_args = []
    template_args.append(dtype)
    template_args.extend(shape)

    return ([], template_args)


add_builtin(
    "tile_ones",
    input_types={"shape": Tuple[int, ...], "dtype": Any, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_ones_value_func,
    dispatch_func=tile_ones_dispatch_func,
    missing_grad=True,
    doc="""Allocate a tile of one-initialized items.

    :param shape: Shape of the output tile
    :param dtype: Data type of output tile's elements
    :param storage: The storage location for the tile: ``"register"`` for registers
      (default) or ``"shared"`` for shared memory.
    :returns: A one-initialized tile with shape and data type as specified""",
    group="Tile Primitives",
    export=False,
)

# overload for scalar shape
add_builtin(
    "tile_ones",
    input_types={"shape": int, "dtype": Any, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_ones_value_func,
    dispatch_func=tile_ones_dispatch_func,
    missing_grad=True,
    hidden=True,
    group="Tile Primitives",
    export=False,
)


def tile_arange_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Scalar, shape=Tuple[int])

    if "args" not in arg_values:
        raise TypeError("tile_arange() requires at least one positional argument specifying the range")

    args = arg_values["args"]

    start = 0
    stop = 0
    step = 1

    if len(args) == 1:
        start = 0
        stop = args[0]

    elif len(args) == 2:
        start = args[0]
        stop = args[1]

    elif len(args) == 3:
        start = args[0]
        stop = args[1]
        step = args[2]

    if start is None or stop is None or step is None:
        raise RuntimeError("tile_arange() arguments must be compile time constants")

    if "dtype" in arg_values:
        dtype = arg_values["dtype"]
    else:
        dtype = float

    if arg_values["storage"] not in {"shared", "register"}:
        raise ValueError(f"Invalid value for 'storage': {arg_values['storage']!r}. Expected 'shared' or 'register'.")

    n = int((stop - start) / step)
    return tile(dtype=dtype, shape=(n,), storage=arg_values["storage"])


def tile_arange_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    size, dtype = return_type.size, return_type.dtype

    template_args = []
    template_args.append(dtype)
    template_args.append(size)

    if "args" not in arg_values:
        raise TypeError("tile_arange() requires at least one positional argument specifying the range")

    args = arg_values["args"]

    if len(args) == 1:
        start = warp.codegen.Var(label=None, type=return_type.dtype, constant=0)
        stop = args[0]
        step = warp.codegen.Var(label=None, type=return_type.dtype, constant=1)
    elif len(args) == 2:
        start = args[0]
        stop = args[1]
        step = warp.codegen.Var(label=None, type=return_type.dtype, constant=1)
    elif len(args) == 3:
        start = args[0]
        stop = args[1]
        step = args[2]
    else:
        raise TypeError(f"tile_arange() accepts at most 3 positional arguments, got {len(args)}")

    function_args = []
    function_args.append(start)
    function_args.append(stop)
    function_args.append(step)

    return (function_args, template_args)


add_builtin(
    "tile_arange",
    input_types={"*args": Scalar, "dtype": Scalar, "storage": str},
    defaults={"dtype": None, "storage": "register"},
    value_func=tile_arange_value_func,
    dispatch_func=tile_arange_dispatch_func,
    variadic=True,
    missing_grad=True,
    doc="""Generate a tile of linearly spaced elements.

    :param args: Variable-length positional arguments, interpreted as:

        - ``(stop,)``: Generates values from ``0`` to ``stop - 1``
        - ``(start, stop)``: Generates values from ``start`` to ``stop - 1``
        - ``(start, stop, step)``: Generates values from ``start`` to ``stop - 1`` with a step size

    :param dtype: Data type of output tile's elements (optional, default: ``float``)
    :param storage: The storage location for the tile: ``"register"`` for registers
      (default) or ``"shared"`` for shared memory.
    :returns: A tile with ``shape=(n)`` with linearly spaced elements of specified data type""",
    group="Tile Primitives",
    export=False,
)


def tile_load_tuple_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, ...])

    a = arg_types["a"]

    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    if "offset" in arg_values:
        offset = extract_tuple(arg_values["offset"])
    else:
        offset = (0,) * a.ndim

    if a.ndim != len(shape):
        raise ValueError(
            f"tile_load() array argument must have same number of dimensions as the tile shape, trying to perform an {len(shape)} dimensional load from an array with {a.ndim} dimensions."
        )

    if a.ndim != len(offset):
        raise ValueError(
            f"tile_load() offset argument must have the same number of dimensions as the array to load from, got {len(offset)} indices for an array with {a.ndim} dimensions"
        )

    if arg_values["storage"] not in {"shared", "register"}:
        raise ValueError(f"Invalid value for 'storage': {arg_values['storage']!r}. Expected 'shared' or 'register'.")

    return tile(dtype=a.dtype, shape=shape, storage=arg_values["storage"])


def tile_load_tuple_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    a = args["a"]
    shape = extract_tuple(args["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    if "offset" in args:
        offset = extract_tuple(args["offset"])
    else:
        offset = (0,) * a.type.ndim

    func_args = (a, *offset)
    template_args = shape

    return (func_args, template_args)


add_builtin(
    "tile_load",
    input_types={"a": array(dtype=Any), "shape": Tuple[int, ...], "offset": Tuple[int, ...], "storage": str},
    value_func=tile_load_tuple_value_func,
    dispatch_func=tile_load_tuple_dispatch_func,
    defaults={"offset": None, "storage": "register"},
    variadic=False,
    doc="""Loads a tile from a global memory array.

    This method will cooperatively load a tile from global memory using all threads in the block.

    :param a: The source array in global memory
    :param shape: Shape of the tile to load, must have the same number of dimensions as ``a``
    :param offset: Offset in the source array to begin reading from (optional)
    :param storage: The storage location for the tile: ``"register"`` for registers
      (default) or ``"shared"`` for shared memory.
    :returns: A tile with shape as specified and data type the same as the source array""",
    group="Tile Primitives",
    export=False,
)

# overload for scalar shape
add_builtin(
    "tile_load",
    input_types={"a": array(dtype=Any), "shape": int, "offset": int, "storage": str},
    value_func=tile_load_tuple_value_func,
    dispatch_func=tile_load_tuple_dispatch_func,
    defaults={"offset": None, "storage": "register"},
    group="Tile Primitives",
    hidden=True,
    export=False,
)


def tile_store_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return None

    a = arg_types["a"]
    t = arg_types["t"]

    if "offset" in arg_types:
        c = extract_tuple(arg_values["offset"])
    else:
        c = (0,) * a.ndim

    if len(c) != a.ndim:
        raise ValueError(
            f"tile_store() 'a' argument must have {len(c)} dimensions, "
            f"calculated based on the provided offset arguments, but got {a.ndim} dimensions."
        )

    if len(t.shape) != a.ndim:
        raise ValueError(
            f"tile_store() 'a' argument must have the same number of dimensions as the 't' argument, "
            f"but got {a.ndim} dimensions for 'a' and {len(t.shape)} dimensions for 't'"
        )

    if not types_equal(arg_types["a"].dtype, arg_types["t"].dtype):
        raise TypeError(
            f"tile_store() 'a' and 't' arguments must have the same dtype, got {arg_types['a'].dtype} and {arg_types['t'].dtype}"
        )

    return None


def tile_store_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    a = args["a"]
    t = args["t"]

    if "offset" in args:
        offset = extract_tuple(args["offset"])
    else:
        offset = (0,) * a.type.ndim

    func_args = (a, *offset, t)
    template_args = []

    return (func_args, template_args)


add_builtin(
    "tile_store",
    input_types={"a": array(dtype=Any), "t": tile(dtype=Any, shape=Tuple[int, ...]), "offset": Tuple[int, ...]},
    value_func=tile_store_value_func,
    dispatch_func=tile_store_dispatch_func,
    defaults={"offset": None},
    variadic=False,
    skip_replay=True,
    doc="""Store a tile to a global memory array.

    This method will cooperatively store a tile to global memory using all threads in the block.

    :param a: The destination array in global memory
    :param t: The source tile to store data from, must have the same data type and number of dimensions as the destination array
    :param offset: Offset in the destination array (optional)""",
    group="Tile Primitives",
    export=False,
)

# overload for scalar offset
add_builtin(
    "tile_store",
    input_types={"a": array(dtype=Any), "t": tile(dtype=Any, shape=Tuple[int, ...]), "offset": int},
    value_func=tile_store_value_func,
    dispatch_func=tile_store_dispatch_func,
    defaults={"offset": None},
    variadic=False,
    skip_replay=True,
    group="Tile Primitives",
    hidden=True,
    export=False,
)


def tile_atomic_add_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, ...])

    a = arg_types["a"]
    t = arg_types["t"]

    if "offset" in arg_types:
        c = extract_tuple(arg_values["offset"])
    else:
        c = (0,) * a.ndim

    if len(c) != a.ndim:
        raise ValueError(
            f"tile_atomic_add() 'a' argument must have {len(c)} dimensions, "
            f"calculated based on the provided offset arguments, but got {a.ndim} dimensions."
        )

    if a.ndim != len(t.shape):
        raise ValueError(
            f"tile_atomic_add() 'a' argument must have the same number of dimensions as the 't' argument, "
            f"but got {a.ndim} dimensions for 'a' and {len(t.shape)} dimensions for 't'"
        )

    if not types_equal(arg_types["a"].dtype, arg_types["t"].dtype):
        raise TypeError(
            f"tile_atomic_add() 'a' and 't' arguments must have the same dtype, got {arg_types['a'].dtype} and {arg_types['t'].dtype}"
        )

    return tile(
        dtype=arg_types["t"].dtype,
        shape=arg_types["t"].shape,
        storage=arg_types["t"].storage,
    )


def tile_atomic_add_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    a = args["a"]
    t = args["t"]

    if "offset" in args:
        offset = extract_tuple(args["offset"])
    else:
        offset = (0,) * a.type.ndim

    func_args = (a, *offset, t)
    template_args = []

    return (func_args, template_args)


add_builtin(
    "tile_atomic_add",
    input_types={"a": array(dtype=Any), "t": tile(dtype=Any, shape=Tuple[int, ...]), "offset": Tuple[int, ...]},
    value_func=tile_atomic_add_value_func,
    dispatch_func=tile_atomic_add_dispatch_func,
    defaults={"offset": None},
    variadic=False,
    skip_replay=True,
    doc="""Atomically add a tile onto the array `a`, each element will be updated atomically.

    :param a: Array in global memory, should have the same ``dtype`` as the input tile
    :param t: Source tile to add to the destination array
    :param offset: Offset in the destination array (optional)
    :returns: A tile with the same dimensions and data type as the source tile, holding the original value of the destination elements""",
    group="Tile Primitives",
    export=False,
)

# overload for scalar offset
add_builtin(
    "tile_atomic_add",
    input_types={"a": array(dtype=Any), "t": tile(dtype=Any, shape=Tuple[int, ...]), "offset": int},
    value_func=tile_atomic_add_value_func,
    dispatch_func=tile_atomic_add_dispatch_func,
    defaults={"offset": None},
    variadic=False,
    skip_replay=True,
    group="Tile Primitives",
    hidden=True,
    export=False,
)


def tile_view_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, ...])

    tile_type = arg_types["t"]
    offset = extract_tuple(arg_values["offset"])

    if len(offset) > len(tile_type.shape):
        raise ValueError(f"tile_view() specified too many offset coordinates {len(offset)} > {len(tile_type.shape)}")

    if "shape" in arg_values:
        # if shape is specified take it directly, e.g.:
        # tile_view(t, offset=(i,j), shape=(m,n))
        shape = extract_tuple(arg_values["shape"], as_constant=True)
        strides = tile_type.strides

        if len(shape) != len(tile_type.shape):
            raise ValueError(
                f"tile_view() if shape is specified it must have same number of dimensions as source tile, expected {len(tile_type.shape)}, got {len(shape)}"
            )
    else:
        # if not specified, then take output shape from unspecified src dimensions
        # e.g.: tile[i] will return a whole row of a 2D tile
        shape = tile_type.shape[len(offset) :]
        strides = tile_type.strides[len(offset) :]

    assert len(shape) == len(strides)

    # force source tile to shared memory
    tile_type.storage = "shared"

    output = tile(
        dtype=tile_type.dtype, shape=shape, strides=strides, layout=tile_type.layout, storage="shared", owner=False
    )
    return output


def tile_view_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    tile = arg_values["t"]
    coord = extract_tuple(arg_values["offset"])

    # zero-pad coord to match source array
    view_coord = [0] * len(tile.type.shape)
    for i in range(len(coord)):
        view_coord[i] = coord[i]

    func_args = (tile, *view_coord)
    template_args = (return_type,)

    return (func_args, template_args)


add_builtin(
    "tile_view",
    input_types={"t": tile(dtype=Any, shape=Tuple[int, ...]), "offset": Tuple[int, ...], "shape": Tuple[int, ...]},
    value_func=tile_view_value_func,
    dispatch_func=tile_view_dispatch_func,
    defaults={"shape": None},
    variadic=False,
    doc="""Return a slice of a given tile [offset, offset+shape], if shape is not specified it will be inferred from the unspecified offset dimensions.

    :param t: Input tile to extract a subrange from
    :param offset: Offset in the source tile
    :param shape: Shape of the returned slice
    :returns: A tile with dimensions given by the specified shape or the remaining source tile dimensions""",
    group="Tile Primitives",
    missing_grad=True,
    export=False,
)


def tile_squeeze_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, ...])

    tile_type = arg_types["t"]
    shape = tile_type.shape
    strides = tile_type.strides
    ndim = len(shape)

    if "axis" in arg_values:
        axis = arg_values["axis"]

        if not isinstance(axis, Sequence):
            # promote to tuple
            axis = (axis,)

        # promote negative indices to their positive equivalents
        axis = tuple([a if a >= 0 else a + ndim for a in axis])

        # validate that specified axes are size 1
        for a in axis:
            if shape[a] != 1:
                raise ValueError(
                    f"Cannot select an axis to squeeze out which has size not equal to one, axis={a}, size={shape[a]}"
                )

        # build new shape by skipping specified axes (if size is 1)
        new_shape = tuple(dim for i, dim in enumerate(shape) if i not in axis)
        new_strides = tuple(stride for i, stride in enumerate(strides) if i not in axis)

    else:
        # no axis specified: remove all singleton dimensions
        new_shape = tuple(dim for dim in shape if dim != 1)
        new_strides = tuple(stride for i, stride in enumerate(strides) if shape[i] != 1)

    # force source tile to shared memory
    tile_type.storage = "shared"

    output = tile(
        dtype=tile_type.dtype,
        shape=new_shape,
        strides=new_strides,
        layout=tile_type.layout,
        storage="shared",
        owner=False,
    )
    return output


def tile_squeeze_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    source_tile = arg_values["t"]

    return ((source_tile,), (return_type,))


add_builtin(
    "tile_squeeze",
    input_types={"t": tile(dtype=Any, shape=Tuple[int, ...]), "axis": Tuple[int, ...]},
    defaults={"axis": None},
    value_func=tile_squeeze_value_func,
    dispatch_func=tile_squeeze_dispatch_func,
    variadic=False,
    doc="""Return a squeezed view of a tile with the same data.

    :param t: Input tile to squeeze
    :param axis: A subset of the entries of length one in the shape (optional)
    :returns: The input tile but with all or a subset of the dimensions of length one removed.""",
    group="Tile Primitives",
    export=False,
)


def tile_reshape_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, ...])

    tile_type = arg_types["t"]

    # calculate total size of tile_type
    size = 1
    for s in tile_type.shape:
        size *= int(s)

    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    # check for -1 dimension and reformat
    if -1 in shape:
        idx = size
        denom = 1
        minus_one_count = 0
        for i, d in enumerate(shape):
            if d == -1:
                idx = i
                minus_one_count += 1
            else:
                denom *= d
        if minus_one_count > 1:
            raise RuntimeError("Cannot infer shape if more than one index is -1.")
        new_shape = list(shape)
        new_shape[idx] = int(size / denom)
        shape = tuple(new_shape)

    # calculate total size of new shape
    new_size = 1
    for s in shape:
        new_size *= int(s)

    if new_size != size:
        raise ValueError(f"New shape {shape} has total size {new_size} which does not match original size {size}")

    # compute new strides matching shape
    strides = []
    stride = 1
    for s in reversed(shape):
        strides.append(stride)
        stride *= s
    strides = tuple(reversed(strides))

    # force source tile to shared memory
    tile_type.storage = "shared"

    output = tile(
        dtype=tile_type.dtype, shape=shape, strides=strides, layout=tile_type.layout, storage="shared", owner=False
    )
    return output


def tile_reshape_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    tile = arg_values["t"]

    return ((tile,), (return_type,))


add_builtin(
    "tile_reshape",
    input_types={"t": tile(dtype=Any, shape=Tuple[int, ...]), "shape": Tuple[int, ...]},
    value_func=tile_reshape_value_func,
    dispatch_func=tile_reshape_dispatch_func,
    variadic=False,
    doc="""Return a reshaped view of a tile with the same data.

    :param t: Input tile to reshape
    :param shape: New shape for the tile
    :returns: A tile containing the same data as the input tile, but arranged in a new shape.""",
    group="Tile Primitives",
    export=False,
)


def tile_astype_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, ...])

    tile_type = arg_types["t"]
    dtype = arg_values["dtype"]

    return tile(dtype=dtype, shape=tile_type.shape)


def tile_astype_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    tile = arg_values["t"]

    return ((tile,), (return_type,))


add_builtin(
    "tile_astype",
    input_types={"t": tile(dtype=Scalar, shape=Tuple[int, ...]), "dtype": Scalar},
    value_func=tile_astype_value_func,
    dispatch_func=tile_astype_dispatch_func,
    variadic=False,
    doc="""Return a new tile with the same data as the input tile, but with a different data type.

    :param t: Input tile
    :param dtype: New data type for the tile
    :returns: A tile with the same data as the input tile, but with a different data type""",
    group="Tile Primitives",
    export=False,
)


def tile_assign_value_func(arg_types, arg_values):
    if arg_types is None:
        return None

    # force the destination tile to shared memory
    arg_types["dst"].storage = "shared"
    return None


def tile_assign_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    dst = args["dst"]
    src = args["src"]

    if "offset" in args:
        offset = extract_tuple(args["offset"])
    else:
        offset = (0,) * len(dst.type.shape)

    func_args = (dst, src, *offset)
    template_args = []

    return (func_args, template_args)


add_builtin(
    "tile_assign",
    input_types={
        "dst": tile(dtype=Any, shape=Tuple[int, ...]),
        "src": tile(dtype=Any, shape=Tuple[int, ...]),
        "offset": Tuple[int, ...],
    },
    value_func=tile_assign_value_func,
    dispatch_func=tile_assign_dispatch_func,
    defaults={"offset": None},
    doc="""Assign a tile to a subrange of a destination tile.

    :param dst: The destination tile to assign to
    :param src: The source tile to read values from
    :param offset: Offset in the destination tile to write to""",
    group="Tile Primitives",
    export=False,
)

# handles expressions like tile[i,j] = 1.0
add_builtin(
    "assign",
    input_types={"dst": tile(dtype=Any, shape=Tuple[int]), "i": int, "src": Any},
    value_func=tile_assign_value_func,
    group="Tile Primitives",
    export=False,
    hidden=True,
)

add_builtin(
    "assign",
    input_types={"dst": tile(dtype=Any, shape=Tuple[int, int]), "i": int, "j": int, "src": Any},
    value_func=tile_assign_value_func,
    group="Tile Primitives",
    export=False,
    hidden=True,
)

add_builtin(
    "assign",
    input_types={"dst": tile(dtype=Any, shape=Tuple[int, int, int]), "i": int, "j": int, "k": int, "src": Any},
    value_func=tile_assign_value_func,
    group="Tile Primitives",
    export=False,
    hidden=True,
)

add_builtin(
    "assign",
    input_types={
        "dst": tile(dtype=Any, shape=Tuple[int, int, int, int]),
        "i": int,
        "j": int,
        "k": int,
        "l": int,
        "src": Any,
    },
    value_func=tile_assign_value_func,
    group="Tile Primitives",
    export=False,
    hidden=True,
)


def tile_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple)

    if len(arg_types) > 2:
        raise TypeError(f"tile() takes 1 positional argument and 1 optional argument but {len(arg_types)} were given")

    preserve_type = arg_values["preserve_type"]

    if preserve_type:
        dtype = arg_types["x"]
        shape = (warp.codegen.options["block_dim"],)

        return tile(dtype=dtype, shape=shape)

    else:
        if type_is_vector(arg_types["x"]):
            dtype = arg_types["x"]._wp_scalar_type_
            length = arg_types["x"]._shape_[0]
            shape = (length, warp.codegen.options["block_dim"])
        elif type_is_quaternion(arg_types["x"]):
            dtype = arg_types["x"]._wp_scalar_type_
            shape = (4, warp.codegen.options["block_dim"])
        elif type_is_matrix(arg_types["x"]):
            dtype = arg_types["x"]._wp_scalar_type_
            rows = arg_types["x"]._shape_[0]
            cols = arg_types["x"]._shape_[1]
            shape = (rows, cols, warp.codegen.options["block_dim"])
        else:
            dtype = arg_types["x"]
            shape = (warp.codegen.options["block_dim"],)

        return tile(dtype=dtype, shape=shape)


def tile_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    x = arg_values["x"]
    preserve_type = get_arg_value(arg_values["preserve_type"])

    if preserve_type:
        dtype = x.type
        return ((x,), (dtype,))

    else:
        if type_is_vector(x.type):
            dtype = x.type._wp_scalar_type_
            length = x.type._shape_[0]
            return ((x,), (dtype, length))
        elif type_is_quaternion(x.type):
            dtype = x.type._wp_scalar_type_
            return ((x,), (dtype, 4))
        elif type_is_matrix(x.type):
            dtype = x.type._wp_scalar_type_
            rows = x.type._shape_[0]
            cols = x.type._shape_[1]
            return ((x,), (rows, cols, dtype))
        else:
            dtype = x.type
            return ((x,), (dtype,))


add_builtin(
    "tile",
    input_types={"x": Any, "preserve_type": bool},
    value_func=tile_value_func,
    dispatch_func=tile_dispatch_func,
    variadic=True,
    defaults={"preserve_type": False},
    doc="""Construct a new tile from per-thread kernel values.

    This function converts values computed using scalar kernel code to a tile representation for input into collective operations.

    * If the input value is a scalar, then the resulting tile has ``shape=(block_dim,)``
    * If the input value is a vector, then the resulting tile has ``shape=(length(vector), block_dim)``
    * If the input value is a vector, and ``preserve_type=True``, then the resulting tile has ``dtype=vector`` and ``shape=(block_dim,)``
    * If the input value is a matrix, then the resulting tile has ``shape=(rows, cols, block_dim)``
    * If the input value is a matrix, and ``preserve_type=True``, then the resulting tile has ``dtype=matrix`` and ``shape=(block_dim,)``

    :param x: A per-thread local value, e.g. scalar, vector, or matrix.
    :param preserve_type: If true, the tile will have the same data type as the input value.
    :returns: If ``preserve_type=True``, a tile of type ``x.type`` of length ``block_dim``. Otherwise, an N-dimensional tile such that the first N-1 dimensions match the shape of ``x`` and the final dimension is of size ``block_dim``.

    This example shows how to create a linear sequence from thread variables:

    .. code-block:: python

        @wp.kernel
        def compute():
            i = wp.tid()
            t = wp.tile(i*2)
            print(t)

        wp.launch(compute, dim=16, inputs=[], block_dim=16)

    Prints:

    .. code-block:: text

        [0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30] = tile(shape=(16), storage=register)

    """,
    group="Tile Primitives",
    export=False,
    hidden=True,
)


def untile_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return Any

    if len(arg_types) != 1:
        raise TypeError(f"untile() takes exactly 1 positional argument but {len(arg_types)} were given")

    t = arg_types["a"]

    if not is_tile(t):
        raise TypeError(f"untile() argument must be a tile, got {t!r}")

    if t.shape[-1] != warp.codegen.options["block_dim"]:
        raise ValueError(
            f"untile() argument last dimension {t.shape[-1]} does not match the expected block width {warp.codegen.options['block_dim']}"
        )

    if len(t.shape) == 1:
        return t.dtype
    elif len(t.shape) == 2:
        return warp.types.vector(t.shape[0], t.dtype)
    elif len(t.shape) == 3:
        return warp.types.matrix((t.shape[0], t.shape[1]), t.dtype)
    else:
        raise ValueError(f"untile() argument must have a positive size in dimension 0, but got {t.shape[0]}")


add_builtin(
    "untile",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...])},
    value_func=untile_value_func,
    variadic=True,
    doc="""Convert a tile back to per-thread values.

    This function converts a block-wide tile back to per-thread values.

    * If the input tile is 1D, then the resulting value will be a per-thread scalar
    * If the input tile is 2D, then the resulting value will be a per-thread vector of length M

    :param a: A tile with dimensions ``shape=(M, block_dim)``
    :returns: A single value per-thread with the same data type as the tile

    This example shows how to create a linear sequence from thread variables:

    .. code-block:: python

        @wp.kernel
        def compute():
            i = wp.tid()

            # create block-wide tile
            t = wp.tile(i)*2

            # convert back to per-thread values
            s = wp.untile(t)

            print(s)

        wp.launch(compute, dim=16, inputs=[], block_dim=16)

    Prints:

    .. code-block:: text

        0
        2
        4
        6
        8
        ...
    """,
    group="Tile Primitives",
    export=False,
)


def tile_extract_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return Any

    # force the input tile to shared memory
    arg_types["a"].storage = "shared"

    return arg_types["a"].dtype


add_builtin(
    "tile_extract",
    input_types={"a": tile(dtype=Any, shape=Tuple[int]), "i": int},
    value_func=tile_extract_value_func,
    variadic=False,
    doc="""Extract a single element from the tile.

    This function will extract an element from the tile and broadcast its value to all threads in the block.

    Note that this may incur additional synchronization if the source tile is a register tile.

    :param a: Tile to extract the element from
    :param i: Coordinate of element on first dimension
    :returns: The value of the element at the specified tile location with the same data type as the input tile""",
    group="Tile Primitives",
    hidden=True,
    export=False,
)

add_builtin(
    "tile_extract",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, int]), "i": int, "j": int},
    value_func=tile_extract_value_func,
    variadic=False,
    doc="""Extract a single element from the tile.

    This function will extract an element from the tile and broadcast its value to all threads in the block.

    Note that this may incur additional synchronization if the source tile is a register tile.

    :param a: Tile to extract the element from
    :param i: Coordinate of element on first dimension
    :param j: Coordinate of element on the second dimension
    :returns: The value of the element at the specified tile location with the same data type as the input tile""",
    group="Tile Primitives",
    hidden=True,
    export=False,
)

add_builtin(
    "tile_extract",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, int, int]), "i": int, "j": int, "k": int},
    value_func=tile_extract_value_func,
    variadic=False,
    doc="""Extract a single element from the tile.

    This function will extract an element from the tile and broadcast its value to all threads in the block.

    Note that this may incur additional synchronization if the source tile is a register tile.

    :param a: Tile to extract the element from
    :param i: Coordinate of element on first dimension
    :param j: Coordinate of element on the second dimension
    :param k: Coordinate of element on the third dimension
    :returns: The value of the element at the specified tile location with the same data type as the input tile""",
    group="Tile Primitives",
    hidden=True,
    export=False,
)

add_builtin(
    "tile_extract",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, int, int, int]), "i": int, "j": int, "k": int, "l": int},
    value_func=tile_extract_value_func,
    variadic=False,
    doc="""Extract a single element from the tile.

    This function will extract an element from the tile and broadcast its value to all threads in the block.

    Note that this may incur additional synchronization if the source tile is a register tile.

    :param a: Tile to extract the element from
    :param i: Coordinate of element on first dimension
    :param j: Coordinate of element on the second dimension
    :param k: Coordinate of element on the third dimension
    :param l: Coordinate of element on the fourth dimension
    :returns: The value of the element at the specified tile location, with the same data type as the input tile""",
    group="Tile Primitives",
    hidden=True,
    export=False,
)


def tile_inplace_value_func(arg_types, arg_values):
    if not types_equal(arg_types["a"].dtype, arg_types["value"]):
        raise TypeError(
            f"'value' must have the same dtype as target tile for inplace ops, got {arg_types['a'].dtype} and {arg_types['value']}"
        )

    # force the input tile to shared memory
    # as inplace addition/subtraction relies on shared memory atomics
    arg_types["a"].storage = "shared"

    return None


add_builtin(
    "tile_add_inplace",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "i": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_add_inplace",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "i": int, "j": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_add_inplace",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "i": int, "j": int, "k": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_add_inplace",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "i": int, "j": int, "k": int, "l": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)

add_builtin(
    "tile_sub_inplace",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "i": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_sub_inplace",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "i": int, "j": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_sub_inplace",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "i": int, "j": int, "k": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_sub_inplace",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "i": int, "j": int, "k": int, "l": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)


def tile_transpose_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, int])

    if len(arg_types) != 1:
        raise TypeError(f"tile_transpose() takes exactly 1 positional argument but {len(arg_types)} were given")

    t = arg_types["a"]

    if not is_tile(t):
        raise TypeError(f"tile_transpose() argument must be a tile, got {t!r}")

    layout = None

    # flip layout
    if t.layout == "rowmajor":
        layout = "colmajor"
    elif t.layout == "colmajor":
        layout = "rowmajor"

    # force the input tile to shared memory
    t.storage = "shared"

    return tile(
        dtype=t.dtype,
        shape=t.shape[::-1],
        storage=t.storage,
        strides=t.strides[::-1],
        layout=layout,
        owner=False,
    )


add_builtin(
    "tile_transpose",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, int])},
    value_func=tile_transpose_value_func,
    variadic=True,
    doc="""Transpose a tile.

    For shared memory tiles, this operation will alias the input tile.
    Register tiles will first be transferred to shared memory before transposition.

    :param a: Tile to transpose with ``shape=(M,N)``
    :returns: Tile with ``shape=(N,M)``""",
    group="Tile Primitives",
    export=False,
)


def tile_broadcast_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, ...])

    t = arg_types["a"]

    # target shape and strides
    target_shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in target_shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    target_strides = [0] * len(target_shape)

    offset = len(target_shape) - len(t.shape)

    # compute target strides
    for i in reversed(range(len(target_shape))):
        j = i - offset

        if j < 0:
            target_strides[i] = 0
        else:
            # try to broadcast each dimension
            if t.shape[j] == 1:
                target_strides[i] = 0
            elif t.shape[j] == target_shape[i]:
                target_strides[i] = t.strides[j]
            else:
                raise ValueError(
                    f"tile_broadcast() cannot broadcast dimension {t.shape[j]} into {target_shape[i]} at index {i}"
                )

    # force the input tile to shared memory
    t.storage = "shared"

    return tile(dtype=t.dtype, shape=target_shape, storage=t.storage, strides=target_strides, owner=False)


def tile_broadcast_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    tile = arg_values["a"]

    assert len(return_type.shape) == len(return_type.strides)
    assert 1 <= len(return_type.shape) <= 4
    template_args = [*return_type.shape, *return_type.strides]

    return ((tile,), template_args)


add_builtin(
    "tile_broadcast",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "shape": Tuple[int, ...]},
    value_func=tile_broadcast_value_func,
    dispatch_func=tile_broadcast_dispatch_func,
    variadic=False,
    doc="""Broadcast a tile.

    Broadcasts the input tile ``a`` to the destination shape.
    Broadcasting follows NumPy broadcast rules.

    :param a: Tile to broadcast
    :param shape: The shape to broadcast to
    :returns: Tile with broadcast shape""",
    group="Tile Primitives",
    export=False,
)


def tile_sum_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Scalar, shape=(1,))

    if len(arg_types) != 1:
        raise TypeError(f"tile_sum() takes exactly 1 positional argument but {len(arg_types)} were given")

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_sum() argument must be a tile, got {a!r}")

    return tile(dtype=a.dtype, shape=(1,))


add_builtin(
    "tile_sum",
    input_types={"a": tile(dtype=Scalar, shape=Tuple[int, ...])},
    value_func=tile_sum_value_func,
    variadic=True,
    doc="""Cooperatively compute the sum of the tile elements using all threads in the block.

    :param a: The tile to compute the sum of
    :returns: A single-element tile holding the sum

    Example:

    .. code-block:: python

        @wp.kernel
        def compute():

            t = wp.tile_ones(dtype=float, shape=(16, 16))
            s = wp.tile_sum(t)

            print(s)

        wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

    Prints:

    .. code-block:: text

        [256] = tile(shape=(1), storage=register)

    """,
    group="Tile Primitives",
    export=False,
)


def tile_sort_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return None

    if len(arg_types) != 2:
        raise TypeError(
            f"tile_sort() takes exactly 2 positional arguments (keys and values) but {len(arg_types)} were given"
        )

    a = arg_types["keys"]
    b = arg_types["values"]

    if not is_tile(a):
        raise TypeError(f"First tile_sort() argument must be a tile, got {a!r}")

    if not is_tile(b):
        raise TypeError(f"Second tile_sort() argument must be a tile, got {b!r}")

    if not (a.dtype is warp.float32 or a.dtype is warp.int32 or a.dtype is warp.uint32):
        raise TypeError(f"First tile_sort() argument must be a tile of type float or int, got {a.dtype}")

    # set the storage type to the inputs to shared
    a.storage = "shared"
    b.storage = "shared"

    if len(a.shape) != len(b.shape):
        raise ValueError(
            f"tile_sort() shapes must have the same number of dimensions, got {len(a.shape)} and {len(b.shape)}"
        )

    for i in range(len(a.shape)):
        if a.shape[i] != b.shape[i]:
            raise ValueError(f"tile_sort() shapes do not match on dimension {i}, got {a.shape} and {b.shape}")

    return None


add_builtin(
    "tile_sort",
    input_types={"keys": tile(dtype=Any, shape=Tuple[int]), "values": tile(dtype=Any, shape=Tuple[int])},
    value_func=tile_sort_value_func,
    variadic=True,
    doc="""Cooperatively sort the elements of two tiles in ascending order based on the keys, using all threads in the block.

    :param keys: Keys to sort by. Supported key types: :class:`float32`, :class:`int32`, :class:`uint32`. Must be in shared memory.
    :param values: Values to sort along with keys. No type restrictions. Must be in shared memory.
    :returns: No return value. Sorts both tiles in-place.

    Example:

    .. code-block:: python

        @wp.kernel
        def compute():

            keys = wp.tile_arange(32, 0, -1, dtype=int, storage="shared")
            values = wp.tile_arange(0, 32, 1, dtype=int, storage="shared")
            wp.tile_sort(keys, values)

            print(keys)
            print(values)


        wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

    Prints:

    .. code-block:: text

        [1, 2, ..., 32] = tile(shape=(32), storage=shared)
        [31, 30, 29, ..., 0] = tile(shape=(32), storage=shared)

    """,
    group="Tile Primitives",
    export=False,
)


def tile_min_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Scalar, shape=(1,))

    if len(arg_types) != 1:
        raise TypeError(f"tile_min() takes exactly 1 positional argument but {len(arg_types)} were given")

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_min() argument must be a tile, got {a!r}")

    return tile(dtype=a.dtype, shape=(1,))


add_builtin(
    "tile_min",
    input_types={"a": tile(dtype=Scalar, shape=Tuple[int, ...])},
    value_func=tile_min_value_func,
    variadic=True,
    doc="""Cooperatively compute the minimum of the tile elements using all threads in the block.

    :param a: The tile to compute the minimum of
    :returns: A single-element tile holding the minimum value

    Example:

    .. code-block:: python

        @wp.kernel
        def compute():

            t = wp.tile_arange(64, 128)
            s = wp.tile_min(t)

            print(s)


        wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

    Prints:

    .. code-block:: text

        [64] = tile(shape=(1), storage=register)

    """,
    group="Tile Primitives",
    export=False,
)


def tile_argmin_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Int, shape=(1,))

    if len(arg_types) != 1:
        raise TypeError(f"tile_argmin() takes exactly 1 positional argument but {len(arg_types)} were given")

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_argmin() argument must be a tile, got {a!r}")

    return tile(dtype=warp.int32, shape=(1,))


add_builtin(
    "tile_argmin",
    input_types={"a": tile(dtype=Scalar, shape=Tuple[int, ...])},
    value_func=tile_argmin_value_func,
    variadic=True,
    doc="""Cooperatively compute the index of the minimum element in the tile using all threads in the block.

    :param a: The tile to compute the argmin from
    :returns: A single-element tile holding the index of the minimum value

    Example:

    .. code-block:: python

        @wp.kernel
        def compute():

            t = wp.tile_arange(64, 128)
            s = wp.tile_argmin(t)

            print(s)


        wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

    Prints:

    .. code-block:: text

        [0] = tile(shape=(1), storage=register)

    """,
    group="Tile Primitives",
    export=False,
)


def tile_max_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Scalar, shape=(1,))

    if len(arg_types) != 1:
        raise TypeError(f"tile_max() takes exactly 1 positional argument but {len(arg_types)} were given")

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_max() argument must be a tile, got {a!r}")

    return tile(dtype=a.dtype, shape=(1,))


add_builtin(
    "tile_max",
    input_types={"a": tile(dtype=Scalar, shape=Tuple[int, ...])},
    value_func=tile_max_value_func,
    variadic=False,
    doc="""Cooperatively compute the maximum of the tile elements using all threads in the block.

    :param a: The tile to compute the maximum from
    :returns: A single-element tile holding the maximum value

    Example:

    .. code-block:: python

        @wp.kernel
        def compute():

            t = wp.tile_arange(64, 128)
            s = wp.tile_max(t)

            print(s)

        wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

    Prints:

    .. code-block:: text

        [127] = tile(shape=(1), storage=register)

    """,
    group="Tile Primitives",
    export=False,
)


def tile_argmax_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Int, shape=(1,))

    if len(arg_types) != 1:
        raise TypeError(f"tile_argmax() takes exactly 1 positional argument but {len(arg_types)} were given")

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_argmax() argument must be a tile, got {a!r}")

    return tile(dtype=warp.int32, shape=(1,))


add_builtin(
    "tile_argmax",
    input_types={"a": tile(dtype=Scalar, shape=Tuple[int, ...])},
    value_func=tile_argmax_value_func,
    variadic=False,
    doc="""Cooperatively compute the index of the maximum element in the tile using all threads in the block.

    :param a: The tile to compute the argmax from
    :returns: A single-element tile holding the index of the maximum value

    Example:

    .. code-block:: python

        @wp.kernel
        def compute():

            t = wp.tile_arange(64, 128)
            s = wp.tile_argmax(t)

            print(s)

        wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

    Prints:

    .. code-block:: text

        [63] = tile(shape=(1), storage=register)

    """,
    group="Tile Primitives",
    export=False,
)


# does type propagation for load()
def tile_reduce_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Scalar, shape=(1,))

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_reduce() 'a' argument must be a tile, got {a!r}")

    return tile(dtype=a.dtype, shape=(1,))


def tile_reduce_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (args["op"], *args["args"])
    template_args = ()
    return (func_args, template_args)


add_builtin(
    "tile_reduce",
    input_types={"op": Callable, "a": tile(dtype=Scalar, shape=Tuple[int, ...])},
    value_func=tile_reduce_value_func,
    native_func="tile_reduce",
    doc="""Apply a custom reduction operator across the tile.

    This function cooperatively performs a reduction using the provided operator across the tile.

    :param op: A callable function that accepts two arguments and returns one argument, may be a user function or builtin
    :param a: The input tile, the operator (or one of its overloads) must be able to accept the tile's data type
    :returns: A single-element tile with the same data type as the input tile.

    Example:

    .. code-block:: python

        @wp.kernel
        def factorial():

            t = wp.tile_arange(1, 10, dtype=int)
            s = wp.tile_reduce(wp.mul, t)

            print(s)

        wp.launch_tiled(factorial, dim=[1], inputs=[], block_dim=16)

    Prints:

    .. code-block:: text

        [362880] = tile(shape=(1), storage=register)
    """,
    group="Tile Primitives",
    export=False,
)


def tile_scan_inclusive_value_func(arg_types, arg_values):
    # Return type is the same as input type
    if arg_types is None:
        return tile(dtype=Scalar, shape=Tuple[int, ...])

    if len(arg_types) != 1:
        raise TypeError(f"tile_scan_inclusive() takes exactly 1 positional argument but {len(arg_types)} were given")

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_scan_inclusive() argument must be a tile, got {a!r}")

    # Only allow float32, int32, or uint32 for scan (like tile_sort)
    if not (a.dtype is warp.float32 or a.dtype is warp.int32 or a.dtype is warp.uint32):
        raise TypeError(
            f"tile_scan_inclusive() argument must be a tile of type float32, int32, or uint32, got {a.dtype}"
        )

    return tile(dtype=a.dtype, shape=a.shape)


def tile_scan_inclusive_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (args["a"],)
    template_args = ()
    return (func_args, template_args)


add_builtin(
    "tile_scan_inclusive",
    input_types={"a": tile(dtype=Scalar, shape=Tuple[int, ...])},
    value_func=tile_scan_inclusive_value_func,
    native_func="tile_scan_inclusive",
    doc="""Inclusive scan (prefix sum) across the tile.

    This function cooperatively performs an inclusive scan (cumulative sum) across the tile.

    :param a: The input tile. Must be a tile of type float32, int32, or uint32.
    :returns: A new tile containing the inclusive scan result.

    Example:

    .. code-block:: python

        @wp.kernel
        def scan_example():
            t = wp.tile_arange(1, 5, dtype=int)
            s = wp.tile_scan_inclusive(t)
            print(s)

        wp.launch_tiled(scan_example, dim=[1], inputs=[], block_dim=16)

    Prints:

    .. code-block:: text

        [1, 3, 6, 10] = tile(shape=(4), storage=register)
    """,
    group="Tile Primitives",
    export=False,
)


def tile_scan_exclusive_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Scalar, shape=Tuple[int, ...])

    if len(arg_types) != 1:
        raise TypeError(f"tile_scan_exclusive() takes exactly 1 positional argument but {len(arg_types)} were given")

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_scan_exclusive() argument must be a tile, got {a!r}")

    # Only allow float32, int32, or uint32 for scan (like tile_sort)
    if not (a.dtype is warp.float32 or a.dtype is warp.int32 or a.dtype is warp.uint32):
        raise TypeError(
            f"tile_scan_exclusive() argument must be a tile of type float32, int32, or uint32, got {a.dtype}"
        )

    return tile(dtype=a.dtype, shape=a.shape)


def tile_scan_exclusive_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (args["a"],)
    template_args = ()
    return (func_args, template_args)


add_builtin(
    "tile_scan_exclusive",
    input_types={"a": tile(dtype=Scalar, shape=Tuple[int, ...])},
    value_func=tile_scan_exclusive_value_func,
    native_func="tile_scan_exclusive",
    doc="""Exclusive scan (prefix sum) across the tile.

    This function cooperatively performs an exclusive scan (cumulative sum) across the tile.

    :param a: The input tile. Must be a tile of type float32, int32, or uint32.
    :returns: A new tile containing the exclusive scan result.

    Example:

    .. code-block:: python

        @wp.kernel
        def scan_example():
            t = wp.tile_arange(1, 5, dtype=int)
            s = wp.tile_scan_exclusive(t)
            print(s)

        wp.launch_tiled(scan_example, dim=[1], inputs=[], block_dim=16)

    Prints:

    .. code-block:: text

        [0, 1, 3, 6] = tile(shape=(4), storage=register)
    """,
    group="Tile Primitives",
    export=False,
)


# maps


# does type propagation for load()
def tile_unary_map_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Scalar, shape=Tuple[int, ...])

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_map() 'a' argument must be a tile, got {a!r}")

    if "op" in arg_values:
        op = arg_values["op"]
        try:
            overload = op.get_overload([a.dtype], {})
        except KeyError as exc:
            raise RuntimeError(f"No overload of {op} found for tile element type {type_repr(a.dtype)}") from exc

        # build the right overload on demand
        if overload.value_func is None:
            overload.build(None)

        value_type = overload.value_func(None, None)

        if not type_is_scalar(value_type) and not type_is_vector(value_type) and not type_is_matrix(value_type):
            raise TypeError(f"Operator {op} returns unsupported type {type_repr(value_type)} for a tile element")

        return tile(dtype=value_type, shape=a.shape)

    else:
        return tile(dtype=a.dtype, shape=a.shape)


def tile_unary_map_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    op = arg_values["op"]
    tile_a = arg_values["a"]

    overload = op.get_overload([tile_a.type.dtype], {})

    # necessary, in case return type is different from input tile types
    tile_r = Var(label=None, type=return_type)

    return ((overload, tile_a, tile_r), ())


add_builtin(
    "tile_map",
    input_types={"op": Callable, "a": tile(dtype=Scalar, shape=Tuple[int, ...])},
    value_func=tile_unary_map_value_func,
    dispatch_func=tile_unary_map_dispatch_func,
    # variadic=True,
    native_func="tile_unary_map",
    doc="""Apply a unary function onto the tile.

    This function cooperatively applies a unary function to each element of the tile using all threads in the block.

    :param op: A callable function that accepts one argument and returns one argument, may be a user function or builtin
    :param a: The input tile, the operator (or one of its overloads) must be able to accept the tile's data type
    :returns: A tile with the same dimensions as the input tile. Its datatype is specified by the return type of op

    Example:

    .. code-block:: python

        @wp.kernel
        def compute():

            t = wp.tile_arange(0.0, 1.0, 0.1, dtype=float)
            s = wp.tile_map(wp.sin, t)

            print(s)

        wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=16)

    Prints:

    .. code-block:: text

        [0 0.0998334 0.198669 0.29552 0.389418 0.479426 0.564642 0.644218 0.717356 0.783327] = tile(shape=(10), storage=register)
    """,
    group="Tile Primitives",
    export=False,
)


def tile_binary_map_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Scalar, shape=Tuple[int, ...])

    a = arg_types["a"]
    b = arg_types["b"]

    # check all args are tiles
    if not is_tile(a):
        raise TypeError(f"tile_map() 'a' argument must be a tile, got {a!r}")

    if not is_tile(b):
        raise TypeError(f"tile_map() 'b' argument must be a tile, got {b!r}")

    if len(a.shape) != len(b.shape):
        raise ValueError(
            f"tile_map() shapes must have the same number of dimensions, got {len(a.shape)} and {len(b.shape)}"
        )

    for i in range(len(a.shape)):
        if a.shape[i] != b.shape[i]:
            raise ValueError(f"tile_map() shapes do not match on dimension {i}, got {a.shape} and {b.shape}")

    if "op" in arg_values:
        op = arg_values["op"]
        try:
            overload = op.get_overload([a.dtype, b.dtype], {})
        except KeyError as exc:
            raise RuntimeError(
                f"No overload of {op} found for tile element types {type_repr(a.dtype)}, {type_repr(b.dtype)}"
            ) from exc

        # build the right overload on demand
        if overload.value_func is None:
            overload.build(None)

        value_type = overload.value_func(None, None)

        if not type_is_scalar(value_type) and not type_is_vector(value_type) and not type_is_matrix(value_type):
            raise TypeError(f"Operator {op} returns unsupported type {type_repr(value_type)} for a tile element")

        return tile(dtype=value_type, shape=a.shape)

    else:
        # ensure types equal
        if not types_equal(a.dtype, b.dtype):
            raise TypeError(
                f"tile_map() arguments must have the same dtype for this operation, got {a.dtype} and {b.dtype}"
            )

        return tile(dtype=a.dtype, shape=a.shape)


def tile_binary_map_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    op = arg_values["op"]
    tile_a = arg_values["a"]
    tile_b = arg_values["b"]

    overload = op.get_overload([tile_a.type.dtype, tile_b.type.dtype], {})

    # necessary, in case return type is different from input tile types
    tile_r = Var(label=None, type=return_type)

    return ((overload, tile_a, tile_b, tile_r), ())


add_builtin(
    "tile_map",
    input_types={
        "op": Callable,
        "a": tile(dtype=Scalar, shape=Tuple[int, ...]),
        "b": tile(dtype=Scalar, shape=Tuple[int, ...]),
    },
    value_func=tile_binary_map_value_func,
    dispatch_func=tile_binary_map_dispatch_func,
    # variadic=True,
    native_func="tile_binary_map",
    doc="""Apply a binary function onto the tile.

    This function cooperatively applies a binary function to each element of the tiles using all threads in the block.
    Both input tiles must have the same dimensions, and if using a builtin op, the same datatypes.

    :param op: A callable function that accepts two arguments and returns one argument, all of the same type, may be a user function or builtin
    :param a: The first input tile, the operator (or one of its overloads) must be able to accept the tile's dtype
    :param b: The second input tile, the operator (or one of its overloads) must be able to accept the tile's dtype
    :returns: A tile with the same dimensions as the input tiles. Its datatype is specified by the return type of op

    Example:

    .. code-block:: python

        @wp.kernel
        def compute():

            a = wp.tile_arange(0.0, 1.0, 0.1, dtype=float)
            b = wp.tile_ones(shape=10, dtype=float)

            s = wp.tile_map(wp.add, a, b)

            print(s)

        wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=16)

    Prints:

    .. code-block:: text

        [1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9] = tile(shape=(10), storage=register)""",
    group="Tile Primitives",
    export=False,
)


# ---------------------------------
# Linear Algebra

add_builtin(
    "dense_gemm",
    input_types={
        "m": int,
        "n": int,
        "p": int,
        "t1": int,
        "t2": int,
        "A": array(dtype=float),
        "B": array(dtype=float),
        "C": array(dtype=float),
    },
    value_type=None,
    doc="",
    group="Utility",
    hidden=True,
)

add_builtin(
    "dense_gemm_batched",
    input_types={
        "m": array(dtype=int),
        "n": array(dtype=int),
        "p": array(dtype=int),
        "t1": int,
        "t2": int,
        "A_start": array(dtype=int),
        "B_start": array(dtype=int),
        "C_start": array(dtype=int),
        "A": array(dtype=float),
        "B": array(dtype=float),
        "C": array(dtype=float),
    },
    value_type=None,
    doc="",
    group="Utility",
    hidden=True,
)


add_builtin(
    "dense_chol",
    input_types={"n": int, "A": array(dtype=float), "regularization": float, "L": array(dtype=float)},
    value_type=None,
    doc="WIP",
    group="Utility",
    hidden=True,
)

add_builtin(
    "dense_chol_batched",
    input_types={
        "A_start": array(dtype=int),
        "A_dim": array(dtype=int),
        "A": array(dtype=float),
        "regularization": float,
        "L": array(dtype=float),
    },
    value_type=None,
    doc="WIP",
    group="Utility",
    hidden=True,
)

add_builtin(
    "dense_subs",
    input_types={"n": int, "L": array(dtype=float), "b": array(dtype=float), "x": array(dtype=float)},
    value_type=None,
    doc="WIP",
    group="Utility",
    hidden=True,
)

add_builtin(
    "dense_solve",
    input_types={
        "n": int,
        "A": array(dtype=float),
        "L": array(dtype=float),
        "b": array(dtype=float),
        "x": array(dtype=float),
    },
    value_type=None,
    doc="WIP",
    group="Utility",
    hidden=True,
)

add_builtin(
    "dense_solve_batched",
    input_types={
        "b_start": array(dtype=int),
        "A_start": array(dtype=int),
        "A_dim": array(dtype=int),
        "A": array(dtype=float),
        "L": array(dtype=float),
        "b": array(dtype=float),
        "x": array(dtype=float),
    },
    value_type=None,
    doc="WIP",
    group="Utility",
    hidden=True,
)


# ---------------------------------
# Geometry

add_builtin(
    "bvh_query_aabb",
    input_types={"id": uint64, "low": vec3, "high": vec3},
    value_type=BvhQuery,
    group="Geometry",
    doc="""Construct an axis-aligned bounding box query against a BVH object.

    This query can be used to iterate over all bounds inside a BVH.

    :param id: The BVH identifier
    :param low: The lower bound of the bounding box in BVH space
    :param high: The upper bound of the bounding box in BVH space""",
    export=False,
)

add_builtin(
    "bvh_query_ray",
    input_types={"id": uint64, "start": vec3, "dir": vec3},
    value_type=BvhQuery,
    group="Geometry",
    doc="""Construct a ray query against a BVH object.

    This query can be used to iterate over all bounds that intersect the ray.

    :param id: The BVH identifier
    :param start: The start of the ray in BVH space
    :param dir: The direction of the ray in BVH space""",
    export=False,
)

add_builtin(
    "bvh_query_next",
    input_types={"query": BvhQuery, "index": int},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Move to the next bound returned by the query.
    The index of the current bound is stored in ``index``, returns ``False`` if there are no more overlapping bound.""",
    export=False,
)

add_builtin(
    "mesh_query_point",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
        "inside": float,
        "face": int,
        "bary_u": float,
        "bary_v": float,
    },
    value_type=builtins.bool,
    group="Geometry",
    doc="""Computes the closest point on the :class:`Mesh` with identifier ``id`` to the given ``point`` in space. Returns ``True`` if a point < ``max_dist`` is found.

    Identifies the sign of the distance using additional ray-casts to determine if the point is inside or outside.
    This method is relatively robust, but does increase computational cost.
    See below for additional sign determination methods.

    :param id: The mesh identifier
    :param point: The point in space to query
    :param max_dist: Mesh faces above this distance will not be considered by the query
    :param inside: Returns a value < 0 if query point is inside the mesh, >=0 otherwise.
                   Note that mesh must be watertight for this to be robust
    :param face: Returns the index of the closest face
    :param bary_u: Returns the barycentric u coordinate of the closest point
    :param bary_v: Returns the barycentric v coordinate of the closest point""",
    export=False,
    hidden=True,
)

add_builtin(
    "mesh_query_point",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
    },
    value_type=MeshQueryPoint,
    group="Geometry",
    doc="""Computes the closest point on the :class:`Mesh` with identifier ``id`` to the given ``point`` in space.

    Identifies the sign of the distance using additional ray-casts to determine if the point is inside or outside.
    This method is relatively robust, but does increase computational cost.
    See below for additional sign determination methods.

    :param id: The mesh identifier
    :param point: The point in space to query
    :param max_dist: Mesh faces above this distance will not be considered by the query""",
    require_original_output_arg=True,
    export=False,
)

add_builtin(
    "mesh_query_point_no_sign",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
        "face": int,
        "bary_u": float,
        "bary_v": float,
    },
    value_type=builtins.bool,
    group="Geometry",
    doc="""Computes the closest point on the :class:`Mesh` with identifier ``id`` to the given ``point`` in space. Returns ``True`` if a point < ``max_dist`` is found.

    This method does not compute the sign of the point (inside/outside) which makes it faster than other point query methods.

    :param id: The mesh identifier
    :param point: The point in space to query
    :param max_dist: Mesh faces above this distance will not be considered by the query
    :param face: Returns the index of the closest face
    :param bary_u: Returns the barycentric u coordinate of the closest point
    :param bary_v: Returns the barycentric v coordinate of the closest point""",
    export=False,
    hidden=True,
)

add_builtin(
    "mesh_query_point_no_sign",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
    },
    value_type=MeshQueryPoint,
    group="Geometry",
    doc="""Computes the closest point on the :class:`Mesh` with identifier ``id`` to the given ``point`` in space.

    This method does not compute the sign of the point (inside/outside) which makes it faster than other point query methods.

    :param id: The mesh identifier
    :param point: The point in space to query
    :param max_dist: Mesh faces above this distance will not be considered by the query""",
    require_original_output_arg=True,
    export=False,
)

add_builtin(
    "mesh_query_furthest_point_no_sign",
    input_types={
        "id": uint64,
        "point": vec3,
        "min_dist": float,
        "face": int,
        "bary_u": float,
        "bary_v": float,
    },
    value_type=builtins.bool,
    group="Geometry",
    doc="""Computes the furthest point on the mesh with identifier `id` to the given point in space. Returns ``True`` if a point > ``min_dist`` is found.

    This method does not compute the sign of the point (inside/outside).

    :param id: The mesh identifier
    :param point: The point in space to query
    :param min_dist: Mesh faces below this distance will not be considered by the query
    :param face: Returns the index of the furthest face
    :param bary_u: Returns the barycentric u coordinate of the furthest point
    :param bary_v: Returns the barycentric v coordinate of the furthest point""",
    export=False,
    hidden=True,
)

add_builtin(
    "mesh_query_furthest_point_no_sign",
    input_types={
        "id": uint64,
        "point": vec3,
        "min_dist": float,
    },
    value_type=MeshQueryPoint,
    group="Geometry",
    doc="""Computes the furthest point on the mesh with identifier `id` to the given point in space.

    This method does not compute the sign of the point (inside/outside).

    :param id: The mesh identifier
    :param point: The point in space to query
    :param min_dist: Mesh faces below this distance will not be considered by the query""",
    require_original_output_arg=True,
    export=False,
)

add_builtin(
    "mesh_query_point_sign_normal",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
        "inside": float,
        "face": int,
        "bary_u": float,
        "bary_v": float,
        "epsilon": float,
    },
    defaults={"epsilon": 1.0e-3},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Computes the closest point on the :class:`Mesh` with identifier ``id`` to the given ``point`` in space. Returns ``True`` if a point < ``max_dist`` is found.

    Identifies the sign of the distance (inside/outside) using the angle-weighted pseudo normal.
    This approach to sign determination is robust for well conditioned meshes that are watertight and non-self intersecting.
    It is also comparatively fast to compute.

    :param id: The mesh identifier
    :param point: The point in space to query
    :param max_dist: Mesh faces above this distance will not be considered by the query
    :param inside: Returns a value < 0 if query point is inside the mesh, >=0 otherwise.
                   Note that mesh must be watertight for this to be robust
    :param face: Returns the index of the closest face
    :param bary_u: Returns the barycentric u coordinate of the closest point
    :param bary_v: Returns the barycentric v coordinate of the closest point
    :param epsilon: Epsilon treating distance values as equal, when locating the minimum distance vertex/face/edge, as a
                    fraction of the average edge length, also for treating closest point as being on edge/vertex default 1e-3""",
    export=False,
    hidden=True,
)

add_builtin(
    "mesh_query_point_sign_normal",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
        "epsilon": float,
    },
    defaults={"epsilon": 1.0e-3},
    value_type=MeshQueryPoint,
    group="Geometry",
    doc="""Computes the closest point on the :class:`Mesh` with identifier ``id`` to the given ``point`` in space.

    Identifies the sign of the distance (inside/outside) using the angle-weighted pseudo normal.
    This approach to sign determination is robust for well conditioned meshes that are watertight and non-self intersecting.
    It is also comparatively fast to compute.

    :param id: The mesh identifier
    :param point: The point in space to query
    :param max_dist: Mesh faces above this distance will not be considered by the query
    :param epsilon: Epsilon treating distance values as equal, when locating the minimum distance vertex/face/edge, as a
                    fraction of the average edge length, also for treating closest point as being on edge/vertex default 1e-3""",
    require_original_output_arg=True,
    export=False,
)

add_builtin(
    "mesh_query_point_sign_winding_number",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
        "inside": float,
        "face": int,
        "bary_u": float,
        "bary_v": float,
        "accuracy": float,
        "threshold": float,
    },
    defaults={"accuracy": 2.0, "threshold": 0.5},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Computes the closest point on the :class:`Mesh` with identifier ``id`` to the given point in space. Returns ``True`` if a point < ``max_dist`` is found.

    Identifies the sign using the winding number of the mesh relative to the query point. This method of sign determination is robust for poorly conditioned meshes
    and provides a smooth approximation to sign even when the mesh is not watertight. This method is the most robust and accurate of the sign determination meshes
    but also the most expensive.

    .. note:: The :class:`Mesh` object must be constructed with ``support_winding_number=True`` for this method to return correct results.

    :param id: The mesh identifier
    :param point: The point in space to query
    :param max_dist: Mesh faces above this distance will not be considered by the query
    :param inside: Returns a value < 0 if query point is inside the mesh, >=0 otherwise.
                   Note that mesh must be watertight for this to be robust
    :param face: Returns the index of the closest face
    :param bary_u: Returns the barycentric u coordinate of the closest point
    :param bary_v: Returns the barycentric v coordinate of the closest point
    :param accuracy: Accuracy for computing the winding number with fast winding number method utilizing second-order dipole approximation, default 2.0
    :param threshold: The threshold of the winding number to be considered inside, default 0.5""",
    export=False,
    hidden=True,
)

add_builtin(
    "mesh_query_point_sign_winding_number",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
        "accuracy": float,
        "threshold": float,
    },
    defaults={"accuracy": 2.0, "threshold": 0.5},
    value_type=MeshQueryPoint,
    group="Geometry",
    doc="""Computes the closest point on the :class:`Mesh` with identifier ``id`` to the given point in space.

    Identifies the sign using the winding number of the mesh relative to the query point. This method of sign determination is robust for poorly conditioned meshes
    and provides a smooth approximation to sign even when the mesh is not watertight. This method is the most robust and accurate of the sign determination meshes
    but also the most expensive.

    .. note:: The :class:`Mesh` object must be constructed with ``support_winding_number=True`` for this method to return correct results.

    :param id: The mesh identifier
    :param point: The point in space to query
    :param max_dist: Mesh faces above this distance will not be considered by the query
    :param accuracy: Accuracy for computing the winding number with fast winding number method utilizing second-order dipole approximation, default 2.0
    :param threshold: The threshold of the winding number to be considered inside, default 0.5""",
    require_original_output_arg=True,
    export=False,
)

add_builtin(
    "mesh_query_ray",
    input_types={
        "id": uint64,
        "start": vec3,
        "dir": vec3,
        "max_t": float,
        "t": float,
        "bary_u": float,
        "bary_v": float,
        "sign": float,
        "normal": vec3,
        "face": int,
    },
    value_type=builtins.bool,
    group="Geometry",
    doc="""Computes the closest ray hit on the :class:`Mesh` with identifier ``id``, returns ``True`` if a hit < ``max_t`` is found.

    :param id: The mesh identifier
    :param start: The start point of the ray
    :param dir: The ray direction (should be normalized)
    :param max_t: The maximum distance along the ray to check for intersections
    :param t: Returns the distance of the closest hit along the ray
    :param bary_u: Returns the barycentric u coordinate of the closest hit
    :param bary_v: Returns the barycentric v coordinate of the closest hit
    :param sign: Returns a value > 0 if the ray hit in front of the face, returns < 0 otherwise
    :param normal: Returns the face normal
    :param face: Returns the index of the hit face""",
    export=False,
    hidden=True,
)

add_builtin(
    "mesh_query_ray",
    input_types={
        "id": uint64,
        "start": vec3,
        "dir": vec3,
        "max_t": float,
    },
    value_type=MeshQueryRay,
    group="Geometry",
    doc="""Computes the closest ray hit on the :class:`Mesh` with identifier ``id``.

    :param id: The mesh identifier
    :param start: The start point of the ray
    :param dir: The ray direction (should be normalized)
    :param max_t: The maximum distance along the ray to check for intersections""",
    require_original_output_arg=True,
    export=False,
)

add_builtin(
    "mesh_query_aabb",
    input_types={"id": uint64, "low": vec3, "high": vec3},
    value_type=MeshQueryAABB,
    group="Geometry",
    doc="""Construct an axis-aligned bounding box query against a :class:`Mesh`.

    This query can be used to iterate over all triangles inside a volume.

    :param id: The mesh identifier
    :param low: The lower bound of the bounding box in mesh space
    :param high: The upper bound of the bounding box in mesh space""",
    export=False,
)

add_builtin(
    "mesh_query_aabb_next",
    input_types={"query": MeshQueryAABB, "index": int},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Move to the next triangle overlapping the query bounding box.

    The index of the current face is stored in ``index``, returns ``False`` if there are no more overlapping triangles.""",
    export=False,
)

add_builtin(
    "mesh_eval_position",
    input_types={"id": uint64, "face": int, "bary_u": float, "bary_v": float},
    value_type=vec3,
    group="Geometry",
    doc="""Evaluates the position on the :class:`Mesh` given a face index and barycentric coordinates.""",
    export=False,
)

add_builtin(
    "mesh_eval_velocity",
    input_types={"id": uint64, "face": int, "bary_u": float, "bary_v": float},
    value_type=vec3,
    group="Geometry",
    doc="""Evaluates the velocity on the :class:`Mesh` given a face index and barycentric coordinates.""",
    export=False,
)

add_builtin(
    "hash_grid_query",
    input_types={"id": uint64, "point": vec3, "max_dist": float},
    value_type=HashGridQuery,
    group="Geometry",
    doc="""Construct a point query against a :class:`HashGrid`.

    This query can be used to iterate over all neighboring point within a fixed radius from the query point.""",
    export=False,
)

add_builtin(
    "hash_grid_query_next",
    input_types={"query": HashGridQuery, "index": int},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Move to the next point in the hash grid query.

    The index of the current neighbor is stored in ``index``, returns ``False`` if there are no more neighbors.""",
    export=False,
)

add_builtin(
    "hash_grid_point_id",
    input_types={"id": uint64, "index": int},
    value_type=int,
    group="Geometry",
    doc="""Return the index of a point in the :class:`HashGrid`.

    This can be used to reorder threads such that grid traversal occurs in a spatially coherent order.

    Returns -1 if the :class:`HashGrid` has not been reserved.""",
    export=False,
)

add_builtin(
    "intersect_tri_tri",
    input_types={"v0": vec3, "v1": vec3, "v2": vec3, "u0": vec3, "u1": vec3, "u2": vec3},
    value_type=int,
    group="Geometry",
    doc="""Tests for intersection between two triangles (v0, v1, v2) and (u0, u1, u2) using Moller's method.

    Returns > 0 if triangles intersect.""",
    export=False,
)

add_builtin(
    "mesh_get",
    input_types={"id": uint64},
    value_type=Mesh,
    missing_grad=True,
    group="Geometry",
    doc="""Retrieves the mesh given its index.""",
    export=False,
)

add_builtin(
    "mesh_eval_face_normal",
    input_types={"id": uint64, "face": int},
    value_type=vec3,
    group="Geometry",
    doc="""Evaluates the face normal the mesh given a face index.""",
    export=False,
)

add_builtin(
    "mesh_get_point",
    input_types={"id": uint64, "index": int},
    value_type=vec3,
    group="Geometry",
    doc="""Returns the point of the mesh given a index.""",
    export=False,
)

add_builtin(
    "mesh_get_velocity",
    input_types={"id": uint64, "index": int},
    value_type=vec3,
    group="Geometry",
    doc="""Returns the velocity of the mesh given a index.""",
    export=False,
)

add_builtin(
    "mesh_get_index",
    input_types={"id": uint64, "index": int},
    value_type=int,
    group="Geometry",
    doc="""Returns the point-index of the mesh given a face-vertex index.""",
    export=False,
)


add_builtin(
    "closest_point_edge_edge",
    input_types={"p1": vec3, "q1": vec3, "p2": vec3, "q2": vec3, "epsilon": float},
    value_type=vec3,
    group="Geometry",
    doc="""Finds the closest points between two edges.

    Returns barycentric weights to the points on each edge, as well as the closest distance between the edges.

    :param p1: First point of first edge
    :param q1: Second point of first edge
    :param p2: First point of second edge
    :param q2: Second point of second edge
    :param epsilon: Zero tolerance for determining if points in an edge are degenerate.
    :param out: vec3 output containing (s,t,d), where `s` in [0,1] is the barycentric weight for the first edge, `t` is the barycentric weight for the second edge, and `d` is the distance between the two edges at these two closest points.""",
    export=False,
)

# ---------------------------------
# Ranges

add_builtin("range", input_types={"end": int}, value_type=range_t, group="Utility", export=False, hidden=True)
add_builtin(
    "range", input_types={"start": int, "end": int}, value_type=range_t, group="Utility", export=False, hidden=True
)
add_builtin(
    "range",
    input_types={"start": int, "end": int, "step": int},
    value_type=range_t,
    group="Utility",
    export=False,
    hidden=True,
)

# ---------------------------------
# Iterators

add_builtin("iter_next", input_types={"range": range_t}, value_type=int, group="Utility", export=False, hidden=True)
add_builtin(
    "iter_next", input_types={"query": HashGridQuery}, value_type=int, group="Utility", export=False, hidden=True
)
add_builtin(
    "iter_next", input_types={"query": MeshQueryAABB}, value_type=int, group="Utility", export=False, hidden=True
)

add_builtin(
    "reversed",
    input_types={"range": range_t},
    value_type=range_t,
    native_func="iter_reverse",
    group="Utility",
    doc="""Returns the range in reversed order.""",
    export=False,
)

# ---------------------------------
# Volumes

_volume_supported_value_types = {
    int32,
    int64,
    uint32,
    float32,
    float64,
    vec3f,
    vec3d,
    vec4f,
    vec4d,
}


def _is_volume_type_supported(dtype):
    for typ in _volume_supported_value_types:
        if types_equal(typ, dtype):
            return True
    return False


def _check_volume_type_is_supported(dtype):
    if not _is_volume_type_supported(dtype):
        raise RuntimeError(f"unsupported volume type `{type_repr(dtype)}`")


def check_volume_value_grad_compatibility(dtype, grad_dtype):
    if type_is_vector(dtype):
        expected = matrix(shape=(type_size(dtype), 3), dtype=type_scalar_type(dtype))
    else:
        expected = vector(length=3, dtype=dtype)

    if not types_equal(grad_dtype, expected):
        raise RuntimeError(f"Incompatible gradient type, expected {type_repr(expected)}, got {type_repr(grad_dtype)}")


def volume_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Any

    dtype = arg_values["dtype"]
    _check_volume_type_is_supported(dtype)

    return dtype


def volume_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = args["dtype"]

    func_args = tuple(v for k, v in args.items() if k != "dtype")
    template_args = (dtype,)
    return (func_args, template_args)


add_builtin(
    "volume_sample",
    input_types={"id": uint64, "uvw": vec3, "sampling_mode": int, "dtype": Any},
    value_func=volume_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=volume_dispatch_func,
    export=False,
    group="Volumes",
    doc="""Sample the volume of type `dtype` given by ``id`` at the volume local-space point ``uvw``.

    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`wp.Volume.LINEAR.`""",
)


def volume_sample_grad_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Any

    dtype = arg_values["dtype"]
    _check_volume_type_is_supported(dtype)

    check_volume_value_grad_compatibility(dtype, arg_types["grad"])

    return dtype


def volume_sample_grad_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = args["dtype"]

    func_args = tuple(v for k, v in args.items() if k != "dtype")
    template_args = (dtype,)
    return (func_args, template_args)


add_builtin(
    "volume_sample_grad",
    input_types={"id": uint64, "uvw": vec3, "sampling_mode": int, "grad": Any, "dtype": Any},
    value_func=volume_sample_grad_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=volume_sample_grad_dispatch_func,
    export=False,
    group="Volumes",
    doc="""Sample the volume given by ``id`` and its gradient at the volume local-space point ``uvw``.

    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`wp.Volume.LINEAR.`""",
)


def volume_lookup_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Any

    dtype = arg_values["dtype"]
    _check_volume_type_is_supported(dtype)

    return dtype


def volume_lookup_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = args["dtype"]

    func_args = tuple(v for k, v in args.items() if k != "dtype")
    template_args = (dtype,)
    return (func_args, template_args)


add_builtin(
    "volume_lookup",
    input_types={"id": uint64, "i": int, "j": int, "k": int, "dtype": Any},
    value_type=int,
    value_func=volume_lookup_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=volume_lookup_dispatch_func,
    export=False,
    group="Volumes",
    doc="""Returns the value of voxel with coordinates ``i``, ``j``, ``k`` for a volume of type type `dtype`.

    If the voxel at this index does not exist, this function returns the background value.""",
)


def volume_store_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return None

    dtype = arg_types["value"]
    _check_volume_type_is_supported(dtype)

    return None


add_builtin(
    "volume_store",
    value_func=volume_store_value_func,
    input_types={"id": uint64, "i": int, "j": int, "k": int, "value": Any},
    export=False,
    group="Volumes",
    doc="""Store ``value`` at the voxel with coordinates ``i``, ``j``, ``k``.""",
)

add_builtin(
    "volume_sample_f",
    input_types={"id": uint64, "uvw": vec3, "sampling_mode": int},
    value_type=float,
    group="Volumes",
    doc="""Sample the volume given by ``id`` at the volume local-space point ``uvw``.

    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`wp.Volume.LINEAR.`""",
)

add_builtin(
    "volume_sample_grad_f",
    input_types={"id": uint64, "uvw": vec3, "sampling_mode": int, "grad": vec3},
    value_type=float,
    group="Volumes",
    doc="""Sample the volume and its gradient given by ``id`` at the volume local-space point ``uvw``.

    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`wp.Volume.LINEAR.`""",
)

add_builtin(
    "volume_lookup_f",
    input_types={"id": uint64, "i": int, "j": int, "k": int},
    value_type=float,
    group="Volumes",
    doc="""Returns the value of voxel with coordinates ``i``, ``j``, ``k``.

    If the voxel at this index does not exist, this function returns the background value""",
)

add_builtin(
    "volume_store_f",
    input_types={"id": uint64, "i": int, "j": int, "k": int, "value": float},
    group="Volumes",
    doc="""Store ``value`` at the voxel with coordinates ``i``, ``j``, ``k``.""",
    export=False,
)

add_builtin(
    "volume_sample_v",
    input_types={"id": uint64, "uvw": vec3, "sampling_mode": int},
    value_type=vec3,
    group="Volumes",
    doc="""Sample the vector volume given by ``id`` at the volume local-space point ``uvw``.

    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`wp.Volume.LINEAR.`""",
)

add_builtin(
    "volume_lookup_v",
    input_types={"id": uint64, "i": int, "j": int, "k": int},
    value_type=vec3,
    group="Volumes",
    doc="""Returns the vector value of voxel with coordinates ``i``, ``j``, ``k``.

    If the voxel at this index does not exist, this function returns the background value.""",
)

add_builtin(
    "volume_store_v",
    input_types={"id": uint64, "i": int, "j": int, "k": int, "value": vec3},
    group="Volumes",
    doc="""Store ``value`` at the voxel with coordinates ``i``, ``j``, ``k``.""",
    export=False,
)

add_builtin(
    "volume_sample_i",
    input_types={"id": uint64, "uvw": vec3},
    value_type=int,
    group="Volumes",
    doc="""Sample the :class:`int32` volume given by ``id`` at the volume local-space point ``uvw``. """,
)

add_builtin(
    "volume_lookup_i",
    input_types={"id": uint64, "i": int, "j": int, "k": int},
    value_type=int,
    group="Volumes",
    doc="""Returns the :class:`int32` value of voxel with coordinates ``i``, ``j``, ``k``.

    If the voxel at this index does not exist, this function returns the background value.""",
)

add_builtin(
    "volume_store_i",
    input_types={"id": uint64, "i": int, "j": int, "k": int, "value": int},
    group="Volumes",
    doc="""Store ``value`` at the voxel with coordinates ``i``, ``j``, ``k``.""",
    export=False,
)


def volume_sample_index_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Any

    dtype = arg_types["voxel_data"].dtype

    if dtype not in _volume_supported_value_types:
        raise RuntimeError(f"unsupported volume type `{dtype.__name__}`")

    if not types_equal(dtype, arg_types["background"]):
        raise RuntimeError("the `voxel_data` array and the `background` value must have the same dtype")

    return dtype


add_builtin(
    "volume_sample_index",
    input_types={"id": uint64, "uvw": vec3, "sampling_mode": int, "voxel_data": array(dtype=Any), "background": Any},
    value_func=volume_sample_index_value_func,
    export=False,
    group="Volumes",
    doc="""Sample the volume given by ``id`` at the volume local-space point ``uvw``.

    Values for allocated voxels are read from the ``voxel_data`` array, and `background` is used as the value of non-existing voxels.
    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`wp.Volume.LINEAR`.
    This function is available for both index grids and classical volumes.
    """,
)


def volume_sample_grad_index_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Any

    dtype = arg_types["voxel_data"].dtype

    if dtype not in _volume_supported_value_types:
        raise RuntimeError(f"unsupported volume type `{dtype.__name__}`")

    if not types_equal(dtype, arg_types["background"]):
        raise RuntimeError("the `voxel_data` array and the `background` value must have the same dtype")

    check_volume_value_grad_compatibility(dtype, arg_types["grad"])

    return dtype


add_builtin(
    "volume_sample_grad_index",
    input_types={
        "id": uint64,
        "uvw": vec3,
        "sampling_mode": int,
        "voxel_data": array(dtype=Any),
        "background": Any,
        "grad": Any,
    },
    value_func=volume_sample_grad_index_value_func,
    export=False,
    group="Volumes",
    doc="""Sample the volume given by ``id`` and its gradient at the volume local-space point ``uvw``.

    Values for allocated voxels are read from the ``voxel_data`` array, and `background` is used as the value of non-existing voxels.
    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`wp.Volume.LINEAR`.
    This function is available for both index grids and classical volumes.
   """,
)

add_builtin(
    "volume_lookup_index",
    input_types={"id": uint64, "i": int, "j": int, "k": int},
    value_type=int32,
    group="Volumes",
    doc="""Returns the index associated to the voxel with coordinates ``i``, ``j``, ``k``.

    If the voxel at this index does not exist, this function returns -1.
    This function is available for both index grids and classical volumes.
    """,
)

add_builtin(
    "volume_index_to_world",
    input_types={"id": uint64, "uvw": vec3},
    value_type=vec3,
    group="Volumes",
    doc="""Transform a point ``uvw`` defined in volume index space to world space given the volume's intrinsic affine transformation.""",
)
add_builtin(
    "volume_world_to_index",
    input_types={"id": uint64, "xyz": vec3},
    value_type=vec3,
    group="Volumes",
    doc="""Transform a point ``xyz`` defined in volume world space to the volume's index space given the volume's intrinsic affine transformation.""",
)
add_builtin(
    "volume_index_to_world_dir",
    input_types={"id": uint64, "uvw": vec3},
    value_type=vec3,
    group="Volumes",
    doc="""Transform a direction ``uvw`` defined in volume index space to world space given the volume's intrinsic affine transformation.""",
)
add_builtin(
    "volume_world_to_index_dir",
    input_types={"id": uint64, "xyz": vec3},
    value_type=vec3,
    group="Volumes",
    doc="""Transform a direction ``xyz`` defined in volume world space to the volume's index space given the volume's intrinsic affine transformation.""",
)


# ---------------------------------
# Random

add_builtin(
    "rand_init",
    input_types={"seed": int},
    value_type=uint32,
    group="Random",
    doc="Initialize a new random number generator given a user-defined seed. Returns a 32-bit integer representing the RNG state.",
)

add_builtin(
    "rand_init",
    input_types={"seed": int, "offset": int},
    value_type=uint32,
    group="Random",
    doc="""Initialize a new random number generator given a user-defined seed and an offset.

    This alternative constructor can be useful in parallel programs, where a kernel as a whole should share a seed,
    but each thread should generate uncorrelated values. In this case usage should be ``r = rand_init(seed, tid)``""",
)

add_builtin(
    "randi",
    input_types={"state": uint32},
    value_type=int,
    group="Random",
    doc="Return a random integer in the range [-2^31, 2^31).",
)
add_builtin(
    "randi",
    input_types={"state": uint32, "low": int, "high": int},
    value_type=int,
    group="Random",
    doc="Return a random integer between [low, high).",
)
add_builtin(
    "randu",
    input_types={"state": uint32},
    value_type=uint32,
    group="Random",
    doc="Return a random unsigned integer in the range [0, 2^32).",
)
add_builtin(
    "randu",
    input_types={"state": uint32, "low": uint32, "high": uint32},
    value_type=uint32,
    group="Random",
    doc="Return a random unsigned integer between [low, high).",
)
add_builtin(
    "randf",
    input_types={"state": uint32},
    value_type=float,
    group="Random",
    doc="Return a random float between [0.0, 1.0).",
)
add_builtin(
    "randf",
    input_types={"state": uint32, "low": float, "high": float},
    value_type=float,
    group="Random",
    doc="Return a random float between [low, high).",
)
add_builtin(
    "randn",
    input_types={"state": uint32},
    value_type=float,
    group="Random",
    doc="Sample a normal (Gaussian) distribution of mean 0 and variance 1. ",
)

add_builtin(
    "sample_cdf",
    input_types={"state": uint32, "cdf": array(dtype=float)},
    value_type=int,
    group="Random",
    doc="Inverse-transform sample a cumulative distribution function.",
)
add_builtin(
    "sample_triangle",
    input_types={"state": uint32},
    value_type=vec2,
    group="Random",
    doc="Uniformly sample a triangle. Returns sample barycentric coordinates.",
)
add_builtin(
    "sample_unit_ring",
    input_types={"state": uint32},
    value_type=vec2,
    group="Random",
    doc="Uniformly sample a ring in the xy plane.",
)
add_builtin(
    "sample_unit_disk",
    input_types={"state": uint32},
    value_type=vec2,
    group="Random",
    doc="Uniformly sample a disk in the xy plane.",
)
add_builtin(
    "sample_unit_sphere_surface",
    input_types={"state": uint32},
    value_type=vec3,
    group="Random",
    doc="Uniformly sample a unit sphere surface.",
)
add_builtin(
    "sample_unit_sphere",
    input_types={"state": uint32},
    value_type=vec3,
    group="Random",
    doc="Uniformly sample a unit sphere.",
)
add_builtin(
    "sample_unit_hemisphere_surface",
    input_types={"state": uint32},
    value_type=vec3,
    group="Random",
    doc="Uniformly sample a unit hemisphere surface.",
)
add_builtin(
    "sample_unit_hemisphere",
    input_types={"state": uint32},
    value_type=vec3,
    group="Random",
    doc="Uniformly sample a unit hemisphere.",
)
add_builtin(
    "sample_unit_square",
    input_types={"state": uint32},
    value_type=vec2,
    group="Random",
    doc="Uniformly sample a unit square.",
)
add_builtin(
    "sample_unit_cube",
    input_types={"state": uint32},
    value_type=vec3,
    group="Random",
    doc="Uniformly sample a unit cube.",
)

add_builtin(
    "poisson",
    input_types={"state": uint32, "lam": float},
    value_type=uint32,
    group="Random",
    doc="""Generate a random sample from a Poisson distribution.

    :param state: RNG state
    :param lam: The expected value of the distribution""",
)

add_builtin(
    "noise",
    input_types={"state": uint32, "x": float},
    value_type=float,
    group="Random",
    doc="Non-periodic Perlin-style noise in 1D.",
)
add_builtin(
    "noise",
    input_types={"state": uint32, "xy": vec2},
    value_type=float,
    group="Random",
    doc="Non-periodic Perlin-style noise in 2D.",
)
add_builtin(
    "noise",
    input_types={"state": uint32, "xyz": vec3},
    value_type=float,
    group="Random",
    doc="Non-periodic Perlin-style noise in 3D.",
)
add_builtin(
    "noise",
    input_types={"state": uint32, "xyzt": vec4},
    value_type=float,
    group="Random",
    doc="Non-periodic Perlin-style noise in 4D.",
)

add_builtin(
    "pnoise",
    input_types={"state": uint32, "x": float, "px": int},
    value_type=float,
    group="Random",
    doc="Periodic Perlin-style noise in 1D.",
)
add_builtin(
    "pnoise",
    input_types={"state": uint32, "xy": vec2, "px": int, "py": int},
    value_type=float,
    group="Random",
    doc="Periodic Perlin-style noise in 2D.",
)
add_builtin(
    "pnoise",
    input_types={"state": uint32, "xyz": vec3, "px": int, "py": int, "pz": int},
    value_type=float,
    group="Random",
    doc="Periodic Perlin-style noise in 3D.",
)
add_builtin(
    "pnoise",
    input_types={"state": uint32, "xyzt": vec4, "px": int, "py": int, "pz": int, "pt": int},
    value_type=float,
    group="Random",
    doc="Periodic Perlin-style noise in 4D.",
)

add_builtin(
    "curlnoise",
    input_types={"state": uint32, "xy": vec2, "octaves": uint32, "lacunarity": float, "gain": float},
    defaults={"octaves": uint32(1), "lacunarity": 2.0, "gain": 0.5},
    value_type=vec2,
    group="Random",
    doc="Divergence-free vector field based on the gradient of a Perlin noise function.",
    missing_grad=True,
)
add_builtin(
    "curlnoise",
    input_types={"state": uint32, "xyz": vec3, "octaves": uint32, "lacunarity": float, "gain": float},
    defaults={"octaves": uint32(1), "lacunarity": 2.0, "gain": 0.5},
    value_type=vec3,
    group="Random",
    doc="Divergence-free vector field based on the curl of three Perlin noise functions.",
    missing_grad=True,
)
add_builtin(
    "curlnoise",
    input_types={"state": uint32, "xyzt": vec4, "octaves": uint32, "lacunarity": float, "gain": float},
    defaults={"octaves": uint32(1), "lacunarity": 2.0, "gain": 0.5},
    value_type=vec3,
    group="Random",
    doc="Divergence-free vector field based on the curl of three Perlin noise functions.",
    missing_grad=True,
)


def printf_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is not None:
        if len(arg_types.get("args", ())) > 32:
            raise RuntimeError("the maximum number of variadic arguments that can be passed to `printf` is 32")

    return None


def printf_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    func_args = (args["fmt"], *args.get("args", ()))
    template_args = ()
    return (func_args, template_args)


# note printf calls directly to global CRT printf (no wp:: namespace prefix)
add_builtin(
    "printf",
    input_types={"fmt": str, "*args": Any},
    namespace="",
    variadic=True,
    value_func=printf_value_func,
    dispatch_func=printf_dispatch_func,
    group="Utility",
    doc="Allows printing formatted strings using C-style format specifiers.",
)

add_builtin("print", input_types={"value": Any}, doc="Print variable to stdout", export=False, group="Utility")

add_builtin(
    "breakpoint",
    input_types={},
    doc="Debugger breakpoint",
    export=False,
    group="Utility",
    namespace="",
    native_func="__debugbreak",
)

# helpers
add_builtin(
    "tid",
    input_types={},
    value_type=int,
    export=False,
    group="Utility",
    doc="""Return the current thread index for a 1D kernel launch.

    Note that this is the *global* index of the thread in the range [0, dim)
    where dim is the parameter passed to kernel launch.

    This function may not be called from user-defined Warp functions.""",
    namespace="",
    native_func="builtin_tid1d",
)

add_builtin(
    "block_dim",
    input_types={},
    value_type=int,
    group="Utility",
    doc="Returns the number of threads in the current block.",
    namespace="",
    native_func="builtin_block_dim",
)

add_builtin(
    "tid",
    input_types={},
    value_type=[int, int],
    group="Utility",
    doc="""Return the current thread indices for a 2D kernel launch.

    Use ``i,j = wp.tid()`` syntax to retrieve the coordinates inside the kernel thread grid.

    This function may not be called from user-defined Warp functions.""",
    namespace="",
    native_func="builtin_tid2d",
)

add_builtin(
    "tid",
    input_types={},
    value_type=[int, int, int],
    group="Utility",
    doc="""Return the current thread indices for a 3D kernel launch.

    Use ``i,j,k = wp.tid()`` syntax to retrieve the coordinates inside the kernel thread grid.

    This function may not be called from user-defined Warp functions.""",
    namespace="",
    native_func="builtin_tid3d",
)

add_builtin(
    "tid",
    input_types={},
    value_type=[int, int, int, int],
    group="Utility",
    doc="""Return the current thread indices for a 4D kernel launch.

    Use ``i,j,k,l = wp.tid()`` syntax to retrieve the coordinates inside the kernel thread grid.

    This function may not be called from user-defined Warp functions.""",
    namespace="",
    native_func="builtin_tid4d",
)


add_builtin(
    "copy",
    input_types={"a": Any},
    value_func=lambda arg_types, arg_values: arg_types["a"],
    hidden=True,
    export=False,
    group="Utility",
)
add_builtin(
    "assign",
    input_types={"dest": Any, "src": Any},
    hidden=True,
    export=False,
    group="Utility",
)


def select_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    warp.utils.warn(
        "wp.select() is deprecated and will be removed in a future\n"
        "version. Use wp.where(cond, value_if_true, value_if_false) instead.",
        category=DeprecationWarning,
    )

    func_args = tuple(args.values())
    template_args = ()

    return (func_args, template_args)


add_builtin(
    "select",
    input_types={"cond": builtins.bool, "value_if_false": Any, "value_if_true": Any},
    value_func=lambda arg_types, arg_values: Any if arg_types is None else arg_types["value_if_false"],
    dispatch_func=select_dispatch_func,
    doc="""Select between two arguments, if ``cond`` is ``False`` then return ``value_if_false``, otherwise return ``value_if_true``.

    .. deprecated:: 1.7
         Use :func:`where` instead, which has the more intuitive argument order:
         ``where(cond, value_if_true, value_if_false)``.""",
    group="Utility",
)
for t in int_types:
    add_builtin(
        "select",
        input_types={"cond": t, "value_if_false": Any, "value_if_true": Any},
        value_func=lambda arg_types, arg_values: Any if arg_types is None else arg_types["value_if_false"],
        dispatch_func=select_dispatch_func,
        doc="""Select between two arguments, if ``cond`` is ``False`` then return ``value_if_false``, otherwise return ``value_if_true``.

    .. deprecated:: 1.7
         Use :func:`where` instead, which has the more intuitive argument order:
         ``where(cond, value_if_true, value_if_false)``.""",
        group="Utility",
    )
add_builtin(
    "select",
    input_types={"arr": array(dtype=Any), "value_if_false": Any, "value_if_true": Any},
    value_func=lambda arg_types, arg_values: Any if arg_types is None else arg_types["value_if_false"],
    dispatch_func=select_dispatch_func,
    doc="""Select between two arguments, if ``arr`` is null then return ``value_if_false``, otherwise return ``value_if_true``.

    .. deprecated:: 1.7
         Use :func:`where` instead, which has the more intuitive argument order:
         ``where(arr, value_if_true, value_if_false)``.""",
    group="Utility",
)

add_builtin(
    "where",
    input_types={"cond": builtins.bool, "value_if_true": Any, "value_if_false": Any},
    value_func=lambda arg_types, arg_values: Any if arg_types is None else arg_types["value_if_false"],
    doc="Select between two arguments, if ``cond`` is ``True`` then return ``value_if_true``, otherwise return ``value_if_false``.",
    group="Utility",
)
for t in int_types:
    add_builtin(
        "where",
        input_types={"cond": t, "value_if_true": Any, "value_if_false": Any},
        value_func=lambda arg_types, arg_values: Any if arg_types is None else arg_types["value_if_false"],
        doc="Select between two arguments, if ``cond`` is ``True`` then return ``value_if_true``, otherwise return ``value_if_false``.",
        group="Utility",
    )
add_builtin(
    "where",
    input_types={"arr": array(dtype=Any), "value_if_true": Any, "value_if_false": Any},
    value_func=lambda arg_types, arg_values: Any if arg_types is None else arg_types["value_if_false"],
    doc="Select between two arguments, if ``arr`` is not null then return ``value_if_true``, otherwise return ``value_if_false``.",
    group="Utility",
)


def array_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return array(dtype=Scalar)

    dtype = arg_values["dtype"]
    shape = extract_tuple(arg_values["shape"], as_constant=False)
    return array(dtype=dtype, ndim=len(shape))


def array_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = return_type.dtype
    shape = extract_tuple(args["shape"], as_constant=False)

    func_args = (args["ptr"], *shape)
    template_args = (dtype,)
    return (func_args, template_args)


add_builtin(
    "array",
    input_types={"ptr": warp.uint64, "shape": Tuple[int, ...], "dtype": Any},
    value_func=array_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=array_dispatch_func,
    native_func="array_t",
    group="Utility",
    hidden=True,
    export=False,
    missing_grad=True,
)


def zeros_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return fixedarray(dtype=Scalar)

    dtype = arg_values["dtype"]
    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise RuntimeError("the `shape` argument must be specified as a constant when zero-initializing an array")

    return fixedarray(dtype=dtype, shape=shape)


def zeros_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = return_type.dtype
    shape = extract_tuple(args["shape"], as_constant=True)

    size = math.prod(shape)

    func_args = shape
    template_args = (size, dtype)
    return (func_args, template_args)


add_builtin(
    "zeros",
    input_types={"shape": Tuple[int, ...], "dtype": Any},
    value_func=zeros_value_func,
    export_func=lambda input_types: {},
    dispatch_func=zeros_dispatch_func,
    native_func="fixedarray_t",
    group="Utility",
    export=False,
    missing_grad=True,
    hidden=True,  # Unhide once we can document both a built-in and a Python scope function sharing the same name.
)


# does argument checking and type propagation for address()
def address_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    arr_type = arg_types["arr"]
    idx_types = tuple(arg_types[x] for x in "ijkl" if arg_types.get(x, None) is not None)

    if not is_array(arr_type):
        raise RuntimeError("address() first argument must be an array")

    idx_count = len(idx_types)

    if idx_count != arr_type.ndim:
        raise RuntimeError(
            f"The number of indices provided ({idx_count}) does not match the array dimensions ({arr_type.ndim}) for array load"
        )

    # check index types
    for t in idx_types:
        if not type_is_int(t):
            raise RuntimeError(f"address() index arguments must be of integer type, got index of type {type_repr(t)}")

    return Reference(arr_type.dtype)


for array_type in array_types:
    add_builtin(
        "address",
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "l": Int},
        constraint=sametypes,
        defaults={"j": None, "k": None, "l": None},
        hidden=True,
        value_func=address_value_func,
        group="Utility",
    )


# does argument checking and type propagation for view()
def view_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    arr_type = arg_types["arr"]
    idx_types = tuple(arg_types[x] for x in "ijk" if arg_types.get(x, None) is not None)

    if not is_array(arr_type):
        raise RuntimeError("view() first argument must be an array")

    idx_count = len(idx_types)

    if idx_count >= arr_type.ndim:
        raise RuntimeError(
            f"Trying to create an array view with {idx_count} indices, "
            f"but the array only has {arr_type.ndim} dimension(s). "
            f"Ensure that the argument type on the function or kernel specifies "
            f"the expected number of dimensions, e.g.: def func(param: wp.array3d(dtype=float): ..."
        )

    # check index types
    for t in idx_types:
        if not type_is_int(t):
            raise RuntimeError(f"view() index arguments must be of integer type, got index of type {type_repr(t)}")

    # create an array view with leading dimensions removed
    dtype = arr_type.dtype
    ndim = arr_type.ndim - idx_count
    if isinstance(arr_type, (fabricarray, indexedfabricarray)):
        # fabric array of arrays: return array attribute as a regular array
        return array(dtype=dtype, ndim=ndim)

    return type(arr_type)(dtype=dtype, ndim=ndim)


for array_type in array_types:
    add_builtin(
        "view",
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int},
        defaults={"j": None, "k": None},
        constraint=sametypes,
        hidden=True,
        value_func=view_value_func,
        group="Utility",
    )


# does argument checking and type propagation for array_store()
def array_store_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    arr_type = arg_types["arr"]
    value_type = arg_types["value"]
    idx_types = tuple(arg_types[x] for x in "ijkl" if arg_types.get(x, None) is not None)

    if not is_array(arr_type):
        raise RuntimeError("array_store() first argument must be an array")

    idx_count = len(idx_types)

    if idx_count != arr_type.ndim:
        raise RuntimeError(
            f"The number of indices provided ({idx_count}) does not match the array dimensions ({arr_type.ndim}) for array store"
        )

    # check index types
    for t in idx_types:
        if not type_is_int(t):
            raise RuntimeError(
                f"array_store() index arguments must be of integer type, got index of type {type_repr(t)}"
            )

    # check value type
    if not types_equal(arr_type.dtype, value_type):
        raise RuntimeError(
            f"array_store() value argument type ({type_repr(value_type)}) must be of the same type as the array ({type_repr(arr_type.dtype)})"
        )

    return None


for array_type in array_types:
    add_builtin(
        "array_store",
        input_types={"arr": array_type(dtype=Any), "i": Int, "value": Any},
        constraint=sametypes,
        hidden=True,
        value_func=array_store_value_func,
        skip_replay=True,
        group="Utility",
    )
    add_builtin(
        "array_store",
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "value": Any},
        constraint=sametypes,
        hidden=True,
        value_func=array_store_value_func,
        skip_replay=True,
        group="Utility",
    )
    add_builtin(
        "array_store",
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "value": Any},
        constraint=sametypes,
        hidden=True,
        value_func=array_store_value_func,
        skip_replay=True,
        group="Utility",
    )
    add_builtin(
        "array_store",
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "l": Int, "value": Any},
        constraint=sametypes,
        hidden=True,
        value_func=array_store_value_func,
        skip_replay=True,
        group="Utility",
    )


# does argument checking for store()
def store_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # we already stripped the Reference from the argument type prior to this call
    if not types_equal(arg_types["address"], arg_types["value"]):
        raise RuntimeError(
            f"store() value argument type ({arg_types['value']}) must be of the same type as the reference"
        )

    return None


def store_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (Reference(args["address"]), args["value"])
    template_args = ()
    return (func_args, template_args)


add_builtin(
    "store",
    input_types={"address": Any, "value": Any},
    value_func=store_value_func,
    dispatch_func=store_dispatch_func,
    hidden=True,
    skip_replay=True,
    group="Utility",
)


def load_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (Reference(args["address"]),)
    template_args = ()
    return (func_args, template_args)


add_builtin(
    "load",
    input_types={"address": Any},
    value_func=lambda arg_types, arg_values: arg_types["address"],
    dispatch_func=load_dispatch_func,
    hidden=True,
    group="Utility",
)


SUPPORTED_ATOMIC_TYPES = (
    warp.int32,
    warp.int64,
    warp.uint32,
    warp.uint64,
    warp.float32,
    warp.float64,
)


def atomic_op_constraint(arg_types: Mapping[str, Any]):
    idx_types = tuple(arg_types[x] for x in "ijkl" if arg_types.get(x, None) is not None)
    return all(types_equal(idx_types[0], t) for t in idx_types[1:]) and arg_types["arr"].ndim == len(idx_types)


def create_atomic_op_value_func(op: str):
    def fn(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
        if arg_types is None:
            return Any

        arr_type = arg_types["arr"]
        value_type = arg_types["value"]
        idx_types = tuple(arg_types[x] for x in "ijkl" if arg_types.get(x, None) is not None)

        if not is_array(arr_type):
            raise RuntimeError(f"atomic_{op}() first argument must be an array")

        idx_count = len(idx_types)

        if idx_count < arr_type.ndim:
            raise RuntimeError(
                f"Num indices < num dimensions for atomic_{op}(), this is a codegen error, should have generated a view instead"
            )

        if idx_count > arr_type.ndim:
            raise RuntimeError(
                f"Num indices > num dimensions for atomic_{op}(), received {idx_count}, but array only has {arr_type.ndim}"
            )

        # check index types
        for t in idx_types:
            if not type_is_int(t):
                raise RuntimeError(
                    f"atomic_{op}() index arguments must be of integer type, got index of type {type_repr(t)}"
                )

        # check value type
        if not types_equal(arr_type.dtype, value_type):
            raise RuntimeError(
                f"atomic_{op}() value argument type ({type_repr(value_type)}) must be of the same type as the array ({type_repr(arr_type.dtype)})"
            )

        scalar_type = getattr(arr_type.dtype, "_wp_scalar_type_", arr_type.dtype)
        if op in ("add", "sub"):
            supported_atomic_types = (*SUPPORTED_ATOMIC_TYPES, warp.float16)
            if not any(types_equal(scalar_type, x, match_generic=True) for x in supported_atomic_types):
                raise RuntimeError(
                    f"atomic_{op}() operations only work on arrays with [u]int32, [u]int64, float16, float32, or float64 "
                    f"as the underlying scalar types, but got {type_repr(arr_type.dtype)} (with scalar type {type_repr(scalar_type)})"
                )
        elif op in ("min", "max"):
            if not any(types_equal(scalar_type, x, match_generic=True) for x in SUPPORTED_ATOMIC_TYPES):
                raise RuntimeError(
                    f"atomic_{op}() operations only work on arrays with [u]int32, [u]int64, float32, or float64 "
                    f"as the underlying scalar types, but got {type_repr(arr_type.dtype)} (with scalar type {type_repr(scalar_type)})"
                )
        elif op in ("cas", "exch"):
            if not any(types_equal(scalar_type, x, match_generic=True) for x in SUPPORTED_ATOMIC_TYPES):
                raise RuntimeError(
                    f"atomic_{op}() operations only work on arrays with [u]int32, [u]int64, float32, or float64 "
                    f"as the underlying scalar types, but got {type_repr(arr_type.dtype)} (with scalar type {type_repr(scalar_type)})"
                )
        else:
            raise NotImplementedError

        return arr_type.dtype

    return fn


def atomic_op_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # as this is a codegen callback, we can mark the fact that this func writes to an array here
    if warp.config.verify_autograd_array_access:
        arr = args["arr"]
        arr.mark_write()

    func_args = tuple(args.values())
    # we don't need to specify template arguments for atomic ops
    template_args = ()

    return (func_args, template_args)


for array_type in array_types:
    # don't list fixed or indexed array operations explicitly in docs
    hidden = array_type in (indexedarray, fixedarray)

    add_builtin(
        "atomic_add",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("add"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically adds ``value`` onto ``arr[i]`` and returns the original value of ``arr[i]``.
        This function is automatically invoked when using the syntax ``arr[i] += value``.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_add",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("add"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically adds ``value`` onto ``arr[i,j]`` and returns the original value of ``arr[i,j]``.
        This function is automatically invoked when using the syntax ``arr[i,j] += value``.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_add",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("add"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically adds ``value`` onto ``arr[i,j,k]`` and returns the original value of ``arr[i,j,k]``.
        This function is automatically invoked when using the syntax ``arr[i,j,k] += value``.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_add",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "l": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("add"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically adds ``value`` onto ``arr[i,j,k,l]`` and returns the original value of ``arr[i,j,k,l]``.
        This function is automatically invoked when using the syntax ``arr[i,j,k,l] += value``.""",
        group="Utility",
        skip_replay=True,
    )

    add_builtin(
        "atomic_sub",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("sub"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically subtracts ``value`` onto ``arr[i]`` and returns the original value of ``arr[i]``.
        This function is automatically invoked when using the syntax ``arr[i] -= value``.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_sub",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("sub"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically subtracts ``value`` onto ``arr[i,j]`` and returns the original value of ``arr[i,j]``.
        This function is automatically invoked when using the syntax ``arr[i,j] -= value``.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_sub",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("sub"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically subtracts ``value`` onto ``arr[i,j,k]`` and returns the original value of ``arr[i,j,k]``.
        This function is automatically invoked when using the syntax ``arr[i,j,k] -= value``.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_sub",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "l": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("sub"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically subtracts ``value`` onto ``arr[i,j,k,l]`` and returns the original value of ``arr[i,j,k,l]``.
        This function is automatically invoked when using the syntax ``arr[i,j,k,l] -= value``.""",
        group="Utility",
        skip_replay=True,
    )

    add_builtin(
        "atomic_min",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("min"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Compute the minimum of ``value`` and ``arr[i]``, atomically update the array, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_min",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("min"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Compute the minimum of ``value`` and ``arr[i,j]``, atomically update the array, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_min",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("min"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Compute the minimum of ``value`` and ``arr[i,j,k]``, atomically update the array, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_min",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "l": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("min"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Compute the minimum of ``value`` and ``arr[i,j,k,l]``, atomically update the array, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )

    add_builtin(
        "atomic_max",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("max"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Compute the maximum of ``value`` and ``arr[i]``, atomically update the array, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_max",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("max"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Compute the maximum of ``value`` and ``arr[i,j]``, atomically update the array, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_max",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("max"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Compute the maximum of ``value`` and ``arr[i,j,k]``, atomically update the array, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_max",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "l": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("max"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Compute the maximum of ``value`` and ``arr[i,j,k,l]``, atomically update the array, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )

    add_builtin(
        "atomic_cas",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "compare": Any, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("cas"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically compare and swap ``value`` with ``arr[i]`` if ``arr[i]`` equals ``compare``, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_cas",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "compare": Any, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("cas"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically compare and swap ``value`` with ``arr[i,j]`` if ``arr[i,j]`` equals ``compare``, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_cas",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "compare": Any, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("cas"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically compare and swap ``value`` with ``arr[i,j,k]`` if ``arr[i,j,k]`` equals ``compare``, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_cas",
        hidden=hidden,
        input_types={
            "arr": array_type(dtype=Any),
            "i": Int,
            "j": Int,
            "k": Int,
            "l": Int,
            "compare": Any,
            "value": Any,
        },
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("cas"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically compare and swap ``value`` with ``arr[i,j,k,l]`` if ``arr[i,j,k,l]`` equals ``compare``, and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )

    add_builtin(
        "atomic_exch",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("exch"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically exchange ``value`` with ``arr[i]`` and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_exch",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("exch"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically exchange ``value`` with ``arr[i,j]`` and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_exch",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("exch"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically exchange ``value`` with ``arr[i,j,k]`` and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_exch",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "l": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("exch"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically exchange ``value`` with ``arr[i,j,k,l]`` and return the old value.

    The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )


# used to index into builtin types, i.e.: y = vec3[1]
def vector_extract_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    vec_type = arg_types["a"]
    idx_type = arg_types["i"]

    if isinstance(idx_type, slice_t):
        length = idx_type.get_length(vec_type._length_)
        return vector(length=length, dtype=vec_type._wp_scalar_type_)

    return vec_type._wp_scalar_type_


def vector_extract_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = tuple(args.values())
    template_args = getattr(return_type, "_shape_", ())
    return (func_args, template_args)


add_builtin(
    "extract",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": Any},
    value_func=vector_extract_value_func,
    dispatch_func=vector_extract_dispatch_func,
    export=False,
    hidden=True,
    group="Utility",
)
add_builtin(
    "extract",
    input_types={"a": quaternion(dtype=Scalar), "i": Any},
    value_func=vector_extract_value_func,
    dispatch_func=vector_extract_dispatch_func,
    export=False,
    hidden=True,
    group="Utility",
)
add_builtin(
    "extract",
    input_types={"a": transformation(dtype=Scalar), "i": Any},
    value_func=vector_extract_value_func,
    dispatch_func=vector_extract_dispatch_func,
    export=False,
    hidden=True,
    group="Utility",
)


def matrix_extract_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    mat_type = arg_types["a"]
    idx_types = tuple(arg_types[x] for x in "ij" if arg_types.get(x, None) is not None)

    # Compute the resulting shape from the slicing, with -1 being simple indexing.
    shape = tuple(
        idx.get_length(mat_type._shape_[i]) if isinstance(idx, slice_t) else -1 for i, idx in enumerate(idx_types)
    )

    # Append any non indexed slice.
    for i in range(len(idx_types), len(mat_type._shape_)):
        shape += (mat_type._shape_[i],)

    # Count how many dimensions the output value will have.
    ndim = sum(1 for x in shape if x >= 0)

    if ndim == 0:
        return mat_type._wp_scalar_type_

    assert shape[0] != -1 or shape[1] != -1

    if ndim == 1:
        length = shape[0] if shape[0] != -1 else shape[1]
        return vector(length=length, dtype=mat_type._wp_scalar_type_)

    assert ndim == 2

    # When a matrix dimension is 0, all other dimensions are also expected to be 0.
    if any(x == 0 for x in shape):
        shape = (0,) * len(shape)

    return matrix(shape=shape, dtype=mat_type._wp_scalar_type_)


def matrix_extract_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    idx_types = tuple(args[x].type for x in "ij" if args.get(x, None) is not None)
    has_slice = any(isinstance(x, slice_t) for x in idx_types)

    func_args = tuple(args.values())
    template_args = getattr(return_type, "_shape_", ()) if has_slice else ()
    return (func_args, template_args)


add_builtin(
    "extract",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any},
    value_func=matrix_extract_value_func,
    dispatch_func=matrix_extract_dispatch_func,
    export=False,
    hidden=True,
    group="Utility",
)
add_builtin(
    "extract",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "j": Any},
    value_func=matrix_extract_value_func,
    dispatch_func=matrix_extract_dispatch_func,
    export=False,
    hidden=True,
    group="Utility",
)

add_builtin("extract", input_types={"s": shape_t, "i": int}, value_type=int, hidden=True, group="Utility")


def vector_index_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    vec_type = arg_types["a"]
    value_type = vec_type._wp_scalar_type_

    return Reference(value_type)


def vector_index_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (Reference(args["a"]), args["i"])
    template_args = ()
    return (func_args, template_args)


# implements &vector[index]
add_builtin(
    "index",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)
# implements &quaternion[index]
add_builtin(
    "index",
    input_types={"a": quaternion(dtype=Float), "i": int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)
# implements &transformation[index]
add_builtin(
    "index",
    input_types={"a": transformation(dtype=Float), "i": int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)
# implements &(*vector)[index]
add_builtin(
    "indexref",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)
# implements &(*quaternion)[index]
add_builtin(
    "indexref",
    input_types={"a": quaternion(dtype=Float), "i": int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)
# implements &(*transformation)[index]
add_builtin(
    "indexref",
    input_types={"a": transformation(dtype=Float), "i": int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)


def vector_assign_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    vec = args["a"].type
    idx = args["i"].type
    value_type = strip_reference(args["value"].type)

    if isinstance(idx, slice_t):
        length = idx.get_length(vec._length_)

        if type_is_vector(value_type):
            if not types_equal(value_type._wp_scalar_type_, vec._wp_scalar_type_):
                raise ValueError(
                    f"The provided vector is expected to be of length {length} with dtype {type_repr(vec._wp_scalar_type_)}."
                )
            if value_type._length_ != length:
                raise ValueError(
                    f"The length of the provided vector ({args['value'].type._length_}) isn't compatible with the given slice (expected {length})."
                )
            template_args = (length,)
        else:
            if not types_equal(value_type, vec._wp_scalar_type_):
                raise ValueError(
                    f"The provided value is expected to be a scalar, or a vector of length {length}, with dtype {type_repr(vec._wp_scalar_type_)}."
                )
            template_args = ()
    else:
        if not types_equal(value_type, vec._wp_scalar_type_):
            raise ValueError(
                f"The provided value is expected to be a scalar of type {type_repr(vec._wp_scalar_type_)}."
            )
        template_args = ()

    func_args = tuple(args.values())
    return (func_args, template_args)


# implements vector[index] = value
add_builtin(
    "assign_inplace",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements quaternion[index] = value
add_builtin(
    "assign_inplace",
    input_types={"a": quaternion(dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)
# implements transformation[index] = value
add_builtin(
    "assign_inplace",
    input_types={"a": transformation(dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)


def vector_assign_copy_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    vec_type = arg_types["a"]
    return vec_type


# implements vector[index] = value, performs a copy internally if wp.config.enable_vector_component_overwrites is True
add_builtin(
    "assign_copy",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": Any, "value": Any},
    value_func=vector_assign_copy_value_func,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements quaternion[index] = value, performs a copy internally if wp.config.enable_vector_component_overwrites is True
add_builtin(
    "assign_copy",
    input_types={"a": quaternion(dtype=Scalar), "i": Any, "value": Any},
    value_func=vector_assign_copy_value_func,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements transformation[index] = value, performs a copy internally if wp.config.enable_vector_component_overwrites is True
add_builtin(
    "assign_copy",
    input_types={"a": transformation(dtype=Scalar), "i": Any, "value": Any},
    value_func=vector_assign_copy_value_func,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements vector[idx] += scalar
add_builtin(
    "add_inplace",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements quaternion[idx] += scalar
add_builtin(
    "add_inplace",
    input_types={"a": quaternion(dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements transformation[idx] += scalar
add_builtin(
    "add_inplace",
    input_types={"a": transformation(dtype=Float), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements transformation.p += vec3
add_builtin(
    "transform_add_inplace",
    input_types={"a": transformation(dtype=Float), "value": vector(length=3, dtype=Float)},
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
)

# implements vector[idx] -= scalar
add_builtin(
    "sub_inplace",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements quaternion[idx] -= scalar
add_builtin(
    "sub_inplace",
    input_types={"a": quaternion(dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements transformation[idx] -= scalar
add_builtin(
    "sub_inplace",
    input_types={"a": transformation(dtype=Float), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements transformation.p -= vec3
add_builtin(
    "transform_sub_inplace",
    input_types={"a": transformation(dtype=Float), "value": vector(length=3, dtype=Float)},
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
)


def matrix_index_row_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    mat_type = arg_types["a"]
    row_type = mat_type._wp_row_type_

    return Reference(row_type)


# implements &matrix[i] = row
add_builtin(
    "index",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": int},
    value_func=matrix_index_row_value_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)


def matrix_index_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    mat_type = arg_types["a"]
    value_type = mat_type._wp_scalar_type_

    return Reference(value_type)


# implements &matrix[i,j] = scalar
add_builtin(
    "index",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": int, "j": int},
    value_func=matrix_index_value_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)


def matrix_vector_sametype(arg_types: Mapping[str, Any]):
    mat_size = arg_types["a"]._shape_[1]
    vec_size = arg_types["value"]._length_
    mat_type = arg_types["a"]._type_
    vec_type = arg_types["value"]._type_
    return mat_size == vec_size and mat_type == vec_type


def matrix_assign_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    mat = args["a"].type
    value_type = strip_reference(args["value"].type)

    idxs = tuple(args[x].type for x in "ij" if args.get(x, None) is not None)
    has_slice = any(isinstance(x, slice_t) for x in idxs)

    if has_slice:
        # Compute the resulting shape from the slicing, with -1 being simple indexing.
        shape = tuple(idx.get_length(mat._shape_[i]) if isinstance(idx, slice_t) else -1 for i, idx in enumerate(idxs))

        # Append any non indexed slice.
        for i in range(len(idxs), len(mat._shape_)):
            shape += (mat._shape_[i],)

        # Count how many dimensions the output value will have.
        ndim = sum(1 for x in shape if x >= 0)
        assert ndim > 0

        if ndim == 1:
            length = shape[0] if shape[0] != -1 else shape[1]

            if type_is_vector(value_type):
                if not types_equal(value_type._wp_scalar_type_, mat._wp_scalar_type_):
                    raise ValueError(
                        f"The provided vector is expected to be of length {length} with dtype {type_repr(mat._wp_scalar_type_)}."
                    )

                if value_type._length_ != length:
                    raise ValueError(
                        f"The length of the provided vector ({value_type._length_}) isn't compatible with the given slice (expected {length})."
                    )

                template_args = (length,)
            else:
                if not types_equal(value_type, mat._wp_scalar_type_):
                    raise ValueError(
                        f"The provided value is expected to be a scalar, or a vector of length {length}, with dtype {type_repr(mat._wp_scalar_type_)}."
                    )

                template_args = ()
        else:
            assert ndim == 2

            # When a matrix dimension is 0, all other dimensions are also expected to be 0.
            if any(x == 0 for x in shape):
                shape = (0,) * len(shape)

            if type_is_matrix(value_type):
                if not types_equal(value_type._wp_scalar_type_, mat._wp_scalar_type_):
                    raise ValueError(
                        f"The provided matrix is expected to be of shape {shape} with dtype {type_repr(mat._wp_scalar_type_)}."
                    )

                if value_type._shape_ != shape:
                    raise ValueError(
                        f"The shape of the provided matrix ({value_type._shape_}) isn't compatible with the given slice (expected {shape})."
                    )

                template_args = shape
            else:
                if not types_equal(value_type, mat._wp_scalar_type_):
                    raise ValueError(
                        f"The provided value is expected to be a scalar, or a matrix of shape {shape}, with dtype {type_repr(mat._wp_scalar_type_)}."
                    )
                template_args = ()
    elif len(idxs) == 1:
        if not type_is_vector(value_type) or not types_equal(value_type._wp_scalar_type_, mat._wp_scalar_type_):
            raise ValueError(
                f"The provided value is expected to be a vector of length {mat._shape_[1]}, with dtype {type_repr(mat._wp_scalar_type_)}."
            )

        if value_type._length_ != mat._shape_[1]:
            raise ValueError(
                f"The length of the provided vector ({value_type._length_}) isn't compatible with the given slice (expected {mat._shape_[1]})."
            )

        template_args = ()
    elif len(idxs) == 2:
        if not types_equal(value_type, mat._wp_scalar_type_):
            raise ValueError(
                f"The provided value is expected to be a scalar of type {type_repr(mat._wp_scalar_type_)}."
            )

        template_args = ()
    else:
        raise AssertionError

    func_args = tuple(args.values())
    return (func_args, template_args)


# implements matrix[i] = value
add_builtin(
    "assign_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "value": Any},
    constraint=matrix_vector_sametype,
    value_type=None,
    dispatch_func=matrix_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)


# implements matrix[i,j] = value
add_builtin(
    "assign_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "j": Any, "value": Any},
    value_type=None,
    dispatch_func=matrix_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)


def matrix_assign_copy_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    mat_type = arg_types["a"]
    return mat_type


# implements matrix[i] = value
add_builtin(
    "assign_copy",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "value": Any},
    value_func=matrix_assign_copy_value_func,
    dispatch_func=matrix_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)


# implements matrix[i,j] = value
add_builtin(
    "assign_copy",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "j": Any, "value": Any},
    value_func=matrix_assign_copy_value_func,
    dispatch_func=matrix_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)


# implements matrix[i] += value
add_builtin(
    "add_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "value": Any},
    constraint=matrix_vector_sametype,
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
)


# implements matrix[i,j] += value
add_builtin(
    "add_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "j": Any, "value": Any},
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
)


# implements matrix[i] -= value
add_builtin(
    "sub_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
)


# implements matrix[i,j] -= value
add_builtin(
    "sub_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "j": Any, "value": Any},
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
)


for t in scalar_types + vector_types + (bool,):
    if "vec" in t.__name__ or "mat" in t.__name__:
        continue

    add_builtin(
        "expect_eq",
        input_types={"a": t, "b": t},
        value_type=None,
        doc="Prints an error to stdout if ``a`` and ``b`` are not equal",
        group="Utility",
        hidden=True,
    )

    add_builtin(
        "expect_neq",
        input_types={"a": t, "b": t},
        value_type=None,
        doc="Prints an error to stdout if ``a`` and ``b`` are not equal",
        group="Utility",
        hidden=True,
        export=False,
    )


def expect_eq_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if not types_equal(arg_types["a"], arg_types["b"]):
        raise RuntimeError("Can't test equality for objects with different types")

    return None


add_builtin(
    "expect_eq",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=expect_eq_value_func,
    doc="Prints an error to stdout if ``a`` and ``b`` are not equal",
    group="Utility",
    hidden=True,
)
add_builtin(
    "expect_neq",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=expect_eq_value_func,
    doc="Prints an error to stdout if ``a`` and ``b`` are equal",
    group="Utility",
    hidden=True,
    export=False,
)

add_builtin(
    "expect_eq",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=expect_eq_value_func,
    doc="Prints an error to stdout if ``a`` and ``b`` are not equal",
    group="Utility",
    hidden=True,
)
add_builtin(
    "expect_neq",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=expect_eq_value_func,
    doc="Prints an error to stdout if ``a`` and ``b`` are equal",
    group="Utility",
    hidden=True,
    export=False,
)

add_builtin(
    "lerp",
    input_types={"a": Float, "b": Float, "t": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``",
    group="Utility",
)
add_builtin(
    "smoothstep",
    input_types={"a": Float, "b": Float, "x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="""Smoothly interpolate between two values ``a`` and ``b`` using a factor ``x``,
    and return a result between 0 and 1 using a cubic Hermite interpolation after clamping.""",
    group="Utility",
)


def lerp_constraint(arg_types: Mapping[str, type]):
    return types_equal(arg_types["a"], arg_types["b"])


def lerp_create_value_func(default):
    def fn(arg_types, arg_values):
        if arg_types is None:
            return default

        if not lerp_constraint(arg_types):
            raise RuntimeError("Can't lerp between objects with different types")

        if arg_types["a"]._wp_scalar_type_ != arg_types["t"]:
            raise RuntimeError("'t' parameter must have the same scalar type as objects you're lerping between")

        return arg_types["a"]

    return fn


add_builtin(
    "lerp",
    input_types={"a": vector(length=Any, dtype=Float), "b": vector(length=Any, dtype=Float), "t": Float},
    constraint=lerp_constraint,
    value_func=lerp_create_value_func(vector(length=Any, dtype=Float)),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``",
    group="Utility",
)
add_builtin(
    "lerp",
    input_types={"a": matrix(shape=(Any, Any), dtype=Float), "b": matrix(shape=(Any, Any), dtype=Float), "t": Float},
    constraint=lerp_constraint,
    value_func=lerp_create_value_func(matrix(shape=(Any, Any), dtype=Float)),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``",
    group="Utility",
)
add_builtin(
    "lerp",
    input_types={"a": quaternion(dtype=Float), "b": quaternion(dtype=Float), "t": Float},
    value_func=lerp_create_value_func(quaternion(dtype=Float)),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``",
    group="Utility",
)
add_builtin(
    "lerp",
    input_types={"a": transformation(dtype=Float), "b": transformation(dtype=Float), "t": Float},
    value_func=lerp_create_value_func(transformation(dtype=Float)),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``",
    group="Utility",
)


# fuzzy compare for float values
def expect_near_constraint(arg_types: Mapping[str, type]):
    if not types_equal(arg_types["a"], arg_types["b"]):
        return False

    if hasattr(arg_types["a"], "_wp_scalar_type_"):
        return types_equal(arg_types["a"]._wp_scalar_type_, arg_types["tolerance"])

    return types_equal(arg_types["a"], arg_types["tolerance"])


add_builtin(
    "expect_near",
    input_types={"a": Float, "b": Float, "tolerance": Float},
    defaults={"tolerance": 1.0e-6},
    constraint=expect_near_constraint,
    value_type=None,
    doc="Prints an error to stdout if ``a`` and ``b`` are not closer than tolerance in magnitude",
    group="Utility",
)
add_builtin(
    "expect_near",
    input_types={"a": vector(length=Any, dtype=Float), "b": vector(length=Any, dtype=Float), "tolerance": Float},
    defaults={"tolerance": 1.0e-6},
    constraint=expect_near_constraint,
    value_type=None,
    doc="Prints an error to stdout if any element of ``a`` and ``b`` are not closer than tolerance in magnitude",
    group="Utility",
)
add_builtin(
    "expect_near",
    input_types={"a": quaternion(dtype=Float), "b": quaternion(dtype=Float), "tolerance": Float},
    defaults={"tolerance": 1.0e-6},
    constraint=expect_near_constraint,
    value_type=None,
    doc="Prints an error to stdout if any element of ``a`` and ``b`` are not closer than tolerance in magnitude",
    group="Utility",
)
add_builtin(
    "expect_near",
    input_types={
        "a": matrix(shape=(Any, Any), dtype=Float),
        "b": matrix(shape=(Any, Any), dtype=Float),
        "tolerance": Float,
    },
    defaults={"tolerance": 1.0e-6},
    constraint=expect_near_constraint,
    value_type=None,
    doc="Prints an error to stdout if any element of ``a`` and ``b`` are not closer than tolerance in magnitude",
    group="Utility",
)

# ---------------------------------
# Algorithms

add_builtin(
    "lower_bound",
    input_types={"arr": array(dtype=Scalar), "value": Scalar},
    value_type=int,
    doc="Search a sorted array ``arr`` for the closest element greater than or equal to ``value``.",
)

add_builtin(
    "lower_bound",
    input_types={"arr": array(dtype=Scalar), "arr_begin": int, "arr_end": int, "value": Scalar},
    value_type=int,
    doc="Search a sorted array ``arr`` in the range [arr_begin, arr_end) for the closest element greater than or equal to ``value``.",
)

# ---------------------------------
# Operators


def op_scalar_create_constraint_func(type, scalar):
    def fn(arg_types: Mapping[str, type]):
        return types_equal(arg_types[type]._wp_scalar_type_, arg_types[scalar])

    return fn


def op_scalar_create_value_func(type, scalar, default):
    def fn(arg_types, arg_values):
        if arg_types is None:
            return default

        if arg_types[type]._wp_scalar_type_ != arg_types[scalar]:
            raise RuntimeError(f"'{scalar}' parameter must have the same scalar type as the modified object")

        return arg_types[type]

    return fn


add_builtin(
    "add", input_types={"a": Scalar, "b": Scalar}, value_func=sametypes_create_value_func(Scalar), group="Operators"
)
add_builtin(
    "add",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": Scalar},
    constraint=op_scalar_create_constraint_func("a", "b"),
    value_func=op_scalar_create_value_func("a", "b", vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"a": Scalar, "b": vector(length=Any, dtype=Scalar)},
    constraint=op_scalar_create_constraint_func("b", "a"),
    value_func=op_scalar_create_value_func("b", "a", vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"a": quaternion(dtype=Scalar), "b": quaternion(dtype=Scalar)},
    value_func=sametypes_create_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": Scalar},
    constraint=op_scalar_create_constraint_func("a", "b"),
    value_func=op_scalar_create_value_func("a", "b", matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"a": Scalar, "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=op_scalar_create_constraint_func("b", "a"),
    value_func=op_scalar_create_value_func("b", "a", matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"a": transformation(dtype=Scalar), "b": transformation(dtype=Scalar)},
    value_func=sametypes_create_value_func(transformation(dtype=Scalar)),
    doc="",
    group="Operators",
)

add_builtin(
    "sub", input_types={"a": Scalar, "b": Scalar}, value_func=sametypes_create_value_func(Scalar), group="Operators"
)
add_builtin(
    "sub",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": Scalar},
    constraint=op_scalar_create_constraint_func("a", "b"),
    value_func=op_scalar_create_value_func("a", "b", vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"a": Scalar, "b": vector(length=Any, dtype=Scalar)},
    constraint=op_scalar_create_constraint_func("b", "a"),
    value_func=op_scalar_create_value_func("b", "a", vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": Scalar},
    constraint=op_scalar_create_constraint_func("a", "b"),
    value_func=op_scalar_create_value_func("a", "b", matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"a": Scalar, "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=op_scalar_create_constraint_func("b", "a"),
    value_func=op_scalar_create_value_func("b", "a", matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"a": quaternion(dtype=Scalar), "b": quaternion(dtype=Scalar)},
    value_func=sametypes_create_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"a": transformation(dtype=Scalar), "b": transformation(dtype=Scalar)},
    value_func=sametypes_create_value_func(transformation(dtype=Scalar)),
    doc="",
    group="Operators",
)

# bitwise operators
add_builtin("bit_and", input_types={"a": Int, "b": Int}, value_func=sametypes_create_value_func(Int))
add_builtin("bit_or", input_types={"a": Int, "b": Int}, value_func=sametypes_create_value_func(Int))
add_builtin("bit_xor", input_types={"a": Int, "b": Int}, value_func=sametypes_create_value_func(Int))
add_builtin("lshift", input_types={"a": Int, "b": Int}, value_func=sametypes_create_value_func(Int))
add_builtin("rshift", input_types={"a": Int, "b": Int}, value_func=sametypes_create_value_func(Int))
add_builtin("invert", input_types={"a": Int}, value_func=sametypes_create_value_func(Int))


add_builtin(
    "mul", input_types={"a": Scalar, "b": Scalar}, value_func=sametypes_create_value_func(Scalar), group="Operators"
)


def scalar_mul_create_value_func(default):
    def fn(arg_types, arg_values):
        if arg_types is None:
            return default

        scalar = next(t for t in arg_types.values() if t in scalar_types)
        compound = next(t for t in arg_types.values() if t not in scalar_types)
        if scalar != compound._wp_scalar_type_:
            raise RuntimeError("Object and coefficient must have the same scalar type when multiplying by scalar")

        return compound

    return fn


add_builtin(
    "mul",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": Scalar},
    value_func=scalar_mul_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": Scalar, "b": vector(length=Any, dtype=Scalar)},
    value_func=scalar_mul_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": quaternion(dtype=Scalar), "b": Scalar},
    value_func=scalar_mul_create_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": Scalar, "b": quaternion(dtype=Scalar)},
    value_func=scalar_mul_create_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": quaternion(dtype=Scalar), "b": quaternion(dtype=Scalar)},
    value_func=sametypes_create_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": Scalar, "b": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=scalar_mul_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": Scalar},
    value_func=scalar_mul_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)


def matvec_mul_constraint(arg_types: Mapping[str, type]):
    return arg_types["a"]._shape_[1] == arg_types["b"]._length_


def matvec_mul_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return vector(length=Any, dtype=Scalar)

    if arg_types["a"]._wp_scalar_type_ != arg_types["b"]._wp_scalar_type_:
        raise RuntimeError(
            f"Can't multiply matrix and vector with different types {arg_types['a']._wp_scalar_type_}, {arg_types['b']._wp_scalar_type_}"
        )

    if not matvec_mul_constraint(arg_types):
        raise RuntimeError(
            f"Can't multiply matrix of shape {arg_types['a']._shape_} and vector with length {arg_types['b']._length_}"
        )

    return vector(length=arg_types["a"]._shape_[0], dtype=arg_types["a"]._wp_scalar_type_)


add_builtin(
    "mul",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=matvec_mul_constraint,
    value_func=matvec_mul_value_func,
    doc="",
    group="Operators",
)


def mul_vecmat_constraint(arg_types: Mapping[str, type]):
    return arg_types["b"]._shape_[0] == arg_types["a"]._length_


def mul_vecmat_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return vector(length=Any, dtype=Scalar)

    if arg_types["b"]._wp_scalar_type_ != arg_types["a"]._wp_scalar_type_:
        raise RuntimeError(
            f"Can't multiply vector and matrix with different types {arg_types['b']._wp_scalar_type_}, {arg_types['a']._wp_scalar_type_}"
        )

    if not mul_vecmat_constraint(arg_types):
        raise RuntimeError(
            f"Can't multiply vector with length {arg_types['a']._length_} and matrix of shape {arg_types['b']._shape_}"
        )

    return vector(length=arg_types["b"]._shape_[1], dtype=arg_types["b"]._wp_scalar_type_)


add_builtin(
    "mul",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=mul_vecmat_constraint,
    value_func=mul_vecmat_value_func,
    doc="",
    group="Operators",
)


def matmat_mul_constraint(arg_types: Mapping[str, type]):
    return arg_types["a"]._shape_[1] == arg_types["b"]._shape_[0]


def matmat_mul_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return matrix(shape=(Any, Any), dtype=Scalar)

    if arg_types["a"]._wp_scalar_type_ != arg_types["b"]._wp_scalar_type_:
        raise RuntimeError(
            f"Can't multiply matrices with different types {arg_types['a']._wp_scalar_type_}, {arg_types['b']._wp_scalar_type_}"
        )

    if not matmat_mul_constraint(arg_types):
        raise RuntimeError(f"Can't multiply matrix of shapes {arg_types['a']._shape_} and {arg_types['b']._shape_}")

    return matrix(shape=(arg_types["a"]._shape_[0], arg_types["b"]._shape_[1]), dtype=arg_types["a"]._wp_scalar_type_)


add_builtin(
    "mul",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=matmat_mul_constraint,
    value_func=matmat_mul_value_func,
    doc="",
    group="Operators",
)


add_builtin(
    "mul",
    input_types={"a": transformation(dtype=Scalar), "b": transformation(dtype=Scalar)},
    value_func=sametypes_create_value_func(transformation(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": Scalar, "b": transformation(dtype=Scalar)},
    value_func=scalar_mul_create_value_func(transformation(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": transformation(dtype=Scalar), "b": Scalar},
    value_func=scalar_mul_create_value_func(transformation(dtype=Scalar)),
    doc="",
    group="Operators",
)

add_builtin(
    "mod",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Modulo operation using truncated division.",
    group="Operators",
)
add_builtin(
    "mod",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="Modulo operation using truncated division.",
    group="Operators",
)
add_builtin(
    "mod",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": Scalar},
    constraint=op_scalar_create_constraint_func("a", "b"),
    value_func=op_scalar_create_value_func("a", "b", vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mod",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": Scalar},
    constraint=op_scalar_create_constraint_func("a", "b"),
    value_func=op_scalar_create_value_func("a", "b", matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)

add_builtin(
    "div",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="",
    group="Operators",
    require_original_output_arg=True,
)
add_builtin(
    "div",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": Scalar},
    value_func=scalar_mul_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"a": Scalar, "b": vector(length=Any, dtype=Scalar)},
    value_func=scalar_mul_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": Scalar},
    value_func=scalar_mul_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"a": Scalar, "b": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=scalar_mul_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"a": quaternion(dtype=Scalar), "b": Scalar},
    value_func=scalar_mul_create_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"a": Scalar, "b": quaternion(dtype=Scalar)},
    value_func=scalar_mul_create_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)

add_builtin(
    "floordiv",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="",
    group="Operators",
)

add_builtin("pos", input_types={"x": Scalar}, value_func=sametypes_create_value_func(Scalar), group="Operators")
add_builtin(
    "pos",
    input_types={"x": vector(length=Any, dtype=Scalar)},
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "pos",
    input_types={"x": quaternion(dtype=Scalar)},
    value_func=sametypes_create_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "pos",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin("neg", input_types={"x": Scalar}, value_func=sametypes_create_value_func(Scalar), group="Operators")
add_builtin(
    "neg",
    input_types={"x": vector(length=Any, dtype=Scalar)},
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "neg",
    input_types={"x": quaternion(dtype=Scalar)},
    value_func=sametypes_create_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "neg",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)

add_builtin("unot", input_types={"a": builtins.bool}, value_type=builtins.bool, doc="", group="Operators")
for t in int_types:
    add_builtin("unot", input_types={"a": t}, value_type=builtins.bool, doc="", group="Operators")


add_builtin("unot", input_types={"a": array(dtype=Any)}, value_type=builtins.bool, doc="", group="Operators")


# Tile operators
def tile_unary_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Scalar, shape=Tuple[int, ...])

    t = arg_types["x"]

    if not is_tile(t):
        raise TypeError(f"Expected tile for unary expression, got {t}")

    return tile(dtype=t.dtype, shape=t.shape)


def tile_scalar_mul_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, ...])

    x = arg_types["x"]
    y = arg_types["y"]

    # tile*scalar
    if is_tile(x):
        if x.dtype != y:
            raise TypeError(f"Scalar factor type {y} does not match tile type {x.dtype} for tile*scalar")

        return tile(dtype=x.dtype, shape=x.shape)

    # scalar*tile
    if is_tile(y):
        if y.dtype != x:
            raise TypeError(f"Scalar factor type {x} does not match tile type {y.dtype} for scalar*tile")

        return tile(dtype=y.dtype, shape=y.shape)


add_builtin(
    "neg",
    input_types={"x": tile(dtype=Any, shape=Tuple[int, ...])},
    value_func=tile_unary_value_func,
    doc="Negate each element of a tile",
    export=False,
    native_func="tile_neg",
    group="Operators",
)

add_builtin(
    "add",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "b": tile(dtype=Any, shape=Tuple[int, ...])},
    value_func=tile_binary_map_value_func,
    # dispatch_func=tile_map_dispatch_func,
    # variadic=True,
    native_func="tile_add",
    doc="Add each element of two tiles together",
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "sub",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "b": tile(dtype=Any, shape=Tuple[int, ...])},
    value_func=tile_binary_map_value_func,
    # dispatch_func=tile_map_dispatch_func,
    # variadic=True,
    native_func="tile_sub",
    doc="Subtract each element b from a",
    group="Tile Primitives",
    export=False,
)


add_builtin(
    "mul",
    input_types={"x": tile(dtype=Any, shape=Tuple[int, ...]), "y": Scalar},
    value_func=tile_scalar_mul_value_func,
    doc="Multiply each element of a tile by a scalar",
    export=False,
    native_func="tile_mul",
    group="Operators",
)

add_builtin(
    "mul",
    input_types={"x": Scalar, "y": tile(dtype=Any, shape=Tuple[int, ...])},
    value_func=tile_scalar_mul_value_func,
    doc="Multiply each element of a tile by a scalar",
    export=False,
    native_func="tile_mul",
    group="Operators",
)


def tile_inplace_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    a = args["a"]
    b = args["b"]

    a_type = input_types["a"]
    b_type = input_types["b"]

    if a_type.shape != b_type.shape:
        raise ValueError(f"Tile inplace arguments must have the same shape, got {a_type.shape} and {b_type.shape}")

    func_args = (a, b)
    template_args = ()
    return (func_args, template_args)


add_builtin(
    "add_inplace",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "b": tile(dtype=Any, shape=Tuple[int, ...])},
    value_type=None,
    dispatch_func=tile_inplace_dispatch_func,
    export=False,
    hidden=True,
    native_func="tile_add_inplace",
    group="Operators",
)


add_builtin(
    "sub_inplace",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...]), "b": tile(dtype=Any, shape=Tuple[int, ...])},
    value_type=None,
    dispatch_func=tile_inplace_dispatch_func,
    export=False,
    hidden=True,
    native_func="tile_sub_inplace",
    group="Operators",
)


def tile_diag_add_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Any, shape=Tuple[int, int])

    a = arg_types["a"]
    d = arg_types["d"]

    if not is_tile(a):
        raise TypeError(f"tile_diag_add() 'a' argument must be a tile, got {a!r}")

    if not is_tile(d):
        raise TypeError(f"tile_diag_add() 'd' argument must be a tile, got {d!r}")

    if not types_equal(a.dtype, d.dtype):
        raise TypeError(f"tile_diag_add() arguments must have the same dtype, got {a.dtype} and {d.dtype}")

    if len(a.shape) != 2:
        raise TypeError("tile_diag_add() argument 'a' must be a 2D tile")

    if len(d.shape) != 1:
        raise TypeError("tile_diag_add() argument 'd' must be a 1D tile")

    if a.shape[0] != a.shape[1]:
        raise ValueError("tile_diag_add() 'a' argument must be square")

    if a.shape[0] != d.shape[0]:
        raise ValueError(
            f"tile_diag_add() 'd' argument must have the same number of elements as the number of rows in 'a', "
            f"got {d.shape[0]} elements in 'd' and {a.shape[0]} rows in 'a'"
        )

    # use first argument to define output type
    return tile(dtype=a.dtype, shape=a.shape, layout=a.layout, strides=a.strides, storage="shared")


def tile_diag_add_lto_dispatch_func(
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp.context.ModuleBuilder,
):
    a = arg_values["a"]
    d = arg_values["d"]
    # force the storage type of the input variables to shared memory
    a.type.storage = "shared"
    d.type.storage = "shared"
    out = return_values[0]
    return ((a, d, out), [], [], 0)


add_builtin(
    "tile_diag_add",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, int]), "d": tile(dtype=Any, shape=Tuple[int])},
    value_func=tile_diag_add_value_func,
    lto_dispatch_func=tile_diag_add_lto_dispatch_func,
    native_func="tile_diag_add",
    doc="Add a square matrix and a diagonal matrix 'd' represented as a 1D tile",
    group="Tile Primitives",
    export=False,
)


##
## MathDx, LTOIR-based, Tile functions
##


##
## Matmul
##


def tile_matmul_out_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return None

    a = arg_types["a"]
    b = arg_types["b"]

    if not is_tile(a):
        raise TypeError(f"tile_matmul() 'a' argument must be a tile, got {a!r}")

    if not is_tile(b):
        raise TypeError(f"tile_matmul() 'b' argument must be a tile, got {b!r}")

    if not is_tile(arg_types["out"]):
        raise TypeError(f"tile_matmul() 'out' argument must be a tile, got {arg_types['out']!r}")

    return None


def tile_matmul_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Float, shape=Tuple[int, int])

    a = arg_types["a"]
    b = arg_types["b"]

    if not is_tile(a):
        raise TypeError(f"tile_matmul() 'a' argument must be a tile, got {a!r}")

    if not is_tile(b):
        raise TypeError(f"tile_matmul() 'b' argument must be a tile, got {b!r}")

    return tile(dtype=a.dtype, shape=(a.shape[0], b.shape[1]), storage="shared")


def tile_matmul_lto_dispatch_func(
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp.context.ModuleBuilder,
):
    a = arg_values["a"]
    b = arg_values["b"]

    if len(return_values) > 0:
        accumulate = 0  # for c = tile_matmul(a,b) case we want to overwrite c value
        out = return_values[0]
    else:
        accumulate = 1  # for tile_matmul(a,b,c) case we want to add to c value
        out = arg_values["out"]

    if not is_tile(out.type):
        raise TypeError(f"tile_matmul() 'out' argument must be a tile, got {out!r}")

    if any(arg.type.dtype not in [float16, float32, float64, vec2h, vec2f, vec2d] for arg in [a, b, out]):
        raise TypeError(
            "tile_matmul() arguments must be tiles of float16, float32 or float64, vec2h, vec2f, vec2d entries"
        )

    if (
        (a.type.shape[1] != b.type.shape[0])
        or (a.type.shape[0] != out.type.shape[0])
        or (b.type.shape[1] != out.type.shape[1])
    ):
        raise ValueError("tile_matmul(A, B, C) requires sizes of A, B and C to be consistent for a matmul")

    # set the storage type to the inputs to shared
    a.type.storage = "shared"
    b.type.storage = "shared"
    out.type.storage = "shared"
    template_args = [accumulate]

    M, K = a.type.shape[0], a.type.shape[1]
    _, N = b.type.shape[0], b.type.shape[1]
    num_threads = options["block_dim"]
    arch = options["output_arch"]

    if arch is None or not warp.context.runtime.core.wp_is_mathdx_enabled():
        # CPU/no-MathDx dispatch
        return ((0, 0, 0, a, b, out), template_args, [], 0)
    else:

        def tile_flip_layout(layout):
            if layout == "rowmajor":
                return "colmajor"
            elif layout == "colmajor":
                return "rowmajor"

        # generate the LTOs
        #    C += A * B
        (fun_forward, lto_forward) = warp.build.build_lto_dot(
            M,
            N,
            K,
            a.type.dtype,
            b.type.dtype,
            out.type.dtype,
            a.type.layout,
            b.type.layout,
            out.type.layout,
            arch,
            num_threads,
            builder,
        )
        if warp.config.enable_backward:
            # adjA += adjC * B^T - Transpose ~= flipped layout
            (fun_backward_A, lto_backward_A) = warp.build.build_lto_dot(
                M,
                K,
                N,
                out.type.dtype,
                b.type.dtype,
                a.type.dtype,
                out.type.layout,
                tile_flip_layout(b.type.layout),
                a.type.layout,
                arch,
                num_threads,
                builder,
            )
            # adjB += A^T * adjC - Transpose ~= flipped layout
            (fun_backward_B, lto_backward_B) = warp.build.build_lto_dot(
                K,
                N,
                M,
                a.type.dtype,
                out.type.dtype,
                b.type.dtype,
                tile_flip_layout(a.type.layout),
                out.type.layout,
                b.type.layout,
                arch,
                num_threads,
                builder,
            )
        else:
            # adjoints aren't computed, so we reuse fun_forward as a dummy arg
            (fun_backward_A, lto_backward_A) = (fun_forward, None)
            (fun_backward_B, lto_backward_B) = (fun_forward, None)

        return (
            (
                Var(fun_forward, str, False, True, False),
                Var(fun_backward_A, str, False, True, False),
                Var(fun_backward_B, str, False, True, False),
                a,
                b,
                out,
            ),
            template_args,
            [lto_forward, lto_backward_A, lto_backward_B],
            0,
        )


add_builtin(
    "tile_matmul",
    input_types={
        "a": tile(dtype=Float, shape=Tuple[int, int]),
        "b": tile(dtype=Float, shape=Tuple[int, int]),
        "out": tile(dtype=Float, shape=Tuple[int, int]),
    },
    value_func=tile_matmul_out_value_func,
    lto_dispatch_func=tile_matmul_lto_dispatch_func,
    variadic=False,
    doc="""Computes the matrix product and accumulates ``out += a*b``.

    Supported datatypes are:
        * fp16, fp32, fp64 (real)
        * vec2h, vec2f, vec2d (complex)

    All input and output tiles must have the same datatype. Tile data will automatically be migrated
    to shared memory if necessary and will use TensorCore operations when available.

    :param a: A tile with ``shape=(M, K)``
    :param b: A tile with ``shape=(K, N)``
    :param out: A tile with ``shape=(M, N)``
    """,
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "tile_matmul",
    input_types={"a": tile(dtype=Float, shape=Tuple[int, int]), "b": tile(dtype=Float, shape=Tuple[int, int])},
    value_func=tile_matmul_value_func,
    lto_dispatch_func=tile_matmul_lto_dispatch_func,
    variadic=False,
    doc="""Computes the matrix product ``out = a*b``.

    Supported datatypes are:
        * fp16, fp32, fp64 (real)
        * vec2h, vec2f, vec2d (complex)

    Both input tiles must have the same datatype. Tile data will automatically be migrated
    to shared memory if necessary and will use TensorCore operations when available.

    :param a: A tile with ``shape=(M, K)``
    :param b: A tile with ``shape=(K, N)``
    :returns: A tile with ``shape=(M, N)``
    """,
    group="Tile Primitives",
    export=False,
)


##
## FFT
##
def tile_fft_generic_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=vector(length=2, dtype=Float), shape=Tuple[int, int])

    if len(arg_types) != 1:
        raise TypeError(f"tile_fft() takes exactly 1 positional argument but {len(arg_types)} were given")

    inout = arg_types["inout"]

    if not is_tile(inout):
        raise TypeError(f"tile_fft() argument must be a tile, got {inout!r}")

    if inout.storage != "register":
        raise ValueError(f"tile_fft() argument must have 'register' storage, got {inout.storage}")

    if inout.dtype not in [vec2f, vec2d]:
        raise TypeError(
            f"tile_fft() argument must be a tile of vec2f or vec2d (interpreted as complex) entries, got {inout.dtype!r}"
        )

    return None


def tile_fft_generic_lto_dispatch_func(
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp.context.ModuleBuilder,
    direction: str | None = None,
):
    inout = arg_values["inout"]
    inout.type.storage = "register"

    # see libcufftdx.hpp
    if direction == "forward":
        dir = 0  # CUFFTDX_DIRECTION_FORWARD
    elif direction == "inverse":
        dir = 1  # CUFFTDX_DIRECTION_INVERSE
    else:
        raise ValueError(f"Invalid direction: {direction!r}.  Expected 'forward' or 'inverse'.")

    if inout.type.dtype == vec2f:
        dtype = "wp::vec2f"
        precision = 5  # COMMONDX_PRECISION_F32
    elif inout.type.dtype == vec2d:
        dtype = "wp::vec2d"
        precision = 6  # COMMONDX_PRECISION_F64
    else:
        raise TypeError(f"Unsupported data type, got {dtype!r}")

    # M FFTs of size N each
    batch, size = inout.type.shape[0], inout.type.shape[1]
    num_threads = options["block_dim"]
    arch = options["output_arch"]
    ept = size // num_threads

    if arch is None or not warp.context.runtime.core.wp_is_mathdx_enabled():
        # CPU/no-MathDx dispatch
        return ([], [], [], 0)
    else:
        # generate the LTO
        lto_symbol, lto_code_data, shared_memory_bytes = warp.build.build_lto_fft(
            arch, size, ept, direction, dir, precision, builder
        )

        return (
            (
                Var(lto_symbol, str, False, True, False),
                Var(dtype, str, False, True, False),
                Var(str(shared_memory_bytes), str, False, True, False),
                Var(str(batch), str, False, True, False),
                Var(str(ept), str, False, True, False),
                inout,
            ),
            [],
            [lto_code_data],
            shared_memory_bytes,
        )


add_builtin(
    "tile_fft",
    input_types={"inout": tile(dtype=vector(length=2, dtype=Float), shape=Tuple[int, int])},
    value_func=tile_fft_generic_value_func,
    lto_dispatch_func=functools.partial(tile_fft_generic_lto_dispatch_func, direction="forward"),
    variadic=True,
    doc="""Compute the forward FFT along the second dimension of a 2D tile of data.

    This function cooperatively computes the forward FFT on a tile of data inplace, treating each row individually.

    Note that computing the adjoint is not yet supported.

    Supported datatypes are:
        * vec2f, vec2d

    :param inout: The input/output tile""",
    group="Tile Primitives",
    export=False,
    namespace="",
)

add_builtin(
    "tile_ifft",
    input_types={"inout": tile(dtype=vector(length=2, dtype=Float), shape=Tuple[int, int])},
    value_func=tile_fft_generic_value_func,
    lto_dispatch_func=functools.partial(tile_fft_generic_lto_dispatch_func, direction="inverse"),
    variadic=True,
    doc="""Compute the inverse FFT along the second dimension of a 2D tile of data.

    This function cooperatively computes the inverse FFT on a tile of data inplace, treating each row individually.

    Note that computing the adjoint is not yet supported.

    Supported datatypes are:
        * vec2f, vec2d

    :param inout: The input/output tile""",
    group="Tile Primitives",
    export=False,
    namespace="",
)


##
## Cholesky
##
def tile_cholesky_generic_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Float, shape=Tuple[int, int])

    if len(arg_types) != 1:
        raise TypeError("tile_cholesky() requires 1 positional args")

    a = arg_types["A"]

    if not is_tile(a):
        raise TypeError(f"tile_cholesky() argument must be a tile, got {a!r}")

    if len(a.shape) != 2:
        raise ValueError("tile_cholesky() argument must be a 2D tile")

    if a.shape[0] != a.shape[1]:
        raise ValueError("tile_cholesky() argument must be square")

    return tile(dtype=a.dtype, shape=a.shape, layout=a.layout, strides=a.strides, storage="shared")


cusolver_function_map = {"getrf": 0, "getrf_no_pivot": 1, "potrf": 2, "potrs": 3, "trsm": 4}

cusolver_type_map = {float32: ("wp::float32", 5), float64: ("wp::float64", 6)}

cusolver_fill_mode_map = {"upper": 0, "lower": 1}

cusolver_side_map = {"-": -1, "left": 0, "right": 1}

cusolver_diag_map = {"-": -1, "unit": 0, "nounit": 1}


def tile_cholesky_generic_lto_dispatch_func(
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp.context.ModuleBuilder,
):
    a = arg_values["A"]
    # force source tile to shared memory
    a.type.storage = "shared"

    if a.type.dtype not in cusolver_type_map.keys():
        raise TypeError("tile_cholesky() argument must be a tile of float32 or float64 entries")

    if len(return_values) != 1:
        raise TypeError("tile_cholesky() returns one output")
    out = return_values[0]

    dtype, precision_enum = cusolver_type_map[a.type.dtype]

    # We already ensured a is square in tile_cholesky_generic_value_func()
    M, N = a.type.shape
    if out.type.shape[0] != M or out.type.shape[1] != M:
        raise ValueError("tile_cholesky() output tile must be square")

    solver = "potrf"
    solver_enum = cusolver_function_map[solver]

    side_enum = cusolver_side_map["-"]
    diag_enum = cusolver_diag_map["-"]
    fill_mode = cusolver_fill_mode_map["lower"]

    arch = options["output_arch"]
    num_threads = options["block_dim"]
    parameter_list = f"({dtype}*, int*)"

    if arch is None or not warp.context.runtime.core.wp_is_mathdx_enabled():
        # CPU/no-MathDx dispatch
        return ((0, a, out), [], [], 0)
    else:
        # generate the LTO
        lto_symbol, lto_code_data = warp.build.build_lto_solver(
            M,
            N,
            1,
            solver,
            solver_enum,
            side_enum,
            diag_enum,
            a.type.layout,
            out.type.layout,
            fill_mode,
            arch,
            precision_enum,
            num_threads,
            parameter_list,
            builder,
        )

        return ((Var(lto_symbol, str, False, True, False), a, out), [], [lto_code_data], 0)


add_builtin(
    "tile_cholesky",
    input_types={"A": tile(dtype=Float, shape=Tuple[int, int])},
    value_func=tile_cholesky_generic_value_func,
    lto_dispatch_func=tile_cholesky_generic_lto_dispatch_func,
    variadic=True,
    doc="""Compute the Cholesky factorization L of a matrix A.
    L is lower triangular and satisfies LL^T = A.

    Only the lower triangular portion of A is used for the decomposition;
    the upper triangular part may be left unspecified.

    Note that computing the adjoint is not yet supported.

    Supported datatypes are:
        * float32
        * float64

    :param A: A square, symmetric positive-definite, matrix. Only the lower triangular part of A is needed; the upper part is ignored.
    :returns L: A square, lower triangular, matrix, such that LL^T = A""",
    group="Tile Primitives",
    export=False,
    namespace="",
)


def tile_cholesky_solve_generic_value_func(arg_types, arg_values):
    if arg_types is None:
        return None

    if len(arg_types) != 2:
        raise TypeError("tile_cholesky_solve() requires exactly 2 positional args")

    l = arg_types["L"]
    y = arg_types["y"]

    if not is_tile(l):
        raise TypeError(f"tile_cholesky_solve() 'L' argument must be a tile, got {l!r}")

    if not is_tile(y):
        raise TypeError(f"tile_cholesky_solve() 'y' argument must be a tile, got {l!r}")

    if not types_equal(l.dtype, y.dtype):
        raise TypeError(f"tile_cholesky_solve() arguments must have the same dtype, got {l.dtype} and {y.dtype}")

    if l.shape[0] != l.shape[1]:
        raise ValueError("tile_cholesky_solve() 'L' argument must be square")

    if len(y.shape) > 2 or len(y.shape) < 1:
        raise TypeError("tile_cholesky_solve() 'y' argument must be a 1D or 2D tile")

    if y.shape[0] != l.shape[0]:
        raise ValueError(
            f"tile_cholesky_solve() 'y' argument must have the same number of elements as the number of rows in 'L', "
            f"got {y.shape[0]} elements in 'x' and {l.shape[0]} rows in 'L'"
        )

    return tile(dtype=l.dtype, shape=y.shape, layout=y.layout, strides=y.strides, storage="shared")


def tile_cholesky_solve_generic_lto_dispatch_func(
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp.context.ModuleBuilder,
):
    L = arg_values["L"]
    y = arg_values["y"]
    # force the storage type of the input variables to shared memory
    L.type.storage = "shared"
    y.type.storage = "shared"

    if len(return_values) != 1:
        raise TypeError(f"tile_cholesky_solve() must return exactly one value, got {len(return_values)}")

    x = return_values[0]

    if any(T not in cusolver_type_map.keys() for T in [y.type.dtype, L.type.dtype]):
        raise TypeError("tile_cholesky_solve() arguments be tiles of float64 or float32")

    dtype, precision_enum = cusolver_type_map[L.type.dtype]
    M, N = L.type.shape
    NRHS = x.type.shape[1] if len(x.type.shape) > 1 else 1

    if len(x.type.shape) > 2 or len(x.type.shape) < 1:
        raise TypeError(f"tile_cholesky_solve() output vector must be 1D or 2D, got {len(x.type.shape)}-D")

    if x.type.shape[0] != M:
        raise ValueError(
            "tile_cholesky_solve() output vector must have same number of elements as the number of rows in 'L' "
            f"got {x.type.shape[0]} elements in output and {M} rows in 'L'"
        )

    solver = "potrs"
    solver_enum = cusolver_function_map[solver]

    side_enum = cusolver_side_map["-"]
    diag_enum = cusolver_diag_map["-"]
    fill_mode = cusolver_fill_mode_map["lower"]

    arch = options["output_arch"]
    num_threads = options["block_dim"]
    parameter_list = f"({dtype}*, {dtype}*)"

    if arch is None or not warp.context.runtime.core.wp_is_mathdx_enabled():
        # CPU/no-MathDx dispatch
        return ((0, L, y, x), [], [], 0)
    else:
        # generate the LTO
        lto_symbol, lto_code_data = warp.build.build_lto_solver(
            M,
            N,
            NRHS,
            solver,
            solver_enum,
            side_enum,
            diag_enum,
            L.type.layout,
            y.type.layout,
            fill_mode,
            arch,
            precision_enum,
            num_threads,
            parameter_list,
            builder,
        )

        return ((Var(lto_symbol, str, False, True, False), L, y, x), [], [lto_code_data], 0)


add_builtin(
    "tile_cholesky_solve",
    input_types={"L": tile(dtype=Float, shape=Tuple[int, int]), "y": tile(dtype=Float, shape=Tuple[int])},
    value_func=tile_cholesky_solve_generic_value_func,
    lto_dispatch_func=tile_cholesky_solve_generic_lto_dispatch_func,
    variadic=True,
    doc="""With L such that LL^T = A, solve for x in Ax = y

    Note that computing the adjoint is not yet supported.

    Supported datatypes are:
        * float32
        * float64

    :param L: A square, lower triangular, matrix, such that LL^T = A
    :param y: A 1D or 2D tile of length M
    :returns x: A tile of the same shape as y such that LL^T x = y""",
    group="Tile Primitives",
    export=False,
    namespace="",
)


def tile_lower_solve_generic_lto_dispatch_func(
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp.context.ModuleBuilder,
):
    L = arg_values["L"]
    y = arg_values["y"]
    # force the storage type of the input variables to shared memory
    L.type.storage = "shared"
    y.type.storage = "shared"

    if any(T not in cusolver_type_map.keys() for T in [y.type.dtype, L.type.dtype]):
        raise TypeError("tile_lower_solve() arguments must be tiles of float64 or float32")

    if len(return_values) != 1:
        raise TypeError(f"tile_lower_solve() must return exactly one value, got {len(return_values)}")

    z = return_values[0]

    dtype, precision_enum = cusolver_type_map[L.type.dtype]
    M, N = L.type.shape
    NRHS = z.type.shape[1] if len(z.type.shape) > 1 else 1

    if len(z.type.shape) > 2 or len(z.type.shape) < 1:
        raise TypeError(f"tile_lower_solve() output vector must be 1D or 2D, got {len(z.type.shape)}-D")

    if z.type.shape[0] != M:
        raise ValueError(
            "tile_lower_solve() output vector must have same number of elements as the number of rows in 'L' "
            f"got {z.type.shape[0]} elements in output and {M} rows in 'L'"
        )

    solver = "trsm"
    solver_enum = cusolver_function_map[solver]

    side_enum = cusolver_side_map["left"]
    diag_enum = cusolver_diag_map["nounit"]
    fill_mode = cusolver_fill_mode_map["lower"]

    arch = options["output_arch"]
    num_threads = options["block_dim"]
    parameter_list = f"({dtype}*, {dtype}*)"

    if arch is None or not warp.context.runtime.core.wp_is_mathdx_enabled():
        # CPU/no-MathDx dispatch
        return ((0, L, y, z), [], [], 0)
    else:
        # generate the LTO
        lto_symbol, lto_code_data = warp.build.build_lto_solver(
            M,
            N,
            NRHS,
            solver,
            solver_enum,
            side_enum,
            diag_enum,
            L.type.layout,
            y.type.layout,
            fill_mode,
            arch,
            precision_enum,
            num_threads,
            parameter_list,
            builder,
        )

        return ((Var(lto_symbol, str, False, True, False), L, y, z), [], [lto_code_data], 0)


def tile_lower_solve_generic_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Float, shape=Tuple[int])

    if len(arg_types) != 2:
        raise TypeError("tile_lower_solve() requires exactly 2 positional args")

    l = arg_types["L"]
    y = arg_types["y"]

    if not is_tile(l):
        raise TypeError(f"tile_lower_solve() 'L' argument must be a tile, got {l!r}")

    if not is_tile(y):
        raise TypeError(f"tile_lower_solve() 'y' argument must be a tile, got {y!r}")

    if not types_equal(l.dtype, y.dtype):
        raise TypeError(f"tile_lower_solve() arguments must have the same dtype, got {l.dtype} and {y.dtype}")

    if l.shape[0] != l.shape[1]:
        raise ValueError("tile_lower_solve() 'L' argument must be square")

    if len(y.shape) > 2 or len(y.shape) < 1:
        raise TypeError("tile_lower_solve() 'y' argument must be a 1D or 2D tile")

    if y.shape[0] != l.shape[0]:
        raise ValueError(
            f"tile_lower_solve() 'y' argument must have the same number of elements as the number of rows in 'L', "
            f"got {y.shape[0]} elements in 'y' and {l.shape[0]} rows in 'L'"
        )

    return tile(dtype=l.dtype, shape=y.shape, layout=y.layout, strides=y.strides, storage="shared")


add_builtin(
    "tile_lower_solve",
    input_types={"L": tile(dtype=Float, shape=Tuple[int, int]), "y": tile(dtype=Float, shape=Tuple[int])},
    value_func=tile_lower_solve_generic_value_func,
    lto_dispatch_func=tile_lower_solve_generic_lto_dispatch_func,
    variadic=True,
    doc="""Solve for z in Lz = y, where L is a lower triangular matrix.

    This performs general forward substitution for a lower triangular system.

    Note that computing the adjoint is not yet supported.

    Supported datatypes are:
        * float32
        * float64

    :param L: A square, non-singular, lower triangular matrix
    :param y: A 1D or 2D tile with compatible shape
    :returns z: A tile of the same shape as y such that Lz = y""",
    group="Tile Primitives",
    export=False,
    namespace="",
)


def tile_upper_solve_generic_lto_dispatch_func(
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp.context.ModuleBuilder,
):
    U = arg_values["U"]
    z = arg_values["z"]
    # force the storage type of the input variables to shared memory
    U.type.storage = "shared"
    z.type.storage = "shared"

    if any(T not in cusolver_type_map.keys() for T in [z.type.dtype, U.type.dtype]):
        raise TypeError("tile_upper_solve() arguments must be tiles of float64 or float32")

    if len(return_values) != 1:
        raise TypeError(f"tile_upper_solve() must return exactly one value, got {len(return_values)}")

    x = return_values[0]

    dtype, precision_enum = cusolver_type_map[U.type.dtype]
    M, N = U.type.shape
    NRHS = x.type.shape[1] if len(x.type.shape) > 1 else 1

    if len(z.type.shape) > 2 or len(z.type.shape) < 1:
        raise TypeError(f"tile_upper_solve() output tile must be 1D or 2D, got {len(z.type.shape)}-D")

    if z.type.shape[0] != M:
        raise ValueError(
            "tile_upper_solve() output tile must have same number of elements as the number of rows in 'U' "
            f"got {z.type.shape[0]} elements in output and {M} rows in 'U'"
        )

    solver = "trsm"
    solver_enum = cusolver_function_map[solver]

    side_enum = cusolver_side_map["left"]
    diag_enum = cusolver_diag_map["nounit"]
    fill_mode = cusolver_fill_mode_map["upper"]

    arch = options["output_arch"]
    num_threads = options["block_dim"]
    parameter_list = f"({dtype}*, {dtype}*)"

    if arch is None or not warp.context.runtime.core.wp_is_mathdx_enabled():
        # CPU/no-MathDx dispatch
        return ((0, U, z, x), [], [], 0)
    else:
        # generate the LTO
        lto_symbol, lto_code_data = warp.build.build_lto_solver(
            M,
            N,
            NRHS,
            solver,
            solver_enum,
            side_enum,
            diag_enum,
            U.type.layout,
            z.type.layout,
            fill_mode,
            arch,
            precision_enum,
            num_threads,
            parameter_list,
            builder,
        )

        return ((Var(lto_symbol, str, False, True, False), U, z, x), [], [lto_code_data], 0)


def tile_upper_solve_generic_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Float, shape=Tuple[int])

    if len(arg_types) != 2:
        raise TypeError("tile_upper_solve() requires exactly 2 positional args")

    u = arg_types["U"]
    z = arg_types["z"]

    if not is_tile(u):
        raise TypeError(f"tile_upper_solve() 'U' argument must be a tile, got {u!r}")

    if not is_tile(z):
        raise TypeError(f"tile_upper_solve() 'z' argument must be a tile, got {z!r}")

    if not types_equal(u.dtype, z.dtype):
        raise TypeError(f"tile_upper_solve() arguments must have the same dtype, got {u.dtype} and {z.dtype}")

    if u.shape[0] != u.shape[1]:
        raise ValueError("tile_upper_solve() 'U' argument must be square")

    if len(z.shape) > 2 or len(z.shape) < 1:
        raise TypeError("tile_upper_solve() 'z' argument must be a 1D or 2D tile")

    if z.shape[0] != u.shape[0]:
        raise ValueError(
            f"tile_upper_solve() 'z' argument must have the same number of elements as the number of rows in 'U', "
            f"got {z.shape[0]} elements in 'z' and {u.shape[0]} rows in 'U'"
        )

    return tile(dtype=u.dtype, shape=z.shape, layout=z.layout, strides=z.strides, storage="shared")


add_builtin(
    "tile_upper_solve",
    input_types={"U": tile(dtype=Float, shape=Tuple[int, int]), "z": tile(dtype=Float, shape=Tuple[int])},
    value_func=tile_upper_solve_generic_value_func,
    lto_dispatch_func=tile_upper_solve_generic_lto_dispatch_func,
    variadic=True,
    doc="""Solve for x in U x = z, where U is an upper triangular matrix.

    This performs general back substitution for upper triangular systems.

    Note that computing the adjoint is not yet supported.

    Supported datatypes are:
        * float32
        * float64

    :param U: A square, non-singular, upper triangular matrix
    :param z: A 1D or 2D tile with compatible shape
    :returns x: A tile of the same shape as z such that U x = z""",
    group="Tile Primitives",
    export=False,
    namespace="",
)


# ---------------------------------
# Code Generation

add_builtin(
    "static",
    input_types={"expr": Any},
    value_type=Any,
    doc="""Evaluate a static Python expression and replaces it with its result.

    See the :ref:`code generation guide <static_expressions>` for more details.

    The inner expression must only reference variables that are available from the current scope where the Warp kernel or function containing the expression is defined,
    which includes constant variables and variables captured in the current closure in which the function or kernel is implemented.
    The return type of the expression must be either a Warp function, a string, or a type that is supported inside Warp kernels and functions
    (excluding Warp arrays since they cannot be created in a Warp kernel at the moment).""",
    group="Code Generation",
)


def static(expr):
    """
    Evaluates a static expression and replaces the expression with its result.

    Args:
        expr: A Python expression to evaluate. Must return a non-null value which must be either a Warp function, a string, or a type that is supported inside Warp kernels and functions (excluding Warp arrays since they cannot be created in a Warp kernel at the moment).

    Note:
        The inner expression must only reference variables that are available from the current scope where the Warp kernel or function containing the expression is defined,
        which includes constant variables and variables captured in the current closure in which the function or kernel is implemented.
    """
    return expr


add_builtin(
    "len",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_func=static_len_value_func,
    doc="Return the number of elements in a vector.",
    group="Utility",
    export=False,
)

add_builtin(
    "len",
    input_types={"a": quaternion(dtype=Scalar)},
    value_func=static_len_value_func,
    doc="Return the number of elements in a quaternion.",
    group="Utility",
    export=False,
)

add_builtin(
    "len",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=static_len_value_func,
    doc="Return the number of rows in a matrix.",
    group="Utility",
    export=False,
)

add_builtin(
    "len",
    input_types={"a": transformation(dtype=Float)},
    value_func=static_len_value_func,
    doc="Return the number of elements in a transformation.",
    group="Utility",
    export=False,
)

add_builtin(
    "len",
    input_types={"a": array(dtype=Any)},
    value_type=int,
    doc="Return the size of the first dimension in an array.",
    group="Utility",
    export=False,
)

add_builtin(
    "len",
    input_types={"a": tile(dtype=Any, shape=Tuple[int, ...])},
    value_func=static_len_value_func,
    doc="Return the number of rows in a tile.",
    group="Utility",
    export=False,
)


# ---------------------------------
# Tuple


def tuple_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    return tuple_t(arg_types["args"], arg_values["args"])


def tuple_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = args.get("args", ())
    template_args = ()
    return (func_args, template_args)


add_builtin(
    "tuple",
    input_types={"*args": Any},
    value_func=tuple_value_func,
    dispatch_func=tuple_dispatch_func,
    variadic=True,
    doc="Construct a tuple from a list of values",
    group="Utility",
    hidden=True,
    missing_grad=True,
    export=False,
)


def tuple_extract_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    tuple_type = arg_types["a"]
    elements = tuple_type.types if is_tuple(tuple_type) else tuple_type

    if "i" not in arg_values:
        raise RuntimeError("Tuple index must be a compile time expression.")

    index = arg_values["i"]
    if isinstance(index, Var):
        raise RuntimeError("Tuple index must be a compile time expression.")

    length = len(elements)
    if index >= length:
        raise RuntimeError(f"Tuple index out of bounds, {index} >= {length}")

    value_type = elements[index]
    return value_type


def tuple_extract_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (args["a"],)
    template_args = (args["i"].constant,)
    return (func_args, template_args)


add_builtin(
    "extract",
    input_types={"a": Tuple, "i": int},
    value_func=tuple_extract_value_func,
    dispatch_func=tuple_extract_dispatch_func,
    group="Utility",
    hidden=True,
    missing_grad=True,
)


add_builtin(
    "len",
    input_types={"a": Tuple},
    value_func=static_len_value_func,
    doc="Return the number of elements in a tuple.",
    group="Utility",
    export=False,
)

# ---------------------------------
# Slicing


def slice_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    return slice_t(**arg_values)


add_builtin(
    "slice",
    input_types={"start": int, "stop": int, "step": int},
    value_func=slice_value_func,
    native_func="slice_t",
    export=False,
    group="Utility",
    hidden=True,
    missing_grad=True,
)
