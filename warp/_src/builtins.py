# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import functools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import warp._src.build
import warp._src.context
from warp._src.codegen import Var, get_arg_value
from warp._src.logger import log_warning
from warp._src.types import *

from .context import add_builtin


def seq_check_equal(seq_1, seq_2):
    if not isinstance(seq_1, Sequence) or not isinstance(seq_2, Sequence):
        return False

    if len(seq_1) != len(seq_2):
        return False

    return all(x == y for x, y in zip(seq_1, seq_2, strict=True))


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
        if isinstance(arg.type, warp._src.types.tuple_t):
            out = arg.type.values
        else:
            out = (arg,)
    elif isinstance(arg, warp._src.types.tuple_t):
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

    length = warp._src.types.type_length(arg_types["a"])
    return Var(None, type=int, constant=length)


# ---------------------------------
# Scalar Math

add_builtin(
    "min",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="""Compute the minimum value.

    On float types, NaN is treated as missing (C ``fmin`` semantics): the
    operation returns the non-NaN operand when exactly one is NaN, and NaN
    only when both are NaN.""",
    group="Scalar Math",
)

add_builtin(
    "max",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="""Compute the maximum value.

    On float types, NaN is treated as missing (C ``fmax`` semantics): the
    operation returns the non-NaN operand when exactly one is NaN, and NaN
    only when both are NaN.""",
    group="Scalar Math",
)

add_builtin(
    "clamp",
    input_types={"x": Scalar, "low": Scalar, "high": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="""Clamp the value of ``x`` to the range [low, high].

    Equivalent to ``wp.min(wp.max(low, x), high)``. On float types this means
    NaN values of ``x`` produce ``wp.min(low, high)`` rather than propagating
    NaN.""",
    group="Scalar Math",
)

add_builtin(
    "abs",
    input_types={"x": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Compute the absolute value of ``x``.",
    group="Scalar Math",
)
add_builtin(
    "sign",
    input_types={"x": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="""Compute the sign of ``x``.

    Returns:
        -1 if ``x`` < 0 and 1 otherwise.""",
    group="Scalar Math",
    is_differentiable=False,
)

add_builtin(
    "copysign",
    input_types={"x": Float, "y": Float},
    value_func=sametypes_create_value_func(Float),
    doc="""Return a value with the magnitude of ``x`` and the sign of ``y``.

    For example, ``wp.copysign(3.0, -1.0)`` returns ``-3.0`` and
    ``wp.copysign(-3.0, 1.0)`` returns ``3.0``. Useful for forcing a
    specific sign on a result whose signed-zero behavior is otherwise
    implementation-defined (e.g. ``wp.min(-0.0, +0.0)``).""",
    group="Scalar Math",
)

add_builtin(
    "step",
    input_types={"x": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Compute 1.0 if ``x`` < 0.0, otherwise 0.0.",
    group="Scalar Math",
    is_differentiable=False,
)
add_builtin(
    "nonzero",
    input_types={"x": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Compute 1.0 if ``x`` is not equal to zero, otherwise 0.0.",
    group="Scalar Math",
    is_differentiable=False,
)

add_builtin(
    "sin",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the sine of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "cos",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the cosine of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "acos",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="""Compute arccos of ``x`` in radians.

    Inputs are automatically clamped to [-1.0, 1.0].""",
    group="Scalar Math",
)
add_builtin(
    "asin",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="""Compute arcsin of ``x`` in radians.

    Inputs are automatically clamped to [-1.0, 1.0].""",
    group="Scalar Math",
)
add_builtin(
    "sqrt",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the square root of ``x``, where ``x`` is positive.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "cbrt",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the cube root of ``x``.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "tan",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the tangent of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "atan",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the arctangent of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "atan2",
    input_types={"y": Float, "x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the 2-argument arctangent, atan2, of the point ``(x, y)`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "sinh",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the sinh of ``x``.",
    group="Scalar Math",
)
add_builtin(
    "cosh",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the cosh of ``x``.",
    group="Scalar Math",
)
add_builtin(
    "tanh",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the tanh of ``x``.",
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
    doc="Compute the natural logarithm (base-e) of ``x``, where ``x`` is positive.",
    group="Scalar Math",
)
add_builtin(
    "log2",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the binary logarithm (base-2) of ``x``, where ``x`` is positive.",
    group="Scalar Math",
)
add_builtin(
    "log10",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the common logarithm (base-10) of ``x``, where ``x`` is positive.",
    group="Scalar Math",
)
add_builtin(
    "exp",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the value of the exponential function :math:`e^x`.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "pow",
    input_types={"x": Float, "y": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute ``x`` raised to the power of ``y``.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "erf",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the error function of ``x``.",
    group="Scalar Math",
)
add_builtin(
    "erfc",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the complementary error function of ``x``.",
    group="Scalar Math",
)
add_builtin(
    "erfinv",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the inverse error function of ``x``.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "erfcinv",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Compute the inverse complementary error function of ``x``.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "round",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Compute the nearest integer value to ``x``, rounding halfway cases away from zero.

    This is the most intuitive form of rounding in the colloquial sense, but can be slower than other options like
    :func:`~warp._src.lang.rint`.
    Differs from :func:`numpy.round`, which behaves the same way as :obj:`numpy.rint`.""",
    is_differentiable=False,
)

add_builtin(
    "rint",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Compute the nearest integer value to ``x``, rounding halfway cases to nearest even integer.

    It is generally faster than :func:`~warp._src.lang.round`.
    Equivalent to :obj:`numpy.rint`.""",
    is_differentiable=False,
)

add_builtin(
    "trunc",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Compute the nearest integer that is closer to zero than ``x``.

    In other words, it discards the fractional part of ``x``.
    It is similar to casting ``float(int(a))``, but preserves the negative sign when ``x`` is in the range [-0.0, -1.0).
    Equivalent to :obj:`numpy.trunc` and :func:`numpy.fix`.""",
    is_differentiable=False,
)

add_builtin(
    "floor",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Compute the largest integer that is less than or equal to ``x``.""",
    is_differentiable=False,
)

add_builtin(
    "ceil",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Compute the smallest integer that is greater than or equal to ``x``.""",
    is_differentiable=False,
)

add_builtin(
    "frac",
    input_types={"x": Float},
    value_func=sametypes_create_value_func(Float),
    group="Scalar Math",
    doc="""Retrieve the fractional part of ``x``.

    In other words, it discards the integer part of ``x`` and is equivalent to ``x - trunc(x)``.""",
    is_differentiable=False,
)


add_builtin(
    "isfinite",
    input_types={"a": Float},
    value_type=builtins.bool,
    group="Scalar Math",
    doc="Check if ``a`` is finite.",
    is_differentiable=False,
)
add_builtin(
    "isfinite",
    input_types={"a": vector(length=Any, dtype=Float)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Check if all elements of ``a`` are finite.",
    is_differentiable=False,
)
add_builtin(
    "isfinite",
    input_types={"a": quaternion(dtype=Float)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Check if all elements of ``a`` are finite.",
    is_differentiable=False,
)
add_builtin(
    "isfinite",
    input_types={"a": matrix(shape=(Any, Any), dtype=Float)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Check if all elements of ``a`` are finite.",
    is_differentiable=False,
)

add_builtin(
    "isnan",
    input_types={"a": Float},
    value_type=builtins.bool,
    group="Scalar Math",
    doc="Check if ``a`` is NaN.",
    is_differentiable=False,
)
add_builtin(
    "isnan",
    input_types={"a": vector(length=Any, dtype=Float)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Check if any element of ``a`` is NaN.",
    is_differentiable=False,
)
add_builtin(
    "isnan",
    input_types={"a": quaternion(dtype=Float)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Check if any element of ``a`` is NaN.",
    is_differentiable=False,
)
add_builtin(
    "isnan",
    input_types={"a": matrix(shape=(Any, Any), dtype=Float)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Check if any element of ``a`` is NaN.",
    is_differentiable=False,
)

add_builtin(
    "isinf",
    input_types={"a": Float},
    value_type=builtins.bool,
    group="Scalar Math",
    doc="Check if ``a`` is positive or negative infinity.",
    is_differentiable=False,
)
add_builtin(
    "isinf",
    input_types={"a": vector(length=Any, dtype=Float)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Check if any element of ``a`` is positive or negative infinity.",
    is_differentiable=False,
)
add_builtin(
    "isinf",
    input_types={"a": quaternion(dtype=Float)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Check if any element of ``a`` is positive or negative infinity.",
    is_differentiable=False,
)
add_builtin(
    "isinf",
    input_types={"a": matrix(shape=(Any, Any), dtype=Float)},
    value_type=builtins.bool,
    group="Vector Math",
    doc="Check if any element of ``a`` is positive or negative infinity.",
    is_differentiable=False,
)


def _cast_scalar_constant(arg, target_dtype):
    """Cast a scalar constant Var to ``target_dtype`` for typed constructors.

    When a typed constructor like ``vec3d()`` receives float literal arguments,
    the literals have already been canonicalized to ``float32`` by
    ``Var.__init__``.  This function creates a new ``Var`` at the target
    precision, preserving the original Python float/int value so the emitted
    C++ declaration uses full precision (e.g. ``wp::float64`` instead of
    ``wp::float32``).

    Returns the original *arg* unchanged when no cast is needed.
    """
    if not isinstance(arg, Var):
        return arg
    if arg.type == target_dtype:
        return arg
    if arg.constant is None:
        return arg
    if arg.type not in scalar_types and arg.type not in (bool,):
        return arg

    raw = arg.constant
    if type(raw) in scalar_types:
        raw = raw.value

    return Var(None, type=target_dtype, constant=target_dtype(raw))


def _check_vars_match_dtype(arg_values, arg_types, dtype, msg):
    """Validate that runtime variables in constructor args match *dtype*.

    Compile-time constants (non-``Var`` values) are accepted regardless of
    their inferred type — ``_cast_scalar_constant`` in the dispatch function
    will cast them to *dtype*.  Runtime variables must already have a type
    that satisfies ``scalars_equal(arg_type, dtype)``.

    Handles both variadic constructors (with an ``"args"`` key) and
    named-parameter constructors (e.g. quaternion's ``x``, ``y``, ``z``,
    ``w``).  Only ``"dtype"``, ``"length"``, and ``"shape"`` are skipped —
    other non-scalar keys (e.g. ``"p"``, ``"q"`` in transformation) are
    deliberately included so that compound ``Var`` arguments still trigger
    the type error.
    """
    skip_keys = {"dtype", "length", "shape"}
    if "args" in arg_values:
        values = arg_values["args"]
    else:
        values = tuple(v for k, v in arg_values.items() if k not in skip_keys)

    for t, v in zip(arg_types, values, strict=True):
        if not isinstance(v, Var):
            continue  # compile-time constant — will be cast in dispatch
        # Extract the scalar type from compound types (vec, mat, quat).
        scalar_t = getattr(t, "_wp_scalar_type_", t)
        if not warp._src.types.scalars_equal(scalar_t, dtype):
            raise RuntimeError(msg)


def scalar_infer_type(arg_types: Mapping[str, type] | tuple[type, ...] | None):
    if arg_types is None:
        return Scalar

    if isinstance(arg_types, Mapping):
        arg_types = tuple(arg_types.values())

    scalar_types = set()
    for t in arg_types:
        if hasattr(t, "_wp_scalar_type_"):
            scalar_types.add(t._wp_scalar_type_)
        elif t in scalar_and_bool_types:
            scalar_types.add(t)

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
    doc="""Compute the dot product.""",
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
    doc="""Compute the minimum value.

    Returns:
        The element-wise minimum of ``a`` and ``b``.""",
    group="Vector Math",
)
add_builtin(
    "max",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="""Compute the maximum value.

    Returns:
        The element-wise maximum of ``a`` and ``b``.""",
    group="Vector Math",
)

add_builtin(
    "min",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_func=scalar_sametypes_value_func,
    doc="""Compute the minimum value.

    On float types, NaN elements are treated as missing (C ``fmin`` semantics);
    the reduction returns the smallest non-NaN element, or NaN only if every
    element is NaN.

    Returns:
        The minimum element of ``a``.""",
    group="Vector Math",
)
add_builtin(
    "max",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_func=scalar_sametypes_value_func,
    doc="""Compute the maximum value.

    On float types, NaN elements are treated as missing (C ``fmax`` semantics);
    the reduction returns the largest non-NaN element, or NaN only if every
    element is NaN.

    Returns:
        The maximum element of ``a``.""",
    group="Vector Math",
)

add_builtin(
    "argmin",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_func=lambda arg_types, arg_values: warp.uint32,
    doc="""Compute the index of the minimum element of vector ``a``.

    On float types, NaN elements are skipped; the result is the index of the
    smallest non-NaN element. If every element is NaN, returns ``0``.""",
    group="Vector Math",
    is_differentiable=False,
)
add_builtin(
    "argmax",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_func=lambda arg_types, arg_values: warp.uint32,
    doc="""Compute the index of the maximum element of vector ``a``.

    On float types, NaN elements are skipped; the result is the index of the
    largest non-NaN element. If every element is NaN, returns ``0``.""",
    group="Vector Math",
    is_differentiable=False,
)

add_builtin(
    "abs",
    input_types={"x": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="""Compute the absolute value of ``x``.

    Returns:
        The element-wise absolute value of ``x``.""",
    group="Vector Math",
)

add_builtin(
    "sign",
    input_types={"x": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="""Compute the sign of ``x``.

    Returns:
        -1 for negative elements of ``x`` and 1 otherwise.""",
    group="Vector Math",
    is_differentiable=False,
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
    value_func=lambda arg_types, arg_values: (
        matrix(shape=(3, 3), dtype=Scalar)
        if arg_types is None
        else matrix(shape=(3, 3), dtype=arg_types["vec"]._wp_scalar_type_)
    ),
    group="Vector Math",
    doc="Compute the skew-symmetric 3x3 matrix for a 3D vector ``vec``.",
)

add_builtin(
    "length",
    input_types={"a": vector(length=Any, dtype=Float)},
    value_func=float_sametypes_value_func,
    group="Vector Math",
    doc="""Compute the length of ``a``.

    Compute the length of a floating-point vector.""",
    require_original_output_arg=True,
)
add_builtin(
    "length",
    input_types={"a": quaternion(dtype=Float)},
    value_func=float_sametypes_value_func,
    group="Vector Math",
    doc="""Compute the length of ``a``.

    Compute the length of a quaternion.""",
    require_original_output_arg=True,
)
add_builtin(
    "length_sq",
    input_types={"a": vector(length=Any, dtype=Scalar)},
    value_func=scalar_sametypes_value_func,
    group="Vector Math",
    doc="""Compute the squared length of ``a``.

    Compute the squared length of a vector.""",
)
add_builtin(
    "length_sq",
    input_types={"a": quaternion(dtype=Float)},
    value_func=float_sametypes_value_func,
    group="Vector Math",
    doc="""Compute the squared length of ``a``.

    Compute the squared length of a quaternion.""",
)
add_builtin(
    "normalize",
    input_types={"a": vector(length=Any, dtype=Float)},
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Float)),
    group="Vector Math",
    doc="""Compute the normalized value of ``a``.

    If ``length(a)`` is 0, the zero vector is returned.""",
    require_original_output_arg=True,
)
add_builtin(
    "normalize",
    input_types={"a": quaternion(dtype=Float)},
    value_func=sametypes_create_value_func(quaternion(dtype=Float)),
    group="Vector Math",
    doc="""Compute the normalized value of ``a``.

    If ``length(a)`` is 0, the zero quaternion is returned.""",
)

add_builtin(
    "transpose",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=lambda arg_types, arg_values: (
        matrix(shape=(Any, Any), dtype=Scalar)
        if arg_types is None
        else matrix(shape=(arg_types["a"]._shape_[1], arg_types["a"]._shape_[0]), dtype=arg_types["a"]._wp_scalar_type_)
    ),
    group="Vector Math",
    doc="Compute the transpose of matrix ``a``.",
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
    doc="""Compute the inverse of matrix ``a``.""",
    require_original_output_arg=True,
)

add_builtin(
    "inverse",
    input_types={"a": matrix(shape=(3, 3), dtype=Float)},
    value_func=inverse_value_func,
    group="Vector Math",
    doc="""Compute the inverse of matrix ``a``.""",
    require_original_output_arg=True,
)

add_builtin(
    "inverse",
    input_types={"a": matrix(shape=(4, 4), dtype=Float)},
    value_func=inverse_value_func,
    group="Vector Math",
    doc="""Compute the inverse of matrix ``a``.""",
    require_original_output_arg=True,
)

add_builtin(
    "inverse_approx",
    input_types={"a": matrix(shape=(2, 2), dtype=Float)},
    value_func=inverse_value_func,
    native_func="approx_inverse",
    group="Vector Math",
    doc="""Compute the inverse of matrix ``a`` using approximate GPU intrinsics.

    Falls back to exact inverse on CPU.""",
    require_original_output_arg=True,
    export=False,
)

add_builtin(
    "inverse_approx",
    input_types={"a": matrix(shape=(3, 3), dtype=Float)},
    value_func=inverse_value_func,
    native_func="approx_inverse",
    group="Vector Math",
    doc="""Compute the inverse of matrix ``a`` using approximate GPU intrinsics.

    Falls back to exact inverse on CPU.""",
    require_original_output_arg=True,
    export=False,
)

add_builtin(
    "inverse_approx",
    input_types={"a": matrix(shape=(4, 4), dtype=Float)},
    value_func=inverse_value_func,
    native_func="approx_inverse",
    group="Vector Math",
    doc="""Compute the inverse of matrix ``a`` using approximate GPU intrinsics.

    Falls back to exact inverse on CPU.""",
    require_original_output_arg=True,
    export=False,
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
    doc="""Compute the determinant of matrix ``a``.""",
)

add_builtin(
    "determinant",
    input_types={"a": matrix(shape=(3, 3), dtype=Float)},
    value_func=determinant_value_func,
    group="Vector Math",
    doc="""Compute the determinant of matrix ``a``.""",
)

add_builtin(
    "determinant",
    input_types={"a": matrix(shape=(4, 4), dtype=Float)},
    value_func=determinant_value_func,
    group="Vector Math",
    doc="""Compute the determinant of matrix ``a``.""",
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
    doc="Compute the trace of matrix ``a``.",
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
    doc="Construct a matrix with the components of vector ``vec`` on the diagonal.",
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
    doc="Extract a vector containing the diagonal elements of square matrix ``mat``.",
)

add_builtin(
    "cw_mul",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    group="Vector Math",
    doc="""Compute the component-wise product of ``a`` and ``b``.""",
)
add_builtin(
    "cw_div",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    group="Vector Math",
    doc="""Compute the component-wise division of ``a`` by ``b``.""",
    require_original_output_arg=True,
)

add_builtin(
    "cw_mul",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    group="Vector Math",
    doc="""Compute the component-wise product of ``a`` and ``b``.""",
)
add_builtin(
    "cw_div",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    group="Vector Math",
    doc="""Compute the component-wise division of ``a`` by ``b``.""",
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
            is_differentiable=type_is_float(t),
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
        value_type = variadic_arg_types[0]
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
            elif not warp._src.types.scalars_equal(value_type, dtype):
                _check_vars_match_dtype(
                    arg_values,
                    variadic_arg_types,
                    dtype,
                    f"the value used to fill this vector is expected to be of the type `{dtype.__name__}`",
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

        if dtype is not None:
            # dtype is known from a typed constructor (e.g. vec3d).
            # Constants will be cast in dispatch; just validate variables.
            _check_vars_match_dtype(
                arg_values,
                variadic_arg_types,
                dtype,
                f"all values used to initialize this vector are expected to be of the type `{dtype.__name__}`",
            )
        else:
            try:
                dtype = scalar_infer_type(variadic_arg_types)
            except RuntimeError:
                raise RuntimeError("all values given when constructing a vector must have the same type") from None

    if length is None:
        raise RuntimeError("could not infer the `length` argument when calling the `wp.types.vector()` function")

    if dtype is None:
        raise RuntimeError("could not infer the `dtype` argument when calling the `wp.types.vector()` function")

    return vector(length=length, dtype=dtype)


def vector_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    length = return_type._length_
    dtype = return_type._wp_scalar_type_

    variadic_args = args.get("args", ())

    # Cast scalar constant args to the target dtype so that the emitted
    # C++ declarations preserve full precision (e.g. wp::float64 instead of
    # wp::float32 for vec3d literals).
    func_args = tuple(_cast_scalar_constant(a, dtype) for a in variadic_args)
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
    doc="""Construct a vector of given length and dtype.

    If no arguments are given, the vector is zero-initialized.""",
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
        value_type = variadic_arg_types[0]
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
            elif not warp._src.types.scalars_equal(value_type, dtype):
                _check_vars_match_dtype(
                    arg_values,
                    variadic_arg_types,
                    dtype,
                    f"the value used to fill this matrix is expected to be of the type `{dtype.__name__}`",
                )
    else:
        # Initializing by value, e.g.: `wp.mat22(1, 2, 3, 4)`, `wp.matrix(1, 2, 3, 4, shape=(2, 2))`.
        if shape is None:
            raise RuntimeError("the `shape` argument must be specified when initializing a matrix by value")

        if all(type_is_vector(x) for x in variadic_arg_types):
            raise TypeError(
                "Passing vectors to `wp.matrix()` isn't supported, use `wp.matrix_from_rows()` or `wp.matrix_from_cols()` instead."
            )
        elif shape[0] * shape[1] != variadic_arg_count:
            raise RuntimeError(
                f"incompatible number of values given ({variadic_arg_count}) "
                f"when constructing a matrix of shape {tuple(shape)}"
            )

        if dtype is not None:
            _check_vars_match_dtype(
                arg_values,
                variadic_arg_types,
                dtype,
                f"all values used to initialize this matrix are expected to be of the type `{dtype.__name__}`",
            )
        else:
            try:
                dtype = scalar_infer_type(variadic_arg_types)
            except RuntimeError:
                raise RuntimeError("all values given when constructing a matrix must have the same type") from None

    if shape is None:
        raise RuntimeError("could not infer the `shape` argument when calling the `wp.types.matrix()` function")

    if dtype is None:
        raise RuntimeError("could not infer the `dtype` argument when calling the `wp.types.matrix()` function")

    return matrix(shape=shape, dtype=dtype)


def matrix_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    shape = return_type._shape_
    dtype = return_type._wp_scalar_type_

    variadic_args = args.get("args", ())

    func_args = tuple(_cast_scalar_constant(a, dtype) for a in variadic_args)
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
    input_types={"*args": Scalar, "shape": tuple[int, int], "dtype": Scalar},
    defaults={"shape": None, "dtype": None},
    variadic=True,
    value_func=matrix_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k not in ("shape", "dtype")},
    dispatch_func=matrix_dispatch_func,
    initializer_list_func=matrix_initializer_list_func,
    native_func="mat_t",
    doc="""Construct a matrix.

    Construct a matrix with the given shape and dtype.

    If no positional arguments are given, the matrix is zero-initialized.""",
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
    doc="Construct a matrix with each vector argument as a column.",
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
    doc="Construct a matrix with each vector argument as a row.",
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
    is_differentiable=False,
)


def matrix_transform_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return matrix(shape=(4, 4), dtype=Float)

    raise RuntimeError(
        "the built-in `wp.types.matrix()` to construct a 4x4 matrix from a 3D position, quaternion, "
        "and 3D scale vector has been removed in favor of `wp.transform_compose()`."
    )


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
    native_func="mat_t",
    doc="""Construct a matrix.

    Construct a 4x4 transformation matrix that applies the transformations as
    ``Translation(pos)*Rotation(rot)*Scaling(scale)`` when applied to column vectors, i.e.: ``y = (TRS)*x``.

    .. versionremoved:: 1.10
        This function has been removed in favor of :func:`~warp._src.lang.transform_compose`.

    .. deprecated:: 1.8""",
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
    doc="""Compute the SVD of a 3x3 matrix ``A``.

    The singular values are returned in ``sigma``, while the left and right basis vectors are returned in ``U`` and ``V``.""",
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
    doc="""Compute the SVD of a 3x3 matrix ``A``.

    The singular values are returned in ``sigma``, while the left and right basis vectors are returned in ``U`` and ``V``.""",
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
    doc="""Compute the SVD of a 2x2 matrix ``A``.

    The singular values are returned in ``sigma``, while the left and right basis vectors are returned in ``U`` and ``V``.""",
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
    doc="""Compute the SVD of a 2x2 matrix ``A``.

    The singular values are returned in ``sigma``, while the left and right basis vectors are returned in ``U`` and ``V``.""",
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
    doc="""Compute the QR decomposition of a 3x3 matrix ``A``.

    The orthogonal matrix is returned in ``Q``, while the upper triangular matrix is returned in ``R``.""",
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
    doc="""Compute the QR decomposition of a 3x3 matrix ``A``.

    The orthogonal matrix is returned in ``Q``, while the upper triangular matrix is returned in ``R``.""",
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
    doc="""Compute the eigendecomposition of a 3x3 matrix ``A``.

    The eigenvectors are returned as the columns of ``Q``, while the corresponding eigenvalues are returned in ``d``.""",
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
    doc="""Compute the eigendecomposition of a 3x3 matrix ``A``.

    The eigenvectors are returned as the columns of ``Q``, while the corresponding eigenvalues are returned in ``d``.""",
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
        if dtype is not None:
            _check_vars_match_dtype(
                arg_values,
                variadic_arg_types,
                dtype,
                f"all values used to initialize this quaternion are expected to be of the type `{dtype.__name__}`",
            )
        else:
            try:
                value_type = scalar_infer_type(variadic_arg_types)
            except RuntimeError:
                raise RuntimeError("all values given when constructing a quaternion must have the same type") from None
            dtype = value_type

    if dtype is None:
        raise RuntimeError("could not infer the `dtype` argument when calling the `wp.types.quaternion()` function")

    return quaternion(dtype=dtype)


def quaternion_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = return_type._wp_scalar_type_

    variadic_args = tuple(v for k, v in args.items() if k != "dtype")

    func_args = tuple(_cast_scalar_constant(a, dtype) for a in variadic_args)
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
    doc="""Construct a quaternion.

    Zero-initialize the quaternion. Quaternions are laid out as
    ``[ix, iy, iz, r]``, where ``ix``, ``iy``, ``iz`` are the imaginary part, and ``r`` the real part.""",
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
    doc="""Construct a quaternion.

    Use the supplied components (type inferred from component type).""",
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
    doc="""Construct a quaternion.

    Use the supplied vector/scalar (type inferred from scalar type).""",
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
    doc="""Construct a quaternion.

    Convert ``quat`` to the specified ``dtype``.""",
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
    doc="Construct an identity quaternion with zero imaginary part and real part of 1.0.",
    export=True,
    is_differentiable=False,
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
    doc="""Construct a quaternion from a matrix.

    If the matrix is not a pure rotation, but for example includes scaling or skewing, the result is undefined.""",
)
add_builtin(
    "quat_from_matrix",
    input_types={"mat": matrix(shape=(4, 4), dtype=Float)},
    value_func=lambda arg_types, arg_values: quaternion(dtype=float_infer_type(arg_types)),
    group="Quaternion Math",
    doc="""Construct a quaternion from a matrix.

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
    doc="""Compute the dot product.""",
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
        value_type = variadic_arg_types[0]
        if dtype is None:
            dtype = value_type
        elif not warp._src.types.scalars_equal(value_type, dtype):
            _check_vars_match_dtype(
                arg_values,
                variadic_arg_types,
                dtype,
                f"the value used to fill this transform is expected to be of the type `{dtype.__name__}`",
            )
    elif variadic_arg_count == 7:
        # Initializing by value, e.g.: `wp.transform(1, 2, 3, 4, 5, 6, 7)`.
        if dtype is not None:
            _check_vars_match_dtype(
                arg_values,
                variadic_arg_types,
                dtype,
                f"all values used to initialize this transform are expected to be of the type `{dtype.__name__}`",
            )
        else:
            try:
                dtype = scalar_infer_type(variadic_arg_types)
            except RuntimeError:
                raise RuntimeError("all values given when constructing a transform must have the same type") from None

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
    elif not warp._src.types.scalars_equal(value_type, dtype):
        raise RuntimeError(
            f"all values used to initialize this transformation matrix are expected to be of the type `{dtype.__name__}`"
        )

    return transformation(dtype=dtype)


def transformation_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # We're in the codegen stage where we emit the code calling the built-in.
    # Further validate the given argument values if needed and map them
    # to the underlying C++ function's runtime and template params.

    dtype = return_type._wp_scalar_type_

    variadic_args = args.get("args", ())
    variadic_arg_count = len(variadic_args)

    if variadic_arg_count == 7:
        func_args = tuple(_cast_scalar_constant(a, dtype) for a in variadic_args)
    else:
        func_args = tuple(v for k, v in args.items() if k != "dtype")
        if "p" in args and "q" not in args:
            quat_ident = warp._src.codegen.Var(
                label=None, type=quaternion(dtype=dtype), constant=quaternion(dtype=dtype)(0, 0, 0, 1)
            )
            func_args += (quat_ident,)

    template_args = (dtype,)
    return (func_args, template_args)


add_builtin(
    "transformation",
    input_types={"p": vector(length=3, dtype=Float), "q": quaternion(dtype=Float), "dtype": Float},
    defaults={"q": None, "dtype": None},
    value_func=transformation_pq_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=transformation_dispatch_func,
    native_func="transform_t",
    group="Transformations",
    doc="""Construct a transformation.

    Use translation ``p`` and rotation ``q``.""",
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
    doc="""Construct a transformation.

    Build a spatial transform vector from components.""",
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
    is_differentiable=False,
)

add_builtin(
    "transform_get_translation",
    input_types={"xform": transformation(dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=float_infer_type(arg_types)),
    group="Transformations",
    doc="Extract the translational part of transform ``xform``.",
)
add_builtin(
    "transform_get_rotation",
    input_types={"xform": transformation(dtype=Float)},
    value_func=lambda arg_types, arg_values: quaternion(dtype=float_infer_type(arg_types)),
    group="Transformations",
    doc="Extract the rotational part of transform ``xform``.",
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
    doc="""Apply a transform to a point.

    Treat the homogeneous coordinate as w=1 (translation and rotation).""",
)
add_builtin(
    "transform_point",
    input_types={"mat": matrix(shape=(4, 4), dtype=Float), "point": vector(length=3, dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=float_infer_type(arg_types)),
    group="Vector Math",
    doc="""Apply a transform to a point.

    Treat the homogeneous coordinate as w=1.

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
    doc="""Apply a transform to a vector.

    Treat the homogeneous coordinate as w=0 (rotation only).""",
)
add_builtin(
    "transform_vector",
    input_types={"mat": matrix(shape=(4, 4), dtype=Float), "vec": vector(length=3, dtype=Float)},
    value_func=lambda arg_types, arg_values: vector(length=3, dtype=float_infer_type(arg_types)),
    group="Vector Math",
    doc="""Apply a transform to a vector.

    Treat the homogeneous coordinate as w=0.

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
        elif not warp._src.types.scalars_equal(value_type, dtype):
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
    doc="""Construct a 6D screw vector.

    Zero-initialize the vector.""",
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
    doc="""Construct a 6D screw vector.

    Use two 3D vectors.""",
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
    doc="""Construct a 6D screw vector.

    Use six scalar values.""",
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
    value_func=lambda arg_types, arg_values: (
        vector(length=3, dtype=Float)
        if arg_types is None
        else vector(length=3, dtype=arg_types["svec"]._wp_scalar_type_)
    ),
    group="Spatial Math",
    doc="Extract the top (first) part of a 6D screw vector.",
)
add_builtin(
    "spatial_bottom",
    input_types={"svec": vector(length=6, dtype=Float)},
    value_func=lambda arg_types, arg_values: (
        vector(length=3, dtype=Float)
        if arg_types is None
        else vector(length=3, dtype=arg_types["svec"]._wp_scalar_type_)
    ),
    group="Spatial Math",
    doc="Extract the bottom (second) part of a 6D screw vector.",
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
    doc="Compute the spatial Jacobian matrix for a kinematic chain.",
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
    doc="Compute the composite rigid-body mass matrix for a kinematic chain.",
    group="Spatial Math",
)

# ------------------
# Tile-based primitives


def tile_zeros_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

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

    dtype_var = arg_values["dtype"]
    dtype = dtype_var.constant if isinstance(dtype_var, Var) else dtype_var

    template_args = []
    template_args.append(dtype)
    template_args.extend(shape)

    return ([], template_args)


add_builtin(
    "tile_zeros",
    input_types={"shape": tuple[int, ...], "dtype": Any, "storage": str},
    defaults={"storage": "register", "dtype": float},
    value_func=tile_zeros_value_func,
    dispatch_func=tile_zeros_dispatch_func,
    variadic=False,
    is_differentiable=False,
    doc="""Allocate a tile of zero-initialized items.

    Args:
        shape: Shape of the output tile
        dtype: Data type of output tile's elements (default float)
        storage: The storage location for the tile: ``"register"`` for registers
            (default) or ``"shared"`` for shared memory.

    Returns:
        A zero-initialized tile with shape and data type as specified.""",
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
    is_differentiable=False,
    doc="""Allocate a tile of zero-initialized items.""",
    group="Tile Primitives",
    export=False,
)


def tile_ones_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

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
    if type_is_struct(dtype):
        raise TypeError("tile_ones() does not support Warp struct dtypes; use tile_full() with an explicit value")

    return tile(dtype=dtype, shape=shape, storage=arg_values["storage"])


def tile_ones_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    dtype_var = arg_values["dtype"]
    dtype = dtype_var.constant if isinstance(dtype_var, Var) else dtype_var

    template_args = []
    template_args.append(dtype)
    template_args.extend(shape)

    return ([], template_args)


add_builtin(
    "tile_ones",
    input_types={"shape": tuple[int, ...], "dtype": Any, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_ones_value_func,
    dispatch_func=tile_ones_dispatch_func,
    is_differentiable=False,
    doc="""Allocate a tile of one-initialized items.

    Args:
        shape: Shape of the output tile
        dtype: Data type of output tile's elements
        storage: The storage location for the tile: ``"register"`` for registers
            (default) or ``"shared"`` for shared memory.

    Returns:
        A one-initialized tile with shape and data type as specified.""",
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
    is_differentiable=False,
    doc="""Allocate a tile of one-initialized items.""",
    group="Tile Primitives",
    export=False,
)


def tile_empty_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    if "dtype" not in arg_values:
        raise TypeError("tile_empty() missing required keyword argument 'dtype'")

    if "storage" not in arg_values:
        raise TypeError("tile_empty() missing required keyword argument 'storage'")

    if arg_values["storage"] not in {"shared", "register"}:
        raise ValueError(f"Invalid value for 'storage': {arg_values['storage']!r}. Expected 'shared' or 'register'.")

    dtype = arg_values["dtype"]

    return tile(dtype=dtype, shape=shape, storage=arg_values["storage"])


def tile_empty_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    dtype_var = arg_values["dtype"]
    dtype = dtype_var.constant if isinstance(dtype_var, Var) else dtype_var

    template_args = []
    template_args.append(dtype)
    template_args.extend(shape)

    return ([], template_args)


add_builtin(
    "tile_empty",
    input_types={"shape": tuple[int, ...], "dtype": Any, "storage": str},
    defaults={"storage": "register", "dtype": float},
    value_func=tile_empty_value_func,
    dispatch_func=tile_empty_dispatch_func,
    variadic=False,
    is_differentiable=False,
    doc="""Allocate a tile of uninitialized items.

    The tile's contents are undefined; the caller is responsible for overwriting
    every element before any read. This matches the semantics of ``numpy.empty``.

    Because it skips initialization, ``tile_empty`` can avoid unnecessary stores
    when every element will be overwritten, especially for ``"shared"`` tiles.

    For accumulator patterns (``a += ...``), use :func:`tile_zeros` instead -
    accumulation reads the prior value and would propagate uninitialized data.
    Use ``tile_empty`` only when the first operation after construction is a
    full overwrite (a ``tile_load``, a tile-typed assignment, or a complete
    element-wise fill).

    Args:
        shape: Shape of the output tile
        dtype: Data type of output tile's elements (default float)
        storage: The storage location for the tile: ``"register"`` for registers
            (default) or ``"shared"`` for shared memory.

    Returns:
        An uninitialized tile with the requested shape and data type.""",
    group="Tile Primitives",
    export=False,
)

# overload for scalar shape
add_builtin(
    "tile_empty",
    input_types={"shape": int, "dtype": Any, "storage": str},
    defaults={"storage": "register", "dtype": float},
    value_func=tile_empty_value_func,
    dispatch_func=tile_empty_dispatch_func,
    variadic=False,
    is_differentiable=False,
    doc="""Allocate a tile of uninitialized items.""",
    group="Tile Primitives",
    export=False,
)


def tile_full_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    if "value" not in arg_values:
        raise TypeError("tile_full() missing required keyword argument 'value'")

    if "dtype" not in arg_values:
        raise TypeError("tile_full() missing required keyword argument 'dtype'")

    if "storage" not in arg_values:
        raise TypeError("tile_full() missing required keyword argument 'storage'")

    if arg_values["storage"] not in {"shared", "register"}:
        raise ValueError(f"Invalid value for 'storage': {arg_values['storage']!r}. Expected 'shared' or 'register'.")

    dtype = arg_values["dtype"]
    value_type = arg_types["value"]

    if type_is_struct(dtype) and not types_equal(value_type, dtype):
        raise TypeError(
            f"tile_full() value must have dtype {type_repr(dtype)} when filling Warp struct tile elements, got {type_repr(value_type)}"
        )

    return tile(dtype=dtype, shape=shape, storage=arg_values["storage"])


def tile_full_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    dtype_var = arg_values["dtype"]
    dtype = dtype_var.constant if isinstance(dtype_var, Var) else dtype_var
    value = arg_values["value"]

    func_args = [value]

    template_args = []
    template_args.append(dtype)
    template_args.extend(shape)

    return (func_args, template_args)


add_builtin(
    "tile_full",
    input_types={"shape": tuple[int, ...], "value": Any, "dtype": Any, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_full_value_func,
    dispatch_func=tile_full_dispatch_func,
    is_differentiable=False,
    doc="""Allocate a tile filled with the specified value.

    Args:
        shape: Shape of the output tile
        value: Value to fill the tile with
        dtype: Data type of output tile's elements
        storage: The storage location for the tile: ``"register"`` for registers
            (default) or ``"shared"`` for shared memory.

    Returns:
        A tile filled with the specified value.""",
    group="Tile Primitives",
    export=False,
)


# overload for scalar shape
add_builtin(
    "tile_full",
    input_types={"shape": int, "value": Any, "dtype": Any, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_full_value_func,
    dispatch_func=tile_full_dispatch_func,
    is_differentiable=False,
    doc="""Allocate a tile filled with the specified value.""",
    group="Tile Primitives",
    export=False,
)


def tile_from_thread_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    if "value" not in arg_values:
        raise TypeError("tile_from_thread() missing required keyword argument 'value'")

    if "thread_idx" not in arg_values:
        raise TypeError("tile_from_thread() missing required keyword argument 'thread_idx'")

    if arg_values["storage"] not in {"shared", "register"}:
        raise ValueError(f"Invalid value for 'storage': {arg_values['storage']!r}. Expected 'shared' or 'register'.")

    return tile(dtype=arg_types["value"], shape=shape, storage=arg_values["storage"])


def tile_from_thread_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    shape = extract_tuple(arg_values["shape"], as_constant=True)

    func_args = [arg_values["value"], arg_values["thread_idx"]]
    template_args = [return_type.dtype, *shape]

    return (func_args, template_args)


add_builtin(
    "tile_from_thread",
    input_types={"shape": tuple[int, ...], "value": Any, "thread_idx": int, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_from_thread_value_func,
    dispatch_func=tile_from_thread_dispatch_func,
    is_differentiable=False,
    doc="""Allocate a tile filled with a value from a specific thread.

    This function broadcasts a value from one thread to all threads in the block,
    then creates a tile filled with that broadcast value. This is useful for
    efficiently sharing a computed result (e.g., from an atomic operation) with
    all threads in a block using minimal shared memory (only 1 element).

    Args:
        shape: Shape of the output tile
        value: Per-thread value (only the value from ``thread_idx`` is used)
        thread_idx: Index of the thread whose value should fill the tile
        storage: The storage location for the tile: ``"register"`` for registers
            (default) or ``"shared"`` for shared memory.

    Returns:
        A tile filled with the value from the specified thread.

    Example:

        .. code-block:: python

            import warp as wp

            TILE_SIZE = 8

            @wp.kernel
            def compute(output: wp.array[int]):
                i, j = wp.tid()

                # Compute offset on the last thread
                offset = 0
                if j == wp.block_dim() - 1:
                    offset = i * wp.block_dim()

                # Broadcast the last thread's offset to all threads (uses only 1 element of shared memory)
                offset_tile = wp.tile_from_thread(shape=TILE_SIZE, value=offset, thread_idx=wp.block_dim() - 1)

                # Combine with other tiles using tile operations
                indices = wp.tile_arange(0, TILE_SIZE, dtype=int)
                result = offset_tile + indices

                wp.tile_store(output, result, offset=(i * TILE_SIZE,))

            output = wp.zeros(16, dtype=int)
            wp.launch_tiled(compute, dim=[2], inputs=[output], block_dim=TILE_SIZE)

            print(output.numpy())

        .. code-block:: text

            [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]

        The output above assumes GPU execution. On CPU, ``wp.block_dim()`` returns ``1``,
        so ``offset`` becomes ``i * 1`` instead of ``i * TILE_SIZE``, and
        ``thread_idx=wp.block_dim() - 1`` selects thread ``0``. The CPU output is:

        .. code-block:: text

            [0 1 2 3 4 5 6 7 1 2 3 4 5 6 7 8]

        See :ref:`CPU Tile Semantics <cpu_tile_semantics>` for more detail on the CPU/GPU
        differences that affect portable tile code.

    """,
    group="Tile Primitives",
    export=False,
)


# overload for scalar shape
add_builtin(
    "tile_from_thread",
    input_types={"shape": int, "value": Any, "thread_idx": int, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_from_thread_value_func,
    dispatch_func=tile_from_thread_dispatch_func,
    is_differentiable=False,
    doc="""Allocate a tile filled with a value from a specific thread.""",
    group="Tile Primitives",
    export=False,
)


def tile_randi_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    if "rng" not in arg_values:
        raise TypeError("tile_randi() missing required keyword argument 'rng'")

    if "storage" not in arg_values:
        raise TypeError("tile_randi() missing required keyword argument 'storage'")

    if arg_values["storage"] not in {"shared", "register"}:
        raise ValueError(f"Invalid value for 'storage': {arg_values['storage']!r}. Expected 'shared' or 'register'.")

    return tile(dtype=int, shape=shape, storage=arg_values["storage"])


def tile_randi_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    rng = arg_values["rng"]

    func_args = [rng]

    has_min = "min" in arg_values
    has_max = "max" in arg_values

    if has_min != has_max:
        raise KeyError("Both 'min' and 'max' must be provided together")

    if has_min and has_max:
        min_val = arg_values["min"].constant
        max_val = arg_values["max"].constant

        if isinstance(min_val, int) and isinstance(max_val, int):
            func_args.append(min_val)
            func_args.append(max_val)
        else:
            raise TypeError("'min' and 'max' must both be integers")

    template_args = []
    template_args.extend(shape)

    return (func_args, template_args)


add_builtin(
    "tile_randi",
    input_types={"shape": tuple[int, ...], "rng": uint32, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_randi_value_func,
    dispatch_func=tile_randi_dispatch_func,
    is_differentiable=False,
    doc="""Generate a tile of random integers.

    Args:
        shape: Shape of the output tile
        rng: Random number generator state, typically from :func:`~warp._src.lang.rand_init`
        storage: The storage location for the tile: ``"register"`` for registers
            (default) or ``"shared"`` for shared memory.

    Returns:
        A tile of random integers with the specified shape.

    Example:

        .. testcode::

            TILE_M, TILE_N = 2, 2
            M, N = 2, 2
            seed = 42

            @wp.kernel
            def rand_kernel(seed: int, x: wp.array2d[int]):
                i, j = wp.tid()
                rng = wp.rand_init(seed, i * TILE_M + j)
                t = wp.tile_randi(shape=(TILE_M, TILE_N), rng=rng)
                wp.tile_store(x, t, offset=(i * TILE_M, j * TILE_N))

            x = wp.zeros(shape=(M * TILE_M, N * TILE_N), dtype=int)
            wp.launch_tiled(rand_kernel, dim=[M, N], inputs=[seed, x], block_dim=32)
            print(x.numpy())

        .. testoutput::

            [[  798497746  1803297529  -955788638    17806966]
             [ 1788185933  1320194893  2073257406 -2009156320]
             [ -257534450 -1138585923  1145322783  -321794125]
             [-2096177388 -1835610841  1159339128  -652221052]]
    """,
    group="Tile Primitives",
    export=False,
)

# overload for scalar shape
add_builtin(
    "tile_randi",
    input_types={"shape": int, "rng": uint32, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_randi_value_func,
    dispatch_func=tile_randi_dispatch_func,
    is_differentiable=False,
    doc="""Generate a tile of random integers.""",
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "tile_randi",
    input_types={"shape": tuple[int, ...], "rng": uint32, "min": int, "max": int, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_randi_value_func,
    dispatch_func=tile_randi_dispatch_func,
    is_differentiable=False,
    doc="""Generate a tile of random integers.

    Sample values in the range [min, max).

    Args:
        shape: Shape of the output tile
        rng: Random number generator state, typically from :func:`~warp._src.lang.rand_init`
        min: Minimum value (inclusive) for random integers
        max: Maximum value (exclusive) for random integers
        storage: The storage location for the tile: ``"register"`` for registers
            (default) or ``"shared"`` for shared memory.

    Returns:
        A tile of random integers in the range [min, max) with the specified shape.

    Example:

        .. testcode::

            TILE_M, TILE_N = 2, 2
            M, N = 2, 2
            seed = 42

            @wp.kernel
            def rand_range_kernel(seed: int, x: wp.array2d[int]):
                i, j = wp.tid()
                rng = wp.rand_init(seed, i * TILE_M + j)
                t = wp.tile_randi(shape=(TILE_M, TILE_N), rng=rng, min=-5, max=5)
                wp.tile_store(x, t, offset=(i * TILE_M, j * TILE_N))

            x = wp.zeros(shape=(M * TILE_M, N * TILE_N), dtype=int)
            wp.launch_tiled(rand_range_kernel, dim=[M, N], inputs=[seed, x], block_dim=32)
            print(x.numpy())

        .. testoutput::

            [[ 1  4  3  1]
             [-2 -2  1  1]
             [ 1 -2 -2 -4]
             [ 3  0  3 -1]]
    """,
    group="Tile Primitives",
    export=False,
)

# overload for scalar shape
add_builtin(
    "tile_randi",
    input_types={"shape": int, "rng": uint32, "min": int, "max": int, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_randi_value_func,
    dispatch_func=tile_randi_dispatch_func,
    is_differentiable=False,
    doc="""Generate a tile of random integers.

    Sample values in the range [min, max).""",
    group="Tile Primitives",
    export=False,
)


def tile_randf_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    if "rng" not in arg_values:
        raise TypeError("tile_randf() missing required keyword argument 'rng'")

    if "storage" not in arg_values:
        raise TypeError("tile_randf() missing required keyword argument 'storage'")

    if arg_values["storage"] not in {"shared", "register"}:
        raise ValueError(f"Invalid value for 'storage': {arg_values['storage']!r}. Expected 'shared' or 'register'.")

    return tile(dtype=float, shape=shape, storage=arg_values["storage"])


def tile_randf_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    rng = arg_values["rng"]

    func_args = [rng]

    has_min = "min" in arg_values
    has_max = "max" in arg_values

    if has_min != has_max:
        raise KeyError("Both 'min' and 'max' must be provided together")

    if has_min and has_max:
        min_val = arg_values["min"].constant
        max_val = arg_values["max"].constant

        if isinstance(min_val, float) and isinstance(max_val, float):
            func_args.append(min_val)
            func_args.append(max_val)
        else:
            raise TypeError("'min' and 'max' must both be floats")

    template_args = []
    template_args.extend(shape)

    return (func_args, template_args)


add_builtin(
    "tile_randf",
    input_types={"shape": tuple[int, ...], "rng": uint32, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_randf_value_func,
    dispatch_func=tile_randf_dispatch_func,
    is_differentiable=False,
    doc="""Generate a tile of random floats.

    Args:
        shape: Shape of the output tile
        rng: Random number generator state, typically from :func:`~warp._src.lang.rand_init`
        storage: The storage location for the tile: ``"register"`` for registers
            (default) or ``"shared"`` for shared memory.

    Returns:
        A tile of random floats in the range [0, 1) with the specified shape.

    Example:

        .. testcode::

            TILE_M, TILE_N = 2, 2
            M, N = 2, 2
            seed = 42

            @wp.kernel
            def rand_kernel(seed: int, x: wp.array2d[float]):
                i, j = wp.tid()
                rng = wp.rand_init(seed, i * TILE_M + j)
                t = wp.tile_randf(shape=(TILE_M, TILE_N), rng=rng)
                wp.tile_store(x, t, offset=(i * TILE_M, j * TILE_N))

            x = wp.zeros(shape=(M * TILE_M, N * TILE_N), dtype=float)
            wp.launch_tiled(rand_kernel, dim=[M, N], inputs=[seed, x], block_dim=32)
            print(x.numpy())

        .. testoutput::

            [[0.1859147  0.41986287 0.7774631  0.00414598]
             [0.41634446 0.3073818  0.4827178  0.53220683]
             [0.9400381  0.73490226 0.26666623 0.9250764 ]
             [0.51194566 0.57261354 0.26992965 0.8481429 ]]
    """,
    group="Tile Primitives",
    export=False,
)

# overload for scalar shape
add_builtin(
    "tile_randf",
    input_types={"shape": int, "rng": uint32, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_randf_value_func,
    dispatch_func=tile_randf_dispatch_func,
    is_differentiable=False,
    doc="""Generate a tile of random floats.""",
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "tile_randf",
    input_types={"shape": tuple[int, ...], "rng": uint32, "min": float, "max": float, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_randf_value_func,
    dispatch_func=tile_randf_dispatch_func,
    is_differentiable=False,
    doc="""Generate a tile of random floats.

    Sample values in the range [min, max).

    Args:
        shape: Shape of the output tile
        rng: Random number generator state, typically from :func:`~warp._src.lang.rand_init`
        min: Minimum value (inclusive) for random floats
        max: Maximum value (exclusive) for random floats
        storage: The storage location for the tile: ``"register"`` for registers
            (default) or ``"shared"`` for shared memory.

    Returns:
        A tile of random floats in the range [min, max) with the specified shape.

    Example:

        .. testcode::

            TILE_M, TILE_N = 2, 2
            M, N = 2, 2
            seed = 42

            @wp.kernel
            def rand_range_kernel(seed: int, x: wp.array2d[float]):
                i, j = wp.tid()
                rng = wp.rand_init(seed, i * TILE_M + j)
                t = wp.tile_randf(shape=(TILE_M, TILE_N), rng=rng, min=-5.0, max=5.0)
                wp.tile_store(x, t, offset=(i * TILE_M, j * TILE_N))

            x = wp.zeros(shape=(M * TILE_M, N * TILE_N), dtype=float)
            wp.launch_tiled(rand_range_kernel, dim=[M, N], inputs=[seed, x], block_dim=32)
            print(x.numpy())

        .. testoutput::

            [[-3.140853   -0.80137134  2.7746308  -4.95854   ]
             [-0.83655536 -1.9261819  -0.17282188  0.32206833]
             [ 4.400381    2.3490226  -2.3333378   4.2507644 ]
             [ 0.11945665  0.7261354  -2.3007035   3.481429  ]]
    """,
    group="Tile Primitives",
    export=False,
)

# overload for scalar shape
add_builtin(
    "tile_randf",
    input_types={"shape": int, "rng": uint32, "min": float, "max": float, "storage": str},
    defaults={"storage": "register"},
    value_func=tile_randf_value_func,
    dispatch_func=tile_randf_dispatch_func,
    is_differentiable=False,
    doc="""Generate a tile of random floats.

    Sample values in the range [min, max).""",
    group="Tile Primitives",
    export=False,
)


def tile_arange_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Scalar, shape=tuple[int])

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

    # tile_arange() is variadic, which bypasses the Scalar dtype constraint enforced by
    # overload matching, so reject struct dtypes explicitly: a numeric range has no
    # meaning for a struct and the native tile_arange would fail to compile.
    if type_is_struct(dtype):
        raise TypeError("tile_arange() does not support Warp struct dtypes")

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
        start = warp._src.codegen.Var(label=None, type=return_type.dtype, constant=0)
        stop = args[0]
        step = warp._src.codegen.Var(label=None, type=return_type.dtype, constant=1)
    elif len(args) == 2:
        start = args[0]
        stop = args[1]
        step = warp._src.codegen.Var(label=None, type=return_type.dtype, constant=1)
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
    is_differentiable=False,
    doc="""Generate a tile of linearly spaced elements.

    - ``(stop,)``: Generates values from ``0`` to ``stop - 1``
    - ``(start, stop)``: Generates values from ``start`` to ``stop - 1``
    - ``(start, stop, step)``: Generates values from ``start`` to ``stop - 1`` with a step size

    Args:
        args: Variable-length positional arguments, interpreted as:
        dtype: Data type of output tile's elements (optional, default: ``float``)
        storage: The storage location for the tile: ``"register"`` for registers
            (default) or ``"shared"`` for shared memory.

    Returns:
        A tile with ``shape=(n)`` with linearly spaced elements of specified data type.""",
    group="Tile Primitives",
    export=False,
)


def tile_load_tuple_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

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

    if arg_values.get("aligned"):
        if arg_values["storage"] == "register":
            log_warning(
                "tile_load() with aligned=True has no effect for storage='register'. "
                "The aligned parameter only affects shared memory tiles."
            )
        elif arg_values["storage"] == "shared" and len(shape) < 2:
            log_warning(
                "tile_load() with aligned=True has no effect for 1D shared tiles. "
                "The vectorized path requires 2D+ tiles."
            )

    return tile(dtype=a.dtype, shape=shape, storage=arg_values["storage"])


def tile_load_tuple_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    a = args["a"]
    shape = extract_tuple(args["shape"], as_constant=True)
    bounds_check = args["bounds_check"]
    aligned = args["aligned"]

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    if "offset" in args:
        offset = extract_tuple(args["offset"])
    else:
        offset = (0,) * a.type.ndim

    func_args = (a, *offset)
    # Force aligned=False for register tiles — the template arg only affects shared tiles
    # and passing True would generate a needless distinct instantiation.
    aligned_val = aligned.constant if return_type.storage == "shared" else False
    template_args = (return_type.dtype, bounds_check.constant, aligned_val, *shape)

    return (func_args, template_args)


add_builtin(
    "tile_load",
    input_types={
        "a": array(dtype=Any),
        "shape": tuple[int, ...],
        "offset": tuple[int, ...],
        "storage": str,
        "bounds_check": builtins.bool,
        "aligned": builtins.bool,
    },
    value_func=tile_load_tuple_value_func,
    dispatch_func=tile_load_tuple_dispatch_func,
    defaults={"offset": None, "storage": "register", "bounds_check": True, "aligned": False},
    variadic=False,
    doc="""Load a tile from a global memory array.

    This method will cooperatively load a tile from global memory using all threads in the block.

    Args:
        a: The source array in global memory
        shape: Shape of the tile to load, must have the same number of dimensions as ``a``
        offset: Offset in the source array to begin reading from (optional)
        storage: The storage location for the tile: ``"register"`` for registers
            (default) or ``"shared"`` for shared memory.
        bounds_check: Needed for unaligned tiles, but can disable for memory-aligned tiles for faster load times
        aligned: If True, skip runtime alignment checks for vectorized loads (shared memory,
            2D+ tiles only). Has no effect for 1D tiles or register storage. Use when you
            guarantee that: (1) the base address at the tile offset is 16-byte aligned,
            (2) the array is contiguous (dense row-major strides), (3) all outer-dimension
            strides are multiples of 16 bytes, and (4) the tile fits entirely within array
            bounds. Address-alignment violations trap unconditionally (even in release
            builds). Bounds and contiguity violations trigger debug-only asserts; in
            release builds they cause silent data corruption.

    Returns:
        A tile with shape as specified and data type the same as the source array.""",
    group="Tile Primitives",
    export=False,
)

# overload for scalar shape
add_builtin(
    "tile_load",
    input_types={
        "a": array(dtype=Any),
        "shape": int,
        "offset": int,
        "storage": str,
        "bounds_check": builtins.bool,
        "aligned": builtins.bool,
    },
    value_func=tile_load_tuple_value_func,
    dispatch_func=tile_load_tuple_dispatch_func,
    defaults={"offset": None, "storage": "register", "bounds_check": True, "aligned": False},
    doc="""Load a tile from a global memory array.""",
    group="Tile Primitives",
    export=False,
)


def tile_load_indexed_tuple_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    a = arg_types["a"]

    indices_tile = arg_types["indices"]
    indices_tile.storage = "shared"  # force to shared

    axis = arg_values["axis"]
    if axis >= a.ndim:
        raise ValueError(f"tile_load_indexed() axis argument must be valid axis of array {a}, got {axis}.")

    indices_tile_dim = len(indices_tile.shape)
    if indices_tile_dim != 1:
        raise ValueError(
            f"tile_load_indexed() indices argument must be a 1D tile, got {indices_tile_dim} dimensions instead."
        )

    shape = extract_tuple(arg_values["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    num_indices = indices_tile.shape[0]
    if num_indices != shape[axis]:
        raise ValueError(
            "The number of elements in the 1D indices tile must match the output tile shape along the specified axis."
        )

    if "offset" in arg_values:
        offset = extract_tuple(arg_values["offset"])
    else:
        offset = (0,) * a.ndim

    if a.ndim != len(shape):
        raise ValueError(
            f"tile_load_indexed() array argument must have same number of dimensions as the tile shape, trying to perform an {len(shape)} dimensional load from an array with {a.ndim} dimensions."
        )

    if a.ndim != len(offset):
        raise ValueError(
            f"tile_load_indexed() offset argument must have the same number of dimensions as the array to load from, got {len(offset)} indices for an array with {a.ndim} dimensions"
        )

    if arg_values["storage"] not in {"shared", "register"}:
        raise ValueError(f"Invalid value for 'storage': {arg_values['storage']!r}. Expected 'shared' or 'register'.")

    return tile(dtype=a.dtype, shape=shape, storage=arg_values["storage"])


def tile_load_indexed_tuple_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    a = args["a"]
    indices_tile = args["indices"]
    axis = args["axis"]

    shape = extract_tuple(args["shape"], as_constant=True)

    if None in shape:
        raise ValueError("Tile functions require shape to be a compile time constant.")

    if "offset" in args:
        offset = extract_tuple(args["offset"])
    else:
        offset = (0,) * a.type.ndim

    func_args = (a, indices_tile, axis, *offset)
    template_args = shape

    return (func_args, template_args)


add_builtin(
    "tile_load_indexed",
    input_types={
        "a": array(dtype=Any),
        "indices": tile(dtype=int, shape=tuple[int]),
        "shape": tuple[int, ...],
        "offset": tuple[int, ...],
        "axis": int,
        "storage": str,
    },
    value_func=tile_load_indexed_tuple_value_func,
    dispatch_func=tile_load_indexed_tuple_dispatch_func,
    defaults={"offset": None, "axis": 0, "storage": "register"},
    variadic=False,
    doc="""Load a tile from a global memory array, with loads along a specified axis mapped according to a 1D tile of indices.

    Args:
        a: The source array in global memory
        indices: A 1D tile of integer indices mapping to elements in ``a``.
        shape: Shape of the tile to load, must have the same number of dimensions as ``a``, and along ``axis``, it must have the same number of elements as the ``indices`` tile.
        offset: Offset in the source array to begin reading from (optional)
        axis: Axis of ``a`` that indices refer to
        storage: The storage location for the tile: ``"register"`` for registers (default) or ``"shared"`` for shared memory.

    Returns:
        A tile with shape as specified and data type the same as the source array.

    Example:

        This example shows how to select and store the even indexed rows from a 2D array.

        .. code-block:: python

            TILE_M = wp.constant(2)
            TILE_N = wp.constant(2)
            HALF_M = wp.constant(TILE_M // 2)
            HALF_N = wp.constant(TILE_N // 2)

            @wp.kernel
            def compute(x: wp.array2d[float], y: wp.array2d[float]):
                i, j = wp.tid()

                evens = wp.tile_arange(HALF_M, dtype=int, storage="shared") * 2

                t0 = wp.tile_load_indexed(x, indices=evens, shape=(HALF_M, TILE_N), offset=(i*TILE_M, j*TILE_N), axis=0, storage="register")
                wp.tile_store(y, t0, offset=(i*HALF_M, j*TILE_N))

            M = TILE_M * 2
            N = TILE_N * 2

            arr = np.arange(M * N).reshape(M, N)

            x = wp.array(arr, dtype=float)
            y = wp.zeros((M // 2, N), dtype=float)

            wp.launch_tiled(compute, dim=[2,2], inputs=[x], outputs=[y], block_dim=32, device=device)

            print(x.numpy())
            print(y.numpy())

        .. code-block:: text

            [[ 0.  1.  2.  3.]
             [ 4.  5.  6.  7.]
             [ 8.  9. 10. 11.]
             12. 13. 14. 15.]]

            [[ 0.  1.  2.  3.]
             [ 8.  9. 10. 11.]]
    """,
    group="Tile Primitives",
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

    if arg_values.get("aligned"):
        if t.storage == "register":
            log_warning(
                "tile_store() with aligned=True has no effect for register tiles. "
                "The aligned parameter only affects shared memory tiles."
            )
        elif t.storage == "shared" and len(t.shape) < 2:
            log_warning(
                "tile_store() with aligned=True has no effect for 1D shared tiles. "
                "The vectorized path requires 2D+ tiles."
            )

    return None


def tile_store_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    a = args["a"]
    t = args["t"]
    bounds_check = args["bounds_check"]
    aligned = args["aligned"]

    if "offset" in args:
        offset = extract_tuple(args["offset"])
    else:
        offset = (0,) * a.type.ndim

    func_args = (a, *offset, t)
    aligned_val = aligned.constant if t.type.storage == "shared" else False
    template_args = (a.type.dtype, bounds_check.constant, aligned_val)

    return (func_args, template_args)


add_builtin(
    "tile_store",
    input_types={
        "a": array(dtype=Any),
        "t": tile(dtype=Any, shape=tuple[int, ...]),
        "offset": tuple[int, ...],
        "bounds_check": builtins.bool,
        "aligned": builtins.bool,
    },
    value_func=tile_store_value_func,
    dispatch_func=tile_store_dispatch_func,
    defaults={"offset": None, "bounds_check": True, "aligned": False},
    variadic=False,
    skip_replay=True,
    doc="""Store a tile to a global memory array.

    This method will cooperatively store a tile to global memory using all threads in the block.

    Args:
        a: The destination array in global memory
        t: The source tile to store data from, must have the same data type and number of dimensions as the destination array
        offset: Offset in the destination array (optional)
        bounds_check: Needed for unaligned tiles, but can disable for memory-aligned tiles for faster write times.
        aligned: If True, skip runtime alignment checks for vectorized stores (shared memory,
            2D+ tiles only). Has no effect for 1D tiles or register storage. Use when you
            guarantee that: (1) the base address at the tile offset is 16-byte aligned,
            (2) the array is contiguous (dense row-major strides), (3) all outer-dimension
            strides are multiples of 16 bytes, and (4) the tile fits entirely within array
            bounds. Address-alignment violations trap unconditionally (even in release
            builds). Bounds and contiguity violations trigger debug-only asserts; in
            release builds they cause silent data corruption.""",
    group="Tile Primitives",
    export=False,
)

# overload for scalar offset
add_builtin(
    "tile_store",
    input_types={
        "a": array(dtype=Any),
        "t": tile(dtype=Any, shape=tuple[int, ...]),
        "offset": int,
        "bounds_check": builtins.bool,
        "aligned": builtins.bool,
    },
    value_func=tile_store_value_func,
    dispatch_func=tile_store_dispatch_func,
    defaults={"offset": None, "bounds_check": True, "aligned": False},
    variadic=False,
    skip_replay=True,
    doc="""Store a tile to a global memory array.""",
    group="Tile Primitives",
    export=False,
)


def tile_store_indexed_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return None

    a = arg_types["a"]
    t = arg_types["t"]
    indices_tile = arg_types["indices"]
    indices_tile.storage = "shared"  # force to shared

    axis = arg_values["axis"]
    if axis >= a.ndim:
        raise ValueError(f"tile_store_indexed() axis argument must be valid axis of array {a}, got {axis}.")

    indices_tile_dim = len(indices_tile.shape)
    if indices_tile_dim != 1:
        raise ValueError(
            f"tile_store_indexed() indices argument must be a 1D tile, got {indices_tile_dim} dimensions instead."
        )

    num_indices = indices_tile.shape[0]
    if num_indices != t.shape[axis]:
        raise ValueError(
            "The number of elements in the 1D indices tile must match the input tile shape along the specified axis."
        )

    if "offset" in arg_types:
        c = extract_tuple(arg_values["offset"])
    else:
        c = (0,) * a.ndim

    if len(c) != a.ndim:
        raise ValueError(
            f"tile_store_indexed() 'a' argument must have {len(c)} dimensions, "
            f"calculated based on the provided offset arguments, but got {a.ndim} dimensions."
        )

    if len(t.shape) != a.ndim:
        raise ValueError(
            f"tile_store_indexed() 'a' argument must have the same number of dimensions as the 't' argument, "
            f"but got {a.ndim} dimensions for 'a' and {len(t.shape)} dimensions for 't'"
        )

    if not types_equal(arg_types["a"].dtype, arg_types["t"].dtype):
        raise TypeError(
            f"tile_store_indexed() 'a' and 't' arguments must have the same dtype, got {arg_types['a'].dtype} and {arg_types['t'].dtype}"
        )

    return None


def tile_store_indexed_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    a = args["a"]
    indices_tile = args["indices"]
    axis = args["axis"]
    t = args["t"]

    if "offset" in args:
        offset = extract_tuple(args["offset"])
    else:
        offset = (0,) * a.type.ndim

    func_args = (a, indices_tile, axis, *offset, t)
    template_args = []

    return (func_args, template_args)


add_builtin(
    "tile_store_indexed",
    input_types={
        "a": array(dtype=Any),
        "indices": tile(dtype=int, shape=tuple[int]),
        "t": tile(dtype=Any, shape=tuple[int, ...]),
        "offset": tuple[int, ...],
        "axis": int,
    },
    value_func=tile_store_indexed_value_func,
    dispatch_func=tile_store_indexed_dispatch_func,
    defaults={"offset": None, "axis": 0},
    variadic=False,
    skip_replay=True,
    doc="""Store a tile to a global memory array, with storage along a specified axis mapped according to a 1D tile of indices.

    Args:
        a: The destination array in global memory
        indices: A 1D tile of integer indices mapping to elements in ``a``.
        t: The source tile to store data from, must have the same data type and number of dimensions as the destination array, and along ``axis``, it must have the same number of elements as the ``indices`` tile.
        offset: Offset in the destination array (optional)
        axis: Axis of ``a`` that indices refer to.

    Example:

        This example shows how to map tile rows to the even rows of a 2D array.

        .. code-block:: python

            TILE_M = wp.constant(2)
            TILE_N = wp.constant(2)
            TWO_M = wp.constant(TILE_M * 2)
            TWO_N = wp.constant(TILE_N * 2)

            @wp.kernel
            def compute(x: wp.array2d[float], y: wp.array2d[float]):
                i, j = wp.tid()

                t = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i*TILE_M, j*TILE_N), storage="register")

                evens_M = wp.tile_arange(TILE_M, dtype=int, storage="shared") * 2

                wp.tile_store_indexed(y, indices=evens_M, t=t, offset=(i*TWO_M, j*TILE_N), axis=0)

            M = TILE_M * 2
            N = TILE_N * 2

            arr = np.arange(M * N, dtype=float).reshape(M, N)

            x = wp.array(arr, dtype=float, requires_grad=True, device=device)
            y = wp.zeros((M * 2, N), dtype=float, requires_grad=True, device=device)

            wp.launch_tiled(compute, dim=[2,2], inputs=[x], outputs=[y], block_dim=32, device=device)

            print(x.numpy())
            print(y.numpy())

        .. code-block:: text

            [[ 0.  1.  2.  3.]
                [ 4.  5.  6.  7.]
                [ 8.  9. 10. 11.]
                [12. 13. 14. 15.]]

            [[ 0.  1.  2.  3.]
                [ 0.  0.  0.  0.]
                [ 4.  5.  6.  7.]
                [ 0.  0.  0.  0.]
                [ 8.  9. 10. 11.]
                [ 0.  0.  0.  0.]
                [12. 13. 14. 15.]
                [ 0.  0.  0.  0.]]
    """,
    group="Tile Primitives",
    export=False,
)


def check_tile_atomic_add_dtype(dtype, fn_name):
    # Mirror the wp.atomic_add() constraint: only scalar leaf types with a CUDA atomicAdd
    # overload may be accumulated. Struct tiles are allowed; their generated helper carries
    # fields whose scalar type has no atomic add rather than accumulating them.
    if type_is_struct(dtype):
        return
    scalar_type = getattr(dtype, "_wp_scalar_type_", dtype)
    supported_atomic_types = (*SUPPORTED_ATOMIC_TYPES, warp.float16, warp.bfloat16)
    if not any(types_equal_generic(scalar_type, x) for x in supported_atomic_types):
        raise RuntimeError(
            f"{fn_name}() only supports tiles with [u]int32, [u]int64, float16, bfloat16, float32, "
            f"or float64 as the underlying scalar type, but got {type_repr(dtype)}"
        )


def tile_atomic_add_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

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

    check_tile_atomic_add_dtype(t.dtype, "tile_atomic_add")

    return tile(
        dtype=arg_types["t"].dtype,
        shape=arg_types["t"].shape,
        storage=arg_types["t"].storage,
    )


def tile_atomic_add_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    a = args["a"]
    t = args["t"]
    bounds_check = args["bounds_check"]

    if "offset" in args:
        offset = extract_tuple(args["offset"])
    else:
        offset = (0,) * a.type.ndim

    func_args = (a, *offset, t)
    template_args = (a.type.dtype, bounds_check.constant)

    return (func_args, template_args)


add_builtin(
    "tile_atomic_add",
    input_types={
        "a": array(dtype=Any),
        "t": tile(dtype=Any, shape=tuple[int, ...]),
        "offset": tuple[int, ...],
        "bounds_check": builtins.bool,
    },
    value_func=tile_atomic_add_value_func,
    dispatch_func=tile_atomic_add_dispatch_func,
    defaults={"offset": None, "bounds_check": True},
    variadic=False,
    skip_replay=True,
    doc="""Atomically add a tile onto the array ``a``.

    Each element is updated atomically.

    Args:
        a: Array in global memory, should have the same ``dtype`` as the input tile
        t: Source tile to add to the destination array
        offset: Offset in the destination array (optional)
        bounds_check: Needed for unaligned tiles, but can disable for memory-aligned tiles for faster write times

    Returns:
        A tile with the same dimensions and data type as the source tile, holding the original value of the destination elements.""",
    group="Tile Primitives",
    export=False,
)

# overload for scalar offset
add_builtin(
    "tile_atomic_add",
    input_types={
        "a": array(dtype=Any),
        "t": tile(dtype=Any, shape=tuple[int, ...]),
        "offset": int,
        "bounds_check": builtins.bool,
    },
    value_func=tile_atomic_add_value_func,
    dispatch_func=tile_atomic_add_dispatch_func,
    defaults={"offset": None, "bounds_check": True},
    variadic=False,
    skip_replay=True,
    doc="""Atomically add a tile onto the array ``a``.""",
    group="Tile Primitives",
    export=False,
)


def tile_atomic_add_indexed_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    a = arg_types["a"]
    t = arg_types["t"]
    indices_tile = arg_types["indices"]
    indices_tile.storage = "shared"  # force to shared

    axis = arg_values["axis"]
    if axis >= a.ndim:
        raise ValueError(f"tile_atomic_add_indexed() axis argument must be valid axis of array {a}, got {axis}.")

    indices_tile_dim = len(indices_tile.shape)
    if indices_tile_dim != 1:
        raise ValueError(
            f"tile_atomic_add_indexed() indices argument must be a 1D tile, got {indices_tile_dim} dimensions instead."
        )

    num_indices = indices_tile.shape[0]
    if num_indices != t.shape[axis]:
        raise ValueError(
            "The number of elements in the 1D indices tile must match the input tile shape along the specified axis."
        )

    if "offset" in arg_types:
        c = extract_tuple(arg_values["offset"])
    else:
        c = (0,) * a.ndim

    if len(c) != a.ndim:
        raise ValueError(
            f"tile_atomic_add_indexed() 'a' argument must have {len(c)} dimensions, "
            f"calculated based on the provided offset arguments, but got {a.ndim} dimensions."
        )

    if len(t.shape) != a.ndim:
        raise ValueError(
            f"tile_atomic_add_indexed() 'a' argument must have the same number of dimensions as the 't' argument, "
            f"but got {a.ndim} dimensions for 'a' and {len(t.shape)} dimensions for 't'"
        )

    if not types_equal(arg_types["a"].dtype, arg_types["t"].dtype):
        raise TypeError(
            f"tile_atomic_add_indexed() 'a' and 't' arguments must have the same dtype, got {arg_types['a'].dtype} and {arg_types['t'].dtype}"
        )

    check_tile_atomic_add_dtype(t.dtype, "tile_atomic_add_indexed")

    return tile(dtype=t.dtype, shape=t.shape, storage=t.storage)


def tile_atomic_add_indexed_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    a = args["a"]
    indices_tile = args["indices"]
    axis = args["axis"]
    t = args["t"]

    if "offset" in args:
        offset = extract_tuple(args["offset"])
    else:
        offset = (0,) * a.type.ndim

    func_args = (a, indices_tile, axis, *offset, t)
    template_args = []

    return (func_args, template_args)


add_builtin(
    "tile_atomic_add_indexed",
    input_types={
        "a": array(dtype=Any),
        "indices": tile(dtype=int, shape=tuple[int]),
        "t": tile(dtype=Any, shape=tuple[int, ...]),
        "offset": tuple[int, ...],
        "axis": int,
    },
    value_func=tile_atomic_add_indexed_value_func,
    dispatch_func=tile_atomic_add_indexed_dispatch_func,
    defaults={"offset": None, "axis": 0},
    variadic=False,
    skip_replay=True,
    doc="""Atomically add a tile to a global memory array, with storage along a specified axis mapped according to a 1D tile of indices.

    Args:
        a: The destination array in global memory
        indices: A 1D tile of integer indices mapping to elements in ``a``.
        t: The source tile to extract data from, must have the same data type and number of dimensions as the destination array, and along ``axis``, it must have the same number of elements as the ``indices`` tile.
        offset: Offset in the destination array (optional)
        axis: Axis of ``a`` that indices refer to.

    Example:

        This example shows how to compute a blocked, row-wise reduction.

        .. code-block:: python

            TILE_M = wp.constant(2)
            TILE_N = wp.constant(2)

            @wp.kernel
            def tile_atomic_add_indexed(x: wp.array2d[float], y: wp.array2d[float]):
                i, j = wp.tid()

                t = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i*TILE_M, j*TILE_N), storage="register")

                zeros = wp.tile_zeros(TILE_M, dtype=int, storage="shared")

                wp.tile_atomic_add_indexed(y, indices=zeros, t=t, offset=(i, j*TILE_N), axis=0)

            M = TILE_M * 2
            N = TILE_N * 2

            arr = np.arange(M * N, dtype=float).reshape(M, N)

            x = wp.array(arr, dtype=float, requires_grad=True, device=device)
            y = wp.zeros((2, N), dtype=float, requires_grad=True, device=device)

            wp.launch_tiled(tile_atomic_add_indexed, dim=[2,2], inputs=[x], outputs=[y], block_dim=32, device=device)

            print(x.numpy())
            print(y.numpy())

        .. code-block:: text

            [[ 0.  1.  2.  3.]
                [ 4.  5.  6.  7.]
                [ 8.  9. 10. 11.]
                [12. 13. 14. 15.]]

            [[ 4.  6.  8. 10.]
                [20. 22. 24. 26.]]
    """,
    group="Tile Primitives",
    export=False,
)


def tile_view_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

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
    input_types={"t": tile(dtype=Any, shape=tuple[int, ...]), "offset": tuple[int, ...], "shape": tuple[int, ...]},
    value_func=tile_view_value_func,
    dispatch_func=tile_view_dispatch_func,
    defaults={"shape": None},
    variadic=False,
    doc="""Extract a slice of a given tile [offset, offset+shape], if shape is not specified it will be inferred from the unspecified offset dimensions.

    Args:
        t: Input tile to extract a subrange from
        offset: Offset in the source tile
        shape: Shape of the returned slice

    Returns:
        A tile with dimensions given by the specified shape or the remaining source tile dimensions.""",
    group="Tile Primitives",
    is_differentiable=False,
    export=False,
)


def tile_squeeze_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

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
    input_types={"t": tile(dtype=Any, shape=tuple[int, ...]), "axis": tuple[int, ...]},
    defaults={"axis": None},
    value_func=tile_squeeze_value_func,
    dispatch_func=tile_squeeze_dispatch_func,
    variadic=False,
    doc="""Create a squeezed view of a tile with the same data.

    Args:
        t: Input tile to squeeze
        axis: A subset of the entries of length one in the shape (optional)

    Returns:
        The input tile but with all or a subset of the dimensions of length one removed.""",
    group="Tile Primitives",
    export=False,
)


def tile_reshape_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

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
    input_types={"t": tile(dtype=Any, shape=tuple[int, ...]), "shape": tuple[int, ...]},
    value_func=tile_reshape_value_func,
    dispatch_func=tile_reshape_dispatch_func,
    variadic=False,
    doc="""Create a reshaped view of a tile with the same data.

    Args:
        t: Input tile to reshape
        shape: New shape for the tile

    Returns:
        A tile containing the same data as the input tile, but arranged in a new shape.""",
    group="Tile Primitives",
    export=False,
)


def tile_astype_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    tile_type = arg_types["t"]
    dtype = arg_values["dtype"]

    return tile(dtype=dtype, shape=tile_type.shape)


def tile_astype_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    tile = arg_values["t"]

    return ((tile,), (return_type,))


add_builtin(
    "tile_astype",
    input_types={"t": tile(dtype=Scalar, shape=tuple[int, ...]), "dtype": Scalar},
    value_func=tile_astype_value_func,
    dispatch_func=tile_astype_dispatch_func,
    variadic=False,
    doc="""Create a new tile with the same data as the input tile, but with a different data type.

    Args:
        t: Input tile
        dtype: New data type for the tile

    Returns:
        A tile with the same data as the input tile, but with a different data type.""",
    group="Tile Primitives",
    export=False,
)


def tile_assign_value_func(arg_types, arg_values):
    if arg_types is None:
        return None

    dst_type = arg_types["dst"]
    src_type = arg_types.get("src")

    # When both operands are tiles (tile_assign), enforce rank compatibility.
    # For scalar/element-wise assign overloads where src is non-tile, skip this
    # check and just force dst to shared as before.
    if src_type is not None and is_tile(src_type):
        if len(dst_type.shape) != len(src_type.shape):
            raise ValueError(
                f"tile_assign() destination and source tiles must have the same rank, "
                f"got {len(dst_type.shape)} and {len(src_type.shape)}"
            )

    # force the destination tile to shared memory
    dst_type.storage = "shared"
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
        "dst": tile(dtype=Any, shape=tuple[int, ...]),
        "src": tile(dtype=Any, shape=tuple[int, ...]),
        "offset": tuple[int, ...],
    },
    value_func=tile_assign_value_func,
    dispatch_func=tile_assign_dispatch_func,
    defaults={"offset": None},
    doc="""Assign a tile to a subrange of a destination tile.

    Args:
        dst: The destination tile to assign to
        src: The source tile to read values from
        offset: Offset in the destination tile to write to.""",
    group="Tile Primitives",
    export=False,
)

# handles expressions like tile[i,j] = 1.0
add_builtin(
    "assign",
    input_types={"dst": tile(dtype=Any, shape=tuple[int]), "i": int, "src": Any},
    value_func=tile_assign_value_func,
    group="Tile Primitives",
    export=False,
    hidden=True,
)

add_builtin(
    "assign",
    input_types={"dst": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "src": Any},
    value_func=tile_assign_value_func,
    group="Tile Primitives",
    export=False,
    hidden=True,
)

add_builtin(
    "assign",
    input_types={"dst": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "src": Any},
    value_func=tile_assign_value_func,
    group="Tile Primitives",
    export=False,
    hidden=True,
)

add_builtin(
    "assign",
    input_types={
        "dst": tile(dtype=Any, shape=tuple[int, ...]),
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

add_builtin(
    "assign",
    input_types={
        "dst": tile(dtype=Any, shape=tuple[int, ...]),
        "i": int,
        "j": int,
        "k": int,
        "l": int,
        "m": int,
        "src": Any,
    },
    value_func=tile_assign_value_func,
    group="Tile Primitives",
    export=False,
    hidden=True,
)

add_builtin(
    "assign",
    input_types={
        "dst": tile(dtype=Any, shape=tuple[int, int, int, int]),
        "i": int,
        "j": int,
        "k": int,
        "l": int,
        "m": int,
        "n": int,
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
        return tile(dtype=Any, shape=tuple)

    if len(arg_types) > 2:
        raise TypeError(f"tile() takes 1 positional argument and 1 optional argument but {len(arg_types)} were given")

    preserve_type = arg_values["preserve_type"]

    if preserve_type:
        dtype = arg_types["x"]
        shape = (warp._src.codegen.options["block_dim"],)

        return tile(dtype=dtype, shape=shape)

    else:
        if type_is_vector(arg_types["x"]):
            dtype = arg_types["x"]._wp_scalar_type_
            length = arg_types["x"]._shape_[0]
            shape = (length, warp._src.codegen.options["block_dim"])
        elif type_is_quaternion(arg_types["x"]):
            dtype = arg_types["x"]._wp_scalar_type_
            shape = (4, warp._src.codegen.options["block_dim"])
        elif type_is_matrix(arg_types["x"]):
            dtype = arg_types["x"]._wp_scalar_type_
            rows = arg_types["x"]._shape_[0]
            cols = arg_types["x"]._shape_[1]
            shape = (rows, cols, warp._src.codegen.options["block_dim"])
        else:
            dtype = arg_types["x"]
            shape = (warp._src.codegen.options["block_dim"],)

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

    Args:
        x: A per-thread local value, e.g. scalar, vector, or matrix.
        preserve_type: If true, the tile will have the same data type as the input value.

    Returns:
        If ``preserve_type=True``, a tile of type ``x.type`` of length ``block_dim``. Otherwise, an N-dimensional tile such that the first N-1 dimensions match the shape of ``x`` and the final dimension is of size ``block_dim``.

    Example:

        This example shows how to create a linear sequence from thread variables.

        .. code-block:: python

            @wp.kernel
            def compute():
                i = wp.tid()
                t = wp.tile(i*2)
                print(t)

            wp.launch(compute, dim=16, inputs=[], block_dim=16)

        .. code-block:: text

            [0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30] = tile(shape=(16), storage=register)
    """,
    group="Tile Primitives",
    export=False,
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

    if t.shape[-1] != warp._src.codegen.options["block_dim"]:
        raise ValueError(
            f"untile() argument last dimension {t.shape[-1]} does not match the expected block width {warp._src.codegen.options['block_dim']}"
        )

    if len(t.shape) == 1:
        return t.dtype
    elif len(t.shape) == 2:
        return warp._src.types.vector(t.shape[0], t.dtype)
    elif len(t.shape) == 3:
        return warp._src.types.matrix((t.shape[0], t.shape[1]), t.dtype)
    else:
        raise ValueError(f"untile() argument must have a positive size in dimension 0, but got {t.shape[0]}")


add_builtin(
    "untile",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=untile_value_func,
    variadic=True,
    doc="""Convert a tile back to per-thread values.

    This function converts a block-wide tile back to per-thread values.

    * If the input tile is 1D, then the resulting value will be a per-thread scalar
    * If the input tile is 2D, then the resulting value will be a per-thread vector of length M

    Args:
        a: A tile with dimensions ``shape=(M, block_dim)``

    Returns:
        A single value per-thread with the same data type as the tile.

    Example:

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

    # count the number of indices (all parameters except the tile "a")
    num_indices = len(arg_types) - 1
    tile_dtype = arg_types["a"].dtype
    tile_shape = arg_types["a"].shape

    if type_is_vector(tile_dtype):
        if num_indices == len(tile_shape):
            return tile_dtype
        elif num_indices == len(tile_shape) + 1:
            return tile_dtype._wp_scalar_type_
        else:
            raise IndexError(
                f"tile_extract: incorrect number of indices ({num_indices}) for tile shape {tuple(tile_shape)}"
            )
    elif type_is_matrix(tile_dtype):
        if num_indices == len(tile_shape):
            return tile_dtype
        elif num_indices == len(tile_shape) + 2:
            return tile_dtype._wp_scalar_type_
        else:
            raise IndexError(
                f"tile_extract: incorrect number of indices ({num_indices}) for matrix tile shape {tuple(tile_shape)}"
            )
    else:
        # scalar element: index count must exactly match tile rank
        if num_indices == len(tile_shape):
            return tile_dtype
        raise IndexError(
            f"tile_extract: incorrect number of indices ({num_indices}) for tile shape {tuple(tile_shape)}"
        )


add_builtin(
    "tile_extract",
    input_types={"a": tile(dtype=Any, shape=tuple[int]), "i": int},
    value_func=tile_extract_value_func,
    variadic=False,
    doc="""Extract a single element from the tile.

    This function will extract an element from the tile and broadcast its value to all threads in the block.

    Note that this may incur additional synchronization if the source tile is a register tile.

    Args:
        a: Tile to extract the element from
        i: Coordinate of element on first dimension

    Returns:
        The value of the element at the specified tile location with the same data type as the input tile.""",
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "tile_extract",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int},
    value_func=tile_extract_value_func,
    variadic=False,
    doc="""Extract a single element from the tile.

    This function will extract an element from the tile and broadcast its value to all threads in the block.

    Note that this may incur additional synchronization if the source tile is a register tile.

    Args:
        a: Tile to extract the element from
        i: Coordinate of element on first dimension
        j: Coordinate of element on the second dimension, or vector index

    Returns:
        The value of the element at the specified tile location with the same data type as the input tile.""",
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "tile_extract",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int},
    value_func=tile_extract_value_func,
    variadic=False,
    doc="""Extract a single element from the tile.

    This function will extract an element from the tile and broadcast its value to all threads in the block.

    Note that this may incur additional synchronization if the source tile is a register tile.

    Args:
        a: Tile to extract the element from
        i: Coordinate of element on first dimension
        j: Coordinate of element on the second dimension, or first matrix index
        k: Coordinate of element on the third dimension, or vector index, or second matrix index

    Returns:
        The value of the element at the specified tile location with the same data type as the input tile.""",
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "tile_extract",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "l": int},
    value_func=tile_extract_value_func,
    variadic=False,
    doc="""Extract a single element from the tile.

    This function will extract an element from the tile and broadcast its value to all threads in the block.

    Note that this may incur additional synchronization if the source tile is a register tile.

    Args:
        a: Tile to extract the element from
        i: Coordinate of element on first dimension
        j: Coordinate of element on the second dimension
        k: Coordinate of element on the third dimension, or first matrix index
        l: Coordinate of element on the fourth dimension, or vector index, or second matrix index

    Returns:
        The value of the element at the specified tile location, with the same data type as the input tile.""",
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "tile_extract",
    input_types={
        "a": tile(dtype=Any, shape=tuple[int, ...]),
        "i": int,
        "j": int,
        "k": int,
        "l": int,
        "m": int,
    },
    value_func=tile_extract_value_func,
    variadic=False,
    doc="""Extract a single element from the tile.

    This function will extract an element from the tile and broadcast its value to all threads in the block.

    Note that this may incur additional synchronization if the source tile is a register tile.

    Args:
        a: Tile to extract the element from
        i: Coordinate of element on first dimension
        j: Coordinate of element on the second dimension
        k: Coordinate of element on the third dimension
        l: Coordinate of element on the fourth dimension, or first matrix index
        m: Vector index, or second matrix index

    Returns:
        The value of the element at the specified tile location, with the same data type as the input tile.""",
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "tile_extract",
    input_types={
        "a": tile(dtype=Any, shape=tuple[int, int, int, int]),
        "i": int,
        "j": int,
        "k": int,
        "l": int,
        "m": int,
        "n": int,
    },
    value_func=tile_extract_value_func,
    variadic=False,
    doc="""Extract a single element from the tile.

    This function will extract an element from the tile and broadcast its value to all threads in the block.

    Note that this may incur additional synchronization if the source tile is a register tile.

    Args:
        a: Tile to extract the element from
        i: Coordinate of element on first dimension
        j: Coordinate of element on the second dimension
        k: Coordinate of element on the third dimension
        l: Coordinate of element on the fourth dimension
        m: Vector index, or first matrix index
        n: Second matrix index

    Returns:
        The value of the element at the specified tile location, with the same data type as the input tile.""",
    group="Tile Primitives",
    export=False,
)


def tile_scatter_add_value_func(arg_types, arg_values):
    if arg_types is None:
        return None

    t = arg_types["a"]
    if not is_tile(t):
        raise TypeError(f"tile_scatter_add() 'a' argument must be a tile, got {t!r}")

    t.storage = "shared"

    num_indices = sum(1 for k in arg_types if k in {"i", "j", "k", "l"})
    if num_indices != len(t.shape):
        raise IndexError(
            f"tile_scatter_add: incorrect number of indices ({num_indices}) for tile shape {tuple(t.shape)}"
        )

    value_type = arg_types["value"]
    if not types_equal(t.dtype, value_type):
        raise TypeError(
            f"tile_scatter_add() 'value' type must match tile dtype, got {value_type} and tile dtype {t.dtype}"
        )

    return None


def tile_scatter_add_dispatch_func(input_types, return_type, args):
    atomic = args["atomic"]
    if atomic.constant is None:
        raise ValueError(
            "tile_scatter_add() 'atomic' must be a compile-time constant (True or False), not a runtime variable"
        )
    idx_names = [x for x in "ijkl" if args.get(x) is not None]
    func_args = (args["a"], *(args[x] for x in idx_names), args["value"], args["has_value"])
    if atomic.constant is False:
        template_args = (False,)
    else:
        template_args = ()
    return (func_args, template_args)


add_builtin(
    "tile_scatter_add",
    input_types={
        "a": tile(dtype=Any, shape=tuple[int, ...]),
        "i": int,
        "value": Any,
        "has_value": builtins.bool,
        "atomic": builtins.bool,
    },
    value_func=tile_scatter_add_value_func,
    dispatch_func=tile_scatter_add_dispatch_func,
    defaults={"atomic": True},
    doc="""Scatter-add a per-thread value into a shared-memory tile.

    Cooperative operation -- all threads in the block must call this function.
    Each thread whose ``has_value`` is ``True`` adds ``value`` at index ``i``.

    A synchronization barrier is included so the updated values are visible to
    all threads after the call returns.

    Args:
        a: A shared-memory tile to scatter-add into.
        i: Index of the element to add to.
        value: The value to add (must match the tile's dtype).
        has_value: Whether this thread should perform the add.
        atomic: If True (default), use atomic add for safe concurrent writes.
            Set to False when indices are guaranteed unique across threads
            (e.g., lane-parallel writes) for better performance.

    Example:

        .. code-block:: python

            @wp.kernel
            def histogram(data: wp.array[float], out: wp.array[float]):

                bins = wp.tile_zeros(dtype=float, shape=4, storage="shared")
                i = wp.tid()
                # Bin values in [0, 8) into 4 bins of width 2
                b = int(data[i] / 2.0)
                wp.tile_scatter_add(bins, b, 1.0, True)
                wp.tile_store(out, bins, offset=0)

            data = wp.array([0.5, 1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0], dtype=float)
            output = wp.zeros(4, dtype=float)
            wp.launch_tiled(histogram, dim=[1], inputs=[data, output], block_dim=8)

            print(output.numpy())

        .. code-block:: text

            [2. 2. 2. 2.]""",
    group="Tile Primitives",
    export=False,
)
add_builtin(
    "tile_scatter_add",
    input_types={
        "a": tile(dtype=Any, shape=tuple[int, ...]),
        "i": int,
        "j": int,
        "value": Any,
        "has_value": builtins.bool,
        "atomic": builtins.bool,
    },
    value_func=tile_scatter_add_value_func,
    dispatch_func=tile_scatter_add_dispatch_func,
    defaults={"atomic": True},
    group="Tile Primitives",
    export=False,
)
add_builtin(
    "tile_scatter_add",
    input_types={
        "a": tile(dtype=Any, shape=tuple[int, ...]),
        "i": int,
        "j": int,
        "k": int,
        "value": Any,
        "has_value": builtins.bool,
        "atomic": builtins.bool,
    },
    value_func=tile_scatter_add_value_func,
    dispatch_func=tile_scatter_add_dispatch_func,
    defaults={"atomic": True},
    group="Tile Primitives",
    export=False,
)
add_builtin(
    "tile_scatter_add",
    input_types={
        "a": tile(dtype=Any, shape=tuple[int, ...]),
        "i": int,
        "j": int,
        "k": int,
        "l": int,
        "value": Any,
        "has_value": builtins.bool,
        "atomic": builtins.bool,
    },
    value_func=tile_scatter_add_value_func,
    dispatch_func=tile_scatter_add_dispatch_func,
    defaults={"atomic": True},
    group="Tile Primitives",
    export=False,
)


def tile_scatter_masked_value_func(arg_types, arg_values):
    if arg_types is None:
        return None

    t = arg_types["a"]
    if not is_tile(t):
        raise TypeError(f"tile_scatter_masked() 'a' argument must be a tile, got {t!r}")

    t.storage = "shared"

    num_indices = len(arg_types) - 3  # subtract 'a', 'value', 'has_value'
    if num_indices != len(t.shape):
        raise IndexError(
            f"tile_scatter_masked() incorrect number of indices ({num_indices}) for tile shape {tuple(t.shape)}"
        )

    value_type = arg_types["value"]
    if not types_equal(t.dtype, value_type):
        raise TypeError(
            f"tile_scatter_masked() 'value' type must match tile dtype, got {value_type} and tile dtype {t.dtype}"
        )

    return None


add_builtin(
    "tile_scatter_masked",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "value": Any, "has_value": builtins.bool},
    value_func=tile_scatter_masked_value_func,
    doc="""Write a value into a shared-memory tile from the calling thread.

    All threads in the block must call this function cooperatively.
    Each thread whose ``has_value`` is ``True`` writes ``value`` at the
    specified index.  A synchronization barrier is included so the written
    values are visible to all threads after the call returns.

    Each index should be written by at most one thread per call.  If multiple
    threads write to the same index, the result is undefined (data race in the
    forward pass, incorrect gradients in the backward pass).

    Example:

        .. code-block:: python

            @wp.kernel
            def write_kernel(out: wp.array[int]):
                tile_idx, thread_idx = wp.tid()

                # Allocate a shared-memory tile
                t = wp.tile_zeros(shape=64, dtype=int, storage="shared")

                # Each thread writes its own slot
                wp.tile_scatter_masked(t, thread_idx, thread_idx + 1, True)

                wp.tile_store(out, t)

    Args:
        a: The tile to write into (will use shared memory).
        i: Index of the element to write.
        value: The value to write (must match the tile's dtype).
        has_value: Whether this thread should perform the write.""",
    group="Tile Primitives",
    export=False,
)
add_builtin(
    "tile_scatter_masked",
    input_types={
        "a": tile(dtype=Any, shape=tuple[int, ...]),
        "i": int,
        "j": int,
        "value": Any,
        "has_value": builtins.bool,
    },
    value_func=tile_scatter_masked_value_func,
    group="Tile Primitives",
    export=False,
)
add_builtin(
    "tile_scatter_masked",
    input_types={
        "a": tile(dtype=Any, shape=tuple[int, ...]),
        "i": int,
        "j": int,
        "k": int,
        "value": Any,
        "has_value": builtins.bool,
    },
    value_func=tile_scatter_masked_value_func,
    group="Tile Primitives",
    export=False,
)
add_builtin(
    "tile_scatter_masked",
    input_types={
        "a": tile(dtype=Any, shape=tuple[int, ...]),
        "i": int,
        "j": int,
        "k": int,
        "l": int,
        "value": Any,
        "has_value": builtins.bool,
    },
    value_func=tile_scatter_masked_value_func,
    group="Tile Primitives",
    export=False,
)


def tile_inplace_value_func(arg_types, arg_values):
    if arg_types is None:
        return None

    if not types_equal(arg_types["a"].dtype, arg_types["value"]):
        raise TypeError(
            f"'value' must have the same dtype as target tile for inplace ops, got {arg_types['a'].dtype} and {arg_types['value']}"
        )

    # force the input tile to shared memory
    # as inplace addition/subtraction relies on shared memory atomics
    arg_types["a"].storage = "shared"

    return None


def tile_inplace_tile_value_func(arg_types, arg_values):
    if arg_types is None:
        return None

    a = arg_types["a"]
    b = arg_types["b"]

    if not is_tile(a):
        raise TypeError(f"Tile inplace operator left-hand side must be a tile, got {a!r}")
    if not is_tile(b):
        raise TypeError(f"Tile inplace operator right-hand side must be a tile, got {b!r}")

    if a.shape != b.shape:
        raise ValueError(f"Tile inplace arguments must have the same shape, got {a.shape} and {b.shape}")

    if not types_equal(a.dtype, b.dtype):
        raise TypeError(f"Tile inplace arguments must have the same dtype, got {a.dtype} and {b.dtype}")

    return None


def tile_inplace_bitwise_value_func(arg_types, arg_values):
    if arg_types is not None and type_is_struct(arg_types["a"].dtype):
        raise TypeError("Tile bitwise inplace operators do not support Warp struct tile elements")

    return tile_inplace_value_func(arg_types, arg_values)


def tile_inplace_tile_bitwise_value_func(arg_types, arg_values):
    if arg_types is not None and type_is_struct(arg_types["a"].dtype):
        raise TypeError("Tile bitwise inplace operators do not support Warp struct tile elements")

    return tile_inplace_tile_value_func(arg_types, arg_values)


add_builtin(
    "tile_add_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_add_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_add_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_add_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "l": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)

add_builtin(
    "tile_sub_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_sub_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_sub_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)
add_builtin(
    "tile_sub_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "l": int, "value": Any},
    value_func=tile_inplace_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
)

add_builtin(
    "tile_bit_and_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)
add_builtin(
    "tile_bit_and_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)
add_builtin(
    "tile_bit_and_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)
add_builtin(
    "tile_bit_and_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "l": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)

add_builtin(
    "tile_bit_or_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)
add_builtin(
    "tile_bit_or_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)
add_builtin(
    "tile_bit_or_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)
add_builtin(
    "tile_bit_or_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "l": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)

add_builtin(
    "tile_bit_xor_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)
add_builtin(
    "tile_bit_xor_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)
add_builtin(
    "tile_bit_xor_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)
add_builtin(
    "tile_bit_xor_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "i": int, "j": int, "k": int, "l": int, "value": Any},
    value_func=tile_inplace_bitwise_value_func,
    group="Tile Primitives",
    hidden=True,
    export=False,
    is_differentiable=False,
)


def tile_transpose_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, int])

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
    input_types={"a": tile(dtype=Any, shape=tuple[int, int])},
    value_func=tile_transpose_value_func,
    variadic=True,
    doc="""Transpose a tile.

    For shared memory tiles, this operation will alias the input tile.
    Register tiles will first be transferred to shared memory before transposition.

    Args:
        a: Tile to transpose with ``shape=(M,N)``

    Returns:
        Tile with ``shape=(N,M)``.""",
    group="Tile Primitives",
    export=False,
)


def tile_broadcast_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

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
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "shape": tuple[int, ...]},
    value_func=tile_broadcast_value_func,
    dispatch_func=tile_broadcast_dispatch_func,
    variadic=False,
    doc="""Broadcast a tile.

    Broadcasts the input tile ``a`` to the destination shape.
    Broadcasting follows NumPy broadcast rules.

    Args:
        a: Tile to broadcast
        shape: The shape to broadcast to

    Returns:
        Tile with broadcast shape.""",
    group="Tile Primitives",
    export=False,
)


def tile_sum_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=(1,))

    if len(arg_types) != 1:
        raise TypeError(f"tile_sum() takes exactly 1 positional argument but {len(arg_types)} were given")

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_sum() argument must be a tile, got {a!r}")

    return tile(dtype=a.dtype, shape=(1,))


add_builtin(
    "tile_sum",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_sum_value_func,
    variadic=True,
    doc="""Cooperatively compute the sum of the tile elements.

    Reduce across all elements using all threads in the block.

    Args:
        a: The tile to compute the sum of

    Returns:
        A single-element tile holding the sum.

    Example:

        .. code-block:: python

            @wp.kernel
            def compute():

                t = wp.tile_ones(dtype=float, shape=(16, 16))
                s = wp.tile_sum(t)

                print(s)

            wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

        .. code-block:: text

            [256] = tile(shape=(1), storage=register)
    """,
    group="Tile Primitives",
    export=False,
)


def tile_dot_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Any, shape=(1,))

    a = arg_types["a"]
    b = arg_types["b"]

    if not is_tile(a):
        raise TypeError(f"tile_dot() first argument must be a tile, got {a!r}")
    if not is_tile(b):
        raise TypeError(f"tile_dot() second argument must be a tile, got {b!r}")
    if a.shape != b.shape:
        raise TypeError(f"tile_dot() arguments must have the same shape, got {a.shape} and {b.shape}")
    if not types_equal(a.dtype, b.dtype):
        raise TypeError(f"tile_dot() arguments must have the same dtype, got {a.dtype} and {b.dtype}")
    if type_is_struct(a.dtype):
        raise TypeError("tile_dot() does not support Warp struct tile elements")

    return tile(dtype=type_scalar_type(a.dtype), shape=(1,))


add_builtin(
    "tile_dot",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_dot_value_func,
    doc="""Compute the dot product of two tiles.

    Computes a full contraction between corresponding elements and sums
    the results. For scalar tiles this is the standard dot product; for
    vector tiles each pair is contracted via ``wp.dot``; for matrix tiles
    it is the Frobenius inner product (the sum of element-wise products
    over all axes).

    Equivalent in Python to ``wp.tile_sum(a * b)`` for scalar tiles,
    ``wp.tile_sum(wp.tile_map(wp.dot, a, b))`` for vector tiles, and
    ``wp.tile_sum(wp.tile_map(wp.ddot, a, b))`` for matrix tiles, but
    without the intermediate tile and shared-memory round trip the
    explicit forms would require.

    Args:
        a: First tile operand.
        b: Second tile operand (must have same shape and dtype as ``a``).

    Returns:
        A single-element tile holding the dot-product result. Index the
        tile at ``[0]`` to obtain the scalar value.

    Example:

        .. code-block:: python

            @wp.kernel
            def compute():

                a = wp.tile_ones(dtype=float, shape=64)
                b = wp.tile_ones(dtype=float, shape=64) * 2.0
                d = wp.tile_dot(a, b)

                print(d)

            wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

        .. code-block:: text

            [128] = tile(shape=(1), storage=register)""",
    group="Tile Primitives",
    export=False,
)


def tile_axpy_value_func(arg_types, arg_values):
    if arg_types is None:
        return None

    alpha = arg_types["alpha"]
    src = arg_types["src"]
    dest = arg_types["dest"]

    if not is_tile(src):
        raise TypeError(f"tile_axpy() 'src' argument must be a tile, got {src!r}")
    if not is_tile(dest):
        raise TypeError(f"tile_axpy() 'dest' argument must be a tile, got {dest!r}")
    if dest.shape != src.shape:
        raise TypeError(f"tile_axpy() 'dest' and 'src' must have the same shape, got {dest.shape} and {src.shape}")
    if not types_equal(dest.dtype, src.dtype):
        raise TypeError(f"tile_axpy() 'dest' and 'src' must have the same dtype, got {dest.dtype} and {src.dtype}")
    if type_is_struct(dest.dtype):
        raise TypeError("tile_axpy() does not support Warp struct tile elements")
    if is_tile(alpha):
        raise TypeError(f"tile_axpy() 'alpha' must be a scalar, got tile {alpha!r}")
    if not type_is_scalar(alpha):
        raise TypeError(f"tile_axpy() 'alpha' must be a scalar type, got {alpha}")
    tile_scalar = type_scalar_type(dest.dtype)
    if not types_equal(tile_scalar, alpha):
        raise TypeError(
            f"tile_axpy() 'alpha' must match the tile's scalar type, got {alpha} for tile scalar type {tile_scalar}"
        )

    return None


add_builtin(
    "tile_axpy",
    input_types={
        "alpha": Any,
        "src": tile(dtype=Any, shape=tuple[int, ...]),
        "dest": tile(dtype=Any, shape=tuple[int, ...]),
    },
    value_func=tile_axpy_value_func,
    doc="""Scale ``src`` by ``alpha`` and accumulate into ``dest``.

    Performs a fused multiply-add directly into the destination tile without
    creating an intermediate scaled tile.

    Args:
        alpha: Scalar multiplier (must match the tile's underlying scalar type).
        src: Source tile (must have same shape and dtype as ``dest``).
        dest: Destination tile, modified in place.

    Example:

        .. code-block:: python

            @wp.kernel
            def compute():

                dest = wp.tile_ones(dtype=float, shape=4) * 2.0
                src = wp.tile_ones(dtype=float, shape=4) * 3.0
                wp.tile_axpy(5.0, src, dest)

                print(dest)

            wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

        .. code-block:: text

            [17 17 17 17] = tile(shape=(4), storage=register)""",
    group="Tile Primitives",
    export=False,
)


def tile_sum_axis_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_sum() 'a' argument must be a tile, got {a!r}")

    # force input tile to shared
    a.storage = "shared"

    axis = arg_values["axis"]
    shape = a.shape

    if axis < 0 or axis >= len(shape):
        raise ValueError(f"tile_sum() axis {axis} is out of bounds for tile with {len(shape)} dimensions")

    # shape is identical less the axis reduction is along
    if len(shape) > 1:
        new_shape = shape[:axis] + shape[axis + 1 :]
    else:
        new_shape = (1,)

    return tile(dtype=a.dtype, shape=new_shape)


def tile_sum_axis_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    tile = arg_values["a"]
    axis_var = arg_values["axis"]
    if not hasattr(axis_var, "constant") or axis_var.constant is None:
        raise ValueError("tile_sum() axis must be a compile-time constant")
    axis = axis_var.constant

    return ((tile,), (axis,))


add_builtin(
    "tile_sum",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "axis": int},
    value_func=tile_sum_axis_value_func,
    dispatch_func=tile_sum_axis_dispatch_func,
    doc="""Cooperatively compute the sum of the tile elements.

    Reduce across a tile axis using all threads in the block.

    Args:
        a: The input tile. Must reside in shared memory.
        axis: The tile axis to compute the sum across. Must be a compile-time constant.

    Returns:
        A tile with the same shape as the input tile less the axis dimension and the same data type as the input tile.

    Example:

        .. code-block:: python

            @wp.kernel
            def compute():

                t = wp.tile_ones(dtype=float, shape=(8, 8))
                s = wp.tile_sum(t, axis=0)

                print(s)

            wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

        .. code-block:: text

            [8 8 8 8 8 8 8 8] = tile(shape=(8), storage=register)
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

    if not (
        a.dtype is warp.float32
        or a.dtype is warp.int32
        or a.dtype is warp.uint32
        or a.dtype is warp.int64
        or a.dtype is warp.uint64
    ):
        raise TypeError(
            f"First tile_sort() argument must be a tile of type float32, int32, uint32, int64, or uint64, got {a.dtype}"
        )

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
    input_types={"keys": tile(dtype=Any, shape=tuple[int]), "values": tile(dtype=Any, shape=tuple[int])},
    value_func=tile_sort_value_func,
    variadic=True,
    doc="""Cooperatively sort the elements of two tiles in ascending order based on the keys, using all threads in the block.

    Args:
        keys: Keys to sort by. Supported key types: :class:`warp.float32`, :class:`warp.int32`, :class:`warp.uint32`, :class:`warp.int64`, :class:`warp.uint64`. Must be in shared memory.
        values: Values to sort along with keys. No type restrictions. Must be in shared memory.

    Returns:
        No return value. Sorts both tiles in-place.

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

        .. code-block:: text

            [1, 2, ..., 32] = tile(shape=(32), storage=shared)
            [31, 30, 29, ..., 0] = tile(shape=(32), storage=shared)
    """,
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
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

    # tile_min() is variadic, which bypasses the Scalar dtype constraint enforced by
    # overload matching, so reject struct tiles explicitly: there is no canonical
    # ordering for a struct and the native reduction would fail to compile.
    if type_is_struct(a.dtype):
        raise TypeError("tile_min() does not support Warp struct tile elements")

    return tile(dtype=a.dtype, shape=(1,))


add_builtin(
    "tile_min",
    input_types={"a": tile(dtype=Scalar, shape=tuple[int, ...])},
    value_func=tile_min_value_func,
    variadic=True,
    doc="""Cooperatively compute the minimum of the tile elements using all threads in the block.

    Args:
        a: The tile to compute the minimum of

    Returns:
        A single-element tile holding the minimum value.

    Example:

        .. code-block:: python

            @wp.kernel
            def compute():

                t = wp.tile_arange(64, 128)
                s = wp.tile_min(t)

                print(s)


            wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

        .. code-block:: text

            [64] = tile(shape=(1), storage=register)
    """,
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
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

    # tile_argmin() is variadic, which bypasses the Scalar dtype constraint enforced by
    # overload matching, so reject struct tiles explicitly: there is no canonical
    # ordering for a struct and the native reduction would fail to compile.
    if type_is_struct(a.dtype):
        raise TypeError("tile_argmin() does not support Warp struct tile elements")

    return tile(dtype=warp.int32, shape=(1,))


add_builtin(
    "tile_argmin",
    input_types={"a": tile(dtype=Scalar, shape=tuple[int, ...])},
    value_func=tile_argmin_value_func,
    variadic=True,
    doc="""Cooperatively compute the index of the minimum element in the tile using all threads in the block.

    Args:
        a: The tile to compute the argmin from

    Returns:
        A single-element tile holding the index of the minimum value.

    Example:

        .. code-block:: python

            @wp.kernel
            def compute():

                t = wp.tile_arange(64, 128)
                s = wp.tile_argmin(t)

                print(s)


            wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

        .. code-block:: text

            [0] = tile(shape=(1), storage=register)
    """,
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
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
    input_types={"a": tile(dtype=Scalar, shape=tuple[int, ...])},
    value_func=tile_max_value_func,
    variadic=False,
    doc="""Cooperatively compute the maximum of the tile elements using all threads in the block.

    Args:
        a: The tile to compute the maximum from

    Returns:
        A single-element tile holding the maximum value.

    Example:

        .. code-block:: python

            @wp.kernel
            def compute():

                t = wp.tile_arange(64, 128)
                s = wp.tile_max(t)

                print(s)

            wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

        .. code-block:: text

            [127] = tile(shape=(1), storage=register)
    """,
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
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
    input_types={"a": tile(dtype=Scalar, shape=tuple[int, ...])},
    value_func=tile_argmax_value_func,
    variadic=False,
    doc="""Cooperatively compute the index of the maximum element in the tile using all threads in the block.

    Args:
        a: The tile to compute the argmax from

    Returns:
        A single-element tile holding the index of the maximum value.

    Example:

        .. code-block:: python

            @wp.kernel
            def compute():

                t = wp.tile_arange(64, 128)
                s = wp.tile_argmax(t)

                print(s)

            wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=64)

        .. code-block:: text

            [63] = tile(shape=(1), storage=register)
    """,
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)


def tile_reduce_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Any, shape=(1,))

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_reduce() 'a' argument must be a tile, got {a!r}")

    # Struct element types require the reduction operator to have an overload that both
    # accepts and returns the struct; otherwise codegen emits an unsupported native call
    # (e.g. wp::min on a struct) that fails to compile, or silently reinterprets a
    # mismatched return type as the struct. Resolve and validate the overload up front so
    # either problem surfaces as a clean Python error. Scalar dtypes keep their existing
    # behavior (their builtin operator overloads always resolve in codegen).
    if type_is_struct(a.dtype) and arg_values is not None and "op" in arg_values:
        op = arg_values["op"]
        overload = _get_tile_map_overload(
            op,
            [a.dtype, a.dtype],
            f"tile_reduce() operator {op} has no overload accepting Warp struct tile element type {type_repr(a.dtype)}",
        )

        if overload.value_func is None:
            overload.build(None)

        param_names = iter(overload.input_types)
        param_a, param_b = next(param_names), next(param_names)
        return_type = overload.value_func({param_a: a.dtype, param_b: a.dtype}, None)

        if not types_equal(return_type, a.dtype):
            raise TypeError(
                f"tile_reduce() operator {op} must return the tile element type {type_repr(a.dtype)}, "
                f"got {type_repr(return_type)}"
            )

    return tile(dtype=a.dtype, shape=(1,))


def tile_reduce_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (args["op"], *args["args"])
    template_args = ()
    return (func_args, template_args)


add_builtin(
    "tile_reduce",
    input_types={"op": Callable, "a": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_reduce_value_func,
    native_func="tile_reduce",
    doc="""Apply a custom reduction operator across a tile.

    Reduce across all elements using the provided operator.

    Args:
        op: A callable function that accepts two arguments and returns one argument, may be a user function or builtin
        a: The input tile, the operator (or one of its overloads) must be able to accept the tile's data type

    Returns:
        A single-element tile with the same data type as the input tile.

    Example:

        .. code-block:: python

            @wp.kernel
            def factorial():

                t = wp.tile_arange(1, 10, dtype=int)
                s = wp.tile_reduce(wp.mul, t)

                print(s)

            wp.launch_tiled(factorial, dim=[1], inputs=[], block_dim=16)

        .. code-block:: text

            [362880] = tile(shape=(1), storage=register)
    """,
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)


def tile_reduce_axis_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Scalar, shape=tuple[int, ...])

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_reduce() 'a' argument must be a tile, got {a!r}")

    # force input tile to shared memory
    a.storage = "shared"

    axis = arg_values["axis"]
    shape = a.shape

    if axis < 0 or axis >= len(shape):
        raise ValueError(f"tile_reduce() axis {axis} is out of bounds for tile with {len(shape)} dimensions")

    # shape is identical less the axis reduction is along
    if len(shape) > 1:
        new_shape = shape[:axis] + shape[axis + 1 :]
    else:
        new_shape = (1,)

    return tile(dtype=a.dtype, shape=new_shape)


add_builtin(
    "tile_reduce",
    input_types={"op": Callable, "a": tile(dtype=Scalar, shape=tuple[int, ...]), "axis": int},
    value_func=tile_reduce_axis_value_func,
    native_func="tile_reduce_axis",
    doc="""Apply a custom reduction operator across a tile.

    Reduce across a tile axis using the provided operator.

    Args:
        op: A callable function that accepts two arguments and returns one argument, may be a user function or builtin
        a: The input tile, the operator (or one of its overloads) must be able to accept the tile's data type. Must reside in shared memory.
        axis: The tile axis to perform the reduction across. Must be a compile-time constant.

    Returns:
        A tile with the same shape as the input tile less the axis dimension and the same data type as the input tile.

    Example:

        .. code-block:: python

            TILE_M = wp.constant(4)
            TILE_N = wp.constant(2)

            @wp.kernel
            def compute(x: wp.array2d[float], y: wp.array[float]):

                a = wp.tile_load(x, shape=(TILE_M, TILE_N))
                b = wp.tile_reduce(wp.add, a, axis=1)
                wp.tile_store(y, b)

            arr = np.arange(TILE_M * TILE_N).reshape(TILE_M, TILE_N)

            x = wp.array(arr, dtype=float)
            y = wp.zeros(TILE_M, dtype=float)

            wp.launch_tiled(compute, dim=[1], inputs=[x], outputs=[y], block_dim=32)

            print(x.numpy())
            print(y.numpy())

        .. code-block:: text

            [[0. 1.]
             [2. 3.]
             [4. 5.]
             [6. 7.]]
            [ 1.  5.  9. 13.]
    """,
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)


def tile_scan_inclusive_value_func(arg_types, arg_values):
    # Return type is the same as input type
    if arg_types is None:
        return tile(dtype=Scalar, shape=tuple[int, ...])

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
    input_types={"a": tile(dtype=Scalar, shape=tuple[int, ...])},
    value_func=tile_scan_inclusive_value_func,
    native_func="tile_scan_inclusive",
    doc="""Inclusive scan (prefix sum) across the tile.

    This function cooperatively performs an inclusive scan (cumulative sum) across the tile.

    Args:
        a: The input tile. Must be a tile of type float32, int32, or uint32.

    Returns:
        A new tile containing the inclusive scan result.

    Example:

        .. code-block:: python

            @wp.kernel
            def scan_example():
                t = wp.tile_arange(1, 5, dtype=int)
                s = wp.tile_scan_inclusive(t)
                print(s)

            wp.launch_tiled(scan_example, dim=[1], inputs=[], block_dim=16)

        .. code-block:: text

            [1, 3, 6, 10] = tile(shape=(4), storage=register)
    """,
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)


def tile_scan_exclusive_value_func(arg_types, arg_values):
    # return generic type (for doc builds)
    if arg_types is None:
        return tile(dtype=Scalar, shape=tuple[int, ...])

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
    input_types={"a": tile(dtype=Scalar, shape=tuple[int, ...])},
    value_func=tile_scan_exclusive_value_func,
    native_func="tile_scan_exclusive",
    doc="""Exclusive scan (prefix sum) across the tile.

    This function cooperatively performs an exclusive scan (cumulative sum) across the tile.

    Args:
        a: The input tile. Must be a tile of type float32, int32, or uint32.

    Returns:
        A new tile containing the exclusive scan result.

    Example:

        .. code-block:: python

            @wp.kernel
            def scan_example():
                t = wp.tile_arange(1, 5, dtype=int)
                s = wp.tile_scan_exclusive(t)
                print(s)

            wp.launch_tiled(scan_example, dim=[1], inputs=[], block_dim=16)

        .. code-block:: text

            [0, 1, 3, 6] = tile(shape=(4), storage=register)
    """,
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)


def tile_scan_max_inclusive_value_func(arg_types, arg_values):
    # Return type is the same as input type
    if arg_types is None:
        return tile(dtype=Scalar, shape=tuple[int, ...])

    if len(arg_types) != 1:
        raise TypeError(
            f"tile_scan_max_inclusive() takes exactly 1 positional argument but {len(arg_types)} were given"
        )

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_scan_max_inclusive() argument must be a tile, got {a!r}")

    # Only allow float32, int32, or uint32 for scan
    if not (a.dtype is warp.float32 or a.dtype is warp.int32 or a.dtype is warp.uint32):
        raise TypeError(
            f"tile_scan_max_inclusive() argument must be a tile of type float32, int32, or uint32, got {a.dtype}"
        )

    return tile(dtype=a.dtype, shape=a.shape)


def tile_scan_max_inclusive_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (args["a"],)
    template_args = ()
    return (func_args, template_args)


add_builtin(
    "tile_scan_max_inclusive",
    input_types={"a": tile(dtype=Scalar, shape=tuple[int, ...])},
    value_func=tile_scan_max_inclusive_value_func,
    native_func="tile_scan_max_inclusive",
    doc="""Inclusive max scan across the tile.

    This function cooperatively performs an inclusive max scan (cumulative maximum) across the tile.

    Args:
        a: The input tile. Must be a tile of type float32, int32, or uint32.

    Returns:
        A new tile containing the inclusive max scan result.

    Example:

        .. code-block:: python

            @wp.kernel
            def scan_example(input: wp.array[int]):
                t = wp.tile_load(input, shape=(4,))
                s = wp.tile_scan_max_inclusive(t)
                print(s)

            input = wp.array([3, 1, 4, 2], dtype=int)
            wp.launch_tiled(scan_example, dim=[1], inputs=[input], block_dim=16)

        .. code-block:: text

            [3, 3, 4, 4] = tile(shape=(4), storage=register)
    """,
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)


def tile_scan_min_inclusive_value_func(arg_types, arg_values):
    # Return type is the same as input type
    if arg_types is None:
        return tile(dtype=Scalar, shape=tuple[int, ...])

    if len(arg_types) != 1:
        raise TypeError(
            f"tile_scan_min_inclusive() takes exactly 1 positional argument but {len(arg_types)} were given"
        )

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_scan_min_inclusive() argument must be a tile, got {a!r}")

    # Only allow float32, int32, or uint32 for scan
    if not (a.dtype is warp.float32 or a.dtype is warp.int32 or a.dtype is warp.uint32):
        raise TypeError(
            f"tile_scan_min_inclusive() argument must be a tile of type float32, int32, or uint32, got {a.dtype}"
        )

    return tile(dtype=a.dtype, shape=a.shape)


def tile_scan_min_inclusive_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (args["a"],)
    template_args = ()
    return (func_args, template_args)


add_builtin(
    "tile_scan_min_inclusive",
    input_types={"a": tile(dtype=Scalar, shape=tuple[int, ...])},
    value_func=tile_scan_min_inclusive_value_func,
    native_func="tile_scan_min_inclusive",
    doc="""Inclusive min scan across the tile.

    This function cooperatively performs an inclusive min scan (cumulative minimum) across the tile.

    Args:
        a: The input tile. Must be a tile of type float32, int32, or uint32.

    Returns:
        A new tile containing the inclusive min scan result.

    Example:

        .. code-block:: python

            @wp.kernel
            def scan_example(input: wp.array[int]):
                t = wp.tile_load(input, shape=(4,))
                s = wp.tile_scan_min_inclusive(t)
                print(s)

            input = wp.array([3, 1, 4, 2], dtype=int)
            wp.launch_tiled(scan_example, dim=[1], inputs=[input], block_dim=16)

        .. code-block:: text

            [3, 1, 1, 1] = tile(shape=(4), storage=register)
    """,
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)


# maps


def _is_tile_map_element_type_supported(t):
    return type_is_scalar(t) or type_is_vector(t) or type_is_matrix(t) or type_is_struct(t)


def _get_tile_map_overload(op, dtypes, message):
    try:
        overload = op.get_overload(dtypes, {})
    except KeyError as exc:
        raise RuntimeError(message) from exc

    if overload is None:
        raise RuntimeError(message)

    return overload


# does type propagation for load()
def tile_unary_map_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    a = arg_types["a"]

    if not is_tile(a):
        raise TypeError(f"tile_map() 'a' argument must be a tile, got {a!r}")

    if "op" in arg_values:
        op = arg_values["op"]
        overload = _get_tile_map_overload(
            op, [a.dtype], f"No overload of {op} found for tile element type {type_repr(a.dtype)}"
        )

        # build the right overload on demand
        if overload.value_func is None:
            overload.build(None)

        param_name = next(iter(overload.input_types))
        value_type = overload.value_func({param_name: a.dtype}, None)

        if not _is_tile_map_element_type_supported(value_type):
            raise TypeError(f"Operator {op} returns unsupported type {type_repr(value_type)} for a tile element")

        return tile(dtype=value_type, shape=a.shape)

    else:
        return tile(dtype=a.dtype, shape=a.shape)


def tile_unary_map_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    op = arg_values["op"]
    tile_a = arg_values["a"]

    overload = op.get_overload([tile_a.type.dtype], {})

    return ((overload, tile_a), ())


add_builtin(
    "tile_map",
    input_types={"op": Callable, "a": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_unary_map_value_func,
    dispatch_func=tile_unary_map_dispatch_func,
    # variadic=True,
    native_func="tile_unary_map",
    doc="""Apply a function to tile elements.

    Apply a unary function to each element using all threads in the block.

    Args:
        op: A callable function that accepts one argument and returns one argument, may be a user function or builtin
        a: The input tile, the operator (or one of its overloads) must be able to accept the tile's data type

    Returns:
        A tile with the same dimensions as the input tile. Its datatype is specified by the return type of ``op``.

    Example:

        .. code-block:: python

            @wp.kernel
            def compute():

                t = wp.tile_arange(0.0, 1.0, 0.1, dtype=float)
                s = wp.tile_map(wp.sin, t)

                print(s)

            wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=16)

        .. code-block:: text

            [0 0.0998334 0.198669 0.29552 0.389418 0.479426 0.564642 0.644218 0.717356 0.783327] = tile(shape=(10), storage=register)
    """,
    group="Tile Primitives",
    export=False,
)


def tile_binary_map_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    a = arg_types["a"]
    b = arg_types["b"]

    # 'a' must be a tile
    if not is_tile(a):
        raise TypeError(f"tile_map() 'a' argument must be a tile, got {a!r}")

    # 'b' can be a tile or a non-tile constant (scalar/vec/mat)
    b_is_tile = is_tile(b)

    if b_is_tile:
        # If both are tiles, shapes must match
        if len(a.shape) != len(b.shape):
            raise ValueError(
                f"tile_map() shapes must have the same number of dimensions, got {len(a.shape)} and {len(b.shape)}"
            )

        for i in range(len(a.shape)):
            if a.shape[i] != b.shape[i]:
                raise ValueError(f"tile_map() shapes do not match on dimension {i}, got {a.shape} and {b.shape}")

        b_dtype = b.dtype
    else:
        # b is a non-tile constant, validate it's a supported type
        if not _is_tile_map_element_type_supported(b):
            raise TypeError(
                f"tile_map() 'b' argument must be a tile, scalar, vector, matrix, or Warp struct, got {b!r}"
            )
        b_dtype = b

    if "op" in arg_values:
        op = arg_values["op"]
        overload = _get_tile_map_overload(
            op,
            [a.dtype, b_dtype],
            f"No overload of {op} found for tile element types {type_repr(a.dtype)}, {type_repr(b_dtype)}",
        )

        # build the right overload on demand
        if overload.value_func is None:
            overload.build(None)

        param_names = iter(overload.input_types)
        param_a_name, param_b_name = next(param_names), next(param_names)
        value_type = overload.value_func({param_a_name: a.dtype, param_b_name: b_dtype}, None)

        if not _is_tile_map_element_type_supported(value_type):
            raise TypeError(f"Operator {op} returns unsupported type {type_repr(value_type)} for a tile element")

        return tile(dtype=value_type, shape=a.shape)

    else:
        # ensure types equal
        if not types_equal(a.dtype, b_dtype):
            raise TypeError(
                f"tile_map() arguments must have the same dtype for this operation, got {a.dtype} and {b_dtype}"
            )

        return tile(dtype=a.dtype, shape=a.shape)


def tile_binary_map_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    op = arg_values["op"]
    tile_a = arg_values["a"]
    arg_b = arg_values["b"]

    # Get dtype for b (either from tile or directly if it's a non-tile constant)
    b_type = arg_types["b"]
    b_dtype = b_type.dtype if is_tile(b_type) else b_type

    overload = op.get_overload([tile_a.type.dtype, b_dtype], {})

    return ((overload, tile_a, arg_b), ())


add_builtin(
    "tile_map",
    input_types={
        "op": Callable,
        "a": tile(dtype=Any, shape=tuple[int, ...]),
        "b": Any,
    },
    value_func=tile_binary_map_value_func,
    dispatch_func=tile_binary_map_dispatch_func,
    # variadic=True,
    native_func="tile_binary_map",
    doc="""Apply a function to tile elements.

    This function cooperatively applies a binary function to each element of the tile using all threads in the block.
    The second argument can be a tile (must have same dimensions as ``a``), or a non-tile constant (scalar, vector,
    matrix, or Warp struct) which will be broadcast across all elements.

    Args:
        op: A callable function that accepts two arguments and returns one argument, all of the same type, may be a user function or builtin.
        a: The first input tile, the operator (or one of its overloads) must be able to accept the tile's dtype.
        b: Either a tile with matching dimensions, or a scalar/vector/matrix/Warp struct constant.

    Returns:
        A tile with the same dimensions as tile ``a``. Its datatype is specified by the return type of ``op``.

    Example:

        .. code-block:: python

            @wp.kernel
            def compute():

                a = wp.tile_arange(0.0, 1.0, 0.1, dtype=float)
                b = wp.tile_ones(shape=10, dtype=float)

                s = wp.tile_map(wp.add, a, b)

                print(s)

            wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=16)

        .. code-block:: text

            [1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9] = tile(shape=(10), storage=register)
    """,
    group="Tile Primitives",
    export=False,
)


def tile_n_map_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    # 'a' is the first tile (required)
    a = arg_types["a"]
    if not is_tile(a):
        raise TypeError(f"tile_map() 'a' argument must be a tile, got {a!r}")

    # Get the variadic arguments from *args
    args = arg_types.get("args", ())

    # Build list of all argument types (a + args)
    all_args = [a, *list(args)]

    # Check that we have at least 3 total arguments for this variadic overload
    if len(all_args) < 3:
        raise ValueError(f"tile_map() with variadic args requires at least 3 arguments, got {len(all_args)}")

    # Validate each arg: if it's a tile, check shape matches; if not, must be scalar/vec/mat
    dtypes = [a.dtype]
    for i, arg in enumerate(args):
        if is_tile(arg):
            # Check shape matches first tile
            if len(arg.shape) != len(a.shape):
                raise ValueError(
                    f"tile_map() shapes must have the same number of dimensions, got {len(a.shape)} and {len(arg.shape)} for arguments 0 and {i + 1}"
                )
            for dim_idx in range(len(a.shape)):
                if arg.shape[dim_idx] != a.shape[dim_idx]:
                    raise ValueError(
                        f"tile_map() shapes do not match on dimension {dim_idx}, got {a.shape} and {arg.shape} for arguments 0 and {i + 1}"
                    )
            dtypes.append(arg.dtype)
        else:
            # Non-tile constant: validate it's a supported type
            if not _is_tile_map_element_type_supported(arg):
                raise TypeError(
                    f"tile_map() argument {i + 1} must be a tile, scalar, vector, matrix, or Warp struct, got {arg!r}"
                )
            dtypes.append(arg)

    if "op" not in arg_values:
        raise ValueError("tile_map() with variadic args requires an 'op' argument")

    op = arg_values["op"]

    dtype_strs = ", ".join(type_repr(dt) for dt in dtypes)
    overload = _get_tile_map_overload(op, dtypes, f"No overload of {op} found for tile element types {dtype_strs}")

    # build the right overload on demand
    if overload.value_func is None:
        overload.build(None)

    assert len(dtypes) == len(overload.input_types), (
        f"Overload parameter count mismatch: expected {len(dtypes)}, got {len(overload.input_types)}"
    )
    arg_type_map = dict(zip(overload.input_types, dtypes, strict=True))
    value_type = overload.value_func(arg_type_map, None)

    if not _is_tile_map_element_type_supported(value_type):
        raise TypeError(f"Operator {op} returns unsupported type {type_repr(value_type)} for a tile element")

    return tile(dtype=value_type, shape=a.shape)


def tile_n_map_dispatch_func(arg_types: Mapping[str, type], return_type: Any, arg_values: Mapping[str, Var]):
    op = arg_values["op"]
    tile_a = arg_values["a"]
    args = arg_values.get("args", ())

    # Get dtypes from the Vars themselves
    dtypes = [tile_a.type.dtype]
    for arg in args:
        arg_type = arg.type
        if is_tile(arg_type):
            dtypes.append(arg_type.dtype)
        else:
            dtypes.append(arg_type)

    overload = op.get_overload(dtypes, {})

    return ((overload, tile_a, *args), ())


add_builtin(
    "tile_map",
    input_types={"op": Callable, "a": tile(dtype=Any, shape=tuple[int, ...]), "*args": Any},
    value_func=tile_n_map_value_func,
    dispatch_func=tile_n_map_dispatch_func,
    variadic=True,
    native_func="tile_map",
    doc="""Apply a function to tile elements.

    This function cooperatively applies a user-defined function to corresponding elements using all threads in the block.
    The first argument 'a' must be a tile (determines output shape). Additional arguments can be tiles (must have same
    dimensions) or non-tile constants (scalar, vector, matrix, or Warp struct) which will be broadcast across all elements.

    Args:
        op: A callable function that accepts N arguments and returns one value, must be a user function.
        a: The first input tile, determines the output shape.
        args: Additional arguments: tiles with matching dimensions, or scalar/vector/matrix/Warp struct constants.

    Returns:
        A tile with the same dimensions as tile ``a``. Its datatype is specified by the return type of ``op``.

    Example:

        .. code-block:: python

            @wp.func
            def weighted_sum(a: float, b: float, c: float):
                return 0.5 * a + 0.3 * b + 0.2 * c

            @wp.kernel
            def compute():

                a = wp.tile_arange(0.0, 1.0, 0.1, dtype=float)
                b = wp.tile_ones(shape=10, dtype=float)
                c = wp.tile_arange(1.0, 2.0, 0.1, dtype=float)

                s = wp.tile_map(weighted_sum, a, b, c)

                print(s)

            wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=16)

        .. code-block:: text

            [0.5 0.57 0.64 0.71 0.78 0.85 0.92 0.99 1.06 1.13] = tile(shape=(10), storage=register)
    """,
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
    is_differentiable=False,
)


add_builtin(
    "dense_chol",
    input_types={"n": int, "A": array(dtype=float), "regularization": float, "L": array(dtype=float)},
    value_type=None,
    doc="WIP",
    group="Utility",
    hidden=True,
    is_differentiable=False,
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
    is_differentiable=False,
)

add_builtin(
    "dense_subs",
    input_types={"n": int, "L": array(dtype=float), "b": array(dtype=float), "x": array(dtype=float)},
    value_type=None,
    doc="WIP",
    group="Utility",
    hidden=True,
    is_differentiable=False,
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
    is_differentiable=False,
)


# ---------------------------------
# Geometry

add_builtin(
    "bvh_query_aabb",
    input_types={"id": uint64, "low": vec3, "high": vec3, "root": int},
    defaults={"root": -1},
    value_type=BvhQuery,
    group="Geometry",
    doc="""Construct an axis-aligned bounding box (AABB) query against a BVH.

    Returns a query that iterates over every item in the BVH whose stored bounding box overlaps
    the query box ``[low, high]``. Advance the query and read each result with :func:`bvh_query_next`.
    ``low`` and ``high`` are given in BVH space, i.e. the same coordinate space
    as the ``lowers``/``uppers`` arrays passed to :class:`warp.Bvh`.

    To restrict traversal to a subtree, set ``root`` to that node's index (for a grouped BVH the
    group root is obtained from :func:`bvh_get_group_root`). If ``root`` is -1 (default),
    traversal starts at the BVH's global root.

    Args:
        id: The BVH identifier
        low: The lower bound of the query box, in BVH space
        high: The upper bound of the query box, in BVH space
        root: The node to begin the query from, or -1 (default) for the BVH's global root

    Returns:
        A :class:`warp.BvhQuery`. It is opaque; pass it to :func:`bvh_query_next`,
        which writes the index of each overlapping item (an index into the arrays passed to :class:`warp.Bvh`)
        to its ``index`` argument.

    Example:

        .. testcode::

            @wp.kernel
            def query_region(bvh_id: wp.uint64, lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3],
                             lo: wp.vec3, hi: wp.vec3, centers: wp.array[wp.vec3]):
                query = wp.bvh_query_aabb(bvh_id, lo, hi)
                item = int(0)
                while wp.bvh_query_next(query, item):
                    centers[item] = 0.5 * (lowers[item] + uppers[item])

            lowers = wp.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=wp.vec3)
            uppers = wp.array([[1, 1, 1], [3, 1, 1], [5, 1, 1]], dtype=wp.vec3)
            bvh = wp.Bvh(lowers=lowers, uppers=uppers)

            centers = wp.zeros(3, dtype=wp.vec3)  # center of each object whose box overlaps the region
            wp.launch(query_region, dim=1, inputs=[bvh.id, lowers, uppers, wp.vec3(0.5, 0.5, 0.5), wp.vec3(2.5, 0.5, 0.5)], outputs=[centers])
            print(centers.numpy().tolist())

        .. testoutput::

            [[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [0.0, 0.0, 0.0]]""",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "bvh_query_ray",
    input_types={"id": uint64, "start": vec3, "dir": vec3, "root": int},
    defaults={"root": -1},
    value_type=BvhQuery,
    group="Geometry",
    doc="""Construct a ray query against a BVH.

    Returns a query that iterates over every item in the BVH whose stored bounding box is
    intersected by the ray. Advance the query and read each result with :func:`bvh_query_next`.
    ``start`` and ``dir`` are given in BVH space, i.e. the same coordinate space as
    the ``lowers``/``uppers`` arrays passed to :class:`warp.Bvh`. ``dir`` need not be normalized,
    but the ``max_dist`` cutoff of :func:`bvh_query_next` is measured in multiples of its length,
    so normalize it for ``max_dist`` to be a distance in BVH-space units.

    To restrict traversal to a subtree, set ``root`` to that node's index (for a grouped BVH the
    group root is obtained from :func:`bvh_get_group_root`). If ``root`` is -1 (default),
    traversal starts at the BVH's global root.

    Args:
        id: The BVH identifier
        start: The ray origin, in BVH space
        dir: The ray direction, in BVH space (see above on normalization)
        root: The node to begin the query from, or -1 (default) for the BVH's global root

    Returns:
        A :class:`warp.BvhQuery`. It is opaque; pass it to :func:`bvh_query_next`, which writes
        the index of each intersected item (an index into the arrays passed to :class:`warp.Bvh`)
        to its ``index`` argument.

    Example:

        .. testcode::

            @wp.kernel
            def cast_ray(bvh_id: wp.uint64, lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3],
                         origin: wp.vec3, dir: wp.vec3, centers: wp.array[wp.vec3]):
                query = wp.bvh_query_ray(bvh_id, origin, dir)
                item = int(0)
                while wp.bvh_query_next(query, item):
                    centers[item] = 0.5 * (lowers[item] + uppers[item])

            lowers = wp.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=wp.vec3)
            uppers = wp.array([[1, 1, 1], [3, 1, 1], [5, 1, 1]], dtype=wp.vec3)
            bvh = wp.Bvh(lowers=lowers, uppers=uppers)

            centers = wp.zeros(3, dtype=wp.vec3)
            wp.launch(cast_ray, dim=1, inputs=[bvh.id, lowers, uppers, wp.vec3(-1.0, 0.5, 0.5), wp.vec3(1.0, 0.0, 0.0)], outputs=[centers])
            print(centers.numpy().tolist())

        .. testoutput::

            [[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [4.5, 0.5, 0.5]]""",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "bvh_query_next",
    input_types={"query": BvhQuery, "index": int, "max_dist": float},
    defaults={"max_dist": math.inf},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Advance a BVH query to the next overlapping item and report whether one was found.

    Writes the index of the current item to ``index`` and returns ``True``; returns ``False`` once
    the query is exhausted (``index`` is then left unchanged). The reported index is the item's
    index into the ``lowers``/``uppers`` arrays passed to :class:`warp.Bvh`. Used in a ``while``
    loop together with :func:`bvh_query_aabb` or :func:`bvh_query_ray`.

    For ray queries, ``max_dist`` bounds how far along the ray to look for intersections, measured
    in multiples of the ray direction's length (so it is a distance only if ``dir`` was normalized).
    It has no effect on AABB queries.

    Note that increasing ``max_dist`` may miss intersections: a subtree already rejected for being
    beyond ``max_dist`` is never revisited, even if a later, larger ``max_dist`` would reach it. It
    is therefore only safe to monotonically *reduce* ``max_dist`` during a query.

    Args:
        query: The query to advance, from :func:`bvh_query_aabb` or :func:`bvh_query_ray`
        index: Output; receives the index of the current overlapping item
        max_dist: For ray queries, the maximum distance along the ray to check for intersections
            (in multiples of ``dir``'s length). Has no effect on AABB queries.

    Returns:
        ``True`` if another overlapping item was found (its index written to ``index``), ``False``
        if the query is exhausted.

    Example:

        .. testcode::

            @wp.kernel
            def query_region(bvh_id: wp.uint64, lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3],
                             lo: wp.vec3, hi: wp.vec3, centers: wp.array[wp.vec3]):
                query = wp.bvh_query_aabb(bvh_id, lo, hi)
                item = int(0)
                while wp.bvh_query_next(query, item):
                    centers[item] = 0.5 * (lowers[item] + uppers[item])

            lowers = wp.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=wp.vec3)
            uppers = wp.array([[1, 1, 1], [3, 1, 1], [5, 1, 1]], dtype=wp.vec3)
            bvh = wp.Bvh(lowers=lowers, uppers=uppers)

            centers = wp.zeros(3, dtype=wp.vec3)  # center of each object whose box overlaps the region
            wp.launch(query_region, dim=1, inputs=[bvh.id, lowers, uppers, wp.vec3(0.5, 0.5, 0.5), wp.vec3(2.5, 0.5, 0.5)], outputs=[centers])
            print(centers.numpy().tolist())

        .. testoutput::

            [[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [0.0, 0.0, 0.0]]""",
    export=False,
    is_differentiable=False,
)

# Primary naming convention (grouped with other geometry functions)
add_builtin(
    "bvh_query_aabb_tiled",
    input_types={"id": uint64, "low": vec3, "high": vec3},
    value_type=BvhQueryTiled,
    group="Geometry",
    doc="""Construct an axis-aligned bounding box (AABB) query against a BVH for thread-block parallel traversal.

    For use in tiled kernels: all threads in the block cooperatively traverse the BVH. Advance the
    query with :func:`bvh_query_next_tiled` (one result index per thread per step) in a loop guarded
    by :func:`tile_query_valid`. ``low`` and ``high`` must be identical across all threads in the
    block and are given in BVH space (the space of the arrays passed to :class:`warp.Bvh`).

    Args:
        id: The BVH identifier
        low: The lower bound of the query box, in BVH space (must be the same for all threads in the block)
        high: The upper bound of the query box, in BVH space (must be the same for all threads in the block)

    Returns:
        A :class:`warp.BvhQueryTiled` to advance with :func:`bvh_query_next_tiled`.

    Example:

        .. testcode::

            @wp.kernel
            def tiled_query(bvh_id: wp.uint64, lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3],
                            lo: wp.vec3, hi: wp.vec3, centers: wp.array[wp.vec3]):
                query = wp.bvh_query_aabb_tiled(bvh_id, lo, hi)
                while wp.tile_query_valid(query):
                    result = wp.bvh_query_next_tiled(query)
                    item = wp.untile(result)
                    if item >= 0:
                        centers[item] = 0.5 * (lowers[item] + uppers[item])

            lowers = wp.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=wp.vec3)
            uppers = wp.array([[1, 1, 1], [3, 1, 1], [5, 1, 1]], dtype=wp.vec3)
            bvh = wp.Bvh(lowers=lowers, uppers=uppers)

            centers = wp.zeros(3, dtype=wp.vec3)
            wp.launch_tiled(tiled_query, dim=[1], inputs=[bvh.id, lowers, uppers, wp.vec3(0.5, 0.5, 0.5), wp.vec3(4.5, 0.5, 0.5)], outputs=[centers], block_dim=32)
            print(centers.numpy().tolist())

        .. testoutput::

            [[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [4.5, 0.5, 0.5]]""",
    native_func="tile_bvh_query_aabb",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "bvh_query_ray_tiled",
    input_types={"id": uint64, "start": vec3, "dir": vec3},
    value_type=BvhQueryTiled,
    group="Geometry",
    doc="""Construct a ray query against a BVH for thread-block parallel traversal.

    For use in tiled kernels: all threads in the block cooperatively traverse the BVH. Advance the
    query with :func:`bvh_query_next_tiled` (one result index per thread per step) in a loop guarded
    by :func:`tile_query_valid`. ``start`` and ``dir`` must be identical across all threads in the
    block and are given in BVH space (the space of the arrays passed to :class:`warp.Bvh`).

    Args:
        id: The BVH identifier
        start: The ray origin, in BVH space (must be the same for all threads in the block)
        dir: The ray direction, in BVH space (must be the same for all threads in the block)

    Returns:
        A :class:`warp.BvhQueryTiled` to advance with :func:`bvh_query_next_tiled`.

    Example:

        .. testcode::

            @wp.kernel
            def tiled_cast(bvh_id: wp.uint64, lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3],
                           origin: wp.vec3, dir: wp.vec3, centers: wp.array[wp.vec3]):
                query = wp.bvh_query_ray_tiled(bvh_id, origin, dir)
                while wp.tile_query_valid(query):
                    result = wp.bvh_query_next_tiled(query)
                    item = wp.untile(result)
                    if item >= 0:
                        centers[item] = 0.5 * (lowers[item] + uppers[item])

            lowers = wp.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=wp.vec3)
            uppers = wp.array([[1, 1, 1], [3, 1, 1], [5, 1, 1]], dtype=wp.vec3)
            bvh = wp.Bvh(lowers=lowers, uppers=uppers)

            centers = wp.zeros(3, dtype=wp.vec3)
            wp.launch_tiled(tiled_cast, dim=[1], inputs=[bvh.id, lowers, uppers, wp.vec3(-1.0, 0.5, 0.5), wp.vec3(1.0, 0.0, 0.0)], outputs=[centers], block_dim=32)
            print(centers.numpy().tolist())

        .. testoutput::

            [[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [4.5, 0.5, 0.5]]""",
    native_func="tile_bvh_query_ray",
    export=False,
    is_differentiable=False,
)


def bvh_query_next_tiled_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=int, shape=tuple[int])

    # Return a register tile of ints with shape (block_dim,)
    block_dim = warp._src.codegen.options.get("block_dim", 256)
    return tile(dtype=int, shape=(block_dim,), storage="register")


def bvh_query_next_tiled_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # This function needs to:
    # 1. Create a temporary per-thread int variable
    # 2. Call bvh_query_next_thread_block with query and the temp variable
    # 3. Wrap the temp variable in a tile
    # The actual implementation is handled via native_func
    query = args["query"]
    return ((query,), ())


add_builtin(
    "bvh_query_next_tiled",
    input_types={"query": BvhQueryTiled},
    value_func=bvh_query_next_tiled_value_func,
    dispatch_func=bvh_query_next_tiled_dispatch_func,
    group="Geometry",
    doc="""Move to the next bound in a thread-block parallel BVH query and return results as a tile.

    Each thread in the block receives one result index in the returned tile, or -1 if no result for that thread.
    The function returns a register tile of shape ``(block_dim,)`` containing the result indices,
    where ``block_dim`` is the kernel's block dimension. All threads in the block must call this
    function cooperatively.

    Call it in a loop guarded by :func:`tile_query_valid` (which returns ``False`` once the query
    is exhausted); within an iteration, check whether any tile element is >= 0 to see if this step
    produced any results.

    Args:
        query: The thread-block BVH query object, from :func:`bvh_query_aabb_tiled` or :func:`bvh_query_ray_tiled`

    Returns:
        A register tile of shape ``(block_dim,)`` with dtype int, where each element contains
            the result index for that thread (-1 if no result)

    Example:

        .. testcode::

            @wp.kernel
            def tiled_query(bvh_id: wp.uint64, lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3],
                            lo: wp.vec3, hi: wp.vec3, centers: wp.array[wp.vec3]):
                query = wp.bvh_query_aabb_tiled(bvh_id, lo, hi)
                while wp.tile_query_valid(query):
                    result = wp.bvh_query_next_tiled(query)
                    item = wp.untile(result)
                    if item >= 0:
                        centers[item] = 0.5 * (lowers[item] + uppers[item])

            lowers = wp.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=wp.vec3)
            uppers = wp.array([[1, 1, 1], [3, 1, 1], [5, 1, 1]], dtype=wp.vec3)
            bvh = wp.Bvh(lowers=lowers, uppers=uppers)

            centers = wp.zeros(3, dtype=wp.vec3)
            wp.launch_tiled(tiled_query, dim=[1], inputs=[bvh.id, lowers, uppers, wp.vec3(0.5, 0.5, 0.5), wp.vec3(4.5, 0.5, 0.5)], outputs=[centers], block_dim=32)
            print(centers.numpy().tolist())

        .. testoutput::

            [[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [4.5, 0.5, 0.5]]""",
    native_func="tile_bvh_query_next",
    export=False,
    is_differentiable=False,
)

# Aliases for backward compatibility (tile_* naming convention)
add_builtin(
    "tile_bvh_query_aabb",
    input_types={"id": uint64, "low": vec3, "high": vec3},
    value_type=BvhQueryTiled,
    group="Tile Primitives",
    doc="""Construct an axis-aligned bounding box query against a BVH object for thread-block parallel traversal.

    This query can be used in tiled kernels to cooperatively traverse a BVH across a thread block.

    .. note:: This is an alias for :func:`bvh_query_aabb_tiled`.

    Args:
        id: The BVH identifier
        low: The lower bound of the bounding box in BVH space (must be the same for all threads in the block)
        high: The upper bound of the bounding box in BVH space (must be the same for all threads in the block)""",
    native_func="tile_bvh_query_aabb",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "tile_bvh_query_ray",
    input_types={"id": uint64, "start": vec3, "dir": vec3},
    value_type=BvhQueryTiled,
    group="Tile Primitives",
    doc="""Construct a ray query against a BVH object for thread-block parallel traversal.

    This query can be used in tiled kernels to cooperatively traverse a BVH across a thread block.

    .. note:: This is an alias for :func:`bvh_query_ray_tiled`.

    Args:
        id: The BVH identifier
        start: The ray origin (must be the same for all threads in the block)
        dir: The ray direction (must be the same for all threads in the block)""",
    native_func="tile_bvh_query_ray",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "tile_bvh_query_next",
    input_types={"query": BvhQueryTiled},
    value_func=bvh_query_next_tiled_value_func,
    dispatch_func=bvh_query_next_tiled_dispatch_func,
    group="Tile Primitives",
    doc="""Move to the next bound in a thread-block parallel BVH query and return results as a tile.

    Each thread in the block receives one result index in the returned tile, or -1 if no result for that thread.
    The function returns a register tile of shape ``(block_dim,)`` containing the result indices.

    To check if any results were found, check if any element in the tile is >= 0.

    .. note:: This is an alias for :func:`bvh_query_next_tiled`.

    Args:
        query: The thread-block BVH query object

    Returns:
        A register tile of shape ``(block_dim,)`` with dtype int, where each element contains
            the result index for that thread (-1 if no result)""",
    native_func="tile_bvh_query_next",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "tile_query_valid",
    input_types={"query": BvhQueryTiled},
    value_type=bool,
    group="Tile Primitives",
    doc="""Return whether there are remaining results in a thread-block parallel BVH query.

    This function returns ``True`` when the query has more results to process, and ``False``
    when the query is fully exhausted. The value is uniform across all threads in the block.

    This can be used as a loop condition instead of :func:`tile_max`:

    .. code-block:: python

        query = wp.tile_bvh_query_aabb(bvh_id, lower, upper)
        while wp.tile_query_valid(query):
            result_tile = wp.tile_bvh_query_next(query)
            result_idx = wp.untile(result_tile)
            if result_idx >= 0:
                ...

    Args:
        query: The thread-block BVH query object

    Returns:
        ``True`` if more results are available, ``False`` if exhausted""",
    native_func="tile_query_valid",
    export=False,
    is_differentiable=False,
)


# ---------------------------------------------------------
# Tile Stack
# ---------------------------------------------------------


def tile_stack_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile_stack(dtype=Any, capacity=Any)

    if "dtype" not in arg_values:
        raise TypeError("tile_stack() missing required keyword argument 'dtype'")

    if "capacity" not in arg_values:
        raise TypeError("tile_stack() missing required keyword argument 'capacity'")

    capacity = arg_values["capacity"]
    if isinstance(capacity, Var):
        capacity = capacity.constant
    if capacity is None:
        raise ValueError("tile_stack() requires capacity to be a compile-time constant.")

    if isinstance(capacity, (builtins.bool, builtins.float, builtins.str)):
        raise ValueError(f"tile_stack() requires capacity to be a positive integer, got {capacity!r}")
    try:
        capacity = builtins.int(capacity)
    except (TypeError, ValueError):
        raise ValueError(f"tile_stack() requires capacity to be a positive integer, got {capacity!r}") from None
    if capacity <= 0:
        raise ValueError(f"tile_stack() requires capacity to be a positive integer, got {capacity!r}")

    return tile_stack(dtype=arg_values["dtype"], capacity=capacity)


def tile_stack_dispatch_func(arg_types, return_type, arg_values):
    # Match the tile_zeros pattern: pass dtype and capacity as template args.
    # The C++ tile_stack_init<T, Capacity>() returns int{}, triggering operator=(int)
    # on the already-allocated struct.
    dtype = arg_values["dtype"]
    capacity = arg_values["capacity"]
    if isinstance(capacity, Var):
        capacity = capacity.constant
    return ([], [dtype, capacity])


add_builtin(
    "tile_stack",
    input_types={"capacity": int, "dtype": Any},
    value_func=tile_stack_value_func,
    dispatch_func=tile_stack_dispatch_func,
    native_func="tile_stack_init",
    variadic=False,
    is_differentiable=False,
    doc="""Allocate a cooperative thread-block stack in shared memory.

    Args:
        capacity: Maximum number of elements (must be a compile-time constant)
        dtype: Data type of stack elements

    Returns:
        A tile stack object for use with :func:`tile_stack_push`, :func:`tile_stack_pop`,
        :func:`tile_stack_clear`, and :func:`tile_stack_count`.

    Example:

        .. code-block:: python

            BLOCK = 8
            CAP = wp.constant(8)

            @wp.kernel
            def compact_kernel(data: wp.array[int], out: wp.array[int], out_count: wp.array[int]):
                _i, j = wp.tid()
                s = wp.tile_stack(capacity=CAP, dtype=int)

                val = data[j]
                wp.tile_stack_push(s, val, val > 5)

                if j == 0:
                    out_count[0] = wp.tile_stack_count(s)

                result, slot = wp.tile_stack_pop(s)
                if slot != -1:
                    out[slot] = result

            data = wp.array([1, 8, 3, 7, 2, 9, 4, 6], dtype=int)
            out = wp.zeros(BLOCK, dtype=int)
            out_count = wp.zeros(1, dtype=int)
            wp.launch_tiled(compact_kernel, dim=[1], inputs=[data, out, out_count], block_dim=BLOCK)

            n = out_count.numpy()[0]
            print(sorted(out.numpy()[:n].tolist()))

        .. code-block:: text

            [6, 7, 8, 9]""",
    group="Tile Primitives",
    export=False,
)


def tile_stack_push_value_func(arg_types, arg_values):
    if arg_types is None:
        return int

    s_type = arg_types["s"]
    if not is_tile_stack(s_type):
        raise TypeError(f"tile_stack_push() first argument must be a tile_stack, got {s_type!r}")

    value_type = arg_types["value"]
    if not types_equal(value_type, s_type.dtype):
        raise TypeError(f"tile_stack_push() value type {value_type} does not match stack dtype {s_type.dtype}")

    return int


def tile_stack_push_dispatch_func(arg_types, return_type, arg_values):
    s = arg_values["s"]
    value = arg_values["value"]
    has_value = arg_values["has_value"]
    return ((s, value, has_value), [])


add_builtin(
    "tile_stack_push",
    input_types={"s": Any, "value": Any, "has_value": builtins.bool},
    value_func=tile_stack_push_value_func,
    dispatch_func=tile_stack_push_dispatch_func,
    variadic=False,
    is_differentiable=False,
    doc="""Push a value onto a tile stack (cooperative).

    All threads in the block must call this function. Only threads with
    ``has_value=True`` write to the stack.

    Args:
        s: The tile stack
        value: The value to push
        has_value: Whether this thread has a value to push

    Returns:
        The slot index where the value was written, or ``-1`` if
        ``has_value`` is ``False`` or the stack overflowed.

    Example:

        .. code-block:: python

            CAP = wp.constant(8)

            @wp.kernel
            def push_kernel(out_idx: wp.array[int]):
                _i, j = wp.tid()
                s = wp.tile_stack(capacity=CAP, dtype=int)
                idx = wp.tile_stack_push(s, j * 10, j < 4)
                out_idx[j] = idx

            out_idx = wp.full(8, -1, dtype=int)
            wp.launch_tiled(push_kernel, dim=[1], inputs=[out_idx], block_dim=8)

            idxs = out_idx.numpy()
            print(sorted(idxs[idxs >= 0].tolist()))
            print(sum(idxs == -1))

        .. code-block:: text

            [0, 1, 2, 3]
            4""",
    group="Tile Primitives",
    export=False,
)


def tile_stack_pop_value_func(arg_types, arg_values):
    if arg_types is None:
        return (Any, int)

    s_type = arg_types["s"]
    if not is_tile_stack(s_type):
        raise TypeError(f"tile_stack_pop() argument must be a tile_stack, got {s_type!r}")

    return (s_type.dtype, int)


def tile_stack_pop_dispatch_func(arg_types, return_type, arg_values):
    s = arg_values["s"]
    return ((s,), [])


add_builtin(
    "tile_stack_pop",
    input_types={"s": Any},
    value_func=tile_stack_pop_value_func,
    dispatch_func=tile_stack_pop_dispatch_func,
    variadic=False,
    is_differentiable=False,
    doc="""Pop a value from a tile stack (cooperative).

    All threads in the block must call this function. Each calling thread
    races for a slot.

    Args:
        s: The tile stack

    Returns:
        A tuple ``(value, slot)`` where ``value`` is the popped element
        (or the default value if the stack was empty) and ``slot`` is the
        index of the popped element (the slot it previously occupied), or
        ``-1`` if the stack was empty. When non-negative, ``slot`` lies in
        ``[0, capacity-1]``. Consistent with :func:`tile_stack_push`
        which also uses ``-1`` to indicate failure.

    Example:

        .. code-block:: python

            CAP = wp.constant(8)

            @wp.kernel
            def pop_kernel(out: wp.array[int]):
                _i, j = wp.tid()
                s = wp.tile_stack(capacity=CAP, dtype=int)
                wp.tile_stack_push(s, j * 10, j < 4)

                val, slot = wp.tile_stack_pop(s)
                if slot != -1:
                    out[slot] = val

            out = wp.full(8, -1, dtype=int)
            wp.launch_tiled(pop_kernel, dim=[1], inputs=[out], block_dim=8)

            vals = out.numpy()
            print(sorted(vals[vals >= 0].tolist()))

        .. code-block:: text

            [0, 10, 20, 30]""",
    group="Tile Primitives",
    export=False,
)


def tile_stack_clear_value_func(arg_types, arg_values):
    if arg_types is None:
        return None

    s_type = arg_types["s"]
    if not is_tile_stack(s_type):
        raise TypeError(f"tile_stack_clear() argument must be a tile_stack, got {s_type!r}")

    return None


def tile_stack_clear_dispatch_func(arg_types, return_type, arg_values):
    s = arg_values["s"]
    return ((s,), [])


add_builtin(
    "tile_stack_clear",
    input_types={"s": Any},
    value_func=tile_stack_clear_value_func,
    dispatch_func=tile_stack_clear_dispatch_func,
    variadic=False,
    is_differentiable=False,
    doc="""Clear a tile stack, resetting the count to zero (cooperative).

    All threads in the block must call this function.

    Args:
        s: The tile stack

    Example:

        .. code-block:: python

            CAP = wp.constant(8)

            @wp.kernel
            def clear_kernel(before: wp.array[int], after: wp.array[int]):
                _i, j = wp.tid()
                s = wp.tile_stack(capacity=CAP, dtype=int)
                wp.tile_stack_push(s, j, True)
                if j == 0:
                    before[0] = wp.tile_stack_count(s)
                wp.tile_stack_clear(s)
                if j == 0:
                    after[0] = wp.tile_stack_count(s)

            before = wp.zeros(1, dtype=int)
            after = wp.zeros(1, dtype=int)
            wp.launch_tiled(clear_kernel, dim=[1], inputs=[before, after], block_dim=8)

            print(f"before: {before.numpy()[0]}, after: {after.numpy()[0]}")

        .. code-block:: text

            before: 8, after: 0""",
    group="Tile Primitives",
    export=False,
)


def tile_stack_count_value_func(arg_types, arg_values):
    if arg_types is None:
        return int

    s_type = arg_types["s"]
    if not is_tile_stack(s_type):
        raise TypeError(f"tile_stack_count() argument must be a tile_stack, got {s_type!r}")

    return int


def tile_stack_count_dispatch_func(arg_types, return_type, arg_values):
    s = arg_values["s"]
    return ((s,), [])


add_builtin(
    "tile_stack_count",
    input_types={"s": Any},
    value_func=tile_stack_count_value_func,
    dispatch_func=tile_stack_count_dispatch_func,
    variadic=False,
    is_differentiable=False,
    doc="""Return the current number of elements in a tile stack.

    Unlike the other tile stack operations this function is **not** cooperative
    — it does not contain a synchronization barrier and may be called by a
    single thread or from within a divergent branch. It is safe to call after
    any :func:`tile_stack_push`, :func:`tile_stack_pop`, or
    :func:`tile_stack_clear` *provided the preceding cooperative call has
    completed on all threads in the block*. Those calls end with a barrier
    that makes ``count`` stable and visible. Calling this after a divergent
    push/pop/clear is undefined.

    Args:
        s: The tile stack

    Returns:
        The current number of elements in the stack.

    Example:

        .. code-block:: python

            CAP = wp.constant(8)

            @wp.kernel
            def count_kernel(out_count: wp.array[int]):
                _i, j = wp.tid()
                s = wp.tile_stack(capacity=CAP, dtype=int)
                wp.tile_stack_push(s, j, j % 2 == 0)
                if j == 0:
                    out_count[0] = wp.tile_stack_count(s)

            out_count = wp.zeros(1, dtype=int)
            wp.launch_tiled(count_kernel, dim=[1], inputs=[out_count], block_dim=8)

            print(out_count.numpy()[0])

        .. code-block:: text

            4""",
    group="Tile Primitives",
    export=False,
)


add_builtin(
    "bvh_get_group_root",
    input_types={"id": uint64, "group": int},
    value_type=int,
    group="Geometry",
    doc="""Get the root of a group in a BVH.

    Args:
        id: The BVH identifier
        group: The group identifier

    Returns:
        The root node index for the specified group. If the group does not exist, returns ``-1``
            (sentinel for the BVH global root). Pass ``-1`` to BVH queries to traverse from the global root.

    Example:

        .. testcode::

            @wp.kernel
            def query_group(bvh_id: wp.uint64, lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3],
                            lo: wp.vec3, hi: wp.vec3, centers: wp.array[wp.vec3]):
                root = wp.bvh_get_group_root(bvh_id, 1)  # restrict the query to group 1
                query = wp.bvh_query_aabb(bvh_id, lo, hi, root)
                item = int(0)
                while wp.bvh_query_next(query, item):
                    centers[item] = 0.5 * (lowers[item] + uppers[item])

            lowers = wp.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=wp.vec3)
            uppers = wp.array([[1, 1, 1], [3, 1, 1], [5, 1, 1]], dtype=wp.vec3)
            groups = wp.array([0, 0, 1], dtype=wp.int32)  # one group index per box (box 2 is alone in group 1)
            bvh = wp.Bvh(lowers=lowers, uppers=uppers, groups=groups)

            centers = wp.zeros(3, dtype=wp.vec3)
            wp.launch(query_group, dim=1, inputs=[bvh.id, lowers, uppers, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(6.0, 6.0, 6.0)], outputs=[centers])
            print(centers.numpy().tolist())

        .. testoutput::

            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.5, 0.5, 0.5]]""",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "mesh_get_group_root",
    input_types={"id": uint64, "group": int},
    value_type=int,
    group="Geometry",
    doc="""Get the root of a group in a :class:`warp.Mesh`.

    Args:
        id: The mesh identifier
        group: The group identifier

    Returns:
        The root node index for the specified group. If the group does not exist, returns ``-1``
            (sentinel for the mesh's global root). Pass ``-1`` to mesh queries to traverse from the global root.

    Example:

        .. testcode::

            @wp.kernel
            def group1_only(mesh_id: wp.uint64, origin: wp.vec3, dir: wp.vec3, out_face: wp.array[wp.int32]):
                root = wp.mesh_get_group_root(mesh_id, 1)
                hit = wp.mesh_query_ray(mesh_id, origin, dir, 1.0e6, root)
                if hit.result:
                    out_face[0] = hit.face

            points = wp.array([[0,0,0],[1,0,0],[0,1,0],  [0,0,5],[1,0,5],[0,1,5]], dtype=wp.vec3)
            indices = wp.array([0,1,2, 3,4,5], dtype=wp.int32)
            groups = wp.array([0, 1], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices, groups=groups)

            out_face = wp.full(1, -1, dtype=wp.int32)
            wp.launch(group1_only, dim=1, inputs=[mesh.id, wp.vec3(0.1, 0.1, -1.0), wp.vec3(0.0, 0.0, 1.0)], outputs=[out_face])
            print("hit face:", out_face.numpy()[0])

        .. testoutput::

            hit face: 1""",
    export=False,
    is_differentiable=False,
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
    doc="""Compute the closest point on the :class:`warp.Mesh` with identifier ``id`` to the given ``point`` in space.

    The sign of the distance (inside/outside) is determined by casting three axis-aligned rays from
    ``point`` and taking a majority vote over the orientation of the closest hit along each ray; a ray
    that misses the mesh counts as an outside vote. Classification therefore relies on consistent,
    outward-facing winding and assumes a watertight mesh, and it is relatively robust but more expensive
    than :func:`mesh_query_point_sign_normal`. Sign classification examines the whole mesh and is not
    bounded by ``max_dist``, which only limits the returned closest point. See the other
    ``mesh_query_point_*`` functions for alternative sign-determination methods.

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the closest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        max_dist: Maximum allowed distance to the returned closest point. The query returns no result if no face is strictly closer than this distance.
        inside: Returns a value < 0 if query point is inside the mesh, >=0 otherwise.
            Note that mesh must be watertight for this to be robust
        face: Returns the index of the closest face
        bary_u: Returns the barycentric u coordinate of the closest point
        bary_v: Returns the barycentric v coordinate of the closest point

    Returns:
        ``True`` if a point < ``max_dist`` is found.""",
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
    doc="""Compute the closest point on the :class:`warp.Mesh` with identifier ``id`` to the given ``point`` in space.

    The sign of the distance (inside/outside) is determined by casting three axis-aligned rays from
    ``point`` and taking a majority vote over the orientation of the closest hit along each ray; a ray
    that misses the mesh counts as an outside vote. Classification therefore relies on consistent,
    outward-facing winding and assumes a watertight mesh, and it is relatively robust but more expensive
    than :func:`mesh_query_point_sign_normal`. Sign classification examines the whole mesh and is not
    bounded by ``max_dist``, which only limits the returned closest point. See the other
    ``mesh_query_point_*`` functions for alternative sign-determination methods.

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the closest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        max_dist: Maximum allowed distance to the returned closest point. The query returns no result if no face is strictly closer than this distance.

    Returns:
        A :class:`warp.MeshQueryPoint`. Check ``result`` first (``True`` if a face within
        ``max_dist`` was found), then read ``sign`` (< 0 if ``point`` is inside the mesh, >= 0 if
        outside), ``face`` (index of the closest face), and the barycentric coordinates ``u`` and
        ``v`` of the closest point on that face. Pass ``face``, ``u`` and ``v`` to
        :func:`mesh_eval_position` to obtain the closest point's position.

    Example:

        .. testcode::

            @wp.kernel
            def nearest(mesh_id: wp.uint64, p: wp.vec3, out_pos: wp.array[wp.vec3], out_inside: wp.array[wp.int32]):
                res = wp.mesh_query_point(mesh_id, p, 1.0e6)
                if res.result:
                    out_pos[0] = wp.mesh_eval_position(mesh_id, res.face, res.u, res.v)
                    out_inside[0] = wp.where(res.sign < 0.0, 1, 0)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_pos = wp.zeros(1, dtype=wp.vec3)
            out_inside = wp.zeros(1, dtype=wp.int32)
            wp.launch(nearest, dim=1, inputs=[mesh.id, wp.vec3(0.5, 0.5, 0.25)], outputs=[out_pos, out_inside])
            print(out_pos.numpy()[0], "inside:", bool(out_inside.numpy()[0]))

        .. testoutput::

            [0.5 0.5 0. ] inside: True""",
    require_original_output_arg=True,
    export=False,
)

add_builtin(
    "mesh_query_point_sign_parity",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
        "inside": float,
        "face": int,
        "bary_u": float,
        "bary_v": float,
        "n_sample": int,
        "perturbation_scale": float,
    },
    defaults={"n_sample": 1, "perturbation_scale": 0.1},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Compute the closest point on the :class:`warp.Mesh` with identifier ``id`` to the given ``point`` in space.

    The sign of the distance (inside/outside) is determined by casting ``n_sample`` rays from ``point``
    and counting how many mesh faces each ray crosses. Sampling is deterministic: every call uses a
    fixed seed, so the same ``n_sample`` and ``perturbation_scale`` always produce the same ray
    directions. Each ray perturbs the base direction (1, 1, 1) by an offset drawn per axis from a
    uniform distribution over [``-perturbation_scale``, ``perturbation_scale``), which avoids
    degeneracies such as rays grazing shared edges or vertices.

    The point is classified as inside when at least half of the rays cross an odd number of faces. With
    an even ``n_sample`` an exact tie therefore counts as inside, so prefer a positive, odd ``n_sample``
    (larger values are more robust on imperfect meshes). A non-positive ``n_sample`` casts no rays and
    always classifies the point as inside. Sign classification examines the whole mesh and is not
    bounded by ``max_dist``, which only limits the returned closest point.

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the closest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        max_dist: Maximum allowed distance to the returned closest point. The query returns no result if no face is strictly closer than this distance.
        inside: Returns a value < 0 if query point is inside the mesh, >=0 otherwise.
            Note that mesh must be watertight for this to be robust
        face: Returns the index of the closest face
        bary_u: Returns the barycentric u coordinate of the closest point
        bary_v: Returns the barycentric v coordinate of the closest point
        n_sample: Number of rays used to classify the sign. Prefer a positive, odd value; larger values
            are more robust. A non-positive value casts no rays and classifies the point as inside.
        perturbation_scale: Scale of the perturbation.

    Returns:
        ``True`` if a point < ``max_dist`` is found.
""",
    export=False,
    hidden=True,
)

add_builtin(
    "mesh_query_point_sign_parity",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
        "n_sample": int,
        "perturbation_scale": float,
    },
    defaults={"n_sample": 1, "perturbation_scale": 0.1},
    value_type=MeshQueryPoint,
    group="Geometry",
    doc="""Compute the closest point on the :class:`warp.Mesh` with identifier ``id`` to the given ``point`` in space.

    The sign of the distance (inside/outside) is determined by casting ``n_sample`` rays from ``point``
    and counting how many mesh faces each ray crosses. Sampling is deterministic: every call uses a
    fixed seed, so the same ``n_sample`` and ``perturbation_scale`` always produce the same ray
    directions. Each ray perturbs the base direction (1, 1, 1) by an offset drawn per axis from a
    uniform distribution over [``-perturbation_scale``, ``perturbation_scale``), which avoids
    degeneracies such as rays grazing shared edges or vertices.

    The point is classified as inside when at least half of the rays cross an odd number of faces. With
    an even ``n_sample`` an exact tie therefore counts as inside, so prefer a positive, odd ``n_sample``
    (larger values are more robust on imperfect meshes). A non-positive ``n_sample`` casts no rays and
    always classifies the point as inside. Sign classification examines the whole mesh and is not
    bounded by ``max_dist``, which only limits the returned closest point.

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the closest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        max_dist: Maximum allowed distance to the returned closest point. The query returns no result if no face is strictly closer than this distance.
        n_sample: Number of rays used to classify the sign. Prefer a positive, odd value; larger values
            are more robust. A non-positive value casts no rays and classifies the point as inside.
        perturbation_scale: Scale of the perturbation.

    Returns:
        A :class:`warp.MeshQueryPoint`. Check ``result`` first (``True`` if a face within
        ``max_dist`` was found), then read ``sign`` (< 0 if ``point`` is inside the mesh, >= 0 if
        outside), ``face`` (index of the closest face), and the barycentric coordinates ``u`` and
        ``v`` of the closest point on that face. Pass ``face``, ``u`` and ``v`` to
        :func:`mesh_eval_position` to obtain the closest point's position.

    Example:

        .. testcode::

            @wp.kernel
            def classify(mesh_id: wp.uint64, p: wp.vec3, out_inside: wp.array[wp.int32]):
                res = wp.mesh_query_point_sign_parity(mesh_id, p, 1.0e6)
                if res.result:
                    out_inside[0] = wp.where(res.sign < 0.0, 1, 0)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_inside = wp.zeros(1, dtype=wp.int32)
            wp.launch(classify, dim=1, inputs=[mesh.id, wp.vec3(0.5, 0.5, 0.5)], outputs=[out_inside])
            print("inside:", bool(out_inside.numpy()[0]))

        .. testoutput::

            inside: True""",
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
    doc="""Compute the closest point on the :class:`warp.Mesh` with identifier ``id`` to the given ``point`` in space.

    This method does not compute the sign of the point (inside/outside) which makes it faster than other point query methods.

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the closest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        max_dist: Maximum allowed distance to the returned closest point. The query returns no result if no face is strictly closer than this distance.
        face: Returns the index of the closest face
        bary_u: Returns the barycentric u coordinate of the closest point
        bary_v: Returns the barycentric v coordinate of the closest point

    Returns:
        ``True`` if a point < ``max_dist`` is found.""",
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
    doc="""Compute the closest point on the :class:`warp.Mesh` with identifier ``id`` to the given ``point`` in space.

    This method does not compute the sign of the point (inside/outside) which makes it faster than other point query methods.

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the closest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        max_dist: Maximum allowed distance to the returned closest point. The query returns no result if no face is strictly closer than this distance.

    Returns:
        A :class:`warp.MeshQueryPoint`. Check ``result`` first, then read ``face`` (index of the
        closest face) and the barycentric coordinates ``u`` and ``v`` of the closest point. This
        method does not compute ``sign``. Pass ``face``, ``u`` and ``v`` to
        :func:`mesh_eval_position` to obtain the closest point's position.

    Example:

        .. testcode::

            @wp.kernel
            def nearest(mesh_id: wp.uint64, p: wp.vec3, out_pos: wp.array[wp.vec3]):
                res = wp.mesh_query_point_no_sign(mesh_id, p, 1.0e6)
                if res.result:
                    out_pos[0] = wp.mesh_eval_position(mesh_id, res.face, res.u, res.v)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_pos = wp.zeros(1, dtype=wp.vec3)
            wp.launch(nearest, dim=1, inputs=[mesh.id, wp.vec3(0.5, 0.5, 0.25)], outputs=[out_pos])
            print(out_pos.numpy()[0])

        .. testoutput::

            [0.5 0.5 0. ]""",
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
    doc="""Compute the furthest point on the :class:`warp.Mesh` with identifier ``id`` to the given point in space.

    This method does not compute the sign of the point (inside/outside).

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the furthest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        min_dist: Minimum allowed distance to the returned furthest point. The query returns no result if no face is strictly farther than this distance.
        face: Returns the index of the furthest face
        bary_u: Returns the barycentric u coordinate of the furthest point
        bary_v: Returns the barycentric v coordinate of the furthest point

    Returns:
        ``True`` if a point > ``min_dist`` is found.""",
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
    doc="""Compute the furthest point on the :class:`warp.Mesh` with identifier ``id`` to the given point in space.

    This method does not compute the sign of the point (inside/outside).

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the furthest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        min_dist: Minimum allowed distance to the returned furthest point. The query returns no result if no face is strictly farther than this distance.

    Returns:
        A :class:`warp.MeshQueryPoint`. Check ``result`` first (``True`` if a face farther than
        ``min_dist`` was found), then read ``face`` (index of the furthest face) and the
        barycentric coordinates ``u`` and ``v`` of the furthest point. This method does not compute
        ``sign``. Pass ``face``, ``u`` and ``v`` to :func:`mesh_eval_position` to obtain its
        position.

    Example:

        .. testcode::

            @wp.kernel
            def farthest(mesh_id: wp.uint64, p: wp.vec3, out_pos: wp.array[wp.vec3]):
                res = wp.mesh_query_furthest_point_no_sign(mesh_id, p, 0.0)
                if res.result:
                    out_pos[0] = wp.mesh_eval_position(mesh_id, res.face, res.u, res.v)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_pos = wp.zeros(1, dtype=wp.vec3)
            wp.launch(farthest, dim=1, inputs=[mesh.id, wp.vec3(0.0, 0.0, 0.0)], outputs=[out_pos])
            print(out_pos.numpy()[0])

        .. testoutput::

            [1. 1. 1.]""",
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
    doc="""Compute the closest point on the :class:`warp.Mesh` with identifier ``id`` to the given ``point`` in space.

    Identifies the sign of the distance (inside/outside) using the angle-weighted pseudo normal.
    This approach to sign determination is robust for well conditioned meshes that are watertight and non-self intersecting.
    It is also comparatively fast to compute.

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the closest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        max_dist: Maximum allowed distance to the returned closest point. The query returns no result if no face is strictly closer than this distance.
        inside: Returns a value < 0 if query point is inside the mesh, >=0 otherwise.
            Note that mesh must be watertight for this to be robust
        face: Returns the index of the closest face
        bary_u: Returns the barycentric u coordinate of the closest point
        bary_v: Returns the barycentric v coordinate of the closest point
        epsilon: Epsilon treating distance values as equal, when locating the minimum distance vertex/face/edge, as a
            fraction of the average edge length, also for treating closest point as being on edge/vertex default 1e-3

    Returns:
        ``True`` if a point < ``max_dist`` is found.""",
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
    doc="""Compute the closest point on the :class:`warp.Mesh` with identifier ``id`` to the given ``point`` in space.

    Identifies the sign of the distance (inside/outside) using the angle-weighted pseudo normal.
    This approach to sign determination is robust for well conditioned meshes that are watertight and non-self intersecting.
    It is also comparatively fast to compute.

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the closest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        max_dist: Maximum allowed distance to the returned closest point. The query returns no result if no face is strictly closer than this distance.
        epsilon: Epsilon treating distance values as equal, when locating the minimum distance vertex/face/edge, as a
            fraction of the average edge length, also for treating closest point as being on edge/vertex default 1e-3.

    Returns:
        A :class:`warp.MeshQueryPoint`. Check ``result`` first (``True`` if a face within
        ``max_dist`` was found), then read ``sign`` (< 0 if ``point`` is inside the mesh, >= 0 if
        outside), ``face`` (index of the closest face), and the barycentric coordinates ``u`` and
        ``v`` of the closest point on that face. Pass ``face``, ``u`` and ``v`` to
        :func:`mesh_eval_position` to obtain the closest point's position.

    Example:

        .. testcode::

            @wp.kernel
            def classify(mesh_id: wp.uint64, p: wp.vec3, out_inside: wp.array[wp.int32]):
                res = wp.mesh_query_point_sign_normal(mesh_id, p, 1.0e6)
                if res.result:
                    out_inside[0] = wp.where(res.sign < 0.0, 1, 0)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_inside = wp.zeros(1, dtype=wp.int32)
            wp.launch(classify, dim=1, inputs=[mesh.id, wp.vec3(0.5, 0.5, 0.5)], outputs=[out_inside])
            print("inside:", bool(out_inside.numpy()[0]))

        .. testoutput::

            inside: True""",
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
    doc="""Compute the closest point on the :class:`warp.Mesh` with identifier ``id`` to the given point in space.

    Identifies the sign using the winding number of the mesh relative to the query point. This method of sign determination is robust for poorly conditioned meshes
    and provides a smooth approximation to sign even when the mesh is not watertight. This method is the most robust and accurate of the sign determination meshes
    but also the most expensive.

    .. note:: The :class:`warp.Mesh` must be constructed with ``support_winding_number=True`` to use
        the winding number for sign determination. If it was not, the sign silently falls back to the
        method used by :func:`mesh_query_point` (a majority vote over the orientation of the closest hit
        of three axis-aligned rays), which is robust only for watertight meshes with consistent winding;
        the closest point, face, and barycentric outputs are unaffected.

    Sign classification examines the whole mesh and is not bounded by ``max_dist``, which only limits the
    returned closest point.

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the closest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        max_dist: Maximum allowed distance to the returned closest point. The query returns no result if no face is strictly closer than this distance.
        inside: Returns a value < 0 if the query point is inside the mesh, >= 0 otherwise, as
            classified by the winding number (subject to the fallback described in the note above)
        face: Returns the index of the closest face
        bary_u: Returns the barycentric u coordinate of the closest point
        bary_v: Returns the barycentric v coordinate of the closest point
        accuracy: Accuracy for computing the winding number with fast winding number method utilizing second-order dipole approximation, default 2.0
        threshold: The threshold of the winding number to be considered inside, default 0.5

    Returns:
        ``True`` if a point < ``max_dist`` is found.""",
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
    doc="""Compute the closest point on the :class:`warp.Mesh` with identifier ``id`` to the given point in space.

    Identifies the sign using the winding number of the mesh relative to the query point. This method of sign determination is robust for poorly conditioned meshes
    and provides a smooth approximation to sign even when the mesh is not watertight. This method is the most robust and accurate of the sign determination meshes
    but also the most expensive.

    .. note:: The :class:`warp.Mesh` must be constructed with ``support_winding_number=True`` to use
        the winding number for sign determination. If it was not, the sign silently falls back to the
        method used by :func:`mesh_query_point` (a majority vote over the orientation of the closest hit
        of three axis-aligned rays), which is robust only for watertight meshes with consistent winding;
        the closest point, face, and barycentric outputs are unaffected.

    Sign classification examines the whole mesh and is not bounded by ``max_dist``, which only limits the
    returned closest point.

    Triangles that are degenerate or nearly degenerate relative to their edge lengths are excluded from
    the closest-point search, so such a face can be skipped even when it satisfies the distance
    constraint. If every face satisfying the distance constraint is excluded, the query returns no result.

    Args:
        id: The mesh identifier
        point: The query point, in the mesh's local space
        max_dist: Maximum allowed distance to the returned closest point. The query returns no result if no face is strictly closer than this distance.
        accuracy: Accuracy for computing the winding number with fast winding number method utilizing second-order dipole approximation, default 2.0
        threshold: The threshold of the winding number to be considered inside, default 0.5.

    Returns:
        A :class:`warp.MeshQueryPoint`. Check ``result`` first (``True`` if a face within
        ``max_dist`` was found), then read ``sign`` (< 0 if ``point`` is inside the mesh, >= 0 if
        outside), ``face`` (index of the closest face), and the barycentric coordinates ``u`` and
        ``v`` of the closest point on that face. Pass ``face``, ``u`` and ``v`` to
        :func:`mesh_eval_position` to obtain the closest point's position.

    Example:

        .. testcode::

            @wp.kernel
            def classify(mesh_id: wp.uint64, p: wp.vec3, out_inside: wp.array[wp.int32]):
                res = wp.mesh_query_point_sign_winding_number(mesh_id, p, 1.0e6)
                if res.result:
                    out_inside[0] = wp.where(res.sign < 0.0, 1, 0)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices, support_winding_number=True)

            out_inside = wp.zeros(1, dtype=wp.int32)
            wp.launch(classify, dim=1, inputs=[mesh.id, wp.vec3(0.5, 0.5, 0.5)], outputs=[out_inside])
            print("inside:", bool(out_inside.numpy()[0]))

        .. testoutput::

            inside: True""",
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
        "root": int,
    },
    defaults={"root": -1},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Compute the closest ray hit on the :class:`warp.Mesh` with identifier ``id``, returns ``True`` if a hit < ``max_t`` is found.

    ``start`` and ``dir`` are given in the mesh's local space. ``dir`` need not be normalized, but
    ``max_t`` and the returned ``t`` are measured in multiples of its length, so normalize it for
    them to be distances in mesh-space units.

    The ``root`` parameter can be obtained using the :func:`mesh_get_group_root` function when creating a grouped mesh.
    When ``root`` is a valid (>=0) value, the traversal will be confined to the subtree starting from the root.
    If ``root`` is -1 (default), traversal starts at the mesh's global root.
    The query will only traverse down from that node, limiting traversal to that subtree.

    Args:
        id: The mesh identifier
        start: The ray origin, in the mesh's local space
        dir: The ray direction, in the mesh's local space (see above on normalization)
        max_t: The maximum distance along the ray to check for intersections (in multiples of ``dir``'s length)
        root: The root node index for grouped BVH queries, or -1 for global root
        t: Returns the distance of the closest hit along the ray (in multiples of ``dir``'s length)
        bary_u: Returns the barycentric u coordinate of the closest hit
        bary_v: Returns the barycentric v coordinate of the closest hit
        sign: Returns a value > 0 if the ray hit in front of the face, < 0 otherwise
        normal: Returns the unit face normal, oriented by the face's winding order
        face: Returns the index of the hit face.""",
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
        "root": int,
    },
    defaults={"root": -1},
    value_type=MeshQueryRay,
    group="Geometry",
    doc="""Compute the closest ray hit on the :class:`warp.Mesh` with identifier ``id``.

    ``start`` and ``dir`` are given in the mesh's local space. ``dir`` need not be normalized, but
    ``max_t`` and the returned ``t`` are measured in multiples of its length, so normalize it for
    them to be distances in mesh-space units.

    The ``root`` parameter can be obtained using the :func:`mesh_get_group_root` function when creating a grouped mesh.
    When ``root`` is a valid (>=0) value, the traversal will be confined to the subtree starting from the root.
    If ``root`` is -1 (default), traversal starts at the mesh's global root.

    Args:
        id: The mesh identifier
        start: The ray origin, in the mesh's local space
        dir: The ray direction, in the mesh's local space (see above on normalization)
        max_t: The maximum distance along the ray to check for intersections (in multiples of ``dir``'s length)
        root: The root node index for grouped BVH queries, or -1 for global root (optional, default: -1)

    Returns:
        A :class:`warp.MeshQueryRay`. Check ``result`` first (``True`` if a hit within ``max_t`` was
        found), then read ``t`` (distance to the hit, in multiples of ``dir``'s length), ``face``
        (index of the hit face), ``normal`` (the unit face normal, oriented by the face's winding
        order), ``sign`` (> 0 if the ray hit the front of the face, < 0 the back), and the
        barycentric coordinates ``u`` and ``v`` of the hit. The hit position is ``start + t * dir``.

    Example:

        .. testcode::

            @wp.kernel
            def cast(mesh_id: wp.uint64, origin: wp.vec3, dir: wp.vec3, out_t: wp.array[wp.float32], out_n: wp.array[wp.vec3]):
                hit = wp.mesh_query_ray(mesh_id, origin, dir, 1.0e6)
                if hit.result:
                    out_t[0] = hit.t
                    out_n[0] = hit.normal

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_t = wp.zeros(1, dtype=wp.float32)
            out_n = wp.zeros(1, dtype=wp.vec3)
            wp.launch(cast, dim=1, inputs=[mesh.id, wp.vec3(0.5, 0.5, -2.0), wp.vec3(0.0, 0.0, 1.0)], outputs=[out_t, out_n])
            print("t =", out_t.numpy()[0], "normal =", out_n.numpy()[0])

        .. testoutput::

            t = 2.0 normal = [ 0.  0. -1.]""",
    require_original_output_arg=True,
    export=False,
)

add_builtin(
    "mesh_query_ray_anyhit",
    input_types={
        "id": uint64,
        "start": vec3,
        "dir": vec3,
        "max_t": float,
        "root": int,
    },
    defaults={"root": -1},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Check whether a ray hits the :class:`warp.Mesh` with identifier ``id``, without computing the closest hit.

    Returns as soon as any intersecting face within ``max_t`` is found, so it is cheaper than
    :func:`mesh_query_ray` when only the yes/no answer is needed. ``start`` and ``dir`` are given in
    the mesh's local space; ``max_t`` is measured in multiples of ``dir``'s length, so normalize
    ``dir`` for it to be a distance in mesh-space units.

    The ``root`` parameter can be obtained using the :func:`mesh_get_group_root` function when creating a grouped mesh.
    When ``root`` is a valid (>=0) value, the traversal will be confined to the subtree starting from the root.
    If ``root`` is -1 (default), traversal starts at the mesh's global root.

    Args:
        id: The mesh identifier
        start: The ray origin, in the mesh's local space
        dir: The ray direction, in the mesh's local space
        max_t: The maximum distance along the ray to check for intersections (in multiples of ``dir``'s length)
        root: The root node index for grouped BVH queries, or -1 for global root (optional, default: -1)

    Returns:
        ``True`` if the ray intersects any face within ``max_t``, ``False`` otherwise.

    Example:

        .. testcode::

            @wp.kernel
            def occluded(mesh_id: wp.uint64, origin: wp.vec3, dir: wp.vec3, out_hit: wp.array[wp.int32]):
                out_hit[0] = wp.where(wp.mesh_query_ray_anyhit(mesh_id, origin, dir, 1.0e6), 1, 0)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_hit = wp.zeros(1, dtype=wp.int32)
            wp.launch(occluded, dim=1, inputs=[mesh.id, wp.vec3(0.5, 0.5, -2.0), wp.vec3(0.0, 0.0, 1.0)], outputs=[out_hit])
            print("hit:", bool(out_hit.numpy()[0]))

        .. testoutput::

            hit: True""",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "mesh_query_ray_count_intersections",
    input_types={
        "id": uint64,
        "start": vec3,
        "dir": vec3,
        "root": int,
    },
    defaults={"root": -1},
    value_type=int,
    group="Geometry",
    doc="""Count the number of intersections between a ray and a :class:`warp.Mesh`.

    This function casts a ray through the mesh and counts all triangle intersections with ``t >= 0``.
    Unlike :func:`mesh_query_ray`, this function does not stop at the first hit and continues
    traversing to count all intersections along the entire ray.

    This function can be used to determine whether the ray origin lies inside a watertight, intersection-free mesh.
    An odd number of intersections indicates the origin is inside the mesh, while an even number indicates it is outside.

    The ``root`` parameter can be obtained using the :func:`mesh_get_group_root` function when creating a grouped mesh.
    When ``root`` is a valid (>=0) value, the traversal will be confined to the subtree starting from the root.
    If ``root`` is -1 (default), traversal starts at the mesh's global root.

    Args:
        id: The mesh identifier
        start: The ray origin, in the mesh's local space
        dir: The ray direction, in the mesh's local space (only its direction matters; the count is independent of its length)
        root: The root node index for grouped BVH queries, or -1 for global root (optional, default: -1)

    Returns:
        The number of intersections (with ``t >= 0``) between the ray and the mesh.

    Example:

        .. testcode::

            @wp.kernel
            def crossings(mesh_id: wp.uint64, origin: wp.vec3, dir: wp.vec3, out_n: wp.array[wp.int32]):
                out_n[0] = wp.mesh_query_ray_count_intersections(mesh_id, origin, dir)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_n = wp.zeros(1, dtype=wp.int32)
            wp.launch(crossings, dim=1, inputs=[mesh.id, wp.vec3(0.3, 0.6, -2.0), wp.vec3(0.0, 0.0, 1.0)], outputs=[out_n])
            print("crossings:", out_n.numpy()[0])

        .. testoutput::

            crossings: 2""",
    export=False,
    is_differentiable=False,
)


add_builtin(
    "mesh_query_aabb",
    input_types={"id": uint64, "low": vec3, "high": vec3},
    value_type=MeshQueryAABB,
    group="Geometry",
    doc="""Construct an axis-aligned bounding box (AABB) query against a :class:`warp.Mesh`.

    Returns a query that iterates over every triangle (face) whose own axis-aligned bounding box
    overlaps the query box ``[low, high]``, given in the mesh's local space. This is a broad-phase
    test on bounding boxes: a reported face's triangle may not actually intersect the box, so
    perform an exact test yourself if required. Advance the query and read each result with
    :func:`mesh_query_aabb_next`.

    Args:
        id: The mesh identifier
        low: The lower bound of the query box, in the mesh's local space
        high: The upper bound of the query box, in the mesh's local space

    Returns:
        A :class:`warp.MeshQueryAABB`. It is opaque; pass it to :func:`mesh_query_aabb_next`, which
        writes the index of each overlapping face to its ``index`` argument.

    Example:

        .. testcode::

            @wp.kernel
            def count_faces(mesh_id: wp.uint64, lo: wp.vec3, hi: wp.vec3, out_count: wp.array[wp.int32]):
                query = wp.mesh_query_aabb(mesh_id, lo, hi)
                face = int(0)
                while wp.mesh_query_aabb_next(query, face):
                    wp.atomic_add(out_count, 0, 1)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_count = wp.zeros(1, dtype=wp.int32)
            wp.launch(count_faces, dim=1, inputs=[mesh.id, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(2.0, 2.0, 2.0)], outputs=[out_count])
            print("overlapping faces:", out_count.numpy()[0])

        .. testoutput::

            overlapping faces: 12""",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "mesh_query_aabb_next",
    input_types={"query": MeshQueryAABB, "index": int},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Advance a mesh AABB query to the next overlapping triangle and report whether one was found.

    Writes the index of the current face to ``index`` and returns ``True``; returns ``False`` once
    no overlapping triangles remain (``index`` is then left unchanged). The reported index is a
    face index (0-based, into the mesh's triangles), suitable for :func:`mesh_eval_position`,
    :func:`mesh_eval_face_normal`, and the other face-indexed functions. Used in a ``while`` loop
    together with :func:`mesh_query_aabb`.

    Args:
        query: The query to advance, from :func:`mesh_query_aabb`
        index: Output; receives the index of the current overlapping face

    Returns:
        ``True`` if another overlapping triangle was found (its face index written to ``index``),
        ``False`` if the query is exhausted.

    Example:

        .. testcode::

            @wp.kernel
            def count_faces(mesh_id: wp.uint64, lo: wp.vec3, hi: wp.vec3, out_count: wp.array[wp.int32]):
                query = wp.mesh_query_aabb(mesh_id, lo, hi)
                face = int(0)
                while wp.mesh_query_aabb_next(query, face):
                    wp.atomic_add(out_count, 0, 1)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_count = wp.zeros(1, dtype=wp.int32)
            wp.launch(count_faces, dim=1, inputs=[mesh.id, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(2.0, 2.0, 2.0)], outputs=[out_count])
            print("overlapping faces:", out_count.numpy()[0])

        .. testoutput::

            overlapping faces: 12""",
    export=False,
    is_differentiable=False,
)

# Primary naming convention (grouped with other geometry functions)
add_builtin(
    "mesh_query_aabb_tiled",
    input_types={"id": uint64, "low": vec3, "high": vec3},
    value_type=MeshQueryAABBTiled,
    group="Geometry",
    doc="""Construct an axis-aligned bounding box (AABB) query against a :class:`warp.Mesh` for thread-block parallel traversal.

    For use in tiled kernels: all threads in the block cooperatively traverse the mesh's BVH.
    Advance the query with :func:`mesh_query_aabb_next_tiled` (one face index per thread per step)
    in a loop guarded by :func:`tile_query_valid`. ``low`` and ``high`` must be identical across all
    threads in the block and are given in the mesh's local space.

    Args:
        id: The mesh identifier
        low: The lower bound of the query box, in the mesh's local space (must be the same for all threads in the block)
        high: The upper bound of the query box, in the mesh's local space (must be the same for all threads in the block)

    Returns:
        A :class:`warp.MeshQueryAABBTiled` to advance with :func:`mesh_query_aabb_next_tiled`.

    Example:

        .. testcode::

            @wp.kernel
            def tiled_faces(mesh_id: wp.uint64, lo: wp.vec3, hi: wp.vec3, out_count: wp.array[wp.int32]):
                query = wp.mesh_query_aabb_tiled(mesh_id, lo, hi)
                while wp.tile_query_valid(query):
                    result = wp.mesh_query_aabb_next_tiled(query)
                    face = wp.untile(result)
                    if face >= 0:
                        wp.atomic_add(out_count, 0, 1)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_count = wp.zeros(1, dtype=wp.int32)
            wp.launch_tiled(tiled_faces, dim=[1], inputs=[mesh.id, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(2.0, 2.0, 2.0)], outputs=[out_count], block_dim=32)
            print("overlapping faces:", out_count.numpy()[0])

        .. testoutput::

            overlapping faces: 12""",
    native_func="tile_mesh_query_aabb",
    export=False,
    is_differentiable=False,
)


def mesh_query_aabb_next_tiled_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=int, shape=tuple[int])

    # Return a register tile of ints with shape (block_dim,)
    block_dim = warp._src.codegen.options.get("block_dim", 256)
    return tile(dtype=int, shape=(block_dim,), storage="register")


def mesh_query_aabb_next_tiled_dispatch_func(
    input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]
):
    # This function needs to:
    # 1. Create a temporary per-thread int variable
    # 2. Call mesh_query_aabb_next_thread_block with query and the temp variable
    # 3. Wrap the temp variable in a tile
    # The actual implementation is handled via native_func
    query = args["query"]
    return ((query,), ())


add_builtin(
    "mesh_query_aabb_next_tiled",
    input_types={"query": MeshQueryAABBTiled},
    value_func=mesh_query_aabb_next_tiled_value_func,
    dispatch_func=mesh_query_aabb_next_tiled_dispatch_func,
    group="Geometry",
    doc="""Move to the next triangle in a thread-block parallel mesh AABB query and return results as a tile.

    Each thread in the block receives one result index in the returned tile, or -1 if no result for that thread.
    The function returns a register tile of shape ``(block_dim,)`` containing the result indices.

    To check if any results were found, check if any element in the tile is >= 0. Call this in a
    loop guarded by :func:`tile_query_valid`, which returns ``False`` once the query is exhausted.
    All threads in the block must call it cooperatively.

    Args:
        query: The thread-block mesh query object, from :func:`mesh_query_aabb_tiled`

    Returns:
        A register tile of shape ``(block_dim,)`` with dtype int, where each element contains
            the result index for that thread (-1 if no result)

    Example:

        .. testcode::

            @wp.kernel
            def tiled_faces(mesh_id: wp.uint64, lo: wp.vec3, hi: wp.vec3, out_count: wp.array[wp.int32]):
                query = wp.mesh_query_aabb_tiled(mesh_id, lo, hi)
                while wp.tile_query_valid(query):
                    result = wp.mesh_query_aabb_next_tiled(query)
                    face = wp.untile(result)
                    if face >= 0:
                        wp.atomic_add(out_count, 0, 1)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out_count = wp.zeros(1, dtype=wp.int32)
            wp.launch_tiled(tiled_faces, dim=[1], inputs=[mesh.id, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(2.0, 2.0, 2.0)], outputs=[out_count], block_dim=32)
            print("overlapping faces:", out_count.numpy()[0])

        .. testoutput::

            overlapping faces: 12""",
    native_func="tile_mesh_query_aabb_next",
    export=False,
    is_differentiable=False,
)

# Aliases for backward compatibility (tile_* naming convention)
add_builtin(
    "tile_mesh_query_aabb",
    input_types={"id": uint64, "low": vec3, "high": vec3},
    value_type=MeshQueryAABBTiled,
    group="Tile Primitives",
    doc="""Construct an axis-aligned bounding box query against a :class:`warp.Mesh` for thread-block parallel traversal.

    This query can be used in tiled kernels to cooperatively traverse a mesh's BVH across a thread block.


    .. note:: This is an alias for :func:`mesh_query_aabb_tiled`.

    Args:
        id: The mesh identifier
        low: The lower bound of the bounding box in mesh space (must be the same for all threads in the block)
        high: The upper bound of the bounding box in mesh space (must be the same for all threads in the block)""",
    native_func="tile_mesh_query_aabb",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "tile_mesh_query_aabb_next",
    input_types={"query": MeshQueryAABBTiled},
    value_func=mesh_query_aabb_next_tiled_value_func,
    dispatch_func=mesh_query_aabb_next_tiled_dispatch_func,
    group="Tile Primitives",
    doc="""Move to the next triangle in a thread-block parallel mesh AABB query and return results as a tile.

    Each thread in the block receives one result index in the returned tile, or -1 if no result for that thread.
    The function returns a register tile of shape ``(block_dim,)`` containing the result indices.

    To check if any results were found, check if any element in the tile is >= 0.


    .. note:: This is an alias for :func:`mesh_query_aabb_next_tiled`.

    Args:
        query: The thread-block mesh query object

    Returns:
        A register tile of shape ``(block_dim,)`` with dtype int, where each element contains
            the result index for that thread (-1 if no result)""",
    native_func="tile_mesh_query_aabb_next",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "tile_query_valid",
    input_types={"query": MeshQueryAABBTiled},
    value_type=bool,
    group="Tile Primitives",
    doc="""Return whether there are remaining results in a thread-block parallel mesh AABB query.

    This function returns ``True`` when the query has more results to process, and ``False``
    when the query is fully exhausted. The value is uniform across all threads in the block.

    This can be used as a loop condition instead of :func:`tile_max`:

    .. code-block:: python

        query = wp.tile_mesh_query_aabb(mesh_id, lower, upper)
        while wp.tile_query_valid(query):
            result_tile = wp.tile_mesh_query_aabb_next(query)
            result_idx = wp.untile(result_tile)
            if result_idx >= 0:
                ...

    Args:
        query: The thread-block mesh query object

    Returns:
        ``True`` if more results are available, ``False`` if exhausted""",
    native_func="tile_query_valid",
    export=False,
    is_differentiable=False,
)


add_builtin(
    "mesh_eval_position",
    input_types={"id": uint64, "face": int, "bary_u": float, "bary_v": float},
    value_type=vec3,
    group="Geometry",
    doc="""Evaluate the interpolated position on a face of the :class:`warp.Mesh` from barycentric coordinates.

    Linearly interpolates the face's three vertex positions: with the face's vertices ``v0``,
    ``v1``, ``v2``, returns ``v0 * bary_u + v1 * bary_v + v2 * (1 - bary_u - bary_v)``, in the
    mesh's local space. Use this to turn a ``face``/``u``/``v`` result from a mesh point or ray
    query back into a position.

    Args:
        id: The mesh identifier
        face: The face (triangle) index
        bary_u: Barycentric weight of the face's first vertex
        bary_v: Barycentric weight of the face's second vertex

    Returns:
        The interpolated position, in the mesh's local space.

    Example:

        .. testcode::

            @wp.kernel
            def centroid(mesh_id: wp.uint64, out: wp.array[wp.vec3]):
                out[0] = wp.mesh_eval_position(mesh_id, 0, 1.0 / 3.0, 1.0 / 3.0)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out = wp.zeros(1, dtype=wp.vec3)
            wp.launch(centroid, dim=1, inputs=[mesh.id], outputs=[out])
            p = out.numpy()[0]
            print(f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

        .. testoutput::

            (0.333, 0.667, 0.000)""",
    export=False,
)

add_builtin(
    "mesh_eval_velocity",
    input_types={"id": uint64, "face": int, "bary_u": float, "bary_v": float},
    value_type=vec3,
    group="Geometry",
    doc="""Evaluate the interpolated velocity on a face of the :class:`warp.Mesh` from barycentric coordinates.

    Linearly interpolates the face's three per-vertex velocities the same way
    :func:`mesh_eval_position` interpolates positions:
    ``v0 * bary_u + v1 * bary_v + v2 * (1 - bary_u - bary_v)``. Requires the mesh to have been
    constructed with a ``velocities`` array; returns a zero vector otherwise.

    Args:
        id: The mesh identifier
        face: The face (triangle) index
        bary_u: Barycentric weight of the face's first vertex
        bary_v: Barycentric weight of the face's second vertex

    Returns:
        The interpolated velocity, or a zero vector if the mesh has no velocities.

    Example:

        .. testcode::

            @wp.kernel
            def vel_at(mesh_id: wp.uint64, out: wp.array[wp.vec3]):
                out[0] = wp.mesh_eval_velocity(mesh_id, 0, 0.25, 0.25)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            velocities = wp.array(np.tile([0.0, 0.0, 1.0], (8, 1)), dtype=wp.vec3)
            mesh = wp.Mesh(points=points, indices=indices, velocities=velocities)

            out = wp.zeros(1, dtype=wp.vec3)
            wp.launch(vel_at, dim=1, inputs=[mesh.id], outputs=[out])
            print(out.numpy()[0])

        .. testoutput::

            [0. 0. 1.]""",
    export=False,
)


# Shared runnable example for the default float32 hash-grid query builtins. It builds a float32 grid,
# so it is attached only to the float32 registrations; the float16/float64 overloads omit it.
_HASH_GRID_QUERY_EXAMPLE = """

    Example:

        .. testcode::

            @wp.kernel
            def count_neighbors(grid_id: wp.uint64, pts: wp.array[wp.vec3], radius: wp.float32, out_count: wp.array[wp.int32]):
                i = wp.tid()
                p = pts[i]
                query = wp.hash_grid_query(grid_id, p, radius)
                index = int(0)
                n = int(0)
                while wp.hash_grid_query_next(query, index):
                    if wp.length(p - pts[index]) <= radius:
                        n += 1
                out_count[i] = n

            points = wp.array([[0,0,0],[0.1,0,0],[0.5,0,0],[2.0,0,0]], dtype=wp.vec3)
            grid = wp.HashGrid(dim_x=8, dim_y=8, dim_z=8)
            grid.build(points=points, radius=0.3)

            out_count = wp.zeros(4, dtype=wp.int32)
            wp.launch(count_neighbors, dim=4, inputs=[grid.id, points, 0.3], outputs=[out_count])
            print(out_count.numpy())

        .. testoutput::

            [2 2 1 1]"""


# Hash grid query builtins for all precisions (float16, float32, float64)
def _add_hash_grid_query_builtins(vec_type, scalar_type, query_type, precision_doc=""):
    """Register the ``hash_grid_query`` and ``hash_grid_query_next`` builtins for one coordinate precision.

    The full runnable example is attached only to the default ``float32`` overload; the ``float16``
    and ``float64`` overloads get concise, type-specific documentation that points back to it, so
    their generated reference entries don't display an example that builds a ``float32`` grid and
    calls the ``float32`` overload.
    """
    doc_suffix = f" ({precision_doc} precision)" if precision_doc else ""
    example = "" if precision_doc else _HASH_GRID_QUERY_EXAMPLE

    # Query.

    if precision_doc:
        body = (
            f"The ``{precision_doc}`` overload of :func:`hash_grid_query`. Behavior and usage match the default\n"
            f"    ``float32`` overload; see it for details and a usage example."
        )
    else:
        body = (
            "Returns a query that iterates over candidate neighbors of ``point``: every point in the grid\n"
            "    cells overlapped by the box from ``point - max_dist`` to ``point + max_dist``. These are\n"
            "    *candidates* — the grid does not test distance, so some are farther than ``max_dist``; filter\n"
            "    by actual distance yourself. Advance the query and read each candidate's index with\n"
            "    :func:`hash_grid_query_next`. ``point`` must be in the same coordinate space as the points the\n"
            "    grid was built from (see :class:`warp.HashGrid`)."
        )

    doc = f"""Construct a point query against a :class:`warp.HashGrid`{doc_suffix}.

    {body}

    Args:
        id: The :class:`warp.HashGrid` identifier
        point: The query point
        max_dist: The query radius

    Returns:
        A hash-grid query object to pass to :func:`hash_grid_query_next`.{example}"""

    grouped_doc = f"""Construct a point query against a :class:`warp.HashGrid`, restricted to one point group{doc_suffix}.

    {body}

    If the grid was built with groups, only points whose group id equals ``group`` are returned as
    candidates; any ``int32`` value is a valid group id. Omit the ``group`` argument to visit all
    groups, matching ungrouped behavior. Unlike grouped BVH queries, grouped hash-grid queries do
    not require a root lookup; pass the group id directly.

    Args:
        id: The :class:`warp.HashGrid` identifier
        point: The query point
        max_dist: The query radius
        group: Restrict candidates to points built with this group id

    Returns:
        A hash-grid query object to pass to :func:`hash_grid_query_next`."""

    add_builtin(
        "hash_grid_query",
        input_types={"id": uint64, "point": vec_type, "max_dist": scalar_type},
        value_type=query_type,
        group="Geometry",
        doc=doc,
        export=False,
        is_differentiable=False,
    )

    add_builtin(
        "hash_grid_query",
        input_types={"id": uint64, "point": vec_type, "max_dist": scalar_type, "group": int},
        value_type=query_type,
        group="Geometry",
        doc=grouped_doc,
        export=False,
        is_differentiable=False,
    )

    # Query next.

    if precision_doc:
        body = (
            f"The ``{precision_doc}`` overload of :func:`hash_grid_query_next`, taking the query returned by the\n"
            f"    ``{precision_doc}`` :func:`hash_grid_query`. Behavior and usage match the default ``float32``\n"
            f"    overload; see it for details and a usage example."
        )
    else:
        body = (
            "Writes the candidate's index to ``index`` and returns ``True``; returns ``False`` once no\n"
            "    candidates remain (``index`` is then left unchanged). The index refers to the points the grid\n"
            "    was built from, in their original order. Candidates share a nearby grid cell and may lie farther\n"
            "    than the query radius, so test the actual distance yourself (see :func:`hash_grid_query`).\n"
            "    Supports query objects returned by :func:`wp.hash_grid_query() <warp.hash_grid_query>` for all\n"
            "    coordinate precisions."
        )

    doc = f"""Advance a hash grid query to the next candidate neighbor and report whether one was found{doc_suffix}.

    {body}

    Args:
        query: The query to advance, from :func:`hash_grid_query`
        index: Output; receives the index of the current candidate neighbor

    Returns:
        ``True`` if another candidate was found (its index written to ``index``), ``False`` if the
        query is exhausted.{example}"""

    add_builtin(
        "hash_grid_query_next",
        input_types={"query": query_type, "index": int},
        value_type=builtins.bool,
        group="Geometry",
        doc=doc,
        export=False,
        is_differentiable=False,
    )


_add_hash_grid_query_builtins(vec3, float, hash_grid_query_type(float32))
_add_hash_grid_query_builtins(vec3h, float16, hash_grid_query_type(float16), "float16")
_add_hash_grid_query_builtins(vec3d, float64, hash_grid_query_type(float64), "float64")

add_builtin(
    "hash_grid_point_id",
    input_types={"id": uint64, "index": int},
    value_type=int,
    group="Geometry",
    doc="""Return the original point index stored at a given position in the :class:`warp.HashGrid`'s spatially-sorted order.

    The grid sorts its points by cell so that points sharing a cell are adjacent. Given a position
    ``index`` in that sorted order (typically the thread index), this returns the corresponding
    index into the original points array. Looking points up in this order makes neighboring threads
    access nearby points, improving memory locality.

    Args:
        id: The :class:`warp.HashGrid` identifier
        index: A position in the grid's sorted order, in ``[0, number_of_points)``

    Returns:
        The corresponding index into the original points array, or -1 if the
        :class:`warp.HashGrid` has not been built/reserved.

    Example:

        .. testcode::

            @wp.kernel
            def gather(grid_id: wp.uint64, pts: wp.array[wp.vec3], out: wp.array[wp.vec3]):
                i = wp.tid()
                # visit points in the grid's cell-sorted order for memory locality;
                # hash_grid_point_id maps sorted position i -> the original point index
                out[i] = pts[wp.hash_grid_point_id(grid_id, i)]

            points = wp.array([[0,0,0],[0.1,0,0],[0.5,0,0],[2.0,0,0]], dtype=wp.vec3)
            grid = wp.HashGrid(dim_x=8, dim_y=8, dim_z=8)
            grid.build(points=points, radius=0.3)

            out = wp.zeros(4, dtype=wp.vec3)
            wp.launch(gather, dim=4, inputs=[grid.id, points], outputs=[out])
            # every point is visited exactly once, so the gather is a permutation of the input
            print(sorted(out.numpy().tolist()) == sorted(points.numpy().tolist()))

        .. testoutput::

            True""",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "intersect_tri_tri",
    input_types={"v0": vec3, "v1": vec3, "v2": vec3, "u0": vec3, "u1": vec3, "u2": vec3},
    value_type=int,
    group="Geometry",
    doc="""Test whether two triangles ``(v0, v1, v2)`` and ``(u0, u1, u2)`` intersect, using Möller's method.

    All six vertices must be in the same coordinate space. Coplanar triangles are handled (an
    overlap within their common plane counts as an intersection). This single-precision overload
    may return incorrect results in near-degenerate cases; use the double-precision overload
    (``vec3d`` inputs) for greater robustness.

    Returns:
        ``1`` if the triangles intersect, ``0`` otherwise.

    Example:

        .. testcode::

            @wp.kernel
            def tri_tri(out: wp.array[wp.int32]):
                a0, a1, a2 = wp.vec3(0.0, 0.0, 0.0), wp.vec3(2.0, 0.0, 0.0), wp.vec3(0.0, 2.0, 0.0)
                b0, b1, b2 = wp.vec3(1.0, 1.0, -1.0), wp.vec3(1.0, 1.0, 1.0), wp.vec3(1.0, -1.0, 0.0)
                out[0] = wp.intersect_tri_tri(a0, a1, a2, b0, b1, b2)

            out = wp.zeros(1, dtype=wp.int32)
            wp.launch(tri_tri, dim=1, inputs=[out])
            print("intersect:", out.numpy()[0])

        .. testoutput::

            intersect: 1""",
    export=False,
    is_differentiable=False,
)


add_builtin(
    "intersect_tri_tri",
    input_types={"v0": vec3d, "v1": vec3d, "v2": vec3d, "u0": vec3d, "u1": vec3d, "u2": vec3d},
    value_type=int,
    group="Geometry",
    doc="""Test whether two triangles ``(v0, v1, v2)`` and ``(u0, u1, u2)`` intersect, using Möller's method.

    All six vertices must be in the same coordinate space. Coplanar triangles are handled (an
    overlap within their common plane counts as an intersection). This double-precision overload is
    more accurate than the single-precision (``vec3`` inputs) overload.

    Returns:
        ``1`` if the triangles intersect, ``0`` otherwise.

    Example:

        .. testcode::

            @wp.kernel
            def tri_tri(out: wp.array[wp.int32]):
                a0, a1, a2 = wp.vec3d(0.0, 0.0, 0.0), wp.vec3d(2.0, 0.0, 0.0), wp.vec3d(0.0, 2.0, 0.0)
                b0, b1, b2 = wp.vec3d(1.0, 1.0, -1.0), wp.vec3d(1.0, 1.0, 1.0), wp.vec3d(1.0, -1.0, 0.0)
                out[0] = wp.intersect_tri_tri(a0, a1, a2, b0, b1, b2)

            out = wp.zeros(1, dtype=wp.int32)
            wp.launch(tri_tri, dim=1, inputs=[out])
            print("intersect:", out.numpy()[0])

        .. testoutput::

            intersect: 1""",
    export=False,
    is_differentiable=False,
)


add_builtin(
    "mesh_get",
    input_types={"id": uint64},
    value_type=Mesh,
    is_differentiable=False,
    group="Geometry",
    doc="""Retrieve the :class:`warp.Mesh` object identified by ``id``.

    Example:

        .. testcode::

            @wp.kernel
            def first_face_vertex(mesh_id: wp.uint64, out: wp.array[wp.vec3]):
                m = wp.mesh_get(mesh_id)
                # the returned struct exposes the mesh arrays (points, indices, velocities)
                out[0] = m.points[m.indices[0]]  # position of the first face's first vertex

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out = wp.zeros(1, dtype=wp.vec3)
            wp.launch(first_face_vertex, dim=1, inputs=[mesh.id], outputs=[out])
            print(out.numpy()[0])

        .. testoutput::

            [0. 0. 0.]""",
    export=False,
)

add_builtin(
    "mesh_eval_face_normal",
    input_types={"id": uint64, "face": int},
    value_type=vec3,
    group="Geometry",
    doc="""Evaluate the unit normal of a face of the :class:`warp.Mesh`.

    Returns the face's geometric normal, ``normalize(cross(v1 - v0, v2 - v0))`` for the face's
    vertices ``v0``, ``v1``, ``v2``, in the mesh's local space. Orientation follows the face's
    winding order.

    Args:
        id: The mesh identifier
        face: The face (triangle) index

    Returns:
        The unit-length face normal, in the mesh's local space.

    Example:

        .. testcode::

            @wp.kernel
            def face0_normal(mesh_id: wp.uint64, out: wp.array[wp.vec3]):
                out[0] = wp.mesh_eval_face_normal(mesh_id, 0)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out = wp.zeros(1, dtype=wp.vec3)
            wp.launch(face0_normal, dim=1, inputs=[mesh.id], outputs=[out])
            print(out.numpy()[0])

        .. testoutput::

            [ 0.  0. -1.]""",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "mesh_get_point",
    input_types={"id": uint64, "index": int},
    value_type=vec3,
    group="Geometry",
    doc="""Look up the position of a face's vertex in the :class:`warp.Mesh`.

    ``index`` is a *face-vertex index*: a position in the mesh's index buffer, in
    ``[0, 3 * number_of_faces)``, where positions ``3*f``, ``3*f + 1``, ``3*f + 2`` belong to face
    ``f``. Returns the position of the vertex referenced there, i.e. ``points[indices[index]]``, in
    the mesh's local space. Use :func:`mesh_get_index` to obtain that vertex index itself.

    Args:
        id: The mesh identifier
        index: A face-vertex index, in ``[0, 3 * number_of_faces)``

    Returns:
        The referenced vertex's position, in the mesh's local space.

    Example:

        .. testcode::

            @wp.kernel
            def slot0(mesh_id: wp.uint64, out: wp.array[wp.vec3]):
                out[0] = wp.mesh_get_point(mesh_id, 0)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out = wp.zeros(1, dtype=wp.vec3)
            wp.launch(slot0, dim=1, inputs=[mesh.id], outputs=[out])
            print(out.numpy()[0])

        .. testoutput::

            [0. 0. 0.]""",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "mesh_get_velocity",
    input_types={"id": uint64, "index": int},
    value_type=vec3,
    group="Geometry",
    doc="""Look up the velocity of a face's vertex in the :class:`warp.Mesh`.

    Like :func:`mesh_get_point`, ``index`` is a *face-vertex index* in ``[0, 3 * number_of_faces)``;
    returns the velocity of the vertex referenced there, i.e. ``velocities[indices[index]]``.
    Requires the mesh to have been constructed with a ``velocities`` array; returns a zero vector
    otherwise.

    Args:
        id: The mesh identifier
        index: A face-vertex index, in ``[0, 3 * number_of_faces)``

    Returns:
        The referenced vertex's velocity, or a zero vector if the mesh has no velocities.

    Example:

        .. testcode::

            @wp.kernel
            def slot0(mesh_id: wp.uint64, out: wp.array[wp.vec3]):
                out[0] = wp.mesh_get_velocity(mesh_id, 0)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            velocities = wp.array(np.tile([0.0, 0.0, 1.0], (8, 1)), dtype=wp.vec3)
            mesh = wp.Mesh(points=points, indices=indices, velocities=velocities)

            out = wp.zeros(1, dtype=wp.vec3)
            wp.launch(slot0, dim=1, inputs=[mesh.id], outputs=[out])
            print(out.numpy()[0])

        .. testoutput::

            [0. 0. 1.]""",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "mesh_get_index",
    input_types={"id": uint64, "index": int},
    value_type=int,
    group="Geometry",
    doc="""Look up the vertex index stored at a face-vertex position in the :class:`warp.Mesh`'s index buffer.

    ``index`` is a *face-vertex index* in ``[0, 3 * number_of_faces)``; returns the vertex index it
    stores (``indices[index]``), which in turn indexes the mesh's points array.

    Args:
        id: The mesh identifier
        index: A face-vertex index, in ``[0, 3 * number_of_faces)``

    Returns:
        The vertex index stored at that position, or -1 if the mesh has no index buffer.

    Example:

        .. testcode::

            @wp.kernel
            def face0_verts(mesh_id: wp.uint64, out: wp.array[wp.int32]):
                out[0] = wp.mesh_get_index(mesh_id, 0)
                out[1] = wp.mesh_get_index(mesh_id, 1)
                out[2] = wp.mesh_get_index(mesh_id, 2)

            points = wp.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=wp.vec3)
            indices = wp.array([0,3,2, 0,2,1,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                                2,3,7, 2,7,6,  0,4,7, 0,7,3,  1,2,6, 1,6,5], dtype=wp.int32)
            mesh = wp.Mesh(points=points, indices=indices)

            out = wp.zeros(3, dtype=wp.int32)
            wp.launch(face0_verts, dim=1, inputs=[mesh.id], outputs=[out])
            print(out.numpy())

        .. testoutput::

            [0 3 2]""",
    export=False,
    is_differentiable=False,
)


add_builtin(
    "closest_point_edge_edge",
    input_types={"p1": vec3, "q1": vec3, "p2": vec3, "q2": vec3, "epsilon": float},
    value_type=vec3,
    group="Geometry",
    doc="""Find the closest points between two edges (line segments) ``[p1, q1]`` and ``[p2, q2]``.

    All four endpoints must be in the same coordinate space.

    Args:
        p1: Start point of the first edge
        q1: End point of the first edge
        p2: Start point of the second edge
        q2: End point of the second edge
        epsilon: An edge whose squared length is ``<= epsilon`` is treated as a single point
            (degenerate); its barycentric weight is then returned as 0.

    Returns:
        A ``vec3`` ``(s, t, d)``: ``s`` in [0, 1] is the barycentric weight of the closest point on
        the first edge (the point is ``p1 + s * (q1 - p1)``), ``t`` in [0, 1] is the barycentric
        weight on the second edge (``p2 + t * (q2 - p2)``), and ``d`` is the distance between those
        two closest points.

    Example:

        .. testcode::

            @wp.kernel
            def edge_edge(out: wp.array[wp.vec3]):
                p1, q1 = wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0)
                p2, q2 = wp.vec3(0.5, 1.0, 1.0), wp.vec3(0.5, 1.0, -1.0)
                out[0] = wp.closest_point_edge_edge(p1, q1, p2, q2, 1.0e-6)

            out = wp.zeros(1, dtype=wp.vec3)
            wp.launch(edge_edge, dim=1, inputs=[out])
            s, t, d = out.numpy()[0]
            print(f"s={s:.1f} t={t:.1f} d={d:.1f}")

        .. testoutput::

            s=0.5 t=0.5 d=1.0""",
    export=False,
)

# ---------------------------------
# Ranges

add_builtin(
    "range",
    input_types={"end": int},
    value_type=range_t,
    group="Utility",
    export=False,
    hidden=True,
    is_differentiable=False,
)
add_builtin(
    "range",
    input_types={"start": int, "end": int},
    value_type=range_t,
    group="Utility",
    export=False,
    hidden=True,
    is_differentiable=False,
)
add_builtin(
    "range",
    input_types={"start": int, "end": int, "step": int},
    value_type=range_t,
    group="Utility",
    export=False,
    hidden=True,
    is_differentiable=False,
)

# ---------------------------------
# Iterators

add_builtin(
    "iter_next",
    input_types={"range": range_t},
    value_type=int,
    group="Utility",
    export=False,
    hidden=True,
    is_differentiable=False,
)
for query_type in (
    hash_grid_query_type(float16),
    hash_grid_query_type(float32),
    hash_grid_query_type(float64),
):
    add_builtin(
        "iter_next",
        input_types={"query": query_type},
        value_type=int,
        group="Utility",
        export=False,
        hidden=True,
        is_differentiable=False,
    )
add_builtin(
    "iter_next",
    input_types={"query": MeshQueryAABB},
    value_type=int,
    group="Utility",
    export=False,
    hidden=True,
    is_differentiable=False,
)

add_builtin(
    "reversed",
    input_types={"range": range_t},
    value_type=range_t,
    native_func="iter_reverse",
    group="Utility",
    doc="""Create the range in reversed order.""",
    export=False,
    hidden=True,
    is_differentiable=False,
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
    for value_type in _volume_supported_value_types:
        if types_equal(value_type, dtype):
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
    doc="""Sample the volume of type ``dtype`` given by ``id`` at the volume local-space point ``uvw``.

    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`warp.Volume.LINEAR`.""",
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

    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`warp.Volume.LINEAR`.""",
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
    doc="""Query the value of voxel with coordinates ``i``, ``j``, ``k`` for a volume of type ``dtype``.

    If the voxel at this index does not exist, this function returns the background value.""",
    is_differentiable=False,
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
    is_differentiable=False,
)

add_builtin(
    "volume_sample_f",
    input_types={"id": uint64, "uvw": vec3, "sampling_mode": int},
    value_type=float,
    group="Volumes",
    doc="""Sample the volume given by ``id`` at the volume local-space point ``uvw``.

    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`warp.Volume.LINEAR`.""",
)

add_builtin(
    "volume_sample_grad_f",
    input_types={"id": uint64, "uvw": vec3, "sampling_mode": int, "grad": vec3},
    value_type=float,
    group="Volumes",
    doc="""Sample the volume and its gradient given by ``id`` at the volume local-space point ``uvw``.

    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`warp.Volume.LINEAR`.""",
)

add_builtin(
    "volume_lookup_f",
    input_types={"id": uint64, "i": int, "j": int, "k": int},
    value_type=float,
    group="Volumes",
    doc="""Query the value of voxel with coordinates ``i``, ``j``, ``k``.

    If the voxel at this index does not exist, this function returns the background value.""",
    is_differentiable=False,
)

add_builtin(
    "volume_store_f",
    input_types={"id": uint64, "i": int, "j": int, "k": int, "value": float},
    group="Volumes",
    doc="""Store ``value`` at the voxel with coordinates ``i``, ``j``, ``k``.""",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "volume_sample_v",
    input_types={"id": uint64, "uvw": vec3, "sampling_mode": int},
    value_type=vec3,
    group="Volumes",
    doc="""Sample the vector volume given by ``id`` at the volume local-space point ``uvw``.

    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`warp.Volume.LINEAR`.""",
)

add_builtin(
    "volume_lookup_v",
    input_types={"id": uint64, "i": int, "j": int, "k": int},
    value_type=vec3,
    group="Volumes",
    doc="""Query the vector value of voxel with coordinates ``i``, ``j``, ``k``.

    If the voxel at this index does not exist, this function returns the background value.""",
    is_differentiable=False,
)

add_builtin(
    "volume_store_v",
    input_types={"id": uint64, "i": int, "j": int, "k": int, "value": vec3},
    group="Volumes",
    doc="""Store ``value`` at the voxel with coordinates ``i``, ``j``, ``k``.""",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "volume_sample_i",
    input_types={"id": uint64, "uvw": vec3},
    value_type=int,
    group="Volumes",
    doc="""Sample the :class:`warp.int32` volume given by ``id`` at the volume local-space point ``uvw``.""",
    is_differentiable=False,
)

add_builtin(
    "volume_lookup_i",
    input_types={"id": uint64, "i": int, "j": int, "k": int},
    value_type=int,
    group="Volumes",
    doc="""Query the :class:`warp.int32` value of voxel with coordinates ``i``, ``j``, ``k``.

    If the voxel at this index does not exist, this function returns the background value.""",
    is_differentiable=False,
)

add_builtin(
    "volume_store_i",
    input_types={"id": uint64, "i": int, "j": int, "k": int, "value": int},
    group="Volumes",
    doc="""Store ``value`` at the voxel with coordinates ``i``, ``j``, ``k``.""",
    export=False,
    is_differentiable=False,
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

    Values for allocated voxels are read from the ``voxel_data`` array, and ``background`` is used as the value of non-existing voxels.
    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`warp.Volume.LINEAR`.
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

    Values for allocated voxels are read from the ``voxel_data`` array, and ``background`` is used as the value of non-existing voxels.
    Interpolation should be :attr:`warp.Volume.CLOSEST` or :attr:`warp.Volume.LINEAR`.
    This function is available for both index grids and classical volumes.
   """,
)

add_builtin(
    "volume_lookup_index",
    input_types={"id": uint64, "i": int, "j": int, "k": int},
    value_type=int32,
    group="Volumes",
    doc="""Query the index associated with the voxel at coordinates ``i``, ``j``, ``k``.

    If the voxel at this index does not exist, this function returns -1.
    This function is available for both index grids and classical volumes.
    """,
    is_differentiable=False,
)

add_builtin(
    "volume_voxel_count",
    input_types={"id": uint64},
    value_type=int32,
    group="Volumes",
    doc="""Return the number of indexable voxels in the volume given by ``id``.

    For active-voxel index grids, this is the active voxel count. For dense tile grids, this is the number of
    allocated leaf nodes multiplied by 512. The result is a 32-bit signed integer and is capped at ``2**31 - 1``
    because voxel-index APIs use 32-bit indices.
    """,
    is_differentiable=False,
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

# fp64 overloads for volume transform functions
add_builtin(
    "volume_index_to_world",
    input_types={"id": uint64, "uvw": vec3d},
    value_type=vec3d,
    group="Volumes",
    doc="""Transform a point ``uvw`` defined in volume index space to world space, using double precision.""",
)
add_builtin(
    "volume_world_to_index",
    input_types={"id": uint64, "xyz": vec3d},
    value_type=vec3d,
    group="Volumes",
    doc="""Transform a point ``xyz`` defined in volume world space to index space, using double precision.""",
)
add_builtin(
    "volume_index_to_world_dir",
    input_types={"id": uint64, "uvw": vec3d},
    value_type=vec3d,
    group="Volumes",
    doc="""Transform a direction ``uvw`` defined in volume index space to world space, using double precision.""",
)
add_builtin(
    "volume_world_to_index_dir",
    input_types={"id": uint64, "xyz": vec3d},
    value_type=vec3d,
    group="Volumes",
    doc="""Transform a direction ``xyz`` defined in volume world space to index space, using double precision.""",
)


# ---------------------------------
# Textures

_texture_supported_types = {float, vec2f, vec4f}


def _is_texture_type_supported(dtype):
    return dtype in _texture_supported_types


def _check_texture_type_is_supported(dtype):
    if not _is_texture_type_supported(dtype):
        raise RuntimeError(f"unsupported texture type `{type_repr(dtype)}`. Supported types: float, vec2f, vec4f")


def texture_sample_1d_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Any

    dtype = arg_values["dtype"]
    _check_texture_type_is_supported(dtype)

    return dtype


def texture_sample_1d_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    dtype = args["dtype"]

    func_args = tuple(v for k, v in args.items() if k != "dtype")
    template_args = (dtype,)
    return (func_args, template_args)


# texture_sample for 1D textures with scalar coordinate
add_builtin(
    "texture_sample",
    input_types={"tex": Texture1D, "u": float, "dtype": Any, "lod": float},
    defaults={"lod": -1.0},
    value_func=texture_sample_1d_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=texture_sample_1d_dispatch_func,
    export=False,
    group="Textures",
    doc="""Sample the 1D texture at the given U coordinate.

    .. admonition:: Experimental

        The texture API is experimental and subject to change. See :class:`warp.Texture`.

    Args:
        tex: The 1D texture to sample.
        u: U coordinate. Range is [0, 1] if the texture was created with
            ``normalized_coords=True`` (default), or [0, width] if ``normalized_coords=False``.
        dtype: The return type (``float``, :class:`warp.vec2f`, or :class:`warp.vec4f`).
        lod: Mipmap level-of-detail as a float. When omitted, the base mip level is sampled
            using the non-LOD code path. Fractional values blend between neighbouring mip
            levels when ``mip_filter_mode`` is :attr:`warp.TextureFilterMode.LINEAR`.
            Ignored for textures created with a single mip level.

    Returns:
        The sampled value of the specified ``dtype``.

    Filtering mode is :attr:`warp.TextureFilterMode.CLOSEST` or :attr:`warp.TextureFilterMode.LINEAR`.""",
    is_differentiable=False,
)


def texture_sample_2d_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Any

    dtype = arg_values["dtype"]
    _check_texture_type_is_supported(dtype)

    return dtype


def texture_sample_2d_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    dtype = args["dtype"]

    func_args = tuple(v for k, v in args.items() if k != "dtype")
    template_args = (dtype,)
    return (func_args, template_args)


# texture_sample for 2D textures with vec2 coordinates
add_builtin(
    "texture_sample",
    input_types={"tex": Texture2D, "uv": vec2f, "dtype": Any, "lod": float},
    defaults={"lod": -1.0},
    value_func=texture_sample_2d_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=texture_sample_2d_dispatch_func,
    export=False,
    group="Textures",
    doc="""Sample the 2D texture at the given UV coordinates.

    .. admonition:: Experimental

        The texture API is experimental and subject to change. See :class:`warp.Texture`.

    Args:
        tex: The 2D texture to sample.
        uv: UV coordinates as a :class:`warp.vec2f`. Range is [0, 1] if the texture was created with
            ``normalized_coords=True`` (default), or [0, width] x [0, height] if ``normalized_coords=False``.
        dtype: The return type (``float``, :class:`warp.vec2f`, or :class:`warp.vec4f`).
        lod: Mipmap level-of-detail as a float. When omitted, the base mip level is sampled
            using the non-LOD code path. Fractional values blend between neighbouring mip
            levels when ``mip_filter_mode`` is :attr:`warp.TextureFilterMode.LINEAR`.
            Ignored for textures created with a single mip level.

    Returns:
        The sampled value of the specified ``dtype``.

    Filtering mode is :attr:`warp.TextureFilterMode.CLOSEST` or :attr:`warp.TextureFilterMode.LINEAR`.""",
    is_differentiable=False,
)

# texture_sample for 2D textures with separate u, v coordinates
add_builtin(
    "texture_sample",
    input_types={"tex": Texture2D, "u": float, "v": float, "dtype": Any, "lod": float},
    defaults={"lod": -1.0},
    value_func=texture_sample_2d_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=texture_sample_2d_dispatch_func,
    export=False,
    group="Textures",
    doc="""Sample the 2D texture at the given UV coordinates.

    .. admonition:: Experimental

        The texture API is experimental and subject to change. See :class:`warp.Texture`.

    Args:
        tex: The 2D texture to sample.
        u: U coordinate. Range is [0, 1] if the texture was created with
            ``normalized_coords=True`` (default), or [0, width] if ``normalized_coords=False``.
        v: V coordinate. Range is [0, 1] if the texture was created with
            ``normalized_coords=True`` (default), or [0, height] if ``normalized_coords=False``.
        dtype: The return type (``float``, :class:`warp.vec2f`, or :class:`warp.vec4f`).
        lod: Mipmap level-of-detail as a float. When omitted, the base mip level is sampled
            using the non-LOD code path. Fractional values blend between neighbouring mip
            levels when ``mip_filter_mode`` is :attr:`warp.TextureFilterMode.LINEAR`.
            Ignored for textures created with a single mip level.

    Returns:
        The sampled value of the specified ``dtype``.

    Filtering mode is :attr:`warp.TextureFilterMode.CLOSEST` or :attr:`warp.TextureFilterMode.LINEAR`.""",
    is_differentiable=False,
)


def texture_sample_3d_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Any

    dtype = arg_values["dtype"]
    _check_texture_type_is_supported(dtype)

    return dtype


def texture_sample_3d_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    dtype = args["dtype"]

    func_args = tuple(v for k, v in args.items() if k != "dtype")
    template_args = (dtype,)
    return (func_args, template_args)


# texture_sample for 3D textures with vec3 coordinates
add_builtin(
    "texture_sample",
    input_types={"tex": Texture3D, "uvw": vec3f, "dtype": Any, "lod": float},
    defaults={"lod": -1.0},
    value_func=texture_sample_3d_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=texture_sample_3d_dispatch_func,
    export=False,
    group="Textures",
    doc="""Sample the 3D texture at the given UVW coordinates.

    .. admonition:: Experimental

        The texture API is experimental and subject to change. See :class:`warp.Texture`.

    Args:
        tex: The 3D texture to sample.
        uvw: UVW coordinates as a :class:`warp.vec3f`. Range is [0, 1] if the texture was created with
            ``normalized_coords=True`` (default), or [0, width] x [0, height] x [0, depth] if ``normalized_coords=False``.
        dtype: The return type (``float``, :class:`warp.vec2f`, or :class:`warp.vec4f`).
        lod: Mipmap level-of-detail as a float. When omitted, the base mip level is sampled
            using the non-LOD code path. Fractional values blend between neighbouring mip
            levels when ``mip_filter_mode`` is :attr:`warp.TextureFilterMode.LINEAR`.
            Ignored for textures created with a single mip level.

    Returns:
        The sampled value of the specified ``dtype``.

    Filtering mode is :attr:`warp.TextureFilterMode.CLOSEST` or :attr:`warp.TextureFilterMode.LINEAR`.""",
    is_differentiable=False,
)

# texture_sample for 3D textures with separate u, v, w coordinates
add_builtin(
    "texture_sample",
    input_types={"tex": Texture3D, "u": float, "v": float, "w": float, "dtype": Any, "lod": float},
    defaults={"lod": -1.0},
    value_func=texture_sample_3d_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=texture_sample_3d_dispatch_func,
    export=False,
    group="Textures",
    doc="""Sample the 3D texture at the given UVW coordinates.

    .. admonition:: Experimental

        The texture API is experimental and subject to change. See :class:`warp.Texture`.

    Args:
        tex: The 3D texture to sample.
        u: U coordinate. Range is [0, 1] if the texture was created with
            ``normalized_coords=True`` (default), or [0, width] if ``normalized_coords=False``.
        v: V coordinate. Range is [0, 1] if the texture was created with
            ``normalized_coords=True`` (default), or [0, height] if ``normalized_coords=False``.
        w: W coordinate. Range is [0, 1] if the texture was created with
            ``normalized_coords=True`` (default), or [0, depth] if ``normalized_coords=False``.
        dtype: The return type (``float``, :class:`warp.vec2f`, or :class:`warp.vec4f`).
        lod: Mipmap level-of-detail as a float. When omitted, the base mip level is sampled
            using the non-LOD code path. Fractional values blend between neighbouring mip
            levels when ``mip_filter_mode`` is :attr:`warp.TextureFilterMode.LINEAR`.
            Ignored for textures created with a single mip level.

    Returns:
        The sampled value of the specified ``dtype``.

    Filtering mode is :attr:`warp.TextureFilterMode.CLOSEST` or :attr:`warp.TextureFilterMode.LINEAR`.""",
    is_differentiable=False,
)


# ---------------------------------
# Random

add_builtin(
    "rand_init",
    input_types={"seed": int},
    value_type=uint32,
    group="Random",
    doc="""Initialize a random number generator (RNG) state from a seed.

    Warp's RNG is a stateless PCG hash (Jarzynski & Olano, 2020): ``rand_init``
    turns a seed into a 32-bit ``state`` that is passed to the ``rand*`` and
    ``sample_*`` built-ins. In a kernel, these built-ins advance ``state`` in place,
    so repeated calls on the same local variable return different values. When
    called directly from the Python scope, they do not modify ``state``: calling
    ``wp.randf(state)`` twice with the same ``state`` returns the same value both
    times. Results are deterministic and reproducible for a given seed and call
    order on every device.

    ``state`` is just a local value, not a persistent generator object: it lives
    only for the duration of the kernel invocation and does not carry over between
    launches. Re-launching with the same seed and per-thread offsets reproduces the
    same sequences. To draw different sequences across launches, change the seed or
    include a launch-specific value in the ``rand_init(seed, offset)`` offset. See
    :ref:`Avoiding Correlated Sequences <avoiding_correlated_sequences>` for examples.

    For parallel kernels, prefer the ``rand_init(seed, offset)`` overload with a
    unique per-thread ``offset`` so each thread draws a distinct sequence. See
    the :ref:`Random Number Generation <random_number_generation>` user guide
    section for more details.

    Args:
        seed: Seed value used to derive the initial state.

    Returns:
        A 32-bit unsigned integer holding the initial RNG state.""",
    is_differentiable=False,
)

add_builtin(
    "rand_init",
    input_types={"seed": int, "offset": int},
    value_type=uint32,
    group="Random",
    doc="""Initialize a random number generator (RNG) state from a seed and an offset.

    Both ``seed`` and ``offset`` are hashed into the returned state. This is the
    recommended constructor for parallel kernels: share ``seed`` across the launch
    and pass a unique per-thread ``offset`` (typically ``wp.tid()``) so each thread
    starts from a distinct RNG state and avoids sharing the same sequence:

    .. code-block:: python

        @wp.kernel
        def sample_kernel(seed: int, out: wp.array[float]):
            tid = wp.tid()
            rng = wp.rand_init(seed, tid)
            out[tid] = wp.randf(rng)

    See the :ref:`Random Number Generation <random_number_generation>` user guide
    section for the RNG model and guidance on avoiding correlated sequences.

    Args:
        seed: Seed shared across a kernel launch.
        offset: Per-thread offset selecting a distinct sequence.

    Returns:
        A 32-bit unsigned integer holding the initial RNG state.""",
    is_differentiable=False,
)

add_builtin(
    "randi",
    input_types={"state": uint32},
    value_type=int,
    group="Random",
    doc="""Generate a uniform random 32-bit signed integer in the range [-2^31, 2^31).

    In a kernel, advances ``state`` in place, so successive calls return different
    values; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same value (see :func:`rand_init`).""",
    is_differentiable=False,
)
add_builtin(
    "randi",
    input_types={"state": uint32, "low": int, "high": int},
    value_type=int,
    group="Random",
    doc="""Generate a uniform random integer in the range [low, high).

    In a kernel, advances ``state`` in place, so successive calls return different
    values; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same value (see :func:`rand_init`).
    Requires ``high > low``. Uses modulo reduction, so the distribution is slightly
    biased toward lower values for very large ranges.""",
    is_differentiable=False,
)
add_builtin(
    "randu",
    input_types={"state": uint32},
    value_type=uint32,
    group="Random",
    doc="""Generate a uniform random unsigned 32-bit integer in the range [0, 2^32).

    In a kernel, advances ``state`` in place, so successive calls return different
    values; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same value (see :func:`rand_init`).""",
    is_differentiable=False,
)
add_builtin(
    "randu",
    input_types={"state": uint32, "low": uint32, "high": uint32},
    value_type=uint32,
    group="Random",
    doc="""Generate a uniform random unsigned integer in the range [low, high).

    In a kernel, advances ``state`` in place, so successive calls return different
    values; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same value (see :func:`rand_init`).
    Requires ``high > low``. Uses modulo reduction, so the distribution is slightly
    biased toward lower values for very large ranges.""",
    is_differentiable=False,
)
add_builtin(
    "randf",
    input_types={"state": uint32},
    value_type=float,
    group="Random",
    doc="""Generate a uniform random float in the range [0.0, 1.0).

    In a kernel, advances ``state`` in place, so successive calls return different
    values; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same value (see :func:`rand_init`).
    Values are multiples of 2^-24, matching the 24 bits of precision in a 32-bit float.

    Random built-ins take an RNG ``state`` produced by :func:`rand_init`;
    initialize it once per thread, then draw values:

    .. code-block:: python

        @wp.kernel
        def random_floats(seed: int, out: wp.array[wp.vec2]):
            i = wp.tid()
            rng = wp.rand_init(seed, i)
            # Each call advances rng, so the two draws differ.
            out[i] = wp.vec2(wp.randf(rng), wp.randf(rng))

    :func:`randi`, :func:`randu`, and :func:`randn` are used the same way:
    initialize ``rng`` once with :func:`rand_init`, then pass it to each call.""",
    is_differentiable=False,
)
add_builtin(
    "randf",
    input_types={"state": uint32, "low": float, "high": float},
    value_type=float,
    group="Random",
    doc="""Generate a uniform random float in the range [low, high).

    In a kernel, advances ``state`` in place, so successive calls return different
    values; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same value (see :func:`rand_init`).
    Equivalent to ``low + (high - low) * wp.randf(state)``.""",
    is_differentiable=False,
)
add_builtin(
    "randn",
    input_types={"state": uint32},
    value_type=float,
    group="Random",
    doc="""Sample the standard normal (Gaussian) distribution with mean 0 and variance 1.

    Uses the Box-Muller transform, consuming two uniform draws. In a kernel, this
    advances ``state`` in place; called from the Python scope, it does not modify
    ``state``, so repeated calls with the same ``state`` return the same value
    (see :func:`rand_init`). For a general normal, scale and shift the result:
    ``mean + stddev * wp.randn(state)``.""",
    is_differentiable=False,
)

add_builtin(
    "sample_cdf",
    input_types={"state": uint32, "cdf": array(dtype=float)},
    value_type=int,
    group="Random",
    doc="""Sample a discrete distribution by inverse-transform sampling of a CDF.

    Draws a uniform value ``u`` in [0.0, 1.0) and returns the index of the first
    entry of ``cdf`` greater than or equal to ``u`` via binary search. ``cdf`` must
    be a 1D, monotonically non-decreasing array normalized so its last element is
    1.0 (for example, a cumulative sum of non-negative weights divided by their
    total). The returned index lies in ``[0, len(cdf) - 1]`` and selects the
    sampled bin.

    Unlike the other random built-ins, this function is only callable from within
    kernels, where it advances ``state`` in place (see :func:`rand_init`).

    Build a normalized CDF in Python, then sample it inside a kernel:

    .. code-block:: python

        @wp.kernel
        def sample_bins(seed: int, cdf: wp.array[float], out: wp.array[int]):
            i = wp.tid()
            rng = wp.rand_init(seed, i)
            out[i] = wp.sample_cdf(rng, cdf)  # index in [0, len(cdf) - 1]

        weights = np.array([0.1, 0.4, 0.5], dtype=np.float32)
        cdf = wp.array(np.cumsum(weights) / weights.sum(), dtype=float)
        out = wp.empty(1000, dtype=int)
        wp.launch(sample_bins, dim=out.shape, inputs=[42, cdf], outputs=[out])

    Args:
        state: RNG state, advanced in place (see :func:`rand_init`).
        cdf: Normalized, non-decreasing cumulative distribution (1D float array).

    Returns:
        The sampled index into ``cdf``.""",
    is_differentiable=False,
)
add_builtin(
    "sample_triangle",
    input_types={"state": uint32},
    value_type=vec2,
    group="Random",
    doc="""Uniformly sample a point in a triangle, returned as barycentric coordinates.

    In a kernel, advances ``state`` in place; called from the Python scope, it does
    not modify ``state``, so repeated calls with the same ``state`` return the same
    point (see :func:`rand_init`). The third coordinate is ``w = 1 - u - v``; recover
    a point on a triangle with vertices ``a``, ``b``, ``c`` as ``u*a + v*b + w*c``:

    .. code-block:: python

        @wp.kernel
        def points_in_triangle(
            seed: int, a: wp.vec3, b: wp.vec3, c: wp.vec3, out: wp.array[wp.vec3]
        ):
            i = wp.tid()
            rng = wp.rand_init(seed, i)
            bary = wp.sample_triangle(rng)
            w = 1.0 - bary[0] - bary[1]
            out[i] = bary[0] * a + bary[1] * b + w * c

    Returns:
        The barycentric coordinates ``(u, v)`` of the sampled point.""",
    is_differentiable=False,
)
add_builtin(
    "sample_unit_ring",
    input_types={"state": uint32},
    value_type=vec2,
    group="Random",
    doc="""Uniformly sample a point on the unit circle (radius 1) in the xy-plane.

    Returns a ``vec2`` of unit length. In a kernel, advances ``state`` in
    place; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same point (see :func:`rand_init`).""",
    is_differentiable=False,
)
add_builtin(
    "sample_unit_disk",
    input_types={"state": uint32},
    value_type=vec2,
    group="Random",
    doc="""Uniformly sample a point inside the unit disk (radius <= 1) in the xy-plane.

    Returns a ``vec2``. In a kernel, advances ``state`` in
    place; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same point (see :func:`rand_init`).""",
    is_differentiable=False,
)
add_builtin(
    "sample_unit_sphere_surface",
    input_types={"state": uint32},
    value_type=vec3,
    group="Random",
    doc="""Uniformly sample a point on the surface of the unit sphere (radius 1).

    Returns a ``vec3`` of unit length. In a kernel, advances ``state`` in
    place; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same point (see :func:`rand_init`).""",
    is_differentiable=False,
)
add_builtin(
    "sample_unit_sphere",
    input_types={"state": uint32},
    value_type=vec3,
    group="Random",
    doc="""Uniformly sample a point inside the unit ball (radius <= 1).

    Returns a ``vec3``. In a kernel, advances ``state`` in
    place; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same point (see :func:`rand_init`).

    The other ``sample_unit_*`` helpers are called the same way (initialize ``rng``
    with :func:`rand_init`, then pass it), differing only in the domain they sample:

    .. code-block:: python

        @wp.kernel
        def random_points(seed: int, out: wp.array[wp.vec3]):
            i = wp.tid()
            rng = wp.rand_init(seed, i)
            out[i] = wp.sample_unit_sphere(rng)""",
    is_differentiable=False,
)
add_builtin(
    "sample_unit_hemisphere_surface",
    input_types={"state": uint32},
    value_type=vec3,
    group="Random",
    doc="""Uniformly sample a point on the surface of the unit hemisphere (radius 1, z >= 0).

    Returns a ``vec3`` of unit length. In a kernel, advances ``state`` in
    place; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same point (see :func:`rand_init`).""",
    is_differentiable=False,
)
add_builtin(
    "sample_unit_hemisphere",
    input_types={"state": uint32},
    value_type=vec3,
    group="Random",
    doc="""Uniformly sample a point inside the unit hemisphere (radius <= 1, z >= 0).

    Returns a ``vec3``. In a kernel, advances ``state`` in
    place; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same point (see :func:`rand_init`).""",
    is_differentiable=False,
)
add_builtin(
    "sample_unit_square",
    input_types={"state": uint32},
    value_type=vec2,
    group="Random",
    doc="""Uniformly sample a point in the unit square ``[-0.5, 0.5)^2``, centered at the origin.

    Returns a ``vec2``. In a kernel, advances ``state`` in
    place; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same point (see :func:`rand_init`).""",
    is_differentiable=False,
)
add_builtin(
    "sample_unit_cube",
    input_types={"state": uint32},
    value_type=vec3,
    group="Random",
    doc="""Uniformly sample a point in the unit cube ``[-0.5, 0.5)^3``, centered at the origin.

    Returns a ``vec3``. In a kernel, advances ``state`` in
    place; called from the Python scope, it does not modify ``state``, so repeated
    calls with the same ``state`` return the same point (see :func:`rand_init`).""",
    is_differentiable=False,
)

add_builtin(
    "poisson",
    input_types={"state": uint32, "lam": float},
    value_type=uint32,
    group="Random",
    doc="""Generate a random sample from a Poisson distribution.

    In a kernel, advances ``state`` in place when ``lam > 0`` (and returns ``0``
    without consuming RNG state when ``lam == 0``); called from the Python scope, it
    does not modify ``state``, so repeated calls with the same ``state`` return the
    same count (see :func:`rand_init`). The returned count has mean and variance
    both equal to ``lam``.

    .. code-block:: python

        @wp.kernel
        def counts(seed: int, rate: float, out: wp.array[wp.uint32]):
            i = wp.tid()
            rng = wp.rand_init(seed, i)
            out[i] = wp.poisson(rng, rate)

    Args:
        state: RNG state; advanced in place when called in a kernel (see :func:`rand_init`).
        lam: Expected value (rate) of the distribution; must be non-negative.

    Returns:
        A ``uint32`` sample drawn from ``Poisson(lam)``.""",
    is_differentiable=False,
)

add_builtin(
    "noise",
    input_types={"state": uint32, "x": float},
    value_type=float,
    group="Random",
    doc="""Non-periodic Perlin-style noise.

    Sample 1D noise.""",
)
add_builtin(
    "noise",
    input_types={"state": uint32, "xy": vec2},
    value_type=float,
    group="Random",
    doc="""Non-periodic Perlin-style noise.

    Sample 2D noise.""",
)
add_builtin(
    "noise",
    input_types={"state": uint32, "xyz": vec3},
    value_type=float,
    group="Random",
    doc="""Non-periodic Perlin-style noise.

    Sample 3D noise.""",
)
add_builtin(
    "noise",
    input_types={"state": uint32, "xyzt": vec4},
    value_type=float,
    group="Random",
    doc="""Non-periodic Perlin-style noise.

    Sample 4D noise.""",
)

add_builtin(
    "pnoise",
    input_types={"state": uint32, "x": float, "px": int},
    value_type=float,
    group="Random",
    doc="""Periodic Perlin-style noise.

    Sample 1D noise.""",
)
add_builtin(
    "pnoise",
    input_types={"state": uint32, "xy": vec2, "px": int, "py": int},
    value_type=float,
    group="Random",
    doc="""Periodic Perlin-style noise.

    Sample 2D noise.""",
)
add_builtin(
    "pnoise",
    input_types={"state": uint32, "xyz": vec3, "px": int, "py": int, "pz": int},
    value_type=float,
    group="Random",
    doc="""Periodic Perlin-style noise.

    Sample 3D noise.""",
)
add_builtin(
    "pnoise",
    input_types={"state": uint32, "xyzt": vec4, "px": int, "py": int, "pz": int, "pt": int},
    value_type=float,
    group="Random",
    doc="""Periodic Perlin-style noise.

    Sample 4D noise.""",
)

add_builtin(
    "curlnoise",
    input_types={"state": uint32, "xy": vec2, "octaves": uint32, "lacunarity": float, "gain": float},
    defaults={"octaves": uint32(1), "lacunarity": 2.0, "gain": 0.5},
    value_type=vec2,
    group="Random",
    doc="""Divergence-free vector field based on Perlin noise.

    Use the gradient of a Perlin noise function.""",
)
add_builtin(
    "curlnoise",
    input_types={"state": uint32, "xyz": vec3, "octaves": uint32, "lacunarity": float, "gain": float},
    defaults={"octaves": uint32(1), "lacunarity": 2.0, "gain": 0.5},
    value_type=vec3,
    group="Random",
    doc="""Divergence-free vector field based on Perlin noise.

    Use the curl of three Perlin noise functions.""",
)
add_builtin(
    "curlnoise",
    input_types={"state": uint32, "xyzt": vec4, "octaves": uint32, "lacunarity": float, "gain": float},
    defaults={"octaves": uint32(1), "lacunarity": 2.0, "gain": 0.5},
    value_type=vec3,
    group="Random",
    doc="""Divergence-free vector field based on Perlin noise.

    Use the curl of three Perlin noise functions.""",
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
    is_differentiable=False,
)

add_builtin(
    "print",
    input_types={"value": Any},
    doc="Print a variable to stdout.",
    export=False,
    group="Utility",
)

add_builtin(
    "breakpoint",
    input_types={},
    doc="Trigger a debugger breakpoint.",
    export=False,
    group="Utility",
    namespace="",
    native_func="__debugbreak",
    is_differentiable=False,
)

# helpers
add_builtin(
    "tid",
    input_types={},
    value_type=int,
    export=False,
    group="Utility",
    doc="""Query the current thread index or indices.

    The return type is determined by the unpacking syntax used:

    - ``i = wp.tid()`` - Returns the first thread index as ``int``
    - ``i, j = wp.tid()`` - Returns the first two indices as ``tuple[int, int]``
    - ``i, j, k = wp.tid()`` - Returns the first three indices as ``tuple[int, int, int]``
    - ``i, j, k, l = wp.tid()`` - Returns all four indices as ``tuple[int, int, int, int]``

    The indices correspond to the thread's position in the kernel launch grid.
    If fewer indices are requested than the launch dimensionality, only the
    leading indices are returned.
    For multi-dimensional launches, the linear thread order is unraveled in
    row-major order, with the last launch dimension varying fastest.

    This function may not be called from user-defined Warp functions.""",
    namespace="",
    native_func="builtin_tid1d",
    is_differentiable=False,
)

add_builtin(
    "tid",
    input_types={},
    value_type=[int, int],
    group="Utility",
    doc="""Query the current thread index or indices.

    The return type is determined by the unpacking syntax used:

    - ``i = wp.tid()`` - Returns the first thread index as ``int``
    - ``i, j = wp.tid()`` - Returns the first two indices as ``tuple[int, int]``
    - ``i, j, k = wp.tid()`` - Returns the first three indices as ``tuple[int, int, int]``
    - ``i, j, k, l = wp.tid()`` - Returns all four indices as ``tuple[int, int, int, int]``

    The indices correspond to the thread's position in the kernel launch grid.
    If fewer indices are requested than the launch dimensionality, only the
    leading indices are returned.
    For multi-dimensional launches, the linear thread order is unraveled in
    row-major order, with the last launch dimension varying fastest.

    This function may not be called from user-defined Warp functions.""",
    namespace="",
    native_func="builtin_tid2d",
    is_differentiable=False,
)

add_builtin(
    "tid",
    input_types={},
    value_type=[int, int, int],
    group="Utility",
    doc="""Query the current thread index or indices.

    The return type is determined by the unpacking syntax used:

    - ``i = wp.tid()`` - Returns the first thread index as ``int``
    - ``i, j = wp.tid()`` - Returns the first two indices as ``tuple[int, int]``
    - ``i, j, k = wp.tid()`` - Returns the first three indices as ``tuple[int, int, int]``
    - ``i, j, k, l = wp.tid()`` - Returns all four indices as ``tuple[int, int, int, int]``

    The indices correspond to the thread's position in the kernel launch grid.
    If fewer indices are requested than the launch dimensionality, only the
    leading indices are returned.
    For multi-dimensional launches, the linear thread order is unraveled in
    row-major order, with the last launch dimension varying fastest.

    This function may not be called from user-defined Warp functions.""",
    namespace="",
    native_func="builtin_tid3d",
    is_differentiable=False,
)

add_builtin(
    "tid",
    input_types={},
    value_type=[int, int, int, int],
    group="Utility",
    doc="""Query the current thread index or indices.

    The return type is determined by the unpacking syntax used:

    - ``i = wp.tid()`` - Returns the first thread index as ``int``
    - ``i, j = wp.tid()`` - Returns the first two indices as ``tuple[int, int]``
    - ``i, j, k = wp.tid()`` - Returns the first three indices as ``tuple[int, int, int]``
    - ``i, j, k, l = wp.tid()`` - Returns all four indices as ``tuple[int, int, int, int]``

    The indices correspond to the thread's position in the kernel launch grid.
    If fewer indices are requested than the launch dimensionality, only the
    leading indices are returned.
    For multi-dimensional launches, the linear thread order is unraveled in
    row-major order, with the last launch dimension varying fastest.

    This function may not be called from user-defined Warp functions.""",
    namespace="",
    native_func="builtin_tid4d",
    is_differentiable=False,
)

add_builtin(
    "block_dim",
    input_types={},
    value_type=int,
    group="Utility",
    doc="Query the number of threads in the current block.",
    namespace="",
    native_func="builtin_block_dim",
    is_differentiable=False,
)


def copy_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    a = arg_types["a"]

    # if the input is a shared tile, we force a copy
    if is_tile(a) and a.storage == "shared":
        return tile(
            dtype=a.dtype,
            shape=a.shape,
            storage=a.storage,
            strides=a.strides,
            layout=a.layout,
            owner=True,
        )

    return a


add_builtin(
    "copy",
    input_types={"a": Any},
    value_func=copy_value_func,
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


def select_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Any

    raise RuntimeError("wp.select() has been removed. Use wp.where(cond, value_if_true, value_if_false) instead.")


add_builtin(
    "select",
    input_types={"cond": builtins.bool, "value_if_false": Any, "value_if_true": Any},
    value_func=select_value_func,
    doc="""Select between two arguments, if ``cond`` is ``False`` then return ``value_if_false``, otherwise return ``value_if_true``.

    .. versionremoved:: 1.10
            Use :func:`where` instead, which has the more intuitive argument order:
            ``where(cond, value_if_true, value_if_false)``.

    .. deprecated:: 1.7""",
    group="Utility",
    is_differentiable=False,
)
for t in int_types:
    add_builtin(
        "select",
        input_types={"cond": t, "value_if_false": Any, "value_if_true": Any},
        value_func=select_value_func,
        doc="""Select between two arguments, if ``cond`` is ``False`` then return ``value_if_false``, otherwise return ``value_if_true``.

    .. versionremoved:: 1.10
            Use :func:`where` instead, which has the more intuitive argument order:
            ``where(cond, value_if_true, value_if_false)``.

    .. deprecated:: 1.7""",
        group="Utility",
        is_differentiable=False,
    )
add_builtin(
    "select",
    input_types={"arr": array(dtype=Any), "value_if_false": Any, "value_if_true": Any},
    value_func=select_value_func,
    doc="""Select between two arguments, if ``arr`` is null then return ``value_if_false``, otherwise return ``value_if_true``.

    .. versionremoved:: 1.10
            Use :func:`where` instead, which has the more intuitive argument order:
            ``where(arr, value_if_true, value_if_false)``.

    .. deprecated:: 1.7""",
    group="Utility",
    is_differentiable=False,
)


def where_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    if arg_types is None:
        return Any

    v_true = arg_types["value_if_true"]
    v_false = arg_types["value_if_false"]

    if not types_equal(v_true, v_false):
        raise RuntimeError(f"where() true value type ({v_true}) must be of the same type as the false type ({v_false})")

    if is_tile(v_false):
        if v_true.storage == "register":
            return v_true
        if v_false.storage == "register":
            return v_false

        # both v_true and v_false are shared
        return tile(
            dtype=v_true.dtype,
            shape=v_true.shape,
            storage=v_true.storage,
            strides=v_true.strides,
            layout=v_true.layout,
            owner=True,
        )

    return v_true


add_builtin(
    "where",
    input_types={"cond": builtins.bool, "value_if_true": Any, "value_if_false": Any},
    value_func=where_value_func,
    doc="Select between two arguments, if ``cond`` is ``True`` then return ``value_if_true``, otherwise return ``value_if_false``.",
    group="Utility",
)
for t in int_types:
    add_builtin(
        "where",
        input_types={"cond": t, "value_if_true": Any, "value_if_false": Any},
        value_func=where_value_func,
        doc="Select between two arguments, if ``cond`` is ``True`` then return ``value_if_true``, otherwise return ``value_if_false``.",
        group="Utility",
    )
add_builtin(
    "where",
    input_types={"arr": array(dtype=Any), "value_if_true": Any, "value_if_false": Any},
    value_func=where_value_func,
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
    input_types={"ptr": warp.uint64, "shape": tuple[int, ...], "dtype": Any},
    value_func=array_value_func,
    export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
    dispatch_func=array_dispatch_func,
    native_func="array_t",
    doc="Construct an array from a memory pointer, shape, and data type.",
    group="Utility",
    export=False,
    is_differentiable=False,
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
    input_types={"shape": tuple[int, ...], "dtype": Any},
    value_func=zeros_value_func,
    export_func=lambda input_types: {},
    dispatch_func=zeros_dispatch_func,
    native_func="fixedarray_t",
    doc="Create a zero-initialized fixed-size array of the given shape and dtype.",
    group="Utility",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "zeros",
    input_types={"shape": int, "dtype": Any},
    value_func=zeros_value_func,
    export_func=lambda input_types: {},
    dispatch_func=zeros_dispatch_func,
    native_func="fixedarray_t",
    doc="Create a zero-initialized fixed-size array of the given length and dtype.",
    group="Utility",
    export=False,
    is_differentiable=False,
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
    idx_types = tuple(arg_types[x] for x in "ijkl" if arg_types.get(x, None) is not None)

    if not is_array(arr_type):
        raise RuntimeError("view() first argument must be an array")

    idx_count = len(idx_types)
    if idx_count > arr_type.ndim:
        raise RuntimeError(
            f"Trying to create an array view with {idx_count} indices, "
            f"but the array only has {arr_type.ndim} dimension(s). "
            f"Ensure that the argument type on the function or kernel specifies "
            f"the expected number of dimensions, e.g.: def func(param: wp.array3d[float]): ..."
        )

    has_slice = any(is_slice(x) for x in idx_types)
    if has_slice:
        # check index types
        for t in idx_types:
            if not (type_is_int(t) or is_slice(t)):
                raise RuntimeError(
                    f"view() index arguments must be of integer or slice types, got index of type {type_repr(t)}"
                )

        # Each integer index collapses one dimension.
        int_count = sum(x.step == 0 for x in idx_types)
        ndim = arr_type.ndim - int_count
        assert ndim > 0
    else:
        if idx_count == arr_type.ndim:
            raise RuntimeError("Expected to call `address()` instead of `view()`")

        # check index types
        for t in idx_types:
            if not type_is_int(t):
                raise RuntimeError(
                    f"view() index arguments must be of integer or slice types, got index of type {type_repr(t)}"
                )

        # create an array view with leading dimensions removed
        ndim = arr_type.ndim - idx_count
        assert ndim > 0

    dtype = arr_type.dtype
    if (
        matches_array_class(arr_type, fabricarray)
        or matches_array_class(arr_type, indexedfabricarray)
        or isinstance(arr_type, fixedarray)
    ):
        # fabric and fixed arrays: return array attribute as a regular array
        return array(dtype=dtype, ndim=ndim)

    return type(arr_type)(dtype=dtype, ndim=ndim)


for array_type in array_types:
    add_builtin(
        "view",
        input_types={
            "arr": array_type(dtype=Any),
            "i": Any,
            "j": Any,
            "k": Any,
            "l": Any,
        },
        defaults={
            "j": None,
            "k": None,
            "l": None,
        },
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
    is_differentiable=False,
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
    is_differentiable=False,
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
            supported_atomic_types = (*SUPPORTED_ATOMIC_TYPES, warp.float16, warp.bfloat16)
            if not any(types_equal_generic(scalar_type, x) for x in supported_atomic_types):
                raise RuntimeError(
                    f"atomic_{op}() operations only work on arrays with [u]int32, [u]int64, float16, bfloat16, float32, or float64 "
                    f"as the underlying scalar types, but got {type_repr(arr_type.dtype)} (with scalar type {type_repr(scalar_type)})"
                )
        elif op in ("min", "max"):
            supported_atomic_types = (*SUPPORTED_ATOMIC_TYPES, warp.bfloat16)
            if not any(types_equal_generic(scalar_type, x) for x in supported_atomic_types):
                raise RuntimeError(
                    f"atomic_{op}() operations only work on arrays with [u]int32, [u]int64, bfloat16, float32, or float64 "
                    f"as the underlying scalar types, but got {type_repr(arr_type.dtype)} (with scalar type {type_repr(scalar_type)})"
                )
        elif op in ("cas", "exch"):
            if not any(types_equal_generic(scalar_type, x) for x in SUPPORTED_ATOMIC_TYPES):
                raise RuntimeError(
                    f"atomic_{op}() operations only work on arrays with [u]int32, [u]int64, float32, or float64 "
                    f"as the underlying scalar types, but got {type_repr(arr_type.dtype)} (with scalar type {type_repr(scalar_type)})"
                )
        elif op in ("and", "or", "xor"):
            supported_atomic_types = (warp.int32, warp.int64, warp.uint32, warp.uint64)
            if not any(types_equal_generic(scalar_type, x) for x in supported_atomic_types):
                raise RuntimeError(
                    f"atomic_{op}() operations only work on arrays with [u]int32 or [u]int64 "
                    f"as the underlying scalar types, but got {type_repr(arr_type.dtype)} (with scalar type {type_repr(scalar_type)})"
                )
        else:
            raise NotImplementedError

        return arr_type.dtype

    return fn


def atomic_op_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # as this is a codegen callback, we can mark the fact that this func writes to an array here
    if warp._src.codegen.options.get("verify_autograd_array_access", False):
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
        is_differentiable=False,
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
        is_differentiable=False,
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
        is_differentiable=False,
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
        is_differentiable=False,
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
        is_differentiable=False,
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
        is_differentiable=False,
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
        is_differentiable=False,
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
        is_differentiable=False,
    )

    add_builtin(
        "atomic_and",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("and"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise AND between ``value`` and ``arr[i]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i] &= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
    )
    add_builtin(
        "atomic_and",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("and"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise AND between ``value`` and ``arr[i,j]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i,j] &= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
    )
    add_builtin(
        "atomic_and",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("and"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise AND between ``value`` and ``arr[i,j,k]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i,j,k] &= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
    )
    add_builtin(
        "atomic_and",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "l": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("and"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise AND between ``value`` and ``arr[i,j,k,l]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i,j,k,l] &= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
    )

    add_builtin(
        "atomic_or",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("or"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise OR between ``value`` and ``arr[i]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i] |= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
    )
    add_builtin(
        "atomic_or",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("or"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise OR between ``value`` and ``arr[i,j]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i,j] |= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
    )
    add_builtin(
        "atomic_or",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("or"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise OR between ``value`` and ``arr[i,j,k]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i,j,k] |= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
    )
    add_builtin(
        "atomic_or",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "l": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("or"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise OR between ``value`` and ``arr[i,j,k,l]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i,j,k,l] |= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
    )

    add_builtin(
        "atomic_xor",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("xor"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise XOR between ``value`` and ``arr[i]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i] ^= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
    )
    add_builtin(
        "atomic_xor",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("xor"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise XOR between ``value`` and ``arr[i,j]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i,j] ^= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
    )
    add_builtin(
        "atomic_xor",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("xor"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise XOR between ``value`` and ``arr[i,j,k]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i,j,k] ^= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
    )
    add_builtin(
        "atomic_xor",
        hidden=hidden,
        input_types={"arr": array_type(dtype=Any), "i": Int, "j": Int, "k": Int, "l": Int, "value": Any},
        constraint=atomic_op_constraint,
        value_func=create_atomic_op_value_func("xor"),
        dispatch_func=atomic_op_dispatch_func,
        doc="""Atomically performs a bitwise XOR between ``value`` and ``arr[i,j,k,l]``, atomically update the array, and return the old value.

        This function is automatically invoked when using the syntax ``arr[i,j,k,l] ^= value``.""",
        group="Utility",
        skip_replay=True,
        is_differentiable=False,
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
# Bool vector extract (bool is not part of Scalar)
add_builtin(
    "extract",
    input_types={"a": vector(length=Any, dtype=bool), "i": Any},
    value_func=vector_extract_value_func,
    dispatch_func=vector_extract_dispatch_func,
    export=False,
    hidden=True,
    group="Utility",
)
add_builtin(
    "extract",
    input_types={"a": quaternion(dtype=Float), "i": Any},
    value_func=vector_extract_value_func,
    dispatch_func=vector_extract_dispatch_func,
    export=False,
    hidden=True,
    group="Utility",
)
add_builtin(
    "extract",
    input_types={"a": transformation(dtype=Float), "i": Any},
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
# Bool matrix extract (bool is not part of Scalar)
add_builtin(
    "extract",
    input_types={"a": matrix(shape=(Any, Any), dtype=bool), "i": Any},
    value_func=matrix_extract_value_func,
    dispatch_func=matrix_extract_dispatch_func,
    export=False,
    hidden=True,
    group="Utility",
)
add_builtin(
    "extract",
    input_types={"a": matrix(shape=(Any, Any), dtype=bool), "i": Any, "j": Any},
    value_func=matrix_extract_value_func,
    dispatch_func=matrix_extract_dispatch_func,
    export=False,
    hidden=True,
    group="Utility",
)

add_builtin(
    "extract",
    input_types={"s": shape_t, "i": int},
    value_type=int,
    hidden=True,
    group="Utility",
    is_differentiable=False,
)


def vector_index_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    vec_type = arg_types["a"]
    value_type = vec_type._wp_scalar_type_

    return Reference(value_type)


def vector_index_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (Reference(args["a"]), args["i"])
    template_args = ()
    return (func_args, template_args)


def matrix_ij_value_func(arg_types: Mapping[str, type], arg_values: Mapping[str, Any]):
    mat_type = arg_types["a"]
    value_type = mat_type._wp_scalar_type_

    return Reference(value_type)


def matrix_ij_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (Reference(args["a"]), args["i"], args["j"])
    template_args = ()
    return (func_args, template_args)


# implements &vector[index]
add_builtin(
    "index",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": Int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
    is_differentiable=False,
)
# implements &bool_vector[index] (bool is not part of Scalar)
add_builtin(
    "index",
    input_types={"a": vector(length=Any, dtype=bool), "i": Int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
    is_differentiable=False,
)
# implements &quaternion[index]
add_builtin(
    "index",
    input_types={"a": quaternion(dtype=Float), "i": Int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
    is_differentiable=False,
)
# implements &transformation[index]
add_builtin(
    "index",
    input_types={"a": transformation(dtype=Float), "i": Int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
    is_differentiable=False,
)
# implements &(*vector)[index]
add_builtin(
    "indexref",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": Int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
    is_differentiable=False,
)
# implements &(*bool_vector)[index] (bool is not part of Scalar)
add_builtin(
    "indexref",
    input_types={"a": vector(length=Any, dtype=bool), "i": Int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
    is_differentiable=False,
)
# implements &(*matrix)[i, j]
add_builtin(
    "indexref",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Int, "j": Int},
    value_func=matrix_ij_value_func,
    dispatch_func=matrix_ij_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
    is_differentiable=False,
)
# implements &(*bool_matrix)[i, j] (bool is not part of Scalar)
add_builtin(
    "indexref",
    input_types={"a": matrix(shape=(Any, Any), dtype=bool), "i": Int, "j": Int},
    value_func=matrix_ij_value_func,
    dispatch_func=matrix_ij_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
    is_differentiable=False,
)
# implements &(*quaternion)[index]
add_builtin(
    "indexref",
    input_types={"a": quaternion(dtype=Float), "i": Int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
    is_differentiable=False,
)
# implements &(*transformation)[index]
add_builtin(
    "indexref",
    input_types={"a": transformation(dtype=Float), "i": Int},
    value_func=vector_index_value_func,
    dispatch_func=vector_index_dispatch_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
    is_differentiable=False,
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
            # Disallow broadcasting.
            raise ValueError(
                f"The provided value is expected to be a vector of length {length}, with dtype {type_repr(vec._wp_scalar_type_)}."
            )
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

# Bool vector assign_inplace (bool is not part of Scalar)
add_builtin(
    "assign_inplace",
    input_types={"a": vector(length=Any, dtype=bool), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements quaternion[index] = value
add_builtin(
    "assign_inplace",
    input_types={"a": quaternion(dtype=Float), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)
# implements transformation[index] = value
add_builtin(
    "assign_inplace",
    input_types={"a": transformation(dtype=Float), "i": Any, "value": Any},
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

# Bool vector assign_copy (bool is not part of Scalar)
add_builtin(
    "assign_copy",
    input_types={"a": vector(length=Any, dtype=bool), "i": Any, "value": Any},
    value_func=vector_assign_copy_value_func,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements quaternion[index] = value, performs a copy internally if wp.config.enable_vector_component_overwrites is True
add_builtin(
    "assign_copy",
    input_types={"a": quaternion(dtype=Float), "i": Any, "value": Any},
    value_func=vector_assign_copy_value_func,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
)

# implements transformation[index] = value, performs a copy internally if wp.config.enable_vector_component_overwrites is True
add_builtin(
    "assign_copy",
    input_types={"a": transformation(dtype=Float), "i": Any, "value": Any},
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
    input_types={"a": quaternion(dtype=Float), "i": Any, "value": Any},
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
    input_types={"a": quaternion(dtype=Float), "i": Any, "value": Any},
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


# implements vector[idx] &= scalar
add_builtin(
    "bit_and_inplace",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
    is_differentiable=False,
)

# implements vector[idx] |= scalar
add_builtin(
    "bit_or_inplace",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
    is_differentiable=False,
)

# implements vector[idx] ^= scalar
add_builtin(
    "bit_xor_inplace",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    dispatch_func=vector_assign_dispatch_func,
    hidden=True,
    export=False,
    group="Utility",
    is_differentiable=False,
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
    is_differentiable=False,
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
    is_differentiable=False,
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
                # Disallow broadcasting.
                raise ValueError(
                    f"The provided value is expected to be a vector of length {length}, with dtype {type_repr(mat._wp_scalar_type_)}."
                )
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
                # Disallow broadcasting.
                raise ValueError(
                    f"The provided value is expected to be a matrix of shape {shape}, with dtype {type_repr(mat._wp_scalar_type_)}."
                )
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


# implements matrix[i] &= value
add_builtin(
    "bit_and_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
    is_differentiable=False,
)


# implements matrix[i,j] &= value
add_builtin(
    "bit_and_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "j": Any, "value": Any},
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
    is_differentiable=False,
)


# implements matrix[i] |= value
add_builtin(
    "bit_or_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
    is_differentiable=False,
)


# implements matrix[i,j] |= value
add_builtin(
    "bit_or_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "j": Any, "value": Any},
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
    is_differentiable=False,
)


# implements matrix[i] ^= value
add_builtin(
    "bit_xor_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "value": Any},
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
    is_differentiable=False,
)


# implements matrix[i,j] ^= value
add_builtin(
    "bit_xor_inplace",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": Any, "j": Any, "value": Any},
    value_type=None,
    hidden=True,
    export=False,
    group="Utility",
    is_differentiable=False,
)


for t in scalar_types + vector_types + (bool,):
    if "vec" in t.__name__ or "mat" in t.__name__:
        continue

    add_builtin(
        "expect_eq",
        input_types={"a": t, "b": t},
        value_type=None,
        doc="Print an error to stdout if ``a`` and ``b`` are not equal.",
        group="Utility",
        hidden=True,
        is_differentiable=False,
    )

    add_builtin(
        "expect_neq",
        input_types={"a": t, "b": t},
        value_type=None,
        doc="Print an error to stdout if ``a`` and ``b`` are not equal.",
        group="Utility",
        hidden=True,
        export=False,
        is_differentiable=False,
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
    doc="Print an error to stdout if ``a`` and ``b`` are not equal.",
    group="Utility",
    hidden=True,
    is_differentiable=False,
)
add_builtin(
    "expect_neq",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=expect_eq_value_func,
    doc="Print an error to stdout if ``a`` and ``b`` are equal.",
    group="Utility",
    hidden=True,
    export=False,
    is_differentiable=False,
)

add_builtin(
    "expect_eq",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=expect_eq_value_func,
    doc="Print an error to stdout if ``a`` and ``b`` are not equal.",
    group="Utility",
    hidden=True,
    is_differentiable=False,
)
add_builtin(
    "expect_neq",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=expect_eq_value_func,
    doc="Print an error to stdout if ``a`` and ``b`` are equal.",
    group="Utility",
    hidden=True,
    export=False,
    is_differentiable=False,
)

# Bool vector/matrix overloads for expect_eq/expect_neq (bool is not part of Scalar)
add_builtin(
    "expect_eq",
    input_types={"a": vector(length=Any, dtype=bool), "b": vector(length=Any, dtype=bool)},
    constraint=sametypes,
    value_func=expect_eq_value_func,
    doc="Print an error to stdout if ``a`` and ``b`` are not equal.",
    group="Utility",
    hidden=True,
    is_differentiable=False,
)
add_builtin(
    "expect_neq",
    input_types={"a": vector(length=Any, dtype=bool), "b": vector(length=Any, dtype=bool)},
    constraint=sametypes,
    value_func=expect_eq_value_func,
    doc="Print an error to stdout if ``a`` and ``b`` are equal.",
    group="Utility",
    hidden=True,
    export=False,
    is_differentiable=False,
)
add_builtin(
    "expect_eq",
    input_types={"a": matrix(shape=(Any, Any), dtype=bool), "b": matrix(shape=(Any, Any), dtype=bool)},
    constraint=sametypes,
    value_func=expect_eq_value_func,
    doc="Print an error to stdout if ``a`` and ``b`` are not equal.",
    group="Utility",
    hidden=True,
    is_differentiable=False,
)
add_builtin(
    "expect_neq",
    input_types={"a": matrix(shape=(Any, Any), dtype=bool), "b": matrix(shape=(Any, Any), dtype=bool)},
    constraint=sametypes,
    value_func=expect_eq_value_func,
    doc="Print an error to stdout if ``a`` and ``b`` are equal.",
    group="Utility",
    hidden=True,
    export=False,
    is_differentiable=False,
)

add_builtin(
    "lerp",
    input_types={"a": Float, "b": Float, "t": Float},
    value_func=sametypes_create_value_func(Float),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``.",
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
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``.",
    group="Utility",
)
add_builtin(
    "lerp",
    input_types={"a": matrix(shape=(Any, Any), dtype=Float), "b": matrix(shape=(Any, Any), dtype=Float), "t": Float},
    constraint=lerp_constraint,
    value_func=lerp_create_value_func(matrix(shape=(Any, Any), dtype=Float)),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``.",
    group="Utility",
)
add_builtin(
    "lerp",
    input_types={"a": quaternion(dtype=Float), "b": quaternion(dtype=Float), "t": Float},
    value_func=lerp_create_value_func(quaternion(dtype=Float)),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``.",
    group="Utility",
)
add_builtin(
    "lerp",
    input_types={"a": transformation(dtype=Float), "b": transformation(dtype=Float), "t": Float},
    value_func=lerp_create_value_func(transformation(dtype=Float)),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``.",
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
    doc="""Print an error to stdout if ``a`` and ``b`` differ by more than ``tolerance``.

    Compare scalar values.""",
    group="Utility",
    is_differentiable=False,
)
add_builtin(
    "expect_near",
    input_types={"a": vector(length=Any, dtype=Float), "b": vector(length=Any, dtype=Float), "tolerance": Float},
    defaults={"tolerance": 1.0e-6},
    constraint=expect_near_constraint,
    value_type=None,
    doc="""Print an error to stdout if ``a`` and ``b`` differ by more than ``tolerance``.

    Compare each vector element.""",
    group="Utility",
    is_differentiable=False,
)
add_builtin(
    "expect_near",
    input_types={"a": quaternion(dtype=Float), "b": quaternion(dtype=Float), "tolerance": Float},
    defaults={"tolerance": 1.0e-6},
    constraint=expect_near_constraint,
    value_type=None,
    doc="""Print an error to stdout if ``a`` and ``b`` differ by more than ``tolerance``.

    Compare each quaternion component.""",
    group="Utility",
    is_differentiable=False,
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
    doc="""Print an error to stdout if ``a`` and ``b`` differ by more than ``tolerance``.

    Compare each matrix element.""",
    group="Utility",
    is_differentiable=False,
)

# ---------------------------------
# Algorithms

add_builtin(
    "lower_bound",
    input_types={"arr": array(dtype=Scalar), "value": Scalar},
    value_type=int,
    doc="Search a sorted array ``arr`` for the closest element greater than or equal to ``value``.",
    is_differentiable=False,
)

add_builtin(
    "lower_bound",
    input_types={"arr": array(dtype=Scalar), "arr_begin": int, "arr_end": int, "value": Scalar},
    value_type=int,
    doc="""Search a sorted array ``arr`` for the closest element greater than or equal to ``value``.

    Search the range [arr_begin, arr_end).""",
    is_differentiable=False,
)

# ---------------------------------
# Operators


add_builtin(
    "add",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="""Add ``a`` and ``b``.""",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="""Add ``a`` and ``b``.""",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"a": quaternion(dtype=Float), "b": quaternion(dtype=Float)},
    value_func=sametypes_create_value_func(quaternion(dtype=Float)),
    doc="""Add ``a`` and ``b``.""",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="""Add ``a`` and ``b``.""",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"a": transformation(dtype=Float), "b": transformation(dtype=Float)},
    value_func=sametypes_create_value_func(transformation(dtype=Float)),
    doc="""Add ``a`` and ``b``.""",
    group="Operators",
)

add_builtin(
    "sub",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="""Subtract ``b`` from ``a``.""",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="""Subtract ``b`` from ``a``.""",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="""Subtract ``b`` from ``a``.""",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"a": quaternion(dtype=Float), "b": quaternion(dtype=Float)},
    value_func=sametypes_create_value_func(quaternion(dtype=Float)),
    doc="""Subtract ``b`` from ``a``.""",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"a": transformation(dtype=Float), "b": transformation(dtype=Float)},
    value_func=sametypes_create_value_func(transformation(dtype=Float)),
    doc="""Subtract ``b`` from ``a``.""",
    group="Operators",
)

# bitwise operators
add_builtin(
    "bit_and",
    input_types={"a": Int, "b": Int},
    value_func=sametypes_create_value_func(Int),
    doc="""Compute the bitwise AND of ``a`` and ``b``.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "bit_and",
    input_types={"a": vector(length=Any, dtype=Int), "b": vector(length=Any, dtype=Int)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Int)),
    doc="""Compute the bitwise AND of ``a`` and ``b``.

    Apply the operation element-wise to vectors.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "bit_and",
    input_types={"a": matrix(shape=(Any, Any), dtype=Int), "b": matrix(shape=(Any, Any), dtype=Int)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Int)),
    doc="""Compute the bitwise AND of ``a`` and ``b``.

    Apply the operation element-wise to matrices.""",
    group="Operators",
    is_differentiable=False,
)

add_builtin(
    "bit_or",
    input_types={"a": Int, "b": Int},
    value_func=sametypes_create_value_func(Int),
    doc="""Compute the bitwise OR of ``a`` and ``b``.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "bit_or",
    input_types={"a": vector(length=Any, dtype=Int), "b": vector(length=Any, dtype=Int)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Int)),
    doc="""Compute the bitwise OR of ``a`` and ``b``.

    Apply the operation element-wise to vectors.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "bit_or",
    input_types={"a": matrix(shape=(Any, Any), dtype=Int), "b": matrix(shape=(Any, Any), dtype=Int)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Int)),
    doc="""Compute the bitwise OR of ``a`` and ``b``.

    Apply the operation element-wise to matrices.""",
    group="Operators",
    is_differentiable=False,
)

add_builtin(
    "bit_xor",
    input_types={"a": Int, "b": Int},
    value_func=sametypes_create_value_func(Int),
    doc="""Compute the bitwise XOR of ``a`` and ``b``.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "bit_xor",
    input_types={"a": vector(length=Any, dtype=Int), "b": vector(length=Any, dtype=Int)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Int)),
    doc="""Compute the bitwise XOR of ``a`` and ``b``.

    Apply the operation element-wise to vectors.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "bit_xor",
    input_types={"a": matrix(shape=(Any, Any), dtype=Int), "b": matrix(shape=(Any, Any), dtype=Int)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Int)),
    doc="""Compute the bitwise XOR of ``a`` and ``b``.

    Apply the operation element-wise to matrices.""",
    group="Operators",
    is_differentiable=False,
)

add_builtin(
    "lshift",
    input_types={"a": Int, "b": Int},
    value_func=sametypes_create_value_func(Int),
    doc="""Compute ``a`` left-shifted by ``b`` bits.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "lshift",
    input_types={"a": vector(length=Any, dtype=Int), "b": vector(length=Any, dtype=Int)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Int)),
    doc="""Compute ``a`` left-shifted by ``b`` bits.

    Apply the operation element-wise to vectors.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "lshift",
    input_types={"a": matrix(shape=(Any, Any), dtype=Int), "b": matrix(shape=(Any, Any), dtype=Int)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Int)),
    doc="""Compute ``a`` left-shifted by ``b`` bits.

    Apply the operation element-wise to matrices.""",
    group="Operators",
    is_differentiable=False,
)

add_builtin(
    "rshift",
    input_types={"a": Int, "b": Int},
    value_func=sametypes_create_value_func(Int),
    doc="""Compute ``a`` right-shifted by ``b`` bits.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "rshift",
    input_types={"a": vector(length=Any, dtype=Int), "b": vector(length=Any, dtype=Int)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Int)),
    doc="""Compute ``a`` right-shifted by ``b`` bits.

    Apply the operation element-wise to vectors.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "rshift",
    input_types={"a": matrix(shape=(Any, Any), dtype=Int), "b": matrix(shape=(Any, Any), dtype=Int)},
    constraint=sametypes,
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Int)),
    doc="""Compute ``a`` right-shifted by ``b`` bits.

    Apply the operation element-wise to matrices.""",
    group="Operators",
    is_differentiable=False,
)

add_builtin(
    "invert",
    input_types={"a": Int},
    value_func=sametypes_create_value_func(Int),
    doc="""Compute the bitwise complement of ``a``.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "invert",
    input_types={"a": vector(length=Any, dtype=Int)},
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Int)),
    doc="""Compute the bitwise complement of ``a``.

    Apply the operation element-wise to vectors.""",
    group="Operators",
    is_differentiable=False,
)
add_builtin(
    "invert",
    input_types={"a": matrix(shape=(Any, Any), dtype=Int)},
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Int)),
    doc="""Compute the bitwise complement of ``a``.

    Apply the operation element-wise to matrices.""",
    group="Operators",
    is_differentiable=False,
)


add_builtin(
    "mul",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="""Multiply two values.""",
    group="Operators",
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
    doc="""Multiply two values.

    Scale a vector by a scalar.""",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": Scalar, "b": vector(length=Any, dtype=Scalar)},
    value_func=scalar_mul_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="""Multiply two values.

    Scale a vector by a scalar.""",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": quaternion(dtype=Float), "b": Scalar},
    value_func=scalar_mul_create_value_func(quaternion(dtype=Float)),
    doc="""Multiply two values.

    Scale a quaternion by a scalar.""",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": Scalar, "b": quaternion(dtype=Float)},
    value_func=scalar_mul_create_value_func(quaternion(dtype=Float)),
    doc="""Multiply two values.

    Scale a quaternion by a scalar.""",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": quaternion(dtype=Float), "b": quaternion(dtype=Float)},
    value_func=sametypes_create_value_func(quaternion(dtype=Float)),
    doc="""Multiply two values.

    Compute the Hamilton product of two quaternions.""",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": Scalar, "b": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=scalar_mul_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="""Multiply two values.

    Scale a matrix by a scalar.""",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": Scalar},
    value_func=scalar_mul_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="""Multiply two values.

    Scale a matrix by a scalar.""",
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
    doc="""Multiply two values.

    Compute a matrix-vector product.""",
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
    doc="""Multiply two values.

    Compute a row-vector-by-matrix product.""",
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
    doc="""Multiply two values.

    Compute a matrix-matrix product.""",
    group="Operators",
)


add_builtin(
    "mul",
    input_types={"a": transformation(dtype=Float), "b": transformation(dtype=Float)},
    value_func=sametypes_create_value_func(transformation(dtype=Float)),
    doc="""Multiply two values.

    Compose transformations (apply ``b`` then ``a``).""",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": Scalar, "b": transformation(dtype=Float)},
    value_func=scalar_mul_create_value_func(transformation(dtype=Float)),
    doc="""Multiply two values.

    Scale a transformation by a scalar.

    The result has an unnormalized quaternion.""",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"a": transformation(dtype=Float), "b": Scalar},
    value_func=scalar_mul_create_value_func(transformation(dtype=Float)),
    doc="""Multiply two values.

    Scale a transformation by a scalar.

    The result has an unnormalized quaternion.""",
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
    is_differentiable=False,
)

add_builtin(
    "div",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="""Divide two values.""",
    group="Operators",
    require_original_output_arg=True,
)
add_builtin(
    "div",
    input_types={"a": vector(length=Any, dtype=Scalar), "b": Scalar},
    value_func=scalar_mul_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="""Divide two values.

    Divide a vector by a scalar.""",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"a": Scalar, "b": vector(length=Any, dtype=Scalar)},
    value_func=scalar_mul_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="""Divide two values.

    Divide a scalar by each element of a vector.""",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "b": Scalar},
    value_func=scalar_mul_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="""Divide two values.

    Divide a matrix by a scalar.""",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"a": Scalar, "b": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=scalar_mul_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="""Divide two values.

    Divide a scalar by each element of a matrix.""",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"a": quaternion(dtype=Float), "b": Scalar},
    value_func=scalar_mul_create_value_func(quaternion(dtype=Float)),
    doc="""Divide two values.

    Divide a quaternion by a scalar.

    The result is unnormalized.""",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"a": Scalar, "b": quaternion(dtype=Float)},
    value_func=scalar_mul_create_value_func(quaternion(dtype=Float)),
    doc="""Divide two values.

    Divide a scalar by a quaternion.

    The result is unnormalized.""",
    group="Operators",
)

add_builtin(
    "div_approx",
    input_types={"a": Float, "b": Float},
    value_func=sametypes_create_value_func(Float),
    native_func="approx_div",
    doc="""Divide two values using approximate GPU intrinsics.

    Falls back to exact division on CPU.""",
    group="Operators",
    require_original_output_arg=True,
    export=False,
)
add_builtin(
    "div_approx",
    input_types={"a": vector(length=Any, dtype=Float), "b": Float},
    value_func=scalar_mul_create_value_func(vector(length=Any, dtype=Float)),
    native_func="approx_div",
    doc="""Divide two values using approximate GPU intrinsics.

    Divide a vector by a scalar. Falls back to exact division on CPU.""",
    group="Operators",
    export=False,
)
add_builtin(
    "div_approx",
    input_types={"a": Float, "b": vector(length=Any, dtype=Float)},
    value_func=scalar_mul_create_value_func(vector(length=Any, dtype=Float)),
    native_func="approx_div",
    doc="""Divide two values using approximate GPU intrinsics.

    Divide a scalar by each element of a vector. Falls back to exact division on CPU.""",
    group="Operators",
    export=False,
)
add_builtin(
    "div_approx",
    input_types={"a": matrix(shape=(Any, Any), dtype=Float), "b": Float},
    value_func=scalar_mul_create_value_func(matrix(shape=(Any, Any), dtype=Float)),
    native_func="approx_div",
    doc="""Divide two values using approximate GPU intrinsics.

    Divide a matrix by a scalar. Falls back to exact division on CPU.""",
    group="Operators",
    export=False,
)
add_builtin(
    "div_approx",
    input_types={"a": Float, "b": matrix(shape=(Any, Any), dtype=Float)},
    value_func=scalar_mul_create_value_func(matrix(shape=(Any, Any), dtype=Float)),
    native_func="approx_div",
    doc="""Divide two values using approximate GPU intrinsics.

    Divide a scalar by each element of a matrix. Falls back to exact division on CPU.""",
    group="Operators",
    export=False,
)
add_builtin(
    "div_approx",
    input_types={"a": quaternion(dtype=Float), "b": Float},
    value_func=scalar_mul_create_value_func(quaternion(dtype=Float)),
    native_func="approx_div",
    doc="""Divide two values using approximate GPU intrinsics.

    Divide a quaternion by a scalar.

    The result is unnormalized. Falls back to exact division on CPU.""",
    group="Operators",
    export=False,
)
add_builtin(
    "div_approx",
    input_types={"a": Float, "b": quaternion(dtype=Float)},
    value_func=scalar_mul_create_value_func(quaternion(dtype=Float)),
    native_func="approx_div",
    doc="""Divide two values using approximate GPU intrinsics.

    Divide a scalar by a quaternion.

    The result is unnormalized. Falls back to exact division on CPU.""",
    group="Operators",
    export=False,
)

add_builtin(
    "floordiv",
    input_types={"a": Scalar, "b": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="Divide two scalars using floor division.",
    group="Operators",
    is_differentiable=False,
)

add_builtin(
    "pos",
    input_types={"x": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="""Pass ``x`` unchanged.""",
    group="Operators",
)
add_builtin(
    "pos",
    input_types={"x": vector(length=Any, dtype=Scalar)},
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="""Pass ``x`` unchanged.""",
    group="Operators",
)
add_builtin(
    "pos",
    input_types={"x": quaternion(dtype=Float)},
    value_func=sametypes_create_value_func(quaternion(dtype=Float)),
    doc="""Pass ``x`` unchanged.""",
    group="Operators",
)
add_builtin(
    "pos",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="""Pass ``x`` unchanged.""",
    group="Operators",
)
add_builtin(
    "neg",
    input_types={"x": Scalar},
    value_func=sametypes_create_value_func(Scalar),
    doc="""Negate ``x``.""",
    group="Operators",
)
add_builtin(
    "neg",
    input_types={"x": vector(length=Any, dtype=Scalar)},
    value_func=sametypes_create_value_func(vector(length=Any, dtype=Scalar)),
    doc="""Negate ``x``.""",
    group="Operators",
)
add_builtin(
    "neg",
    input_types={"x": quaternion(dtype=Float)},
    value_func=sametypes_create_value_func(quaternion(dtype=Float)),
    doc="""Negate ``x``.""",
    group="Operators",
)
add_builtin(
    "neg",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=sametypes_create_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="""Negate ``x``.""",
    group="Operators",
)

add_builtin(
    "unot",
    input_types={"a": builtins.bool},
    value_type=builtins.bool,
    doc="""Compute logical NOT of ``a``.

    Returns:
        ``True`` if ``a`` is falsy (``False``, zero, or an empty/null array), ``False`` otherwise.""",
    group="Operators",
    is_differentiable=False,
)
for t in int_types:
    add_builtin(
        "unot",
        input_types={"a": t},
        value_type=builtins.bool,
        doc="""Compute logical NOT of ``a``.

    Returns:
        ``True`` if ``a`` is falsy (``False``, zero, or an empty/null array), ``False`` otherwise.""",
        group="Operators",
        is_differentiable=False,
    )


add_builtin(
    "unot",
    input_types={"a": array(dtype=Any)},
    value_type=builtins.bool,
    doc="""Compute logical NOT of ``a``.

    Returns:
        ``True`` if ``a`` is falsy (``False``, zero, or an empty/null array), ``False`` otherwise.""",
    group="Operators",
    is_differentiable=False,
)


# Tile operators
def tile_unary_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Scalar, shape=tuple[int, ...])

    t = arg_types["x"]

    if not is_tile(t):
        raise TypeError(f"Expected tile for unary expression, got {t}")

    if type_is_struct(t.dtype):
        raise TypeError("Tile unary operators do not support Warp struct tile elements")

    return tile(dtype=t.dtype, shape=t.shape)


def tile_binary_bitwise_value_func(arg_types, arg_values):
    result = tile_binary_map_value_func(arg_types, arg_values)

    if arg_types is None:
        return result

    a = arg_types["a"]
    b = arg_types["b"]

    a_dtype = a.dtype if is_tile(a) else a
    b_dtype = b.dtype if is_tile(b) else b

    if type_is_struct(a_dtype) or type_is_struct(b_dtype):
        raise TypeError("Tile bitwise operators do not support Warp struct tile elements")

    return result


def tile_mul_value_func(arg_types, arg_values):
    """Value function for tile multiplication.

    Handles:
    1. tile * tile: element-wise multiplication (shapes must match)
    2. tile * constant: multiply each element by scalar/vec/mat
    3. constant * tile: multiply each element by scalar/vec/mat

    At least one operand must be a scalar type (can't multiply vec by vec).
    Underlying scalar types must match.
    """
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    a = arg_types["a"]
    b = arg_types["b"]

    a_is_tile = is_tile(a)
    b_is_tile = is_tile(b)

    if not (a_is_tile or b_is_tile):
        raise TypeError("tile mul requires at least one tile operand")

    if a_is_tile and b_is_tile:
        # tile * tile: validate shapes match, at least one dtype must be scalar
        if len(a.shape) != len(b.shape):
            raise ValueError(f"Shapes must have same dimensions: {len(a.shape)} vs {len(b.shape)}")
        for i in range(len(a.shape)):
            if a.shape[i] != b.shape[i]:
                raise ValueError(f"Shape mismatch on dim {i}: {a.shape} vs {b.shape}")
        if not type_is_scalar(a.dtype) and not type_is_scalar(b.dtype):
            raise TypeError(
                f"Cannot multiply tile<{type_repr(a.dtype)}> by tile<{type_repr(b.dtype)}>:"
                " at least one element type must be scalar"
            )
        a_scalar = type_scalar_type(a.dtype)
        b_scalar = type_scalar_type(b.dtype)
        if a_scalar != b_scalar:
            raise TypeError(f"Underlying scalar types don't match: {type_repr(a_scalar)} vs {type_repr(b_scalar)}")
        # Result dtype: vec/mat side wins; if both scalar they're equal
        if type_is_vector(a.dtype) or type_is_matrix(a.dtype):
            result_dtype = a.dtype
        elif type_is_vector(b.dtype) or type_is_matrix(b.dtype):
            result_dtype = b.dtype
        else:
            result_dtype = a.dtype
        return tile(dtype=result_dtype, shape=a.shape)

    # tile * const or const * tile
    tile_type = a if a_is_tile else b
    const_type = b if a_is_tile else a

    # Constant must be scalar/vector/matrix
    if not (type_is_scalar(const_type) or type_is_vector(const_type) or type_is_matrix(const_type)):
        raise TypeError(f"Non-tile operand must be scalar/vec/mat, got {type_repr(const_type)}")

    # Underlying scalar-type compatibility
    tile_scalar = type_scalar_type(tile_type.dtype)
    const_scalar = type_scalar_type(const_type)
    if tile_scalar != const_scalar:
        raise TypeError(
            f"Underlying scalar types don't match: tile={type_repr(tile_scalar)}, const={type_repr(const_scalar)}"
        )

    # At least one side must be scalar (can't multiply vec by vec)
    if not type_is_scalar(tile_type.dtype) and not type_is_scalar(const_type):
        if a_is_tile:
            raise TypeError(
                f"Cannot multiply tile<{type_repr(tile_type.dtype)}> by {type_repr(const_type)}:"
                " at least one operand must be a scalar type"
            )
        else:
            raise TypeError(
                f"Cannot multiply {type_repr(const_type)} by tile<{type_repr(tile_type.dtype)}>:"
                " at least one operand must be a scalar type"
            )

    # Result dtype: adopt const dtype if vec/mat; otherwise keep tile's dtype
    result_dtype = const_type if (type_is_vector(const_type) or type_is_matrix(const_type)) else tile_type.dtype
    return tile(dtype=result_dtype, shape=tile_type.shape)


def tile_div_value_func(arg_types, arg_values):
    """Value function for tile division.

    Handles:
    1. tile / tile: element-wise division (shapes must match)
    2. tile / constant: divide each element by scalar/vec/mat
    3. constant / tile: divide scalar/vec/mat by each element

    At least one operand must be a scalar type (can't divide vec by vec).
    Underlying scalar types must match.
    """
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, ...])

    a = arg_types["a"]
    b = arg_types["b"]

    a_is_tile = is_tile(a)
    b_is_tile = is_tile(b)

    if not (a_is_tile or b_is_tile):
        raise TypeError("tile div requires at least one tile operand")

    if a_is_tile and b_is_tile:
        # tile / tile: validate shapes match, at least one dtype must be scalar
        if len(a.shape) != len(b.shape):
            raise ValueError(f"Shapes must have same dimensions: {len(a.shape)} vs {len(b.shape)}")
        for i in range(len(a.shape)):
            if a.shape[i] != b.shape[i]:
                raise ValueError(f"Shape mismatch on dim {i}: {a.shape} vs {b.shape}")
        if not type_is_scalar(a.dtype) and not type_is_scalar(b.dtype):
            raise TypeError(
                f"Cannot divide tile<{type_repr(a.dtype)}> by tile<{type_repr(b.dtype)}>:"
                " at least one element type must be scalar"
            )
        a_scalar = type_scalar_type(a.dtype)
        b_scalar = type_scalar_type(b.dtype)
        if a_scalar != b_scalar:
            raise TypeError(f"Underlying scalar types don't match: {type_repr(a_scalar)} vs {type_repr(b_scalar)}")
        # Result dtype: vec/mat side wins; if both scalar they're equal
        if type_is_vector(a.dtype) or type_is_matrix(a.dtype):
            result_dtype = a.dtype
        elif type_is_vector(b.dtype) or type_is_matrix(b.dtype):
            result_dtype = b.dtype
        else:
            result_dtype = a.dtype
        return tile(dtype=result_dtype, shape=a.shape)

    # tile / const or const / tile
    tile_type = a if a_is_tile else b
    const_type = b if a_is_tile else a

    # Constant must be scalar/vector/matrix
    if not (type_is_scalar(const_type) or type_is_vector(const_type) or type_is_matrix(const_type)):
        raise TypeError(f"Non-tile operand must be scalar/vec/mat, got {type_repr(const_type)}")

    # Underlying scalar-type compatibility
    tile_scalar = type_scalar_type(tile_type.dtype)
    const_scalar = type_scalar_type(const_type)
    if tile_scalar != const_scalar:
        raise TypeError(
            f"Underlying scalar types don't match: tile={type_repr(tile_scalar)}, const={type_repr(const_scalar)}"
        )

    # At least one side must be scalar (can't divide vec by vec)
    if not type_is_scalar(tile_type.dtype) and not type_is_scalar(const_type):
        if a_is_tile:
            raise TypeError(
                f"Cannot divide tile<{type_repr(tile_type.dtype)}> by {type_repr(const_type)}:"
                " at least one operand must be a scalar type"
            )
        else:
            raise TypeError(
                f"Cannot divide {type_repr(const_type)} by tile<{type_repr(tile_type.dtype)}>:"
                " at least one operand must be a scalar type"
            )

    # Result dtype: adopt const dtype if vec/mat; otherwise keep tile's dtype
    result_dtype = const_type if (type_is_vector(const_type) or type_is_matrix(const_type)) else tile_type.dtype
    return tile(dtype=result_dtype, shape=tile_type.shape)


add_builtin(
    "neg",
    input_types={"x": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_unary_value_func,
    doc="""Negate ``x``.

    Negate tiles element-wise.""",
    export=False,
    native_func="tile_neg",
    group="Operators",
)

add_builtin(
    "add",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_binary_map_value_func,
    # dispatch_func=tile_map_dispatch_func,
    # variadic=True,
    native_func="tile_add",
    doc="""Add ``a`` and ``b``.""",
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "sub",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_binary_map_value_func,
    # dispatch_func=tile_map_dispatch_func,
    # variadic=True,
    native_func="tile_sub",
    doc="""Subtract ``b`` from ``a``.""",
    group="Tile Primitives",
    export=False,
)

# NOTE: The tile*tile overload must be registered before the tile*Any overload below.
# Warp's overload resolution tries earlier registrations first, so if tile*Any were
# registered first, tile*tile would silently route to tile_mul instead of
# tile_mul_elementwise. The same applies to the div overloads further below.
add_builtin(
    "mul",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_mul_value_func,
    native_func="tile_mul_elementwise",
    doc="""Element-wise multiplication of tiles.""",
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "bit_and",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_binary_bitwise_value_func,
    # dispatch_func=tile_map_dispatch_func,
    # variadic=True,
    native_func="tile_bit_and",
    doc="""Compute the bitwise AND of ``a`` and ``b``.

    Apply the operation element-wise to tiles.""",
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "bit_or",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_binary_bitwise_value_func,
    # dispatch_func=tile_map_dispatch_func,
    # variadic=True,
    native_func="tile_bit_or",
    doc="""Compute the bitwise OR of ``a`` and ``b``.

    Apply the operation element-wise to tiles.""",
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "bit_xor",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_binary_bitwise_value_func,
    # dispatch_func=tile_map_dispatch_func,
    # variadic=True,
    native_func="tile_bit_xor",
    doc="""Compute the bitwise XOR of ``a`` and ``b``.

    Apply the operation element-wise to tiles.""",
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)


add_builtin(
    "mul",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": Any},
    value_func=tile_mul_value_func,
    doc="""Multiply two values.

    Multiply each element of a tile by a constant (scalar, vector, or matrix).

    At least one of the tile's element type or the constant type must be scalar.
    Underlying scalar types must match.""",
    export=False,
    native_func="tile_mul",
    group="Operators",
)


# Dispatch function for const*tile that reorders args so tile comes first
def tile_mul_const_first_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    # Reorder: (const, tile) -> (tile, const) for C++ tile_mul(Tile&, const S&)
    return ((args["b"], args["a"]), ())


add_builtin(
    "mul",
    input_types={"a": Any, "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_mul_value_func,
    dispatch_func=tile_mul_const_first_dispatch_func,
    doc="""Multiply two values.

    Multiply each element of a tile by a constant (scalar, vector, or matrix).

    At least one of the tile's element type or the constant type must be scalar.
    Underlying scalar types must match.""",
    export=False,
    native_func="tile_mul",
    group="Operators",
)


# NOTE: The tile/tile overload must be registered before the tile/Any and Any/tile overloads below.
# See the equivalent note above tile*tile mul for details.
add_builtin(
    "div",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_div_value_func,
    native_func="tile_div_elementwise",
    doc="""Element-wise division of tiles.""",
    group="Tile Primitives",
    export=False,
)


# tile / scalar
add_builtin(
    "div",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": Any},
    value_func=tile_div_value_func,
    native_func="tile_div",
    doc="""Divide tile elements by a constant.

    Divide each element of a tile by a constant (scalar, vector, or matrix).

    At least one of the tile's element type or the constant type must be scalar.
    Underlying scalar types must match.""",
    export=False,
    group="Operators",
)


# scalar / tile
add_builtin(
    "div",
    input_types={"a": Any, "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_div_value_func,
    native_func="tile_div",
    doc="""Divide a constant by tile elements.

    Divide a constant (scalar, vector, or matrix) by each element of a tile.

    At least one of the tile's element type or the constant type must be scalar.
    Underlying scalar types must match.""",
    export=False,
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
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_inplace_tile_value_func,
    dispatch_func=tile_inplace_dispatch_func,
    export=False,
    hidden=True,
    native_func="tile_add_inplace",
    group="Operators",
)


add_builtin(
    "sub_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_inplace_tile_value_func,
    dispatch_func=tile_inplace_dispatch_func,
    export=False,
    hidden=True,
    native_func="tile_sub_inplace",
    group="Operators",
)


add_builtin(
    "bit_and_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_inplace_tile_bitwise_value_func,
    dispatch_func=tile_inplace_dispatch_func,
    export=False,
    hidden=True,
    native_func="tile_bit_and_inplace",
    group="Operators",
    is_differentiable=False,
)


add_builtin(
    "bit_or_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_inplace_tile_bitwise_value_func,
    dispatch_func=tile_inplace_dispatch_func,
    export=False,
    hidden=True,
    native_func="tile_bit_or_inplace",
    group="Operators",
    is_differentiable=False,
)


add_builtin(
    "bit_xor_inplace",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...]), "b": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=tile_inplace_tile_bitwise_value_func,
    dispatch_func=tile_inplace_dispatch_func,
    export=False,
    hidden=True,
    native_func="tile_bit_xor_inplace",
    group="Operators",
    is_differentiable=False,
)


def tile_diag_add_value_func(arg_types, arg_values):
    if arg_types is None:
        return tile(dtype=Any, shape=tuple[int, int])

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
    builder: warp._src.context.ModuleBuilder,
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
    input_types={"a": tile(dtype=Any, shape=tuple[int, int]), "d": tile(dtype=Any, shape=tuple[int])},
    value_func=tile_diag_add_value_func,
    lto_dispatch_func=tile_diag_add_lto_dispatch_func,
    native_func="tile_diag_add",
    doc="Add a square matrix and a diagonal matrix ``d`` represented as a 1D tile.",
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
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
        return tile(dtype=Float, shape=tuple[int, int])

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
    builder: warp._src.context.ModuleBuilder,
):
    a = arg_values["a"]
    b = arg_values["b"]
    alpha = arg_values["alpha"]

    if len(return_values) > 0:
        # c = tile_matmul(a, b) case: fresh output, don't read from C
        beta = 0.0
        out = return_values[0]
    else:
        # tile_matmul(a, b, out=c) case: accumulate into C
        beta = arg_values["beta"]
        out = arg_values["out"]

    if not is_tile(out.type):
        raise TypeError(f"tile_matmul() 'out' argument must be a tile, got {out!r}")

    if any(arg.type.dtype not in [float16, bfloat16, float32, float64, vec2h, vec2f, vec2d] for arg in [a, b, out]):
        raise TypeError(
            "tile_matmul() arguments must be tiles of float16, bfloat16, float32 or float64, vec2h, vec2f, vec2d entries"
        )

    # Reject bfloat16 as the accumulator precision uniformly across all backends. cuBLASDx
    # disallows bf16 accumulators (a static_assert since cuBLASDx 0.6.0), and the scalar
    # matmul fallback was silently lossy for the same reason. The K-loop reduction precision
    # is derived from 'out's dtype regardless of which calling form is used. If backward is
    # enabled, 'a' and 'b' are also accumulators (for adjA, adjB).
    if out.type.dtype == bfloat16:
        raise TypeError(
            "tile_matmul() does not support a bfloat16 'out' tile. "
            "Allowed 'out' dtypes are float16, float32, and float64."
        )
    if options["enable_backward"] and (a.type.dtype == bfloat16 or b.type.dtype == bfloat16):
        raise TypeError(
            "tile_matmul() does not support bfloat16 'a' or 'b' tiles when the backward pass is enabled. "
            "Allowed accumulator dtypes are float16, float32, and float64 (the backward pass uses 'a' "
            "and 'b' as accumulators for adjA and adjB). If gradients are not needed, set "
            "`enable_backward=False` on the kernel's module, e.g. "
            "`wp.set_module_options({'enable_backward': False})`."
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

    M, K = a.type.shape[0], a.type.shape[1]
    _, N = b.type.shape[0], b.type.shape[1]
    num_threads = options["block_dim"]
    arch = options["output_arch"]

    if (
        arch is None
        or not warp._src.context.runtime.core.wp_is_mathdx_enabled()
        or not options.get("enable_mathdx_gemm", True)
    ):
        # CPU/no-MathDx dispatch (or mathdx GEMM disabled via module option)
        return ((0, 0, 0, a, b, out, alpha, beta), (), [], 0)
    else:

        def tile_flip_layout(layout):
            if layout == "rowmajor":
                return "colmajor"
            elif layout == "colmajor":
                return "rowmajor"
            else:
                raise ValueError(f"unexpected layout {layout!r}")

        # generate the LTOs
        #    C += A * B
        (fun_forward, lto_forward) = warp._src.build.build_lto_dot(
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
        if options["enable_backward"]:
            # adjA += adjC * B^T - Transpose ~= flipped layout
            (fun_backward_A, lto_backward_A) = warp._src.build.build_lto_dot(
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
            (fun_backward_B, lto_backward_B) = warp._src.build.build_lto_dot(
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
                alpha,
                beta,
            ),
            (),
            [lto_forward, lto_backward_A, lto_backward_B],
            0,
        )


add_builtin(
    "tile_matmul",
    input_types={
        "a": tile(dtype=Float, shape=tuple[int, int]),
        "b": tile(dtype=Float, shape=tuple[int, int]),
        "out": tile(dtype=Float, shape=tuple[int, int]),
        "alpha": Float,
        "beta": Float,
    },
    defaults={"alpha": 1.0, "beta": 1.0},
    value_func=tile_matmul_out_value_func,
    lto_dispatch_func=tile_matmul_lto_dispatch_func,
    native_func="tile_matmul_acc",
    variadic=False,
    doc="""Compute the matrix product ``a*b``.

    Compute ``out = alpha * a*b + beta * out``.

    Supported datatypes are:
        * fp16, bf16, fp32, fp64 (real)
        * vec2h, vec2f, vec2d (complex)

    All input and output tiles must have the same datatype. Tile data will automatically be migrated
    to shared memory if necessary and will use TensorCore operations when available.

    Note that computing the adjoints of alpha and beta are not yet supported.

    Args:
        a: A tile with ``shape=(M, K)``
        b: A tile with ``shape=(K, N)``
        out: A tile with ``shape=(M, N)``
        alpha: Scaling factor (default 1.0)
        beta: Accumulator factor (default 1.0)
""",
    group="Tile Primitives",
    export=False,
)

add_builtin(
    "tile_matmul",
    input_types={
        "a": tile(dtype=Float, shape=tuple[int, int]),
        "b": tile(dtype=Float, shape=tuple[int, int]),
        "alpha": Float,
    },
    defaults={"alpha": 1.0},
    value_func=tile_matmul_value_func,
    lto_dispatch_func=tile_matmul_lto_dispatch_func,
    variadic=False,
    doc="""Compute the matrix product ``a*b``.

    Compute ``out = alpha * a*b``.

    Supported datatypes are:
        * fp16, bf16, fp32, fp64 (real)
        * vec2h, vec2f, vec2d (complex)

    Both input tiles must have the same datatype. Tile data will automatically be migrated
    to shared memory if necessary and will use TensorCore operations when available.

    Note that computing the adjoints of alpha is not yet supported.

    Args:
        a: A tile with ``shape=(M, K)``
        b: A tile with ``shape=(K, N)``
        alpha: Scaling factor (default 1.0)

    Returns:
        A tile with ``shape=(M, N)``
""",
    group="Tile Primitives",
    export=False,
)


##
## FFT
##
def tile_fft_generic_value_func(arg_types, arg_values, func_name="tile_fft"):
    if arg_types is None:
        return None

    if len(arg_types) != 1:
        raise TypeError(f"{func_name}() takes exactly 1 positional argument but {len(arg_types)} were given")

    inout = arg_types["inout"]

    if not is_tile(inout):
        raise TypeError(f"{func_name}() argument must be a tile, got {inout!r}")

    if inout.storage != "register":
        raise ValueError(f"{func_name}() argument must have 'register' storage, got {inout.storage}")

    if inout.dtype not in [vec2f, vec2d]:
        raise TypeError(
            f"{func_name}() argument must be a tile of vec2f or vec2d (interpreted as complex) entries, got {inout.dtype!r}"
        )

    if len(inout.shape) < 2:
        raise ValueError(f"{func_name}() argument must be a tile with at least 2 dimensions, got {len(inout.shape)}D")

    return None


def tile_fft_generic_lto_dispatch_func(
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp._src.context.ModuleBuilder,
    direction: str | None = None,
):
    inout = arg_values["inout"]
    inout.type.storage = "register"

    # see libcufftdx.hpp
    if direction == "forward":
        fwd_dir = 0  # CUFFTDX_DIRECTION_FORWARD
        bwd_direction = "inverse"
        bwd_dir = 1  # CUFFTDX_DIRECTION_INVERSE
    elif direction == "inverse":
        fwd_dir = 1  # CUFFTDX_DIRECTION_INVERSE
        bwd_direction = "forward"
        bwd_dir = 0  # CUFFTDX_DIRECTION_FORWARD
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

    # batch = product of all leading dims, size = last (FFT) dim
    batch, size = math.prod(inout.type.shape[:-1]), inout.type.shape[-1]
    num_threads = options["block_dim"]
    arch = options["output_arch"]
    ept = size // num_threads

    func_name = "tile_fft" if direction == "forward" else "tile_ifft"
    use_mathdx = (
        arch is not None
        and warp._src.context.runtime.core.wp_is_mathdx_enabled()
        and options.get("enable_mathdx_fft", True)
    )

    if not use_mathdx:
        # CPU sequential or GPU cooperative scalar path. Both go through
        # `wp::tile_fft_entry` with a literal `0` for the LTO function name;
        # `wp_is_null_func<int>::value` selects the scalar branch at template
        # instantiation, mirroring how `tile_matmul` handles the same case.
        if arch is not None:
            # GPU cooperative path requires power-of-two FFT size (cooperative
            # mixed-radix would mean reimplementing cuFFTDx) and ept >= 1 so
            # every thread participates in at least one butterfly per stage.
            if size <= 0 or (size & (size - 1)) != 0:
                raise ValueError(
                    f"{func_name}() on GPU without libmathdx requires a power-of-two FFT size, "
                    f"got {size}. Build Warp with libmathdx (cuFFTDx) for arbitrary sizes, "
                    f"or run on CPU."
                )
            if size % num_threads != 0:
                raise ValueError(
                    f"{func_name}() on GPU without libmathdx requires fft_size to be divisible "
                    f"by block_dim (got fft_size={size}, block_dim={num_threads})."
                )
            # Shared scratch holds one batch worth of complex data; reused
            # across batches inside `tile_fft_gpu_impl`.
            dtype_size = 2 * (4 if precision == 5 else 8)
            shared_memory_bytes = size * dtype_size
        else:
            # CPU path: non-power-of-two sizes use an O(n^2) DFT with fixed
            # stack buffers capped at WP_FFT_CPU_MAX_DFT_SIZE (4096).
            if (size & (size - 1)) != 0 and size > 4096:
                raise ValueError(
                    f"{func_name}() on CPU with a non-power-of-two FFT size is limited to "
                    f"4096 elements, got {size}. Use a power-of-two size for larger transforms."
                )
            shared_memory_bytes = 0

        lto_placeholder = "/* scalar */ 0"
        return (
            (
                Var(lto_placeholder, str, False, True, False),
                Var(lto_placeholder, str, False, True, False),
                Var(dtype, str, False, True, False),
                Var(str(shared_memory_bytes), str, False, True, False),
                Var(str(batch), str, False, True, False),
                Var(str(ept), str, False, True, False),
                inout,
            ),
            [],
            [],
            shared_memory_bytes,
        )

    # GPU cuFFTDx LTO path.
    if ept < 2:
        raise ValueError(
            f"{func_name}() requires at least 2 elements per thread, but got ept={ept} "
            f"(fft_size={size}, block_dim={num_threads}). "
            f"Reduce block_dim to at most {size // 2} for this FFT size."
        )

    # generate the forward LTO
    lto_symbol_fwd, lto_code_data_fwd, shared_memory_bytes = warp._src.build.build_lto_fft(
        arch, size, ept, direction, fwd_dir, precision, builder
    )

    if options["enable_backward"]:
        # generate the backward LTO (inverse direction for adjoint)
        # shared memory requirements are identical since tile sizes match
        lto_symbol_bwd, lto_code_data_bwd, _ = warp._src.build.build_lto_fft(
            arch, size, ept, bwd_direction, bwd_dir, precision, builder
        )
    else:
        # adjoints aren't computed, so we reuse forward symbol as a dummy arg
        lto_symbol_bwd = lto_symbol_fwd
        lto_code_data_bwd = None

    return (
        (
            Var(lto_symbol_fwd, str, False, True, False),
            Var(lto_symbol_bwd, str, False, True, False),
            Var(dtype, str, False, True, False),
            Var(str(shared_memory_bytes), str, False, True, False),
            Var(str(batch), str, False, True, False),
            Var(str(ept), str, False, True, False),
            inout,
        ),
        [],
        [lto_code_data_fwd, lto_code_data_bwd],
        shared_memory_bytes,
    )


add_builtin(
    "tile_fft",
    input_types={"inout": tile(dtype=vector(length=2, dtype=Float), shape=tuple[int, ...])},
    value_func=functools.partial(tile_fft_generic_value_func, func_name="tile_fft"),
    lto_dispatch_func=functools.partial(tile_fft_generic_lto_dispatch_func, direction="forward"),
    variadic=True,
    doc="""Compute the forward FFT along the last dimension of an N-D tile of data.

    This function cooperatively computes the forward FFT on a tile of data inplace.
    All leading dimensions are treated as independent batch dimensions.
    The tile must have at least two dimensions.

    The transform is unnormalized, meaning that applying :func:`tile_fft` followed by :func:`tile_ifft`
    will scale the data by N, where N is the FFT size (the last dimension of the tile).
    Normalization is left to the user to perform as needed.

    Supported datatypes are:
        * vec2f, vec2d

    Args:
        inout: The input/output tile.

    Notes:
        Supported FFT sizes by backend:

        * **CPU**: Any size. Non-power-of-two sizes are capped at 4096;
          larger non-power-of-two sizes raise ``ValueError``.
        * **GPU with libmathdx**: Any size. This is the default when Warp
          is built with libmathdx.
        * **GPU without libmathdx** (or ``enable_mathdx_fft=False``):
          Power-of-two sizes only, and the FFT size must be divisible by
          ``block_dim``. Other sizes raise ``ValueError``. Slower than the
          libmathdx path.

        See :attr:`warp.config.enable_mathdx_fft` to control GPU backend
        selection.""",
    group="Tile Primitives",
    export=False,
    namespace="",
)

add_builtin(
    "tile_ifft",
    input_types={"inout": tile(dtype=vector(length=2, dtype=Float), shape=tuple[int, ...])},
    value_func=functools.partial(tile_fft_generic_value_func, func_name="tile_ifft"),
    lto_dispatch_func=functools.partial(tile_fft_generic_lto_dispatch_func, direction="inverse"),
    variadic=True,
    doc="""Compute the inverse FFT along the last dimension of an N-D tile of data.

    This function cooperatively computes the inverse FFT on a tile of data inplace.
    All leading dimensions are treated as independent batch dimensions.
    The tile must have at least two dimensions.

    The transform is unnormalized, meaning that applying :func:`tile_fft` followed by :func:`tile_ifft`
    will scale the data by N, where N is the FFT size (the last dimension of the tile).
    Normalization is left to the user to perform as needed.

    Supported datatypes are:
        * vec2f, vec2d

    Args:
        inout: The input/output tile.

    Notes:
        See :func:`tile_fft` for backend selection and supported sizes — the
        same constraints apply to :func:`tile_ifft`.""",
    group="Tile Primitives",
    export=False,
    namespace="",
)


cusolver_function_map = {"getrf": 0, "getrf_no_pivot": 1, "potrf": 2, "potrs": 3, "trsm": 4}

cusolver_type_map = {float32: ("wp::float32", 5), float64: ("wp::float64", 6)}

cusolver_fill_mode_map = {"upper": 0, "lower": 1}

cusolver_side_map = {"-": -1, "left": 0, "right": 1}

cusolver_diag_map = {"-": -1, "unit": 0, "nounit": 1}


##
## Cholesky
##
def _tile_cholesky_generic_value_func(inplace: bool, arg_types, arg_values):
    if arg_types is None:
        if inplace:
            return None
        return tile(dtype=Float, shape=tuple[int, int])

    if len(arg_types) > 2:
        raise TypeError(
            f"tile_cholesky() takes 1 positional argument and 1 optional argument but {len(arg_types)} were given"
        )

    a = arg_types["A"]

    if not is_tile(a):
        raise TypeError(f"tile_cholesky() argument must be a tile, got {a!r}")

    if len(a.shape) != 2:
        raise ValueError("tile_cholesky() argument must be a 2D tile")

    if a.shape[0] != a.shape[1]:
        raise ValueError("tile_cholesky() argument must be square")

    if inplace:
        return None
    return tile(dtype=a.dtype, shape=a.shape, layout=a.layout, strides=a.strides, storage="shared")


def tile_cholesky_generic_value_func(arg_types, arg_values):
    return _tile_cholesky_generic_value_func(False, arg_types, arg_values)


def tile_cholesky_inplace_generic_value_func(arg_types, arg_values):
    return _tile_cholesky_generic_value_func(True, arg_types, arg_values)


def _tile_cholesky_extract_fill_mode(arg_values, func_name="tile_cholesky"):
    """Extract fill_mode from arg_values, returning the upper bool."""
    fill_mode_var = arg_values.get("fill_mode")
    if fill_mode_var is not None:
        if not hasattr(fill_mode_var, "constant") or fill_mode_var.constant is None:
            raise ValueError(f"{func_name}() fill_mode must be a compile-time constant")
        fill_mode_str = fill_mode_var.constant
        if fill_mode_str not in ("lower", "upper"):
            raise ValueError(f'{func_name}() fill_mode must be "lower" or "upper"')
    else:
        fill_mode_str = "lower"
    return fill_mode_str == "upper"


def _tile_cholesky_generic_lto_dispatch_func(
    inplace: bool,
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp._src.context.ModuleBuilder,
):
    upper = _tile_cholesky_extract_fill_mode(arg_values)
    a = arg_values["A"]
    # force source tile to shared memory
    a.type.storage = "shared"

    if a.type.dtype not in cusolver_type_map.keys():
        raise TypeError("tile_cholesky() argument must be a tile of float32 or float64 entries")

    M, N = a.type.shape

    if not inplace:
        if len(return_values) != 1:
            raise TypeError("tile_cholesky() returns one output")
        out = return_values[0]

        # We already ensured a is square in tile_cholesky_generic_value_func()
        if out.type.shape[0] != M or out.type.shape[1] != M:
            raise ValueError("tile_cholesky() output tile must be square")

    arch = options["output_arch"]

    if (
        arch is None
        or not warp._src.context.runtime.core.wp_is_mathdx_enabled()
        or not options.get("enable_mathdx_solver", True)
    ):
        # CPU/no-MathDx/disabled dispatch -- falls into the cooperative scalar
        # branch via wp_is_null_func<Fwd>.
        if inplace:
            return ((0, a), [upper], [], 0)
        return ((0, 0, 0, a, out), [upper], [], 0)
    else:
        solver = "potrf"
        solver_enum = cusolver_function_map[solver]
        side_enum = cusolver_side_map["-"]
        diag_enum = cusolver_diag_map["-"]
        fill_mode = cusolver_fill_mode_map["upper" if upper else "lower"]
        dtype, precision_enum = cusolver_type_map[a.type.dtype]
        num_threads = options["block_dim"]
        parameter_list = f"({dtype}*, int*)"
        req_smem_bytes = a.type.size * type_size_in_bytes(a.type.dtype)
        if not inplace:
            req_smem_bytes *= 2
            if options["enable_backward"]:
                req_smem_bytes += 2 * M * M * type_size_in_bytes(a.type.dtype)

        # generate the forward LTO
        assert M == N
        lto_symbol, lto_code_data = warp._src.build.build_lto_solver(
            M,
            N,
            1,
            solver,
            solver_enum,
            side_enum,
            diag_enum,
            a.type.layout,
            a.type.layout if inplace else out.type.layout,
            fill_mode,
            arch,
            precision_enum,
            num_threads,
            parameter_list,
            builder,
            smem_estimate_bytes=req_smem_bytes,
        )

        if inplace:
            var = Var(lto_symbol, str, False, True, False)
            return ((var, a), [upper], [lto_code_data], 0)

        # for out-of-place Cholesky, build backward LTOs for adjoint
        # we need a GEMM and two trsm solves for the adjoint
        lto_list = [lto_code_data]
        if options["enable_backward"]:

            def tile_flip_layout(layout):
                if layout == "rowmajor":
                    return "colmajor"
                elif layout == "colmajor":
                    return "rowmajor"
                else:
                    raise ValueError(f"unexpected layout {layout!r}")

            # LTO to calculate transpose(L).adj_L or adj_U.transpose(U)
            # Lower: first operand (L) flipped -> L^T; Upper: second operand (U) flipped -> U^T
            gemm_layout_a = out.type.layout if upper else tile_flip_layout(out.type.layout)
            gemm_layout_b = tile_flip_layout(out.type.layout) if upper else out.type.layout
            fun_bkwd_gemm, lto_bkwd_gemm = warp._src.build.build_lto_dot(
                M,
                M,
                M,
                a.type.dtype,
                a.type.dtype,
                a.type.dtype,
                gemm_layout_a,
                gemm_layout_b,
                out.type.layout,
                arch,
                num_threads,
                builder,
            )
            # LTO to solve L^T @ X = Y (lower) or U @ X = Y (upper)
            fun_bkwd_trsm, lto_bkwd_trsm = warp._src.build.build_lto_solver(
                M,
                M,
                1,
                "trsm",
                cusolver_function_map["trsm"],
                cusolver_side_map["left"],
                cusolver_diag_map["nounit"],
                tile_flip_layout(out.type.layout) if not upper else out.type.layout,
                out.type.layout,
                cusolver_fill_mode_map["upper"],
                arch,
                precision_enum,
                num_threads,
                f"({dtype}*, {dtype}*)",
                builder,
                smem_estimate_bytes=req_smem_bytes,
            )
            lto_list.extend([lto_bkwd_gemm, lto_bkwd_trsm])
        else:
            fun_bkwd_gemm = 0
            fun_bkwd_trsm = 0

        var_fwd = Var(lto_symbol, str, False, True, False)
        if options["enable_backward"]:
            var_gemm = Var(fun_bkwd_gemm, str, False, True, False)
            var_trsm = Var(fun_bkwd_trsm, str, False, True, False)
        else:
            var_gemm = 0
            var_trsm = 0
        result = ((var_fwd, var_gemm, var_trsm, a, out), [upper], lto_list, 0)
        return result


def tile_cholesky_generic_lto_dispatch_func(*args, **kwargs):
    return _tile_cholesky_generic_lto_dispatch_func(False, *args, **kwargs)


def tile_cholesky_inplace_generic_lto_dispatch_func(*args, **kwargs):
    return _tile_cholesky_generic_lto_dispatch_func(True, *args, **kwargs)


add_builtin(
    "tile_cholesky",
    input_types={"A": tile(dtype=Float, shape=tuple[int, int]), "fill_mode": str},
    defaults={"fill_mode": "lower"},
    value_func=tile_cholesky_generic_value_func,
    lto_dispatch_func=tile_cholesky_generic_lto_dispatch_func,
    variadic=True,
    doc="""Compute the Cholesky factorization of a symmetric positive-definite matrix ``A``.

    When ``fill_mode="lower"`` (default), returns lower-triangular ``L`` such that ``LL^T = A``.
    When ``fill_mode="upper"``, returns upper-triangular ``U`` such that ``U^T U = A``.

    The ``fill_mode`` parameter must be a compile-time constant.

    Backward propagation computes gradients with respect to the corresponding
    triangular parameterization of ``A`` (lower triangle when ``fill_mode="lower"``,
    upper triangle when ``fill_mode="upper"``).

    Supported datatypes are:
        * float32
        * float64

    Args:
        A: A square, symmetric positive-definite matrix.
        fill_mode: ``"lower"`` (default) or ``"upper"``. Must be a compile-time constant.

    Returns:
        A triangular matrix ``L`` or ``U``.""",
    group="Tile Primitives",
    export=False,
    is_differentiable=True,
)


add_builtin(
    "tile_cholesky_inplace",
    input_types={"A": tile(dtype=Float, shape=tuple[int, int]), "fill_mode": str},
    defaults={"fill_mode": "lower"},
    value_func=tile_cholesky_inplace_generic_value_func,
    lto_dispatch_func=tile_cholesky_inplace_generic_lto_dispatch_func,
    variadic=True,
    doc="""Compute the Cholesky factorization of a symmetric positive-definite matrix ``A`` inplace.

    When ``fill_mode="lower"`` (default), the lower triangle of ``A`` is replaced by ``L``
    such that ``LL^T = A``; the upper triangle is set to zero.
    When ``fill_mode="upper"``, the upper triangle of ``A`` is replaced by ``U``
    such that ``U^T U = A``; the lower triangle is set to zero.

    The ``fill_mode`` parameter must be a compile-time constant.

    Note: This inplace variant does not support automatic differentiation (adjoint computation),
    but offers improved performance and uses half the shared memory compared to the standard version.

    Supported datatypes are:
        * float32
        * float64

    Args:
        A: A square, symmetric positive-definite matrix.
        fill_mode: ``"lower"`` (default) or ``"upper"``. Must be a compile-time constant.""",
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)


def _tile_cholesky_solve_generic_value_func(inplace: bool, arg_types, arg_values):
    if arg_types is None:
        if inplace:
            return None
        return tile(dtype=Float, shape=tuple[int])

    if len(arg_types) > 3:
        raise TypeError(
            f"tile_cholesky_solve() takes 2 positional arguments and 1 optional argument but {len(arg_types)} were given"
        )

    l = arg_types["L"]
    y = arg_types["y"]

    if not is_tile(l):
        raise TypeError(f"tile_cholesky_solve() 'L' argument must be a tile, got {l!r}")

    if not is_tile(y):
        raise TypeError(f"tile_cholesky_solve() 'y' argument must be a tile, got {y!r}")

    if not types_equal(l.dtype, y.dtype):
        raise TypeError(f"tile_cholesky_solve() arguments must have the same dtype, got {l.dtype} and {y.dtype}")

    if l.shape[0] != l.shape[1]:
        raise ValueError("tile_cholesky_solve() 'L' argument must be square")

    if len(y.shape) > 2 or len(y.shape) < 1:
        raise TypeError("tile_cholesky_solve() 'y' argument must be a 1D or 2D tile")

    if y.shape[0] != l.shape[0]:
        raise ValueError(
            f"tile_cholesky_solve() 'y' argument must have the same number of elements as the number of rows in 'L', "
            f"got {y.shape[0]} elements in 'y' and {l.shape[0]} rows in 'L'"
        )

    if inplace:
        return None
    return tile(dtype=l.dtype, shape=y.shape, layout=y.layout, strides=y.strides, storage="shared")


def tile_cholesky_solve_generic_value_func(arg_types, arg_values):
    return _tile_cholesky_solve_generic_value_func(False, arg_types, arg_values)


def tile_cholesky_solve_inplace_generic_value_func(arg_types, arg_values):
    return _tile_cholesky_solve_generic_value_func(True, arg_types, arg_values)


def _tile_cholesky_solve_generic_lto_dispatch_func(
    inplace: bool,
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp._src.context.ModuleBuilder,
):
    upper = _tile_cholesky_extract_fill_mode(arg_values, func_name="tile_cholesky_solve")
    L = arg_values["L"]
    y = arg_values["y"]
    # force the storage type of the input variables to shared memory
    L.type.storage = "shared"
    y.type.storage = "shared"

    M, N = L.type.shape

    if not inplace:
        if len(return_values) != 1:
            raise TypeError(f"tile_cholesky_solve() must return exactly one value, got {len(return_values)}")

        x = return_values[0]

        if len(x.type.shape) > 2 or len(x.type.shape) < 1:
            raise TypeError(f"tile_cholesky_solve() output vector must be 1D or 2D, got {len(x.type.shape)}-D")

        if x.type.shape[0] != M:
            raise ValueError(
                "tile_cholesky_solve() output vector must have same number of elements as the number of rows in 'L' "
                f"got {x.type.shape[0]} elements in output and {M} rows in 'L'"
            )

        if len(x.type.shape) > 1 and y.type.shape[1] != x.type.shape[1]:
            raise ValueError(
                "tile_cholesky_solve() output vector must have the same number of columns as 'y' "
                f"got {x.type.shape[1]} columns in output and {y.type.shape[1]} columns in 'y'"
            )

    if any(T not in cusolver_type_map.keys() for T in [y.type.dtype, L.type.dtype]):
        raise TypeError("tile_cholesky_solve() arguments must be tiles of float64 or float32")

    arch = options["output_arch"]

    if (
        arch is None
        or not warp._src.context.runtime.core.wp_is_mathdx_enabled()
        or not options.get("enable_mathdx_solver", True)
    ):
        # CPU/no-MathDx/disabled dispatch -- falls into the cooperative scalar
        # branch via wp_is_null_func<Fwd>.
        return ((0, L, y) if inplace else (0, L, y, x), [upper], [], 0)
    else:
        NRHS = y.type.shape[1] if len(y.type.shape) > 1 else 1
        solver = "potrs"
        solver_enum = cusolver_function_map[solver]
        side_enum = cusolver_side_map["-"]
        diag_enum = cusolver_diag_map["-"]
        fill_mode = cusolver_fill_mode_map["upper" if upper else "lower"]
        dtype, precision_enum = cusolver_type_map[L.type.dtype]
        num_threads = options["block_dim"]
        parameter_list = f"({dtype}*, {dtype}*)"
        req_smem_bytes = (y.type.size + L.type.size) * type_size_in_bytes(L.type.dtype)
        if not inplace:
            req_smem_bytes += x.type.size * type_size_in_bytes(L.type.dtype)

        # generate the LTO
        lto_symbol, lto_code_data = warp._src.build.build_lto_solver(
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
            smem_estimate_bytes=req_smem_bytes,
        )

        var = Var(lto_symbol, str, False, True, False)
        return ((var, L, y) if inplace else (var, L, y, x), [upper], [lto_code_data], 0)


def tile_cholesky_solve_generic_lto_dispatch_func(*args, **kwargs):
    return _tile_cholesky_solve_generic_lto_dispatch_func(False, *args, **kwargs)


def tile_cholesky_solve_inplace_generic_lto_dispatch_func(*args, **kwargs):
    return _tile_cholesky_solve_generic_lto_dispatch_func(True, *args, **kwargs)


add_builtin(
    "tile_cholesky_solve",
    input_types={
        "L": tile(dtype=Float, shape=tuple[int, int]),
        "y": tile(dtype=Float, shape=tuple[int]),
        "fill_mode": str,
    },
    defaults={"fill_mode": "lower"},
    value_func=tile_cholesky_solve_generic_value_func,
    lto_dispatch_func=tile_cholesky_solve_generic_lto_dispatch_func,
    variadic=True,
    doc="""Solve for ``x`` in ``Ax = y`` given the Cholesky factor of ``A``.

    When ``fill_mode="lower"`` (default), ``L`` is lower-triangular such that ``LL^T = A``.
    When ``fill_mode="upper"``, ``L`` is upper-triangular ``U`` such that ``U^T U = A``.

    The ``fill_mode`` parameter must be a compile-time constant.

    Note that computing the adjoint is not yet supported.

    Supported datatypes are:
        * float32
        * float64

    Args:
        L: A square triangular Cholesky factor of ``A``.
        y: A 1D or 2D tile of length ``M``.
        fill_mode: ``"lower"`` (default) or ``"upper"``. Must be a compile-time constant.

    Returns:
        A tile of the same shape as ``y`` such that ``Ax = y``.""",
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)


add_builtin(
    "tile_cholesky_solve_inplace",
    input_types={
        "L": tile(dtype=Float, shape=tuple[int, int]),
        "y": tile(dtype=Float, shape=tuple[int]),
        "fill_mode": str,
    },
    defaults={"fill_mode": "lower"},
    value_func=tile_cholesky_solve_inplace_generic_value_func,
    lto_dispatch_func=tile_cholesky_solve_inplace_generic_lto_dispatch_func,
    variadic=True,
    doc="""Solve for ``x`` in ``Ax = y`` by overwriting ``y`` with ``x``.

    When ``fill_mode="lower"`` (default), ``L`` is lower-triangular such that ``LL^T = A``.
    When ``fill_mode="upper"``, ``L`` is upper-triangular ``U`` such that ``U^T U = A``.

    The ``fill_mode`` parameter must be a compile-time constant.

    Note: This inplace variant does not support automatic differentiation (adjoint computation),
    but avoids allocating shared memory for the output ``x`` by reusing ``y``'s memory.

    Supported datatypes are:
        * float32
        * float64

    Args:
        L: A square triangular Cholesky factor of ``A``.
        y: A 1D or 2D tile of length ``M`` that gets overwritten by ``x`` where ``Ax = y``.
        fill_mode: ``"lower"`` (default) or ``"upper"``. Must be a compile-time constant.""",
    group="Tile Primitives",
    export=False,
    is_differentiable=False,
)


def _tile_lower_solve_generic_value_func(inplace: bool, arg_types, arg_values):
    if arg_types is None:
        if inplace:
            return None
        return tile(dtype=Float, shape=tuple[int])

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

    if inplace:
        return None
    return tile(dtype=l.dtype, shape=y.shape, layout=y.layout, strides=y.strides, storage="shared")


def tile_lower_solve_generic_value_func(arg_types, arg_values):
    return _tile_lower_solve_generic_value_func(False, arg_types, arg_values)


def tile_lower_solve_inplace_generic_value_func(arg_types, arg_values):
    return _tile_lower_solve_generic_value_func(True, arg_types, arg_values)


def _tile_lower_solve_generic_lto_dispatch_func(
    inplace: bool,
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp._src.context.ModuleBuilder,
):
    L = arg_values["L"]
    y = arg_values["y"]
    # force the storage type of the input variables to shared memory
    L.type.storage = "shared"
    y.type.storage = "shared"

    if any(T not in cusolver_type_map.keys() for T in [y.type.dtype, L.type.dtype]):
        raise TypeError("tile_lower_solve() arguments must be tiles of float64 or float32")

    M, N = L.type.shape

    if not inplace:
        if len(return_values) != 1:
            raise TypeError(f"tile_lower_solve() must return exactly one value, got {len(return_values)}")

        z = return_values[0]

        if len(z.type.shape) > 2 or len(z.type.shape) < 1:
            raise TypeError(f"tile_lower_solve() output vector must be 1D or 2D, got {len(z.type.shape)}-D")

        if z.type.shape[0] != M:
            raise ValueError(
                "tile_lower_solve() output vector must have same number of elements as the number of rows in 'L' "
                f"got {z.type.shape[0]} elements in output and {M} rows in 'L'"
            )

        if len(z.type.shape) > 1 and y.type.shape[1] != z.type.shape[1]:
            raise ValueError(
                "tile_lower_solve() output vector must have the same number of columns as 'y' "
                f"got {z.type.shape[1]} columns in output and {y.type.shape[1]} columns in 'y'"
            )

    arch = options["output_arch"]

    if (
        arch is None
        or not warp._src.context.runtime.core.wp_is_mathdx_enabled()
        or not options.get("enable_mathdx_solver", True)
    ):
        return ((0, L, y) if inplace else (0, 0, L, y, z), [], [], 0)
    else:
        NRHS = y.type.shape[1] if len(y.type.shape) > 1 else 1
        solver = "trsm"
        solver_enum = cusolver_function_map[solver]
        side_enum = cusolver_side_map["left"]
        diag_enum = cusolver_diag_map["nounit"]
        fill_mode = cusolver_fill_mode_map["lower"]
        dtype, precision_enum = cusolver_type_map[L.type.dtype]
        num_threads = options["block_dim"]
        parameter_list = f"({dtype}*, {dtype}*)"
        req_smem_bytes = (y.type.size + L.type.size) * type_size_in_bytes(L.type.dtype)
        if not inplace:
            req_smem_bytes += z.type.size * type_size_in_bytes(L.type.dtype)
            if options["enable_backward"]:
                # backward buffer: one row-major M x NRHS raw buffer,
                # preloaded with adj_ret and overwritten by the transposed solve
                # to hold w in L^T w = adj_ret
                req_smem_bytes += M * NRHS * type_size_in_bytes(L.type.dtype)

        # generate the forward LTO
        assert M == N
        lto_symbol, lto_code_data = warp._src.build.build_lto_solver(
            M,
            NRHS,
            1,
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
            smem_estimate_bytes=req_smem_bytes,
        )

        if inplace:
            var = Var(lto_symbol, str, False, True, False)
            return ((var, L, y), [], [lto_code_data], 0)

        # for out-of-place solves, build the backward TRSM for the adjoint.
        # The adjoint of L Z = Y is the transposed solve L^T W = adj_ret, followed
        # by adj_Y += W and adj_L -= tril(W @ Z^T). For a vector right-hand side,
        # W @ Z^T reduces to outer(w, z).
        lto_list = [lto_code_data]
        if options["enable_backward"]:

            def tile_flip_layout(layout):
                if layout == "rowmajor":
                    return "colmajor"
                elif layout == "colmajor":
                    return "rowmajor"
                else:
                    raise ValueError(f"unexpected layout {layout!r}")

            fun_bkwd_trsm, lto_bkwd_trsm = warp._src.build.build_lto_solver(
                M,
                NRHS,
                1,
                solver,
                solver_enum,
                side_enum,
                diag_enum,
                tile_flip_layout(L.type.layout),
                "rowmajor",
                cusolver_fill_mode_map["upper"],
                arch,
                precision_enum,
                num_threads,
                parameter_list,
                builder,
                smem_estimate_bytes=req_smem_bytes,
            )
            lto_list.append(lto_bkwd_trsm)
        else:
            fun_bkwd_trsm = 0

        var_fwd = Var(lto_symbol, str, False, True, False)
        var_bkwd = Var(fun_bkwd_trsm, str, False, True, False) if options["enable_backward"] else 0
        return ((var_fwd, var_bkwd, L, y, z), [], lto_list, 0)


def tile_lower_solve_generic_lto_dispatch_func(*args, **kwargs):
    return _tile_lower_solve_generic_lto_dispatch_func(False, *args, **kwargs)


def tile_lower_solve_inplace_generic_lto_dispatch_func(*args, **kwargs):
    return _tile_lower_solve_generic_lto_dispatch_func(True, *args, **kwargs)


add_builtin(
    "tile_lower_solve",
    input_types={"L": tile(dtype=Float, shape=tuple[int, int]), "y": tile(dtype=Float, shape=tuple[int])},
    value_func=tile_lower_solve_generic_value_func,
    lto_dispatch_func=tile_lower_solve_generic_lto_dispatch_func,
    variadic=True,
    doc="""Solve for ``z`` in ``Lz = y``, where ``L`` is a lower triangular matrix.

    This performs general forward substitution for a lower triangular system.

    Backward propagation computes gradients with respect to ``y`` and the
    lower-triangular parameterization of ``L``. The strictly upper triangle of
    ``L`` does not affect the solve and receives zero gradient.

    Supported datatypes are:
        * float32
        * float64

    Args:
        L: A square, non-singular, lower triangular matrix
        y: A 1D or 2D tile with compatible shape

    Returns:
        A tile of the same shape as ``y`` such that ``Lz = y``.""",
    group="Tile Primitives",
    export=False,
    namespace="",
    is_differentiable=True,
)


add_builtin(
    "tile_lower_solve_inplace",
    input_types={"L": tile(dtype=Float, shape=tuple[int, int]), "y": tile(dtype=Float, shape=tuple[int])},
    value_func=tile_lower_solve_inplace_generic_value_func,
    lto_dispatch_func=tile_lower_solve_inplace_generic_lto_dispatch_func,
    variadic=True,
    doc="""Solve for ``z`` in ``Lz = y``, where ``L`` is a lower triangular matrix by overwriting ``y`` with ``z``.

    This performs general forward substitution for a lower triangular system inplace.

    Note: This inplace variant does not support automatic differentiation (adjoint computation),
    but avoids allocating shared memory for the output ``z`` by reusing ``y``'s memory.

    Supported datatypes are:
        * float32
        * float64

    Args:
        L: A square, non-singular, lower triangular matrix
        y: A 1D or 2D tile with compatible shape that gets overwritten by ``z`` where ``Lz = y``.""",
    group="Tile Primitives",
    export=False,
    namespace="",
    is_differentiable=False,
)


def _tile_upper_solve_generic_value_func(inplace: bool, arg_types, arg_values):
    if arg_types is None:
        if inplace:
            return None
        return tile(dtype=Float, shape=tuple[int])

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

    if inplace:
        return None
    return tile(dtype=u.dtype, shape=z.shape, layout=z.layout, strides=z.strides, storage="shared")


def tile_upper_solve_generic_value_func(arg_types, arg_values):
    return _tile_upper_solve_generic_value_func(False, arg_types, arg_values)


def tile_upper_solve_inplace_generic_value_func(arg_types, arg_values):
    return _tile_upper_solve_generic_value_func(True, arg_types, arg_values)


def _tile_upper_solve_generic_lto_dispatch_func(
    inplace: bool,
    arg_types: Mapping[str, type],
    return_type: Any,
    return_values: List[Var],
    arg_values: Mapping[str, Var],
    options: Mapping[str, Any],
    builder: warp._src.context.ModuleBuilder,
):
    U = arg_values["U"]
    z = arg_values["z"]
    # force the storage type of the input variables to shared memory
    U.type.storage = "shared"
    z.type.storage = "shared"

    if any(T not in cusolver_type_map.keys() for T in [z.type.dtype, U.type.dtype]):
        raise TypeError("tile_upper_solve() arguments must be tiles of float64 or float32")

    M, N = U.type.shape

    if not inplace:
        if len(return_values) != 1:
            raise TypeError(f"tile_upper_solve() must return exactly one value, got {len(return_values)}")

        x = return_values[0]

        if len(x.type.shape) > 2 or len(x.type.shape) < 1:
            raise TypeError(f"tile_upper_solve() output tile must be 1D or 2D, got {len(x.type.shape)}-D")

        if x.type.shape[0] != M:
            raise ValueError(
                "tile_upper_solve() output tile must have same number of elements as the number of rows in 'U' "
                f"got {x.type.shape[0]} elements in output and {M} rows in 'U'"
            )

        if len(x.type.shape) > 1 and z.type.shape[1] != x.type.shape[1]:
            raise ValueError(
                "tile_upper_solve() output vector must have the same number of columns as 'z' "
                f"got {x.type.shape[1]} columns in output and {z.type.shape[1]} columns in 'z'"
            )

    arch = options["output_arch"]

    if (
        arch is None
        or not warp._src.context.runtime.core.wp_is_mathdx_enabled()
        or not options.get("enable_mathdx_solver", True)
    ):
        # CPU/no-MathDx/disabled dispatch -- falls into the cooperative scalar
        # branch via wp_is_null_func<Fwd>.
        return ((0, U, z) if inplace else (0, U, z, x), [], [], 0)
    else:
        NRHS = z.type.shape[1] if len(z.type.shape) > 1 else 1
        solver = "trsm"
        solver_enum = cusolver_function_map[solver]
        side_enum = cusolver_side_map["left"]
        diag_enum = cusolver_diag_map["nounit"]
        fill_mode = cusolver_fill_mode_map["upper"]
        dtype, precision_enum = cusolver_type_map[U.type.dtype]
        num_threads = options["block_dim"]
        parameter_list = f"({dtype}*, {dtype}*)"
        req_smem_bytes = (z.type.size + U.type.size) * type_size_in_bytes(U.type.dtype)
        if not inplace:
            req_smem_bytes += x.type.size * type_size_in_bytes(U.type.dtype)

        # generate the LTO
        assert M == N
        lto_symbol, lto_code_data = warp._src.build.build_lto_solver(
            M,
            NRHS,
            1,
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
            smem_estimate_bytes=req_smem_bytes,
        )

        var = Var(lto_symbol, str, False, True, False)
        return ((var, U, z) if inplace else (var, U, z, x), [], [lto_code_data], 0)


def tile_upper_solve_generic_lto_dispatch_func(*args, **kwargs):
    return _tile_upper_solve_generic_lto_dispatch_func(False, *args, **kwargs)


def tile_upper_solve_inplace_generic_lto_dispatch_func(*args, **kwargs):
    return _tile_upper_solve_generic_lto_dispatch_func(True, *args, **kwargs)


add_builtin(
    "tile_upper_solve",
    input_types={"U": tile(dtype=Float, shape=tuple[int, int]), "z": tile(dtype=Float, shape=tuple[int])},
    value_func=tile_upper_solve_generic_value_func,
    lto_dispatch_func=tile_upper_solve_generic_lto_dispatch_func,
    variadic=True,
    doc="""Solve for ``x`` in ``Ux = z``, where ``U`` is an upper triangular matrix.

    This performs general back substitution for upper triangular systems.

    Note that computing the adjoint is not yet supported.

    Supported datatypes are:
        * float32
        * float64

    Args:
        U: A square, non-singular, upper triangular matrix
        z: A 1D or 2D tile with compatible shape

    Returns:
        A tile of the same shape as ``z`` such that ``Ux = z``.""",
    group="Tile Primitives",
    export=False,
    namespace="",
    is_differentiable=False,
)


add_builtin(
    "tile_upper_solve_inplace",
    input_types={"U": tile(dtype=Float, shape=tuple[int, int]), "z": tile(dtype=Float, shape=tuple[int])},
    value_func=tile_upper_solve_inplace_generic_value_func,
    lto_dispatch_func=tile_upper_solve_inplace_generic_lto_dispatch_func,
    variadic=True,
    doc="""Solve for ``x`` in ``Ux = z``, where ``U`` is an upper triangular matrix by overwriting ``z`` with ``x``.

    This performs general back substitution for upper triangular systems inplace.

    Note: This inplace variant does not support automatic differentiation (adjoint computation),
    but avoids allocating shared memory for the output ``x`` by reusing ``z``'s memory.

    Supported datatypes are:
        * float32
        * float64

    Args:
        U: A square, non-singular, upper triangular matrix
        z: A 1D or 2D tile with compatible shape that gets overwritten by ``x`` where ``Ux = z``.""",
    group="Tile Primitives",
    export=False,
    namespace="",
    is_differentiable=False,
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
    is_differentiable=False,
)


def static(expr):
    """
    Evaluate a static expression and replace the expression with its result.

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
    doc="""Query the length of ``a``.

    Returns:
        The number of elements for vectors, quaternions, and transformations; the number
        of rows for matrices and tiles; or the size of the leading dimension for arrays.""",
    group="Utility",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "len",
    input_types={"a": quaternion(dtype=Float)},
    value_func=static_len_value_func,
    doc="""Query the length of ``a``.

    Returns:
        The number of elements for vectors, quaternions, and transformations; the number
        of rows for matrices and tiles; or the size of the leading dimension for arrays.""",
    group="Utility",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "len",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=static_len_value_func,
    doc="""Query the length of ``a``.

    Returns:
        The number of elements for vectors, quaternions, and transformations; the number
        of rows for matrices and tiles; or the size of the leading dimension for arrays.""",
    group="Utility",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "len",
    input_types={"a": transformation(dtype=Float)},
    value_func=static_len_value_func,
    doc="""Query the length of ``a``.

    Returns:
        The number of elements for vectors, quaternions, and transformations; the number
        of rows for matrices and tiles; or the size of the leading dimension for arrays.""",
    group="Utility",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "len",
    input_types={"a": array(dtype=Any)},
    value_type=int,
    doc="""Query the length of ``a``.

    Returns:
        The number of elements for vectors, quaternions, and transformations; the number
        of rows for matrices and tiles; or the size of the leading dimension for arrays.""",
    group="Utility",
    export=False,
    is_differentiable=False,
)

add_builtin(
    "len",
    input_types={"a": tile(dtype=Any, shape=tuple[int, ...])},
    value_func=static_len_value_func,
    doc="""Query the length of ``a``.

    Returns:
        The number of elements for vectors, quaternions, and transformations; the number
        of rows for matrices and tiles; or the size of the leading dimension for arrays.""",
    group="Utility",
    export=False,
    is_differentiable=False,
)


def cast_value_func(arg_types, arg_values):
    # Return generic type for doc builds.
    if arg_types is None:
        return Any

    return arg_values["dtype"]


def cast_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Var]):
    func_args = (args["a"],)
    template_args = (args["dtype"],)
    return (func_args, template_args)


add_builtin(
    "cast",
    input_types={"a": Any, "dtype": Any},
    value_func=cast_value_func,
    dispatch_func=cast_dispatch_func,
    group="Utility",
    export=False,
    is_differentiable=False,
    doc="""Reinterpret a value as a different type while preserving its bit pattern.

    Args:
        a: The value to cast
        dtype: The target type.

    Example:

        .. code-block:: python

            @wp.struct
            class MyStruct:
                f: wp.float16
                i: wp.int16


            @wp.kernel
            def compute():
                x = wp.int32(0x40000000)
                x_casted = wp.cast(x, wp.float32)
                wp.expect_eq(x_casted, 2.0) # 0x40000000

                s = MyStruct()
                s.f = wp.float16(2.0) # 0x4000
                s.i = wp.int16(4096) # 0x1000
                s_casted = wp.cast(s, wp.int32)
                wp.expect_eq(s_casted, 0x10004000)


            wp.launch(compute, dim=1)
    """,
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
    doc="Construct a tuple from a list of values.",
    group="Utility",
    hidden=True,
    is_differentiable=False,
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
    input_types={"a": tuple, "i": int},
    value_func=tuple_extract_value_func,
    dispatch_func=tuple_extract_dispatch_func,
    group="Utility",
    hidden=True,
    is_differentiable=False,
)


add_builtin(
    "len",
    input_types={"a": tuple},
    value_func=static_len_value_func,
    doc="""Query the length of ``a``.

    Returns:
        The number of elements for vectors, quaternions, and transformations; the number
        of rows for matrices and tiles; or the size of the leading dimension for arrays.""",
    group="Utility",
    export=False,
    is_differentiable=False,
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
    is_differentiable=False,
)
