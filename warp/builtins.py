# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import builtins
from typing import Any, Callable, Tuple

from warp.codegen import Reference
from warp.types import *

from .context import add_builtin


def sametypes(arg_types):
    return all(types_equal(arg_types[0], t) for t in arg_types[1:])


def sametype_value_func(default):
    def fn(arg_types, kwds, _):
        if arg_types is None:
            return default
        if not sametypes(arg_types):
            raise RuntimeError(f"Input types must be the same, found: {[type_repr(t) for t in arg_types]}")
        return arg_types[0]

    return fn


# ---------------------------------
# Scalar Math

add_builtin(
    "min",
    input_types={"x": Scalar, "y": Scalar},
    value_func=sametype_value_func(Scalar),
    doc="Return the minimum of two scalars.",
    group="Scalar Math",
)

add_builtin(
    "max",
    input_types={"x": Scalar, "y": Scalar},
    value_func=sametype_value_func(Scalar),
    doc="Return the maximum of two scalars.",
    group="Scalar Math",
)

add_builtin(
    "clamp",
    input_types={"x": Scalar, "a": Scalar, "b": Scalar},
    value_func=sametype_value_func(Scalar),
    doc="Clamp the value of ``x`` to the range [a, b].",
    group="Scalar Math",
)

add_builtin(
    "abs",
    input_types={"x": Scalar},
    value_func=sametype_value_func(Scalar),
    doc="Return the absolute value of ``x``.",
    group="Scalar Math",
)
add_builtin(
    "sign",
    input_types={"x": Scalar},
    value_func=sametype_value_func(Scalar),
    doc="Return -1 if ``x`` < 0, return 1 otherwise.",
    group="Scalar Math",
)

add_builtin(
    "step",
    input_types={"x": Scalar},
    value_func=sametype_value_func(Scalar),
    doc="Return 1.0 if ``x`` < 0.0, return 0.0 otherwise.",
    group="Scalar Math",
)
add_builtin(
    "nonzero",
    input_types={"x": Scalar},
    value_func=sametype_value_func(Scalar),
    doc="Return 1.0 if ``x`` is not equal to zero, return 0.0 otherwise.",
    group="Scalar Math",
)

add_builtin(
    "sin",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the sine of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "cos",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the cosine of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "acos",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return arccos of ``x`` in radians. Inputs are automatically clamped to [-1.0, 1.0].",
    group="Scalar Math",
)
add_builtin(
    "asin",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return arcsin of ``x`` in radians. Inputs are automatically clamped to [-1.0, 1.0].",
    group="Scalar Math",
)
add_builtin(
    "sqrt",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the square root of ``x``, where ``x`` is positive.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "cbrt",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the cube root of ``x``.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "tan",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the tangent of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "atan",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the arctangent of ``x`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "atan2",
    input_types={"y": Float, "x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the 2-argument arctangent, atan2, of the point ``(x, y)`` in radians.",
    group="Scalar Math",
)
add_builtin(
    "sinh",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the sinh of ``x``.",
    group="Scalar Math",
)
add_builtin(
    "cosh",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the cosh of ``x``.",
    group="Scalar Math",
)
add_builtin(
    "tanh",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the tanh of ``x``.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "degrees",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Convert ``x`` from radians into degrees.",
    group="Scalar Math",
)
add_builtin(
    "radians",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Convert ``x`` from degrees into radians.",
    group="Scalar Math",
)

add_builtin(
    "log",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the natural logarithm (base-e) of ``x``, where ``x`` is positive.",
    group="Scalar Math",
)
add_builtin(
    "log2",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the binary logarithm (base-2) of ``x``, where ``x`` is positive.",
    group="Scalar Math",
)
add_builtin(
    "log10",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the common logarithm (base-10) of ``x``, where ``x`` is positive.",
    group="Scalar Math",
)
add_builtin(
    "exp",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    doc="Return the value of the exponential function :math:`e^x`.",
    group="Scalar Math",
    require_original_output_arg=True,
)
add_builtin(
    "pow",
    input_types={"x": Float, "y": Float},
    value_func=sametype_value_func(Float),
    doc="Return the result of ``x`` raised to power of ``y``.",
    group="Scalar Math",
    require_original_output_arg=True,
)

add_builtin(
    "round",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    group="Scalar Math",
    doc="""Return the nearest integer value to ``x``, rounding halfway cases away from zero.

    This is the most intuitive form of rounding in the colloquial sense, but can be slower than other options like :func:`warp.rint()`.
    Differs from :func:`numpy.round()`, which behaves the same way as :func:`numpy.rint()`.""",
)

add_builtin(
    "rint",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    group="Scalar Math",
    doc="""Return the nearest integer value to ``x``, rounding halfway cases to nearest even integer.

    It is generally faster than :func:`warp.round()`. Equivalent to :func:`numpy.rint()`.""",
)

add_builtin(
    "trunc",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    group="Scalar Math",
    doc="""Return the nearest integer that is closer to zero than ``x``.

    In other words, it discards the fractional part of ``x``.
    It is similar to casting ``float(int(x))``, but preserves the negative sign when x is in the range [-0.0, -1.0).
    Equivalent to :func:`numpy.trunc()` and :func:`numpy.fix()`.""",
)

add_builtin(
    "floor",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    group="Scalar Math",
    doc="""Return the largest integer that is less than or equal to ``x``.""",
)

add_builtin(
    "ceil",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    group="Scalar Math",
    doc="""Return the smallest integer that is greater than or equal to ``x``.""",
)

add_builtin(
    "frac",
    input_types={"x": Float},
    value_func=sametype_value_func(Float),
    group="Scalar Math",
    doc="""Retrieve the fractional part of x.

    In other words, it discards the integer part of x and is equivalent to ``x - trunc(x)``.""",
)


def infer_scalar_type(arg_types):
    if arg_types is None:
        return Scalar

    def iterate_scalar_types(arg_types):
        for t in arg_types:
            if hasattr(t, "_wp_scalar_type_"):
                yield t._wp_scalar_type_
            elif t in scalar_and_bool_types:
                yield t

    scalarTypes = set(iterate_scalar_types(arg_types))
    if len(scalarTypes) > 1:
        raise RuntimeError(
            f"Couldn't figure out return type as arguments have multiple precisions: {list(scalarTypes)}"
        )
    return list(scalarTypes)[0]


def sametype_scalar_value_func(arg_types, kwds, _):
    if arg_types is None:
        return Scalar
    if not sametypes(arg_types):
        raise RuntimeError(f"Input types must be exactly the same, {list(arg_types)}")

    return infer_scalar_type(arg_types)


# ---------------------------------
# Vector Math

add_builtin(
    "dot",
    input_types={"x": vector(length=Any, dtype=Scalar), "y": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_scalar_value_func,
    group="Vector Math",
    doc="Compute the dot product between two vectors.",
)
add_builtin(
    "ddot",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar), "y": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_scalar_value_func,
    group="Vector Math",
    doc="Compute the double dot product between two matrices.",
)

add_builtin(
    "min",
    input_types={"x": vector(length=Any, dtype=Scalar), "y": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_value_func(vector(length=Any, dtype=Scalar)),
    doc="Return the element-wise minimum of two vectors.",
    group="Vector Math",
)
add_builtin(
    "max",
    input_types={"x": vector(length=Any, dtype=Scalar), "y": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_value_func(vector(length=Any, dtype=Scalar)),
    doc="Return the element-wise maximum of two vectors.",
    group="Vector Math",
)

add_builtin(
    "min",
    input_types={"v": vector(length=Any, dtype=Scalar)},
    value_func=sametype_scalar_value_func,
    doc="Return the minimum element of a vector ``v``.",
    group="Vector Math",
)
add_builtin(
    "max",
    input_types={"v": vector(length=Any, dtype=Scalar)},
    value_func=sametype_scalar_value_func,
    doc="Return the maximum element of a vector ``v``.",
    group="Vector Math",
)

add_builtin(
    "argmin",
    input_types={"v": vector(length=Any, dtype=Scalar)},
    value_func=lambda arg_types, kwds, _: warp.uint32,
    doc="Return the index of the minimum element of a vector ``v``.",
    group="Vector Math",
    missing_grad=True,
)
add_builtin(
    "argmax",
    input_types={"v": vector(length=Any, dtype=Scalar)},
    value_func=lambda arg_types, kwds, _: warp.uint32,
    doc="Return the index of the maximum element of a vector ``v``.",
    group="Vector Math",
    missing_grad=True,
)


def value_func_outer(arg_types, kwds, _):
    if arg_types is None:
        return matrix(shape=(Any, Any), dtype=Scalar)

    scalarType = infer_scalar_type(arg_types)
    vectorLengths = [t._length_ for t in arg_types]
    return matrix(shape=(vectorLengths), dtype=scalarType)


add_builtin(
    "outer",
    input_types={"x": vector(length=Any, dtype=Scalar), "y": vector(length=Any, dtype=Scalar)},
    value_func=value_func_outer,
    group="Vector Math",
    doc="Compute the outer product ``x*y^T`` for two vectors.",
)

add_builtin(
    "cross",
    input_types={"x": vector(length=3, dtype=Scalar), "y": vector(length=3, dtype=Scalar)},
    value_func=sametype_value_func(vector(length=3, dtype=Scalar)),
    group="Vector Math",
    doc="Compute the cross product of two 3D vectors.",
)
add_builtin(
    "skew",
    input_types={"x": vector(length=3, dtype=Scalar)},
    value_func=lambda arg_types, kwds, _: matrix(shape=(3, 3), dtype=arg_types[0]._wp_scalar_type_),
    group="Vector Math",
    doc="Compute the skew-symmetric 3x3 matrix for a 3D vector ``x``.",
)

add_builtin(
    "length",
    input_types={"x": vector(length=Any, dtype=Float)},
    value_func=sametype_scalar_value_func,
    group="Vector Math",
    doc="Compute the length of a floating-point vector ``x``.",
    require_original_output_arg=True,
)
add_builtin(
    "length",
    input_types={"x": quaternion(dtype=Float)},
    value_func=sametype_scalar_value_func,
    group="Vector Math",
    doc="Compute the length of a quaternion ``x``.",
    require_original_output_arg=True,
)
add_builtin(
    "length_sq",
    input_types={"x": vector(length=Any, dtype=Scalar)},
    value_func=sametype_scalar_value_func,
    group="Vector Math",
    doc="Compute the squared length of a vector ``x``.",
)
add_builtin(
    "length_sq",
    input_types={"x": quaternion(dtype=Scalar)},
    value_func=sametype_scalar_value_func,
    group="Vector Math",
    doc="Compute the squared length of a quaternion ``x``.",
)
add_builtin(
    "normalize",
    input_types={"x": vector(length=Any, dtype=Float)},
    value_func=sametype_value_func(vector(length=Any, dtype=Scalar)),
    group="Vector Math",
    doc="Compute the normalized value of ``x``. If ``length(x)`` is 0 then the zero vector is returned.",
    require_original_output_arg=True,
)
add_builtin(
    "normalize",
    input_types={"x": quaternion(dtype=Float)},
    value_func=sametype_value_func(quaternion(dtype=Scalar)),
    group="Vector Math",
    doc="Compute the normalized value of ``x``. If ``length(x)`` is 0, then the zero quaternion is returned.",
)

add_builtin(
    "transpose",
    input_types={"m": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=lambda arg_types, kwds, _: matrix(
        shape=(arg_types[0]._shape_[1], arg_types[0]._shape_[0]), dtype=arg_types[0]._wp_scalar_type_
    ),
    group="Vector Math",
    doc="Return the transpose of the matrix ``m``.",
)


def value_func_mat_inv(arg_types, kwds, _):
    if arg_types is None:
        return matrix(shape=(Any, Any), dtype=Float)
    return arg_types[0]


add_builtin(
    "inverse",
    input_types={"m": matrix(shape=(2, 2), dtype=Float)},
    value_func=value_func_mat_inv,
    group="Vector Math",
    doc="Return the inverse of a 2x2 matrix ``m``.",
    require_original_output_arg=True,
)

add_builtin(
    "inverse",
    input_types={"m": matrix(shape=(3, 3), dtype=Float)},
    value_func=value_func_mat_inv,
    group="Vector Math",
    doc="Return the inverse of a 3x3 matrix ``m``.",
    require_original_output_arg=True,
)

add_builtin(
    "inverse",
    input_types={"m": matrix(shape=(4, 4), dtype=Float)},
    value_func=value_func_mat_inv,
    group="Vector Math",
    doc="Return the inverse of a 4x4 matrix ``m``.",
    require_original_output_arg=True,
)


def value_func_mat_det(arg_types, kwds, _):
    if arg_types is None:
        return Scalar
    return arg_types[0]._wp_scalar_type_


add_builtin(
    "determinant",
    input_types={"m": matrix(shape=(2, 2), dtype=Float)},
    value_func=value_func_mat_det,
    group="Vector Math",
    doc="Return the determinant of a 2x2 matrix ``m``.",
)

add_builtin(
    "determinant",
    input_types={"m": matrix(shape=(3, 3), dtype=Float)},
    value_func=value_func_mat_det,
    group="Vector Math",
    doc="Return the determinant of a 3x3 matrix ``m``.",
)

add_builtin(
    "determinant",
    input_types={"m": matrix(shape=(4, 4), dtype=Float)},
    value_func=value_func_mat_det,
    group="Vector Math",
    doc="Return the determinant of a 4x4 matrix ``m``.",
)


def value_func_mat_trace(arg_types, kwds, _):
    if arg_types is None:
        return Scalar
    if arg_types[0]._shape_[0] != arg_types[0]._shape_[1]:
        raise RuntimeError(f"Matrix shape is {arg_types[0]._shape_}. Cannot find the trace of non square matrices")
    return arg_types[0]._wp_scalar_type_


add_builtin(
    "trace",
    input_types={"m": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=value_func_mat_trace,
    group="Vector Math",
    doc="Return the trace of the matrix ``m``.",
)


def value_func_diag(arg_types, kwds, _):
    if arg_types is None:
        return matrix(shape=(Any, Any), dtype=Scalar)
    else:
        return matrix(shape=(arg_types[0]._length_, arg_types[0]._length_), dtype=arg_types[0]._wp_scalar_type_)


add_builtin(
    "diag",
    input_types={"d": vector(length=Any, dtype=Scalar)},
    value_func=value_func_diag,
    group="Vector Math",
    doc="Returns a matrix with the components of the vector ``d`` on the diagonal.",
)


def value_func_get_diag(arg_types, kwds, _):
    if arg_types is None:
        return vector(length=(Any), dtype=Scalar)
    else:
        if arg_types[0]._shape_[0] != arg_types[0]._shape_[1]:
            raise RuntimeError(
                f"Matrix shape is {arg_types[0]._shape_}; get_diag is only available for square matrices."
            )
        return vector(length=arg_types[0]._shape_[0], dtype=arg_types[0]._wp_scalar_type_)


add_builtin(
    "get_diag",
    input_types={"m": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=value_func_get_diag,
    group="Vector Math",
    doc="Returns a vector containing the diagonal elements of the square matrix ``m``.",
)

add_builtin(
    "cw_mul",
    input_types={"x": vector(length=Any, dtype=Scalar), "y": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_value_func(vector(length=Any, dtype=Scalar)),
    group="Vector Math",
    doc="Component-wise multiplication of two vectors.",
)
add_builtin(
    "cw_div",
    input_types={"x": vector(length=Any, dtype=Scalar), "y": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_value_func(vector(length=Any, dtype=Scalar)),
    group="Vector Math",
    doc="Component-wise division of two vectors.",
    require_original_output_arg=True,
)

add_builtin(
    "cw_mul",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar), "y": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    group="Vector Math",
    doc="Component-wise multiplication of two matrices.",
)
add_builtin(
    "cw_div",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar), "y": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
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
            input_types={"u": u},
            value_type=t,
            doc="",
            hidden=True,
            group="Scalar Math",
            export=False,
            namespace="wp::" if t is not bool else "",
        )


def vector_constructor_func(arg_types, kwds, templates):
    if arg_types is None:
        return vector(length=Any, dtype=Scalar)

    if templates is None or len(templates) == 0:
        # handle construction of anonymous (undeclared) vector types

        if "length" in kwds:
            if len(arg_types) == 0:
                if "dtype" not in kwds:
                    raise RuntimeError(
                        "vec() must have dtype as a keyword argument if it has no positional arguments, e.g.: wp.vector(length=5, dtype=wp.float32)"
                    )

                # zero initialization e.g.: wp.vector(length=5, dtype=wp.float32)
                veclen = kwds["length"]
                vectype = kwds["dtype"]

            elif len(arg_types) == 1:
                # value initialization e.g.: wp.vec(1.0, length=5)
                veclen = kwds["length"]
                vectype = arg_types[0]
                if type_is_vector(vectype):
                    # constructor from another vector
                    if vectype._length_ != veclen:
                        raise RuntimeError(
                            f"Incompatible vector lengths for casting copy constructor, {veclen} vs {vectype._length_}"
                        )
                    vectype = vectype._wp_scalar_type_
            else:
                raise RuntimeError(
                    "vec() must have one scalar argument or the dtype keyword argument if the length keyword argument is specified, e.g.: wp.vec(1.0, length=5)"
                )

        else:
            if len(arg_types) == 0:
                raise RuntimeError(
                    "vec() must have at least one numeric argument, if it's length, dtype is not specified"
                )

            if "dtype" in kwds:
                # casting constructor
                if len(arg_types) == 1 and types_equal(
                    arg_types[0], vector(length=Any, dtype=Scalar), match_generic=True
                ):
                    veclen = arg_types[0]._length_
                    vectype = kwds["dtype"]
                    templates.append(veclen)
                    templates.append(vectype)
                    return vector(length=veclen, dtype=vectype)
                raise RuntimeError(
                    "vec() should not have dtype specified if numeric arguments are given, the dtype will be inferred from the argument types"
                )

            # component wise construction of an anonymous vector, e.g. wp.vec(wp.float16(1.0), wp.float16(2.0), ....)
            # we infer the length and data type from the number and type of the arg values
            veclen = len(arg_types)
            vectype = arg_types[0]

            if len(arg_types) == 1 and type_is_vector(vectype):
                # constructor from another vector
                veclen = vectype._length_
                vectype = vectype._wp_scalar_type_
            elif not all(vectype == t for t in arg_types):
                raise RuntimeError(
                    f"All numeric arguments to vec() constructor should have the same type, expected {veclen} arg_types of type {vectype}, received { ','.join([str(t) for t in arg_types]) }"
                )

        # update the templates list, so we can generate vec<len, type>() correctly in codegen
        templates.append(veclen)
        templates.append(vectype)

    else:
        # construction of a predeclared type, e.g.: vec5d
        veclen, vectype = templates
        if len(arg_types) == 1 and type_is_vector(arg_types[0]):
            # constructor from another vector
            if arg_types[0]._length_ != veclen:
                raise RuntimeError(
                    f"Incompatible matrix sizes for casting copy constructor, {veclen} vs {arg_types[0]._length_}"
                )
        elif not all(vectype == t for t in arg_types):
            raise RuntimeError(
                f"All numeric arguments to vec() constructor should have the same type, expected {veclen} arg_types of type {vectype}, received { ','.join([str(t) for t in arg_types]) }"
            )

    retvalue = vector(length=veclen, dtype=vectype)
    return retvalue


add_builtin(
    "vector",
    input_types={"*arg_types": Scalar, "length": int, "dtype": Scalar},
    variadic=True,
    initializer_list_func=lambda arg_types, _: len(arg_types) > 4,
    value_func=vector_constructor_func,
    native_func="vec_t",
    doc="Construct a vector of with given length and dtype.",
    group="Vector Math",
    export=False,
)


def matrix_constructor_func(arg_types, kwds, templates):
    if arg_types is None:
        return matrix(shape=(Any, Any), dtype=Scalar)

    if len(templates) == 0:
        # anonymous construction
        if "shape" not in kwds:
            raise RuntimeError("shape keyword must be specified when calling matrix() function")

        if len(arg_types) == 0:
            if "dtype" not in kwds:
                raise RuntimeError("matrix() must have dtype as a keyword argument if it has no positional arguments")

            # zero initialization, e.g.: m = matrix(shape=(3,2), dtype=wp.float16)
            shape = kwds["shape"]
            dtype = kwds["dtype"]

        else:
            # value initialization, e.g.: m = matrix(1.0, shape=(3,2))
            shape = kwds["shape"]
            dtype = arg_types[0]

            if len(arg_types) == 1 and type_is_matrix(dtype):
                # constructor from another matrix
                if arg_types[0]._shape_ != shape:
                    raise RuntimeError(
                        f"Incompatible matrix sizes for casting copy constructor, {shape} vs {arg_types[0]._shape_}"
                    )
                dtype = dtype._wp_scalar_type_
            elif len(arg_types) > 1 and len(arg_types) != shape[0] * shape[1]:
                raise RuntimeError(
                    "Wrong number of arguments for matrix() function, must initialize with either a scalar value, or m*n values"
                )

        templates.append(shape[0])
        templates.append(shape[1])
        templates.append(dtype)

    else:
        # predeclared type, e.g.: mat32d
        shape = (templates[0], templates[1])
        dtype = templates[2]

        if len(arg_types) > 0:
            if len(arg_types) == 1 and type_is_matrix(arg_types[0]):
                # constructor from another matrix with same dimension but possibly different type
                if arg_types[0]._shape_ != shape:
                    raise RuntimeError(
                        f"Incompatible matrix sizes for casting copy constructor, {shape} vs {arg_types[0]._shape_}"
                    )
            else:
                # check scalar arg type matches declared type
                if infer_scalar_type(arg_types) != dtype:
                    raise RuntimeError("Wrong scalar type for mat {} constructor".format(",".join(map(str, templates))))

                # check vector arg type matches declared type
                if all(type_is_vector(a) for a in arg_types):
                    cols = len(arg_types)
                    if shape[1] != cols:
                        raise RuntimeError(
                            "Wrong number of vectors when attempting to construct a matrix with column vectors"
                        )

                    if not all(a._length_ == shape[0] for a in arg_types):
                        raise RuntimeError(
                            "Wrong vector row count when attempting to construct a matrix with column vectors"
                        )
                else:
                    # check that we either got 1 arg (scalar construction), or enough values for whole matrix
                    size = shape[0] * shape[1]
                    if len(arg_types) > 1 and len(arg_types) != size:
                        raise RuntimeError(
                            "Wrong number of scalars when attempting to construct a matrix from a list of components"
                        )

    return matrix(shape=shape, dtype=dtype)


# only use initializer list if matrix size < 5x5, or for scalar construction
def matrix_initlist_func(arg_types, templates):
    m, n, dtype = templates
    return not (
        len(arg_types) == 0
        or len(arg_types) == 1  # zero construction
        or (m == n and n < 5)  # scalar construction  # value construction for small matrices
    )


add_builtin(
    "matrix",
    input_types={"*arg_types": Scalar, "shape": Tuple[int, int], "dtype": Scalar},
    variadic=True,
    initializer_list_func=matrix_initlist_func,
    value_func=matrix_constructor_func,
    native_func="mat_t",
    doc="Construct a matrix. If the positional ``arg_types`` are not given, then matrix will be zero-initialized.",
    group="Vector Math",
    export=False,
)


# identity:
def matrix_identity_value_func(arg_types, kwds, templates):
    if arg_types is None:
        return matrix(shape=(Any, Any), dtype=Scalar)

    if len(arg_types):
        raise RuntimeError("identity() function does not accept positional arguments")

    if "n" not in kwds:
        raise RuntimeError("'n' keyword argument must be specified when calling identity() function")

    if "dtype" not in kwds:
        raise RuntimeError("'dtype' keyword argument must be specified when calling identity() function")

    n, dtype = [kwds["n"], kwds["dtype"]]

    if n is None:
        raise RuntimeError("'n' must be a constant when calling identity() function")

    templates.append(n)
    templates.append(dtype)

    return matrix(shape=(n, n), dtype=dtype)


add_builtin(
    "identity",
    input_types={"n": int, "dtype": Scalar},
    value_func=matrix_identity_value_func,
    variadic=True,
    doc="Create an identity matrix with shape=(n,n) with the type given by ``dtype``.",
    group="Vector Math",
    export=False,
)


def matrix_transform_value_func(arg_types, kwds, templates):
    if templates is None:
        return matrix(shape=(4, 4), dtype=Float)

    if len(templates) == 0:
        raise RuntimeError("Cannot use a generic type name in a kernel")

    m, n, dtype = templates
    if (m, n) != (4, 4):
        raise RuntimeError("Can only construct 4x4 matrices with position, rotation and scale")
    if infer_scalar_type(arg_types) != dtype:
        raise RuntimeError("Wrong scalar type for mat<{}> constructor".format(",".join(map(str, templates))))

    return matrix(shape=(4, 4), dtype=dtype)


add_builtin(
    "matrix",
    input_types={
        "pos": vector(length=3, dtype=Float),
        "rot": quaternion(dtype=Float),
        "scale": vector(length=3, dtype=Float),
    },
    value_func=matrix_transform_value_func,
    native_func="mat_t",
    doc="""Construct a 4x4 transformation matrix that applies the transformations as
    Translation(pos)*Rotation(rot)*Scale(scale) when applied to column vectors, i.e.: y = (TRS)*x""",
    group="Vector Math",
    export=False,
)


# not making these functions available outside kernels (export=False) as they
# return data via references, which we don't currently support:
add_builtin(
    "svd3",
    input_types={
        "A": matrix(shape=(3, 3), dtype=Float),
        "U": matrix(shape=(3, 3), dtype=Float),
        "sigma": vector(length=3, dtype=Float),
        "V": matrix(shape=(3, 3), dtype=Scalar),
    },
    value_type=None,
    group="Vector Math",
    export=False,
    doc="""Compute the SVD of a 3x3 matrix ``A``. The singular values are returned in ``sigma``,
    while the left and right basis vectors are returned in ``U`` and ``V``.""",
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


def quaternion_value_func(arg_types, kwds, templates):
    if arg_types is None:
        return quaternion(dtype=Float)

    if len(templates) == 0:
        if "dtype" in kwds:
            # casting constructor
            dtype = kwds["dtype"]
        else:
            # if constructing anonymous quat type then infer output type from arguments
            dtype = infer_scalar_type(arg_types)
        templates.append(dtype)
    else:
        # if constructing predeclared type then check arg_types match expectation
        if len(arg_types) > 0 and infer_scalar_type(arg_types) != templates[0]:
            raise RuntimeError("Wrong scalar type for quat {} constructor".format(",".join(map(str, templates))))

    return quaternion(dtype=templates[0])


def quat_cast_value_func(arg_types, kwds, templates):
    if arg_types is None:
        raise RuntimeError("Missing quaternion argument.")
    if "dtype" not in kwds:
        raise RuntimeError("Missing 'dtype' kwd.")

    dtype = kwds["dtype"]
    templates.append(dtype)

    return quaternion(dtype=dtype)


add_builtin(
    "quaternion",
    input_types={},
    value_func=quaternion_value_func,
    native_func="quat_t",
    group="Quaternion Math",
    doc="""Construct a zero-initialized quaternion. Quaternions are laid out as
    [ix, iy, iz, r], where ix, iy, iz are the imaginary part, and r the real part.""",
    export=False,
)
add_builtin(
    "quaternion",
    input_types={"x": Float, "y": Float, "z": Float, "w": Float},
    value_func=quaternion_value_func,
    native_func="quat_t",
    group="Quaternion Math",
    doc="Create a quaternion using the supplied components (type inferred from component type).",
    export=False,
)
add_builtin(
    "quaternion",
    input_types={"i": vector(length=3, dtype=Float), "r": Float},
    value_func=quaternion_value_func,
    native_func="quat_t",
    group="Quaternion Math",
    doc="Create a quaternion using the supplied vector/scalar (type inferred from scalar type).",
    export=False,
)
add_builtin(
    "quaternion",
    input_types={"q": quaternion(dtype=Float)},
    value_func=quat_cast_value_func,
    native_func="quat_t",
    group="Quaternion Math",
    doc="Construct a quaternion of type dtype from another quaternion of a different dtype.",
    export=False,
)


def quat_identity_value_func(arg_types, kwds, templates):
    # if arg_types is None then we are in 'export' mode
    if arg_types is None:
        return quatf

    if "dtype" not in kwds:
        # defaulting to float32 to preserve current behavior:
        dtype = float32
    else:
        dtype = kwds["dtype"]

    templates.append(dtype)

    return quaternion(dtype=dtype)


add_builtin(
    "quat_identity",
    input_types={},
    value_func=quat_identity_value_func,
    group="Quaternion Math",
    doc="Construct an identity quaternion with zero imaginary part and real part of 1.0",
    export=True,
)

add_builtin(
    "quat_from_axis_angle",
    input_types={"axis": vector(length=3, dtype=Float), "angle": Float},
    value_func=lambda arg_types, kwds, _: quaternion(dtype=infer_scalar_type(arg_types)),
    group="Quaternion Math",
    doc="Construct a quaternion representing a rotation of angle radians around the given axis.",
)
add_builtin(
    "quat_to_axis_angle",
    input_types={"q": quaternion(dtype=Float), "axis": vector(length=3, dtype=Float), "angle": Float},
    value_type=None,
    group="Quaternion Math",
    doc="Extract the rotation axis and angle radians a quaternion represents.",
)
add_builtin(
    "quat_from_matrix",
    input_types={"m": matrix(shape=(3, 3), dtype=Float)},
    value_func=lambda arg_types, kwds, _: quaternion(dtype=infer_scalar_type(arg_types)),
    group="Quaternion Math",
    doc="Construct a quaternion from a 3x3 matrix.",
)
add_builtin(
    "quat_rpy",
    input_types={"roll": Float, "pitch": Float, "yaw": Float},
    value_func=lambda arg_types, kwds, _: quaternion(dtype=infer_scalar_type(arg_types)),
    group="Quaternion Math",
    doc="Construct a quaternion representing a combined roll (z), pitch (x), yaw rotations (y) in radians.",
)
add_builtin(
    "quat_inverse",
    input_types={"q": quaternion(dtype=Float)},
    value_func=lambda arg_types, kwds, _: quaternion(dtype=infer_scalar_type(arg_types)),
    group="Quaternion Math",
    doc="Compute quaternion conjugate.",
)
add_builtin(
    "quat_rotate",
    input_types={"q": quaternion(dtype=Float), "p": vector(length=3, dtype=Float)},
    value_func=lambda arg_types, kwds, _: vector(length=3, dtype=infer_scalar_type(arg_types)),
    group="Quaternion Math",
    doc="Rotate a vector by a quaternion.",
)
add_builtin(
    "quat_rotate_inv",
    input_types={"q": quaternion(dtype=Float), "p": vector(length=3, dtype=Float)},
    value_func=lambda arg_types, kwds, _: vector(length=3, dtype=infer_scalar_type(arg_types)),
    group="Quaternion Math",
    doc="Rotate a vector by the inverse of a quaternion.",
)
add_builtin(
    "quat_slerp",
    input_types={"q0": quaternion(dtype=Float), "q1": quaternion(dtype=Float), "t": Float},
    value_func=lambda arg_types, kwds, _: quaternion(dtype=infer_scalar_type(arg_types)),
    group="Quaternion Math",
    doc="Linearly interpolate between two quaternions.",
    require_original_output_arg=True,
)
add_builtin(
    "quat_to_matrix",
    input_types={"q": quaternion(dtype=Float)},
    value_func=lambda arg_types, kwds, _: matrix(shape=(3, 3), dtype=infer_scalar_type(arg_types)),
    group="Quaternion Math",
    doc="Convert a quaternion to a 3x3 rotation matrix.",
)

add_builtin(
    "dot",
    input_types={"x": quaternion(dtype=Float), "y": quaternion(dtype=Float)},
    value_func=sametype_scalar_value_func,
    group="Quaternion Math",
    doc="Compute the dot product between two quaternions.",
)
# ---------------------------------
# Transformations


def transform_constructor_value_func(arg_types, kwds, templates):
    if templates is None:
        return transformation(dtype=Scalar)

    if len(templates) == 0:
        # if constructing anonymous transform type then infer output type from arguments
        dtype = infer_scalar_type(arg_types)
        templates.append(dtype)
    else:
        # if constructing predeclared type then check arg_types match expectation
        if infer_scalar_type(arg_types) != templates[0]:
            raise RuntimeError(
                f"Wrong scalar type for transform constructor expected {templates[0]}, got {','.join([ str(t) for t in arg_types])}"
            )

    return transformation(dtype=templates[0])


add_builtin(
    "transformation",
    input_types={"p": vector(length=3, dtype=Float), "q": quaternion(dtype=Float)},
    value_func=transform_constructor_value_func,
    native_func="transform_t",
    group="Transformations",
    doc="Construct a rigid-body transformation with translation part ``p`` and rotation ``q``.",
    export=False,
)


def transform_identity_value_func(arg_types, kwds, templates):
    # if arg_types is None then we are in 'export' mode
    if arg_types is None:
        return transformf

    if "dtype" not in kwds:
        # defaulting to float32 to preserve current behavior:
        dtype = float32
    else:
        dtype = kwds["dtype"]

    templates.append(dtype)

    return transformation(dtype=dtype)


add_builtin(
    "transform_identity",
    input_types={},
    value_func=transform_identity_value_func,
    group="Transformations",
    doc="Construct an identity transform with zero translation and identity rotation.",
    export=True,
)

add_builtin(
    "transform_get_translation",
    input_types={"t": transformation(dtype=Float)},
    value_func=lambda arg_types, kwds, _: vector(length=3, dtype=infer_scalar_type(arg_types)),
    group="Transformations",
    doc="Return the translational part of a transform ``t``.",
)
add_builtin(
    "transform_get_rotation",
    input_types={"t": transformation(dtype=Float)},
    value_func=lambda arg_types, kwds, _: quaternion(dtype=infer_scalar_type(arg_types)),
    group="Transformations",
    doc="Return the rotational part of a transform ``t``.",
)
add_builtin(
    "transform_multiply",
    input_types={"a": transformation(dtype=Float), "b": transformation(dtype=Float)},
    value_func=lambda arg_types, kwds, _: transformation(dtype=infer_scalar_type(arg_types)),
    group="Transformations",
    doc="Multiply two rigid body transformations together.",
)
add_builtin(
    "transform_point",
    input_types={"t": transformation(dtype=Scalar), "p": vector(length=3, dtype=Scalar)},
    value_func=lambda arg_types, kwds, _: vector(length=3, dtype=infer_scalar_type(arg_types)),
    group="Transformations",
    doc="Apply the transform to a point ``p`` treating the homogeneous coordinate as w=1 (translation and rotation).",
)
add_builtin(
    "transform_point",
    input_types={"m": matrix(shape=(4, 4), dtype=Scalar), "p": vector(length=3, dtype=Scalar)},
    value_func=lambda arg_types, kwds, _: vector(length=3, dtype=infer_scalar_type(arg_types)),
    group="Vector Math",
    doc="""Apply the transform to a point ``p`` treating the homogeneous coordinate as w=1.

    The transformation is applied treating ``p`` as a column vector, e.g.: ``y = M*p``.
    Note this is in contrast to some libraries, notably USD, which applies transforms to row vectors, ``y^T = p^T*M^T``.
    If the transform is coming from a library that uses row-vectors, then users should transpose the transformation
    matrix before calling this method.""",
)
add_builtin(
    "transform_vector",
    input_types={"t": transformation(dtype=Scalar), "v": vector(length=3, dtype=Scalar)},
    value_func=lambda arg_types, kwds, _: vector(length=3, dtype=infer_scalar_type(arg_types)),
    group="Transformations",
    doc="Apply the transform to a vector ``v`` treating the homogeneous coordinate as w=0 (rotation only).",
)
add_builtin(
    "transform_vector",
    input_types={"m": matrix(shape=(4, 4), dtype=Scalar), "v": vector(length=3, dtype=Scalar)},
    value_func=lambda arg_types, kwds, _: vector(length=3, dtype=infer_scalar_type(arg_types)),
    group="Vector Math",
    doc="""Apply the transform to a vector ``v`` treating the homogeneous coordinate as w=0.

    The transformation is applied treating ``v`` as a column vector, e.g.: ``y = M*v``
    note this is in contrast to some libraries, notably USD, which applies transforms to row vectors, ``y^T = v^T*M^T``.
    If the transform is coming from a library that uses row-vectors, then users should transpose the transformation
    matrix before calling this method.""",
)
add_builtin(
    "transform_inverse",
    input_types={"t": transformation(dtype=Float)},
    value_func=sametype_value_func(transformation(dtype=Float)),
    group="Transformations",
    doc="Compute the inverse of the transformation ``t``.",
)
# ---------------------------------
# Spatial Math


def spatial_vector_constructor_value_func(arg_types, kwds, templates):
    if templates is None:
        return spatial_vector(dtype=Float)

    if len(templates) == 0:
        raise RuntimeError("Cannot use a generic type name in a kernel")

    vectype = templates[1]
    if len(arg_types) and infer_scalar_type(arg_types) != vectype:
        raise RuntimeError("Wrong scalar type for spatial_vector<{}> constructor".format(",".join(map(str, templates))))

    return vector(length=6, dtype=vectype)


add_builtin(
    "vector",
    input_types={"w": vector(length=3, dtype=Float), "v": vector(length=3, dtype=Float)},
    value_func=spatial_vector_constructor_value_func,
    native_func="vec_t",
    group="Spatial Math",
    doc="Construct a 6D screw vector from two 3D vectors.",
    export=False,
)


add_builtin(
    "spatial_adjoint",
    input_types={"r": matrix(shape=(3, 3), dtype=Float), "s": matrix(shape=(3, 3), dtype=Float)},
    value_func=lambda arg_types, kwds, _: matrix(shape=(6, 6), dtype=infer_scalar_type(arg_types)),
    group="Spatial Math",
    doc="Construct a 6x6 spatial inertial matrix from two 3x3 diagonal blocks.",
    export=False,
)
add_builtin(
    "spatial_dot",
    input_types={"a": vector(length=6, dtype=Float), "b": vector(length=6, dtype=Float)},
    value_func=sametype_scalar_value_func,
    group="Spatial Math",
    doc="Compute the dot product of two 6D screw vectors.",
)
add_builtin(
    "spatial_cross",
    input_types={"a": vector(length=6, dtype=Float), "b": vector(length=6, dtype=Float)},
    value_func=sametype_value_func(vector(length=6, dtype=Float)),
    group="Spatial Math",
    doc="Compute the cross product of two 6D screw vectors.",
)
add_builtin(
    "spatial_cross_dual",
    input_types={"a": vector(length=6, dtype=Float), "b": vector(length=6, dtype=Float)},
    value_func=sametype_value_func(vector(length=6, dtype=Float)),
    group="Spatial Math",
    doc="Compute the dual cross product of two 6D screw vectors.",
)

add_builtin(
    "spatial_top",
    input_types={"a": vector(length=6, dtype=Float)},
    value_func=lambda arg_types, kwds, _: vector(length=3, dtype=arg_types[0]._wp_scalar_type_),
    group="Spatial Math",
    doc="Return the top (first) part of a 6D screw vector.",
)
add_builtin(
    "spatial_bottom",
    input_types={"a": vector(length=6, dtype=Float)},
    value_func=lambda arg_types, kwds, _: vector(length=3, dtype=arg_types[0]._wp_scalar_type_),
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


add_builtin(
    "mlp",
    input_types={
        "weights": array(dtype=float, ndim=2),
        "bias": array(dtype=float, ndim=1),
        "activation": Callable,
        "index": int,
        "x": array(dtype=float, ndim=2),
        "out": array(dtype=float, ndim=2),
    },
    value_type=None,
    skip_replay=True,
    doc="""Evaluate a multi-layer perceptron (MLP) layer in the form: ``out = act(weights*x + bias)``.

    :param weights: A layer's network weights with dimensions ``(m, n)``.
    :param bias: An array with dimensions ``(n)``.
    :param activation: A ``wp.func`` function that takes a single scalar float as input and returns a scalar float as output
    :param index: The batch item to process, typically each thread will process one item in the batch, in which case
                  index should be ``wp.tid()``
    :param x: The feature matrix with dimensions ``(n, b)``
    :param out: The network output with dimensions ``(m, b)``

    :note: Feature and output matrices are transposed compared to some other frameworks such as PyTorch.
           All matrices are assumed to be stored in flattened row-major memory layout (NumPy default).""",
    group="Utility",
)


# ---------------------------------
# Geometry

add_builtin(
    "bvh_query_aabb",
    input_types={"id": uint64, "lower": vec3, "upper": vec3},
    value_type=bvh_query_t,
    group="Geometry",
    doc="""Construct an axis-aligned bounding box query against a BVH object.

    This query can be used to iterate over all bounds inside a BVH.

    :param id: The BVH identifier
    :param lower: The lower bound of the bounding box in BVH space
    :param upper: The upper bound of the bounding box in BVH space""",
)

add_builtin(
    "bvh_query_ray",
    input_types={"id": uint64, "start": vec3, "dir": vec3},
    value_type=bvh_query_t,
    group="Geometry",
    doc="""Construct a ray query against a BVH object.

    This query can be used to iterate over all bounds that intersect the ray.

    :param id: The BVH identifier
    :param start: The start of the ray in BVH space
    :param dir: The direction of the ray in BVH space""",
)

add_builtin(
    "bvh_query_next",
    input_types={"query": bvh_query_t, "index": int},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Move to the next bound returned by the query.
    The index of the current bound is stored in ``index``, returns ``False`` if there are no more overlapping bound.""",
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
    hidden=True,
)

add_builtin(
    "mesh_query_point",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
    },
    value_type=mesh_query_point_t,
    group="Geometry",
    doc="""Computes the closest point on the :class:`Mesh` with identifier ``id`` to the given ``point`` in space.

    Identifies the sign of the distance using additional ray-casts to determine if the point is inside or outside.
    This method is relatively robust, but does increase computational cost.
    See below for additional sign determination methods.

    :param id: The mesh identifier
    :param point: The point in space to query
    :param max_dist: Mesh faces above this distance will not be considered by the query""",
    require_original_output_arg=True,
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
    hidden=True,
)

add_builtin(
    "mesh_query_point_no_sign",
    input_types={
        "id": uint64,
        "point": vec3,
        "max_dist": float,
    },
    value_type=mesh_query_point_t,
    group="Geometry",
    doc="""Computes the closest point on the :class:`Mesh` with identifier ``id`` to the given ``point`` in space.

    This method does not compute the sign of the point (inside/outside) which makes it faster than other point query methods.

    :param id: The mesh identifier
    :param point: The point in space to query
    :param max_dist: Mesh faces above this distance will not be considered by the query""",
    require_original_output_arg=True,
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
    hidden=True,
)

add_builtin(
    "mesh_query_furthest_point_no_sign",
    input_types={
        "id": uint64,
        "point": vec3,
        "min_dist": float,
    },
    value_type=mesh_query_point_t,
    group="Geometry",
    doc="""Computes the furthest point on the mesh with identifier `id` to the given point in space.

    This method does not compute the sign of the point (inside/outside).

    :param id: The mesh identifier
    :param point: The point in space to query
    :param min_dist: Mesh faces below this distance will not be considered by the query""",
    require_original_output_arg=True,
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
    value_type=mesh_query_point_t,
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
    value_type=mesh_query_point_t,
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
    value_type=mesh_query_ray_t,
    group="Geometry",
    doc="""Computes the closest ray hit on the :class:`Mesh` with identifier ``id``.

    :param id: The mesh identifier
    :param start: The start point of the ray
    :param dir: The ray direction (should be normalized)
    :param max_t: The maximum distance along the ray to check for intersections""",
    require_original_output_arg=True,
)

add_builtin(
    "mesh_query_aabb",
    input_types={"id": uint64, "lower": vec3, "upper": vec3},
    value_type=mesh_query_aabb_t,
    group="Geometry",
    doc="""Construct an axis-aligned bounding box query against a :class:`Mesh`.

    This query can be used to iterate over all triangles inside a volume.

    :param id: The mesh identifier
    :param lower: The lower bound of the bounding box in mesh space
    :param upper: The upper bound of the bounding box in mesh space""",
)

add_builtin(
    "mesh_query_aabb_next",
    input_types={"query": mesh_query_aabb_t, "index": int},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Move to the next triangle overlapping the query bounding box.

    The index of the current face is stored in ``index``, returns ``False`` if there are no more overlapping triangles.""",
)

add_builtin(
    "mesh_eval_position",
    input_types={"id": uint64, "face": int, "bary_u": float, "bary_v": float},
    value_type=vec3,
    group="Geometry",
    doc="""Evaluates the position on the :class:`Mesh` given a face index and barycentric coordinates.""",
)

add_builtin(
    "mesh_eval_velocity",
    input_types={"id": uint64, "face": int, "bary_u": float, "bary_v": float},
    value_type=vec3,
    group="Geometry",
    doc="""Evaluates the velocity on the :class:`Mesh` given a face index and barycentric coordinates.""",
)

add_builtin(
    "hash_grid_query",
    input_types={"id": uint64, "point": vec3, "max_dist": float},
    value_type=hash_grid_query_t,
    group="Geometry",
    doc="""Construct a point query against a :class:`HashGrid`.

    This query can be used to iterate over all neighboring point within a fixed radius from the query point.""",
)

add_builtin(
    "hash_grid_query_next",
    input_types={"query": hash_grid_query_t, "index": int},
    value_type=builtins.bool,
    group="Geometry",
    doc="""Move to the next point in the hash grid query.

    The index of the current neighbor is stored in ``index``, returns ``False`` if there are no more neighbors.""",
)

add_builtin(
    "hash_grid_point_id",
    input_types={"id": uint64, "index": int},
    value_type=int,
    group="Geometry",
    doc="""Return the index of a point in the :class:`HashGrid`.

    This can be used to reorder threads such that grid traversal occurs in a spatially coherent order.

    Returns -1 if the :class:`HashGrid` has not been reserved.""",
)

add_builtin(
    "intersect_tri_tri",
    input_types={"v0": vec3, "v1": vec3, "v2": vec3, "u0": vec3, "u1": vec3, "u2": vec3},
    value_type=int,
    group="Geometry",
    doc="""Tests for intersection between two triangles (v0, v1, v2) and (u0, u1, u2) using Moller's method.

    Returns > 0 if triangles intersect.""",
)

add_builtin(
    "mesh_get",
    input_types={"id": uint64},
    value_type=Mesh,
    missing_grad=True,
    group="Geometry",
    doc="""Retrieves the mesh given its index.""",
)

add_builtin(
    "mesh_eval_face_normal",
    input_types={"id": uint64, "face": int},
    value_type=vec3,
    group="Geometry",
    doc="""Evaluates the face normal the mesh given a face index.""",
)

add_builtin(
    "mesh_get_point",
    input_types={"id": uint64, "index": int},
    value_type=vec3,
    group="Geometry",
    doc="""Returns the point of the mesh given a index.""",
)

add_builtin(
    "mesh_get_velocity",
    input_types={"id": uint64, "index": int},
    value_type=vec3,
    group="Geometry",
    doc="""Returns the velocity of the mesh given a index.""",
)

add_builtin(
    "mesh_get_index",
    input_types={"id": uint64, "index": int},
    value_type=int,
    group="Geometry",
    doc="""Returns the point-index of the mesh given a face-vertex index.""",
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

add_builtin("iter_next", input_types={"range": range_t}, value_type=int, group="Utility", hidden=True)
add_builtin("iter_next", input_types={"query": hash_grid_query_t}, value_type=int, group="Utility", hidden=True)
add_builtin("iter_next", input_types={"query": mesh_query_aabb_t}, value_type=int, group="Utility", hidden=True)

# ---------------------------------
# Volumes

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
    doc="Return a random integer in the range [0, 2^32).",
)
add_builtin(
    "randi",
    input_types={"state": uint32, "min": int, "max": int},
    value_type=int,
    group="Random",
    doc="Return a random integer between [min, max).",
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
    input_types={"state": uint32, "min": float, "max": float},
    value_type=float,
    group="Random",
    doc="Return a random float between [min, max).",
)
add_builtin(
    "randn", input_types={"state": uint32}, value_type=float, group="Random", doc="Sample a normal distribution."
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
    defaults={"octaves": 1, "lacunarity": 2.0, "gain": 0.5},
    value_type=vec2,
    group="Random",
    doc="Divergence-free vector field based on the gradient of a Perlin noise function.",
    missing_grad=True,
)
add_builtin(
    "curlnoise",
    input_types={"state": uint32, "xyz": vec3, "octaves": uint32, "lacunarity": float, "gain": float},
    defaults={"octaves": 1, "lacunarity": 2.0, "gain": 0.5},
    value_type=vec3,
    group="Random",
    doc="Divergence-free vector field based on the curl of three Perlin noise functions.",
    missing_grad=True,
)
add_builtin(
    "curlnoise",
    input_types={"state": uint32, "xyzt": vec4, "octaves": uint32, "lacunarity": float, "gain": float},
    defaults={"octaves": 1, "lacunarity": 2.0, "gain": 0.5},
    value_type=vec3,
    group="Random",
    doc="Divergence-free vector field based on the curl of three Perlin noise functions.",
    missing_grad=True,
)

# note printf calls directly to global CRT printf (no wp:: namespace prefix)
add_builtin(
    "printf",
    input_types={},
    namespace="",
    variadic=True,
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
    input_types={"value": Any},
    value_func=lambda arg_types, kwds, _: arg_types[0],
    hidden=True,
    export=False,
    group="Utility",
)
add_builtin("assign", variadic=True, hidden=True, export=False, group="Utility")
add_builtin(
    "select",
    input_types={"cond": builtins.bool, "arg1": Any, "arg2": Any},
    value_func=lambda arg_types, kwds, _: arg_types[1],
    doc="Select between two arguments, if ``cond`` is ``False`` then return ``arg1``, otherwise return ``arg2``",
    group="Utility",
)
for t in int_types:
    add_builtin(
        "select",
        input_types={"cond": t, "arg1": Any, "arg2": Any},
        value_func=lambda arg_types, kwds, _: arg_types[1],
        doc="Select between two arguments, if ``cond`` is ``False`` then return ``arg1``, otherwise return ``arg2``",
        group="Utility",
    )
add_builtin(
    "select",
    input_types={"arr": array(dtype=Any), "arg1": Any, "arg2": Any},
    value_func=lambda arg_types, kwds, _: arg_types[1],
    doc="Select between two arguments, if ``arr`` is null then return ``arg1``, otherwise return ``arg2``",
    group="Utility",
)


# does argument checking and type propagation for address()
def address_value_func(arg_types, kwds, _):
    if not is_array(arg_types[0]):
        raise RuntimeError("load() argument 0 must be an array")

    num_indices = len(arg_types) - 1
    num_dims = arg_types[0].ndim

    if num_indices < num_dims:
        raise RuntimeError(
            "Num indices < num dimensions for array load, this is a codegen error, should have generated a view instead"
        )

    if num_indices > num_dims:
        raise RuntimeError(
            f"Num indices > num dimensions for array load, received {num_indices}, but array only has {num_dims}"
        )

    # check index types
    for t in arg_types[1:]:
        if not type_is_int(t):
            raise RuntimeError(f"address() index arguments must be of integer type, got index of type {t}")

    return Reference(arg_types[0].dtype)


# does argument checking and type propagation for view()
def view_value_func(arg_types, kwds, _):
    if not is_array(arg_types[0]):
        raise RuntimeError("view() argument 0 must be an array")

    # check array dim big enough to support view
    num_indices = len(arg_types) - 1
    num_dims = arg_types[0].ndim

    if num_indices >= num_dims:
        raise RuntimeError(
            f"Trying to create an array view with {num_indices} indices, but the array only has {num_dims} dimension(s). Ensure that the argument type on the function or kernel specifies the expected number of dimensions, e.g.: def func(param: wp.array3d(dtype=float):"
        )

    # check index types
    for t in arg_types[1:]:
        if not type_is_int(t):
            raise RuntimeError(f"view() index arguments must be of integer type, got index of type {t}")

    # create an array view with leading dimensions removed
    dtype = arg_types[0].dtype
    ndim = num_dims - num_indices
    if isinstance(arg_types[0], (fabricarray, indexedfabricarray)):
        # fabric array of arrays: return array attribute as a regular array
        return array(dtype=dtype, ndim=ndim)
    else:
        return type(arg_types[0])(dtype=dtype, ndim=ndim)


# does argument checking and type propagation for array_store()
def array_store_value_func(arg_types, kwds, _):
    # check target type
    if not is_array(arg_types[0]):
        raise RuntimeError("array_store() argument 0 must be an array")

    num_indices = len(arg_types[1:-1])
    num_dims = arg_types[0].ndim

    # if this happens we should have generated a view instead of a load during code gen
    if num_indices < num_dims:
        raise RuntimeError("Num indices < num dimensions for array store")

    if num_indices > num_dims:
        raise RuntimeError(
            f"Num indices > num dimensions for array store, received {num_indices}, but array only has {num_dims}"
        )

    # check index types
    for t in arg_types[1:-1]:
        if not type_is_int(t):
            raise RuntimeError(f"array_store() index arguments must be of integer type, got index of type {t}")

    # check value type
    if not types_equal(arg_types[-1], arg_types[0].dtype):
        raise RuntimeError(
            f"array_store() value argument type ({arg_types[2]}) must be of the same type as the array ({arg_types[0].dtype})"
        )

    return None


# does argument checking for store()
def store_value_func(arg_types, kwds, _):
    # we already stripped the Reference from the argument type prior to this call
    if not types_equal(arg_types[0], arg_types[1]):
        raise RuntimeError(f"store() value argument type ({arg_types[1]}) must be of the same type as the reference")

    return None


# does type propagation for load()
def load_value_func(arg_types, kwds, _):
    # we already stripped the Reference from the argument type prior to this call
    return arg_types[0]


add_builtin("address", variadic=True, hidden=True, value_func=address_value_func, group="Utility")
add_builtin("view", variadic=True, hidden=True, value_func=view_value_func, group="Utility")
add_builtin(
    "array_store", variadic=True, hidden=True, value_func=array_store_value_func, skip_replay=True, group="Utility"
)
add_builtin(
    "store",
    input_types={"address": Reference, "value": Any},
    hidden=True,
    value_func=store_value_func,
    skip_replay=True,
    group="Utility",
)
add_builtin(
    "load",
    input_types={"address": Reference},
    hidden=True,
    value_func=load_value_func,
    group="Utility",
)


def atomic_op_value_func(arg_types, kwds, _):
    # check target type
    if not is_array(arg_types[0]):
        raise RuntimeError("atomic() operation argument 0 must be an array")

    num_indices = len(arg_types[1:-1])
    num_dims = arg_types[0].ndim

    # if this happens we should have generated a view instead of a load during code gen
    if num_indices < num_dims:
        raise RuntimeError("Num indices < num dimensions for atomic array operation")

    if num_indices > num_dims:
        raise RuntimeError(
            f"Num indices > num dimensions for atomic array operation, received {num_indices}, but array only has {num_dims}"
        )

    # check index types
    for t in arg_types[1:-1]:
        if not type_is_int(t):
            raise RuntimeError(f"atomic() operation index arguments must be of integer type, got index of type {t}")

    if not types_equal(arg_types[-1], arg_types[0].dtype):
        raise RuntimeError(
            f"atomic() value argument ({arg_types[-1]}) must be of the same type as the array ({arg_types[0].dtype})"
        )

    return arg_types[0].dtype


for array_type in array_types:
    # don't list indexed array operations explicitly in docs
    hidden = array_type == indexedarray

    add_builtin(
        "atomic_add",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="Atomically add ``value`` onto ``a[i]``.",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_add",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="Atomically add ``value`` onto ``a[i,j]``.",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_add",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "k": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="Atomically add ``value`` onto ``a[i,j,k]``.",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_add",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "k": int, "l": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="Atomically add ``value`` onto ``a[i,j,k,l]``.",
        group="Utility",
        skip_replay=True,
    )

    add_builtin(
        "atomic_sub",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="Atomically subtract ``value`` onto ``a[i]``.",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_sub",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="Atomically subtract ``value`` onto ``a[i,j]``.",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_sub",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "k": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="Atomically subtract ``value`` onto ``a[i,j,k]``.",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_sub",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "k": int, "l": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="Atomically subtract ``value`` onto ``a[i,j,k,l]``.",
        group="Utility",
        skip_replay=True,
    )

    add_builtin(
        "atomic_min",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="""Compute the minimum of ``value`` and ``a[i]`` and atomically update the array.

    .. note:: The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_min",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="""Compute the minimum of ``value`` and ``a[i,j]`` and atomically update the array.

    .. note:: The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_min",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "k": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="""Compute the minimum of ``value`` and ``a[i,j,k]`` and atomically update the array.

    .. note:: The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_min",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "k": int, "l": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="""Compute the minimum of ``value`` and ``a[i,j,k,l]`` and atomically update the array.

    .. note:: The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )

    add_builtin(
        "atomic_max",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="""Compute the maximum of ``value`` and ``a[i]`` and atomically update the array.

    .. note:: The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_max",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="""Compute the maximum of ``value`` and ``a[i,j]`` and atomically update the array.

    .. note:: The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_max",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "k": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="""Compute the maximum of ``value`` and ``a[i,j,k]`` and atomically update the array.

    .. note:: The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )
    add_builtin(
        "atomic_max",
        hidden=hidden,
        input_types={"a": array_type(dtype=Any), "i": int, "j": int, "k": int, "l": int, "value": Any},
        value_func=atomic_op_value_func,
        doc="""Compute the maximum of ``value`` and ``a[i,j,k,l]`` and atomically update the array.

    .. note:: The operation is only atomic on a per-component basis for vectors and matrices.""",
        group="Utility",
        skip_replay=True,
    )


# used to index into builtin types, i.e.: y = vec3[1]
def index_value_func(arg_types, kwds, _):
    return arg_types[0]._wp_scalar_type_


add_builtin(
    "extract",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": int},
    value_func=index_value_func,
    hidden=True,
    group="Utility",
)
add_builtin(
    "extract",
    input_types={"a": quaternion(dtype=Scalar), "i": int},
    value_func=index_value_func,
    hidden=True,
    group="Utility",
)

add_builtin(
    "extract",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": int},
    value_func=lambda arg_types, kwds, _: vector(length=arg_types[0]._shape_[1], dtype=arg_types[0]._wp_scalar_type_),
    hidden=True,
    group="Utility",
)
add_builtin(
    "extract",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": int, "j": int},
    value_func=index_value_func,
    hidden=True,
    group="Utility",
)

add_builtin(
    "extract",
    input_types={"a": transformation(dtype=Scalar), "i": int},
    value_func=index_value_func,
    hidden=True,
    group="Utility",
)

add_builtin("extract", input_types={"s": shape_t, "i": int}, value_type=int, hidden=True, group="Utility")


def vector_indexref_element_value_func(arg_types, kwds, _):
    vec_type = arg_types[0]
    # index_type = arg_types[1]
    value_type = vec_type._wp_scalar_type_

    return Reference(value_type)


# implements &vector[index]
add_builtin(
    "index",
    input_types={"a": vector(length=Any, dtype=Scalar), "i": int},
    value_func=vector_indexref_element_value_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)
# implements &(*vector)[index]
add_builtin(
    "indexref",
    input_types={"a": Reference, "i": int},
    value_func=vector_indexref_element_value_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)


def matrix_indexref_element_value_func(arg_types, kwds, _):
    mat_type = arg_types[0]
    # row_type = arg_types[1]
    # col_type = arg_types[2]
    value_type = mat_type._wp_scalar_type_

    return Reference(value_type)


def matrix_indexref_row_value_func(arg_types, kwds, _):
    mat_type = arg_types[0]
    row_type = mat_type._wp_row_type_
    # value_type = arg_types[2]

    return Reference(row_type)


# implements matrix[i] = row
add_builtin(
    "index",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": int},
    value_func=matrix_indexref_row_value_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)

# implements matrix[i,j] = scalar
add_builtin(
    "index",
    input_types={"a": matrix(shape=(Any, Any), dtype=Scalar), "i": int, "j": int},
    value_func=matrix_indexref_element_value_func,
    hidden=True,
    group="Utility",
    skip_replay=True,
)

for t in scalar_types + vector_types + (bool,):
    if "vec" in t.__name__ or "mat" in t.__name__:
        continue
    add_builtin(
        "expect_eq",
        input_types={"arg1": t, "arg2": t},
        value_type=None,
        doc="Prints an error to stdout if ``arg1`` and ``arg2`` are not equal",
        group="Utility",
        hidden=True,
    )


def expect_eq_val_func(arg_types, kwds, _):
    if not types_equal(arg_types[0], arg_types[1]):
        raise RuntimeError("Can't test equality for objects with different types")
    return None


add_builtin(
    "expect_eq",
    input_types={"arg1": vector(length=Any, dtype=Scalar), "arg2": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=expect_eq_val_func,
    doc="Prints an error to stdout if ``arg1`` and ``arg2`` are not equal",
    group="Utility",
    hidden=True,
)
add_builtin(
    "expect_neq",
    input_types={"arg1": vector(length=Any, dtype=Scalar), "arg2": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=expect_eq_val_func,
    doc="Prints an error to stdout if ``arg1`` and ``arg2`` are equal",
    group="Utility",
    hidden=True,
)

add_builtin(
    "expect_eq",
    input_types={"arg1": matrix(shape=(Any, Any), dtype=Scalar), "arg2": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=expect_eq_val_func,
    doc="Prints an error to stdout if ``arg1`` and ``arg2`` are not equal",
    group="Utility",
    hidden=True,
)
add_builtin(
    "expect_neq",
    input_types={"arg1": matrix(shape=(Any, Any), dtype=Scalar), "arg2": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=expect_eq_val_func,
    doc="Prints an error to stdout if ``arg1`` and ``arg2`` are equal",
    group="Utility",
    hidden=True,
)

add_builtin(
    "lerp",
    input_types={"a": Float, "b": Float, "t": Float},
    value_func=sametype_value_func(Float),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``",
    group="Utility",
)
add_builtin(
    "smoothstep",
    input_types={"edge0": Float, "edge1": Float, "x": Float},
    value_func=sametype_value_func(Float),
    doc="""Smoothly interpolate between two values ``edge0`` and ``edge1`` using a factor ``x``,
    and return a result between 0 and 1 using a cubic Hermite interpolation after clamping.""",
    group="Utility",
)


def lerp_constraint(arg_types):
    return types_equal(arg_types[0], arg_types[1])


def lerp_value_func(default):
    def fn(arg_types, kwds, _):
        if arg_types is None:
            return default
        scalar_type = arg_types[-1]
        if not lerp_constraint(arg_types):
            raise RuntimeError("Can't lerp between objects with different types")
        if arg_types[0]._wp_scalar_type_ != scalar_type:
            raise RuntimeError("'t' parameter must have the same scalar type as objects you're lerping between")

        return arg_types[0]

    return fn


add_builtin(
    "lerp",
    input_types={"a": vector(length=Any, dtype=Float), "b": vector(length=Any, dtype=Float), "t": Float},
    constraint=lerp_constraint,
    value_func=lerp_value_func(vector(length=Any, dtype=Float)),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``",
    group="Utility",
)
add_builtin(
    "lerp",
    input_types={"a": matrix(shape=(Any, Any), dtype=Float), "b": matrix(shape=(Any, Any), dtype=Float), "t": Float},
    constraint=lerp_constraint,
    value_func=lerp_value_func(matrix(shape=(Any, Any), dtype=Float)),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``",
    group="Utility",
)
add_builtin(
    "lerp",
    input_types={"a": quaternion(dtype=Float), "b": quaternion(dtype=Float), "t": Float},
    value_func=lerp_value_func(quaternion(dtype=Float)),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``",
    group="Utility",
)
add_builtin(
    "lerp",
    input_types={"a": transformation(dtype=Float), "b": transformation(dtype=Float), "t": Float},
    value_func=lerp_value_func(transformation(dtype=Float)),
    doc="Linearly interpolate two values ``a`` and ``b`` using factor ``t``, computed as ``a*(1-t) + b*t``",
    group="Utility",
)

# fuzzy compare for float values
add_builtin(
    "expect_near",
    input_types={"arg1": Float, "arg2": Float, "tolerance": Float},
    defaults={"tolerance": 1.0e-6},
    value_type=None,
    doc="Prints an error to stdout if ``arg1`` and ``arg2`` are not closer than tolerance in magnitude",
    group="Utility",
)
add_builtin(
    "expect_near",
    input_types={"arg1": vec3, "arg2": vec3, "tolerance": float},
    value_type=None,
    doc="Prints an error to stdout if any element of ``arg1`` and ``arg2`` are not closer than tolerance in magnitude",
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

add_builtin("add", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar))
add_builtin(
    "add",
    input_types={"x": vector(length=Any, dtype=Scalar), "y": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"x": quaternion(dtype=Scalar), "y": quaternion(dtype=Scalar)},
    value_func=sametype_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar), "y": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "add",
    input_types={"x": transformation(dtype=Scalar), "y": transformation(dtype=Scalar)},
    value_func=sametype_value_func(transformation(dtype=Scalar)),
    doc="",
    group="Operators",
)

add_builtin("sub", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar))
add_builtin(
    "sub",
    input_types={"x": vector(length=Any, dtype=Scalar), "y": vector(length=Any, dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar), "y": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=sametypes,
    value_func=sametype_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"x": quaternion(dtype=Scalar), "y": quaternion(dtype=Scalar)},
    value_func=sametype_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "sub",
    input_types={"x": transformation(dtype=Scalar), "y": transformation(dtype=Scalar)},
    value_func=sametype_value_func(transformation(dtype=Scalar)),
    doc="",
    group="Operators",
)

# bitwise operators
add_builtin("bit_and", input_types={"x": Int, "y": Int}, value_func=sametype_value_func(Int))
add_builtin("bit_or", input_types={"x": Int, "y": Int}, value_func=sametype_value_func(Int))
add_builtin("bit_xor", input_types={"x": Int, "y": Int}, value_func=sametype_value_func(Int))
add_builtin("lshift", input_types={"x": Int, "y": Int}, value_func=sametype_value_func(Int))
add_builtin("rshift", input_types={"x": Int, "y": Int}, value_func=sametype_value_func(Int))
add_builtin("invert", input_types={"x": Int}, value_func=sametype_value_func(Int))


def scalar_mul_value_func(default):
    def fn(arg_types, kwds, _):
        if arg_types is None:
            return default
        scalar = [t for t in arg_types if t in scalar_types][0]
        compound = [t for t in arg_types if t not in scalar_types][0]
        if scalar != compound._wp_scalar_type_:
            raise RuntimeError("Object and coefficient must have the same scalar type when multiplying by scalar")
        return compound

    return fn


def mul_matvec_constraint(arg_types):
    return arg_types[0]._shape_[1] == arg_types[1]._length_


def mul_matvec_value_func(arg_types, kwds, _):
    if arg_types is None:
        return vector(length=Any, dtype=Scalar)

    if arg_types[0]._wp_scalar_type_ != arg_types[1]._wp_scalar_type_:
        raise RuntimeError(
            f"Can't multiply matrix and vector with different types {arg_types[0]._wp_scalar_type_}, {arg_types[1]._wp_scalar_type_}"
        )

    if not mul_matmat_constraint(arg_types):
        raise RuntimeError(
            f"Can't multiply matrix of shape {arg_types[0]._shape_} and vector with length {arg_types[1]._length_}"
        )

    return vector(length=arg_types[0]._shape_[0], dtype=arg_types[0]._wp_scalar_type_)


def mul_vecmat_constraint(arg_types):
    return arg_types[1]._shape_[0] == arg_types[0]._length_


def mul_vecmat_value_func(arg_types, kwds, _):
    if arg_types is None:
        return vector(length=Any, dtype=Scalar)

    if arg_types[1]._wp_scalar_type_ != arg_types[0]._wp_scalar_type_:
        raise RuntimeError(
            f"Can't multiply vector and matrix with different types {arg_types[1]._wp_scalar_type_}, {arg_types[0]._wp_scalar_type_}"
        )

    if not mul_vecmat_constraint(arg_types):
        raise RuntimeError(
            f"Can't multiply vector with length {arg_types[0]._length_} and matrix of shape {arg_types[1]._shape_}"
        )

    return vector(length=arg_types[1]._shape_[1], dtype=arg_types[1]._wp_scalar_type_)


def mul_matmat_constraint(arg_types):
    return arg_types[0]._shape_[1] == arg_types[1]._shape_[0]


def mul_matmat_value_func(arg_types, kwds, _):
    if arg_types is None:
        return matrix(length=Any, dtype=Scalar)

    if arg_types[0]._wp_scalar_type_ != arg_types[1]._wp_scalar_type_:
        raise RuntimeError(
            f"Can't multiply matrices with different types {arg_types[0]._wp_scalar_type_}, {arg_types[1]._wp_scalar_type_}"
        )

    if not mul_matmat_constraint(arg_types):
        raise RuntimeError(f"Can't multiply matrix of shapes {arg_types[0]._shape_} and {arg_types[1]._shape_}")

    return matrix(shape=(arg_types[0]._shape_[0], arg_types[1]._shape_[1]), dtype=arg_types[0]._wp_scalar_type_)


add_builtin("mul", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar))
add_builtin(
    "mul",
    input_types={"x": vector(length=Any, dtype=Scalar), "y": Scalar},
    value_func=scalar_mul_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"x": Scalar, "y": vector(length=Any, dtype=Scalar)},
    value_func=scalar_mul_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"x": quaternion(dtype=Scalar), "y": Scalar},
    value_func=scalar_mul_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"x": Scalar, "y": quaternion(dtype=Scalar)},
    value_func=scalar_mul_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"x": quaternion(dtype=Scalar), "y": quaternion(dtype=Scalar)},
    value_func=sametype_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"x": Scalar, "y": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=scalar_mul_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar), "y": Scalar},
    value_func=scalar_mul_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar), "y": vector(length=Any, dtype=Scalar)},
    constraint=mul_matvec_constraint,
    value_func=mul_matvec_value_func,
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"x": vector(length=Any, dtype=Scalar), "y": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=mul_vecmat_constraint,
    value_func=mul_vecmat_value_func,
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar), "y": matrix(shape=(Any, Any), dtype=Scalar)},
    constraint=mul_matmat_constraint,
    value_func=mul_matmat_value_func,
    doc="",
    group="Operators",
)

add_builtin(
    "mul",
    input_types={"x": transformation(dtype=Scalar), "y": transformation(dtype=Scalar)},
    value_func=sametype_value_func(transformation(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"x": Scalar, "y": transformation(dtype=Scalar)},
    value_func=scalar_mul_value_func(transformation(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "mul",
    input_types={"x": transformation(dtype=Scalar), "y": Scalar},
    value_func=scalar_mul_value_func(transformation(dtype=Scalar)),
    doc="",
    group="Operators",
)

add_builtin("mod", input_types={"x": Scalar, "y": Scalar}, value_func=sametype_value_func(Scalar))

add_builtin(
    "div",
    input_types={"x": Scalar, "y": Scalar},
    value_func=sametype_value_func(Scalar),
    doc="",
    group="Operators",
    require_original_output_arg=True,
)
add_builtin(
    "div",
    input_types={"x": vector(length=Any, dtype=Scalar), "y": Scalar},
    value_func=scalar_mul_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"x": Scalar, "y": vector(length=Any, dtype=Scalar)},
    value_func=scalar_mul_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar), "y": Scalar},
    value_func=scalar_mul_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"x": Scalar, "y": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=scalar_mul_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"x": quaternion(dtype=Scalar), "y": Scalar},
    value_func=scalar_mul_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "div",
    input_types={"x": Scalar, "y": quaternion(dtype=Scalar)},
    value_func=scalar_mul_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)

add_builtin(
    "floordiv",
    input_types={"x": Scalar, "y": Scalar},
    value_func=sametype_value_func(Scalar),
    doc="",
    group="Operators",
)

add_builtin("pos", input_types={"x": Scalar}, value_func=sametype_value_func(Scalar))
add_builtin(
    "pos",
    input_types={"x": vector(length=Any, dtype=Scalar)},
    value_func=sametype_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "pos",
    input_types={"x": quaternion(dtype=Scalar)},
    value_func=sametype_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "pos",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=sametype_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin("neg", input_types={"x": Scalar}, value_func=sametype_value_func(Scalar))
add_builtin(
    "neg",
    input_types={"x": vector(length=Any, dtype=Scalar)},
    value_func=sametype_value_func(vector(length=Any, dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "neg",
    input_types={"x": quaternion(dtype=Scalar)},
    value_func=sametype_value_func(quaternion(dtype=Scalar)),
    doc="",
    group="Operators",
)
add_builtin(
    "neg",
    input_types={"x": matrix(shape=(Any, Any), dtype=Scalar)},
    value_func=sametype_value_func(matrix(shape=(Any, Any), dtype=Scalar)),
    doc="",
    group="Operators",
)

add_builtin("unot", input_types={"b": builtins.bool}, value_type=builtins.bool, doc="", group="Operators")
for t in int_types:
    add_builtin("unot", input_types={"b": t}, value_type=builtins.bool, doc="", group="Operators")


add_builtin("unot", input_types={"a": array(dtype=Any)}, value_type=builtins.bool, doc="", group="Operators")
