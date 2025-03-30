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
import ctypes
import inspect
import struct
import zlib
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    Sequence,
    Tuple,
    TypeVar,
    get_args,
    get_origin,
)

import numpy as np
import numpy.typing as npt

import warp

# type hints
T = TypeVar("T")
Length = TypeVar("Length", bound=int)
Rows = TypeVar("Rows")
Cols = TypeVar("Cols")
DType = TypeVar("DType")

Int = TypeVar("Int")
Float = TypeVar("Float")
Scalar = TypeVar("Scalar")


class Vector(Generic[Length, Scalar]):
    pass


class Matrix(Generic[Rows, Cols, Scalar]):
    pass


class Quaternion(Generic[Float]):
    pass


class Transformation(Generic[Float]):
    pass


class Array(Generic[DType]):
    device: warp.context.Device | None
    dtype: type
    size: int


int_tuple_type_hints = {
    Tuple[int]: 1,
    Tuple[int, int]: 2,
    Tuple[int, int, int]: 3,
    Tuple[int, int, int, int]: 4,
    Tuple[int, ...]: -1,
}


def constant(x):
    """Function to declare compile-time constants accessible from Warp kernels

    Args:
        x: Compile-time constant value, can be any of the built-in math types.
    """

    if not is_value(x):
        raise TypeError(f"Invalid constant type: {type(x)}")

    return x


def float_to_half_bits(value):
    return warp.context.runtime.core.float_to_half_bits(value)


def half_bits_to_float(value):
    return warp.context.runtime.core.half_bits_to_float(value)


# ----------------------
# built-in types


def vector(length, dtype):
    # canonicalize dtype
    if dtype == int:
        dtype = int32
    elif dtype == float:
        dtype = float32
    elif dtype == builtins.bool:
        dtype = bool

    class vec_t(ctypes.Array):
        # ctypes.Array data for length, shape and c type:
        _length_ = 0 if length is Any else length
        _shape_ = (_length_,)

        if dtype is bool:
            _type_ = ctypes.c_bool
        elif dtype in (Scalar, Float):
            _type_ = ctypes.c_float
        elif dtype is Int:
            _type_ = ctypes.c_int
        else:
            _type_ = dtype._type_

        # warp scalar type:
        _wp_scalar_type_ = dtype
        _wp_type_params_ = [length, dtype]
        _wp_type_args_ = {"length": length, "dtype": dtype}
        _wp_generic_type_str_ = "vec_t"
        _wp_generic_type_hint_ = Vector
        _wp_constructor_ = "vector"

        # special handling for float16 type: in this case, data is stored
        # as uint16 but it's actually half precision floating point
        # data. This means we need to convert each of the arguments
        # to uint16s containing half float bits before storing them in
        # the array:
        scalar_import = float_to_half_bits if _wp_scalar_type_ == float16 else lambda x: x
        scalar_export = half_bits_to_float if _wp_scalar_type_ == float16 else lambda x: x

        def __init__(self, *args):
            num_args = len(args)
            if num_args == 0:
                super().__init__()
            elif num_args == 1:
                if hasattr(args[0], "__len__"):
                    # try to copy from expanded sequence, e.g. (1, 2, 3)
                    self.__init__(*args[0])
                else:
                    # set all elements to the same value
                    value = vec_t.scalar_import(args[0])
                    for i in range(self._length_):
                        super().__setitem__(i, value)
            elif num_args == self._length_:
                # set all scalar elements
                for i in range(self._length_):
                    super().__setitem__(i, vec_t.scalar_import(args[i]))
            else:
                raise ValueError(
                    f"Invalid number of arguments in vector constructor, expected {self._length_} elements, got {num_args}"
                )

        def __getitem__(self, key):
            if isinstance(key, int):
                return vec_t.scalar_export(super().__getitem__(key))
            elif isinstance(key, slice):
                if self._wp_scalar_type_ == float16:
                    return [vec_t.scalar_export(x) for x in super().__getitem__(key)]
                else:
                    return super().__getitem__(key)
            else:
                raise KeyError(f"Invalid key {key}, expected int or slice")

        def __setitem__(self, key, value):
            if isinstance(key, int):
                try:
                    return super().__setitem__(key, vec_t.scalar_import(value))
                except (TypeError, ctypes.ArgumentError):
                    raise TypeError(
                        f"Expected to assign a `{self._wp_scalar_type_.__name__}` value "
                        f"but got `{type(value).__name__}` instead"
                    ) from None
            elif isinstance(key, slice):
                try:
                    iter(value)
                except TypeError:
                    raise TypeError(
                        f"Expected to assign a slice from a sequence of values but got `{type(value).__name__}` instead"
                    ) from None

                if self._wp_scalar_type_ == float16:
                    converted = []
                    try:
                        for x in value:
                            converted.append(vec_t.scalar_import(x))
                    except ctypes.ArgumentError:
                        raise TypeError(
                            f"Expected to assign a slice from a sequence of `float16` values "
                            f"but got `{type(x).__name__}` instead"
                        ) from None

                    value = converted

                try:
                    return super().__setitem__(key, value)
                except TypeError:
                    for x in value:
                        try:
                            self._type_(x)
                        except TypeError:
                            raise TypeError(
                                f"Expected to assign a slice from a sequence of `{self._wp_scalar_type_.__name__}` values "
                                f"but got `{type(x).__name__}` instead"
                            ) from None
            else:
                raise KeyError(f"Invalid key {key}, expected int or slice")

        def __getattr__(self, name):
            idx = "xyzw".find(name)
            if idx != -1:
                return self.__getitem__(idx)

            return self.__getattribute__(name)

        def __setattr__(self, name, value):
            idx = "xyzw".find(name)
            if idx != -1:
                return self.__setitem__(idx, value)

            return super().__setattr__(name, value)

        def __add__(self, y):
            return warp.add(self, y)

        def __radd__(self, y):
            return warp.add(y, self)

        def __sub__(self, y):
            return warp.sub(self, y)

        def __rsub__(self, y):
            return warp.sub(y, self)

        def __mul__(self, y):
            return warp.mul(self, y)

        def __rmul__(self, x):
            return warp.mul(x, self)

        def __truediv__(self, y):
            return warp.div(self, y)

        def __rtruediv__(self, x):
            return warp.div(x, self)

        def __mod__(self, x):
            return warp.mod(self, x)

        def __rmod__(self, x):
            return warp.mod(x, self)

        def __pos__(self):
            return warp.pos(self)

        def __neg__(self):
            return warp.neg(self)

        def __str__(self):
            return f"[{', '.join(map(str, self))}]"

        def __eq__(self, other):
            for i in range(self._length_):
                if self[i] != other[i]:
                    return False
            return True

        @classmethod
        def from_ptr(cls, ptr):
            if ptr:
                # create a new vector instance and initialize the contents from the binary data
                # this skips float16 conversions, assuming that float16 data is already encoded as uint16
                value = cls()
                ctypes.memmove(ctypes.byref(value), ptr, ctypes.sizeof(cls._type_) * cls._length_)
                return value
            else:
                raise RuntimeError("NULL pointer exception")

    return vec_t


def matrix(shape, dtype):
    assert len(shape) == 2

    # canonicalize dtype
    if dtype == int:
        dtype = int32
    elif dtype == float:
        dtype = float32
    elif dtype == builtins.bool:
        dtype = bool

    class mat_t(ctypes.Array):
        _length_ = 0 if shape[0] == Any or shape[1] == Any else shape[0] * shape[1]
        _shape_ = (0, 0) if _length_ == 0 else shape

        if dtype is bool:
            _type_ = ctypes.c_bool
        elif dtype in (Scalar, Float):
            _type_ = ctypes.c_float
        elif dtype is Int:
            _type_ = ctypes.c_int
        else:
            _type_ = dtype._type_

        # warp scalar type:
        # used in type checking and when writing out c++ code for constructors:
        _wp_scalar_type_ = dtype
        _wp_type_params_ = [shape[0], shape[1], dtype]
        _wp_type_args_ = {"shape": (shape[0], shape[1]), "dtype": dtype}
        _wp_generic_type_str_ = "mat_t"
        _wp_generic_type_hint_ = Matrix
        _wp_constructor_ = "matrix"

        _wp_row_type_ = vector(0 if shape[1] == Any else shape[1], dtype)

        # special handling for float16 type: in this case, data is stored
        # as uint16 but it's actually half precision floating point
        # data. This means we need to convert each of the arguments
        # to uint16s containing half float bits before storing them in
        # the array:
        scalar_import = float_to_half_bits if _wp_scalar_type_ == float16 else lambda x: x
        scalar_export = half_bits_to_float if _wp_scalar_type_ == float16 else lambda x: x

        def __init__(self, *args):
            num_args = len(args)
            if num_args == 0:
                super().__init__()
            elif num_args == 1:
                if hasattr(args[0], "__len__"):
                    # try to copy from expanded sequence, e.g. [[1, 0], [0, 1]]
                    self.__init__(*args[0])
                else:
                    # set all elements to the same value
                    value = mat_t.scalar_import(args[0])
                    for i in range(self._length_):
                        super().__setitem__(i, value)
            elif num_args == self._length_:
                # set all scalar elements
                for i in range(self._length_):
                    super().__setitem__(i, mat_t.scalar_import(args[i]))
            elif num_args == self._shape_[0]:
                # row vectors
                for i, row in enumerate(args):
                    if not hasattr(row, "__len__") or len(row) != self._shape_[1]:
                        raise TypeError(
                            f"Invalid argument in matrix constructor, expected row of length {self._shape_[1]}, got {row}"
                        )
                    offset = i * self._shape_[1]
                    for j in range(self._shape_[1]):
                        super().__setitem__(offset + j, mat_t.scalar_import(row[j]))
            else:
                raise ValueError(
                    f"Invalid number of arguments in matrix constructor, expected {self._length_} elements, got {num_args}"
                )

        def __len__(self):
            return self._shape_[0]

        def __add__(self, y):
            return warp.add(self, y)

        def __radd__(self, y):
            return warp.add(y, self)

        def __sub__(self, y):
            return warp.sub(self, y)

        def __rsub__(self, y):
            return warp.sub(y, self)

        def __mul__(self, y):
            return warp.mul(self, y)

        def __rmul__(self, x):
            return warp.mul(x, self)

        def __matmul__(self, y):
            return warp.mul(self, y)

        def __rmatmul__(self, x):
            return warp.mul(x, self)

        def __truediv__(self, y):
            return warp.div(self, y)

        def __rtruediv__(self, x):
            return warp.div(x, self)

        def __pos__(self):
            return warp.pos(self)

        def __neg__(self):
            return warp.neg(self)

        def __str__(self):
            row_str = []
            for r in range(self._shape_[0]):
                row_val = self.get_row(r)
                row_str.append(f"[{', '.join(map(str, row_val))}]")

            return "[" + ",\n ".join(row_str) + "]"

        def __eq__(self, other):
            for i in range(self._shape_[0]):
                for j in range(self._shape_[1]):
                    if self[i][j] != other[i][j]:
                        return False
            return True

        def get_row(self, r):
            if r < 0 or r >= self._shape_[0]:
                raise IndexError("Invalid row index")
            row_start = r * self._shape_[1]
            row_end = row_start + self._shape_[1]
            row_data = super().__getitem__(slice(row_start, row_end))
            if self._wp_scalar_type_ == float16:
                return self._wp_row_type_(*[mat_t.scalar_export(x) for x in row_data])
            else:
                return self._wp_row_type_(row_data)

        def set_row(self, r, v):
            if r < 0 or r >= self._shape_[0]:
                raise IndexError("Invalid row index")
            try:
                iter(v)
            except TypeError:
                raise TypeError(
                    f"Expected to assign a slice from a sequence of values but got `{type(v).__name__}` instead"
                ) from None

            row_start = r * self._shape_[1]
            row_end = row_start + self._shape_[1]
            if self._wp_scalar_type_ == float16:
                converted = []
                try:
                    for x in v:
                        converted.append(mat_t.scalar_import(x))
                except ctypes.ArgumentError:
                    raise TypeError(
                        f"Expected to assign a slice from a sequence of `float16` values "
                        f"but got `{type(x).__name__}` instead"
                    ) from None

                v = converted
            super().__setitem__(slice(row_start, row_end), v)

        def __getitem__(self, key):
            if isinstance(key, Tuple):
                # element indexing m[i,j]
                if len(key) != 2:
                    raise KeyError(f"Invalid key, expected one or two indices, got {len(key)}")
                if any(isinstance(x, slice) for x in key):
                    raise KeyError("Slices are not supported when indexing matrices using the `m[i, j]` notation")
                return mat_t.scalar_export(super().__getitem__(key[0] * self._shape_[1] + key[1]))
            elif isinstance(key, int):
                # row vector indexing m[r]
                return self.get_row(key)
            else:
                raise KeyError(f"Invalid key {key}, expected int or pair of ints")

        def __setitem__(self, key, value):
            if isinstance(key, Tuple):
                # element indexing m[i,j] = x
                if len(key) != 2:
                    raise KeyError(f"Invalid key, expected one or two indices, got {len(key)}")
                if any(isinstance(x, slice) for x in key):
                    raise KeyError("Slices are not supported when indexing matrices using the `m[i, j]` notation")
                try:
                    return super().__setitem__(key[0] * self._shape_[1] + key[1], mat_t.scalar_import(value))
                except (TypeError, ctypes.ArgumentError):
                    raise TypeError(
                        f"Expected to assign a `{self._wp_scalar_type_.__name__}` value "
                        f"but got `{type(value).__name__}` instead"
                    ) from None
            elif isinstance(key, int):
                # row vector indexing m[r] = v
                return self.set_row(key, value)
            elif isinstance(key, slice):
                raise KeyError("Slices are not supported when indexing matrices using the `m[start:end]` notation")
            else:
                raise KeyError(f"Invalid key {key}, expected int or pair of ints")

        @classmethod
        def from_ptr(cls, ptr):
            if ptr:
                # create a new matrix instance and initialize the contents from the binary data
                # this skips float16 conversions, assuming that float16 data is already encoded as uint16
                value = cls()
                ctypes.memmove(ctypes.byref(value), ptr, ctypes.sizeof(cls._type_) * cls._length_)
                return value
            else:
                raise RuntimeError("NULL pointer exception")

    return mat_t


class void:
    def __init__(self):
        pass


class scalar_base:
    def __init__(self, x=0):
        self.value = x

    def __bool__(self) -> builtins.bool:
        return self.value != 0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __add__(self, y):
        return warp.add(self, y)

    def __radd__(self, y):
        return warp.add(y, self)

    def __sub__(self, y):
        return warp.sub(self, y)

    def __rsub__(self, y):
        return warp.sub(y, self)

    def __mul__(self, y):
        return warp.mul(self, y)

    def __rmul__(self, x):
        return warp.mul(x, self)

    def __truediv__(self, y):
        return warp.div(self, y)

    def __rtruediv__(self, x):
        return warp.div(x, self)

    def __mod__(self, x):
        return warp.mod(self, x)

    def __rmod__(self, x):
        return warp.mod(x, self)

    def __pos__(self):
        return warp.pos(self)

    def __neg__(self):
        return warp.neg(self)


class float_base(scalar_base):
    pass


class int_base(scalar_base):
    def __index__(self) -> int:
        return int(self.value)


class bool:
    _length_ = 1
    _type_ = ctypes.c_bool

    def __init__(self, x=False):
        self.value = x

    def __bool__(self) -> builtins.bool:
        return self.value != 0

    def __float__(self) -> float:
        return float(self.value != 0)

    def __int__(self) -> int:
        return int(self.value != 0)


class float16(float_base):
    _length_ = 1
    _type_ = ctypes.c_uint16


class float32(float_base):
    _length_ = 1
    _type_ = ctypes.c_float


class float64(float_base):
    _length_ = 1
    _type_ = ctypes.c_double


class int8(int_base):
    _length_ = 1
    _type_ = ctypes.c_int8


class uint8(int_base):
    _length_ = 1
    _type_ = ctypes.c_uint8


class int16(int_base):
    _length_ = 1
    _type_ = ctypes.c_int16


class uint16(int_base):
    _length_ = 1
    _type_ = ctypes.c_uint16


class int32(int_base):
    _length_ = 1
    _type_ = ctypes.c_int32


class uint32(int_base):
    _length_ = 1
    _type_ = ctypes.c_uint32


class int64(int_base):
    _length_ = 1
    _type_ = ctypes.c_int64


class uint64(int_base):
    _length_ = 1
    _type_ = ctypes.c_uint64


def quaternion(dtype=Any):
    class quat_t(vector(length=4, dtype=dtype)):
        pass
        # def __init__(self, *args):
        #     super().__init__(args)

    ret = quat_t
    ret._wp_type_params_ = [dtype]
    ret._wp_type_args_ = {"dtype": dtype}
    ret._wp_generic_type_str_ = "quat_t"
    ret._wp_generic_type_hint_ = Quaternion
    ret._wp_constructor_ = "quaternion"

    return ret


class quath(quaternion(dtype=float16)):
    pass


class quatf(quaternion(dtype=float32)):
    pass


class quatd(quaternion(dtype=float64)):
    pass


def transformation(dtype=Any):
    class transform_t(vector(length=7, dtype=dtype)):
        _wp_init_from_components_sig_ = inspect.Signature(
            (
                inspect.Parameter(
                    "p",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=(0.0, 0.0, 0.0),
                ),
                inspect.Parameter(
                    "q",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
        )
        _wp_type_params_ = [dtype]
        _wp_type_args_ = {"dtype": dtype}
        _wp_generic_type_str_ = "transform_t"
        _wp_generic_type_hint_ = Transformation
        _wp_constructor_ = "transformation"

        def __init__(self, *args, **kwargs):
            if len(args) == 1 and len(kwargs) == 0:
                if is_float(args[0]):
                    # Initialize from a single scalar.
                    super().__init__(args[0])
                    return
                if args[0]._wp_generic_type_str_ == self._wp_generic_type_str_:
                    # Copy constructor.
                    super().__init__(*args[0])
                    return

            try:
                # For backward compatibility, try to check if the arguments
                # match the original signature that'd allow initializing
                # the `p` and `q` components separately.
                bound_args = self._wp_init_from_components_sig_.bind(*args, **kwargs)
                bound_args.apply_defaults()
                p, q = bound_args.args
            except (TypeError, ValueError):
                # Fallback to the vector's constructor.
                super().__init__(*args)
                return

            # Even if the arguments match the original “from components”
            # signature, we still need to make sure that they represent
            # sequences that can be unpacked.
            if hasattr(p, "__len__") and hasattr(q, "__len__"):
                # Initialize from the `p` and `q` components.
                super().__init__()
                self[0:3] = vector(length=3, dtype=dtype)(*p)
                self[3:7] = quaternion(dtype=dtype)(*q)
                return

            # Fallback to the vector's constructor.
            super().__init__(*args)

        @property
        def p(self):
            return vec3(self[0:3])

        @property
        def q(self):
            return quat(self[3:7])

    return transform_t


class transformh(transformation(dtype=float16)):
    pass


class transformf(transformation(dtype=float32)):
    pass


class transformd(transformation(dtype=float64)):
    pass


class vec2h(vector(length=2, dtype=float16)):
    pass


class vec3h(vector(length=3, dtype=float16)):
    pass


class vec4h(vector(length=4, dtype=float16)):
    pass


class vec2f(vector(length=2, dtype=float32)):
    pass


class vec3f(vector(length=3, dtype=float32)):
    pass


class vec4f(vector(length=4, dtype=float32)):
    pass


class vec2d(vector(length=2, dtype=float64)):
    pass


class vec3d(vector(length=3, dtype=float64)):
    pass


class vec4d(vector(length=4, dtype=float64)):
    pass


class vec2b(vector(length=2, dtype=int8)):
    pass


class vec3b(vector(length=3, dtype=int8)):
    pass


class vec4b(vector(length=4, dtype=int8)):
    pass


class vec2ub(vector(length=2, dtype=uint8)):
    pass


class vec3ub(vector(length=3, dtype=uint8)):
    pass


class vec4ub(vector(length=4, dtype=uint8)):
    pass


class vec2s(vector(length=2, dtype=int16)):
    pass


class vec3s(vector(length=3, dtype=int16)):
    pass


class vec4s(vector(length=4, dtype=int16)):
    pass


class vec2us(vector(length=2, dtype=uint16)):
    pass


class vec3us(vector(length=3, dtype=uint16)):
    pass


class vec4us(vector(length=4, dtype=uint16)):
    pass


class vec2i(vector(length=2, dtype=int32)):
    pass


class vec3i(vector(length=3, dtype=int32)):
    pass


class vec4i(vector(length=4, dtype=int32)):
    pass


class vec2ui(vector(length=2, dtype=uint32)):
    pass


class vec3ui(vector(length=3, dtype=uint32)):
    pass


class vec4ui(vector(length=4, dtype=uint32)):
    pass


class vec2l(vector(length=2, dtype=int64)):
    pass


class vec3l(vector(length=3, dtype=int64)):
    pass


class vec4l(vector(length=4, dtype=int64)):
    pass


class vec2ul(vector(length=2, dtype=uint64)):
    pass


class vec3ul(vector(length=3, dtype=uint64)):
    pass


class vec4ul(vector(length=4, dtype=uint64)):
    pass


class mat22h(matrix(shape=(2, 2), dtype=float16)):
    pass


class mat33h(matrix(shape=(3, 3), dtype=float16)):
    pass


class mat44h(matrix(shape=(4, 4), dtype=float16)):
    pass


class mat22f(matrix(shape=(2, 2), dtype=float32)):
    pass


class mat33f(matrix(shape=(3, 3), dtype=float32)):
    pass


class mat44f(matrix(shape=(4, 4), dtype=float32)):
    pass


class mat22d(matrix(shape=(2, 2), dtype=float64)):
    pass


class mat33d(matrix(shape=(3, 3), dtype=float64)):
    pass


class mat44d(matrix(shape=(4, 4), dtype=float64)):
    pass


class spatial_vectorh(vector(length=6, dtype=float16)):
    pass


class spatial_vectorf(vector(length=6, dtype=float32)):
    pass


class spatial_vectord(vector(length=6, dtype=float64)):
    pass


class spatial_matrixh(matrix(shape=(6, 6), dtype=float16)):
    pass


class spatial_matrixf(matrix(shape=(6, 6), dtype=float32)):
    pass


class spatial_matrixd(matrix(shape=(6, 6), dtype=float64)):
    pass


# built-in type aliases that default to 32bit precision
vec2 = vec2f
vec3 = vec3f
vec4 = vec4f
mat22 = mat22f
mat33 = mat33f
mat44 = mat44f
quat = quatf
transform = transformf
spatial_vector = spatial_vectorf
spatial_matrix = spatial_matrixf


int_types = (int8, uint8, int16, uint16, int32, uint32, int64, uint64)
float_types = (float16, float32, float64)
scalar_types = int_types + float_types
scalar_and_bool_types = scalar_types + (bool,)

vector_types = (
    vec2b,
    vec2ub,
    vec2s,
    vec2us,
    vec2i,
    vec2ui,
    vec2l,
    vec2ul,
    vec2h,
    vec2f,
    vec2d,
    vec3b,
    vec3ub,
    vec3s,
    vec3us,
    vec3i,
    vec3ui,
    vec3l,
    vec3ul,
    vec3h,
    vec3f,
    vec3d,
    vec4b,
    vec4ub,
    vec4s,
    vec4us,
    vec4i,
    vec4ui,
    vec4l,
    vec4ul,
    vec4h,
    vec4f,
    vec4d,
    mat22h,
    mat22f,
    mat22d,
    mat33h,
    mat33f,
    mat33d,
    mat44h,
    mat44f,
    mat44d,
    quath,
    quatf,
    quatd,
    transformh,
    transformf,
    transformd,
    spatial_vectorh,
    spatial_vectorf,
    spatial_vectord,
    spatial_matrixh,
    spatial_matrixf,
    spatial_matrixd,
)

np_dtype_to_warp_type = {
    # Numpy scalar types
    np.bool_: bool,
    np.int8: int8,
    np.uint8: uint8,
    np.int16: int16,
    np.uint16: uint16,
    np.int32: int32,
    np.int64: int64,
    np.uint32: uint32,
    np.uint64: uint64,
    np.byte: int8,
    np.ubyte: uint8,
    np.float16: float16,
    np.float32: float32,
    np.float64: float64,
    # Numpy dtype objects
    np.dtype(np.bool_): bool,
    np.dtype(np.int8): int8,
    np.dtype(np.uint8): uint8,
    np.dtype(np.int16): int16,
    np.dtype(np.uint16): uint16,
    np.dtype(np.int32): int32,
    np.dtype(np.int64): int64,
    np.dtype(np.uint32): uint32,
    np.dtype(np.uint64): uint64,
    np.dtype(np.byte): int8,
    np.dtype(np.ubyte): uint8,
    np.dtype(np.float16): float16,
    np.dtype(np.float32): float32,
    np.dtype(np.float64): float64,
}

warp_type_to_np_dtype = {
    bool: np.bool_,
    int8: np.int8,
    int16: np.int16,
    int32: np.int32,
    int64: np.int64,
    uint8: np.uint8,
    uint16: np.uint16,
    uint32: np.uint32,
    uint64: np.uint64,
    float16: np.float16,
    float32: np.float32,
    float64: np.float64,
}

non_atomic_types = (
    int8,
    uint8,
    int16,
    uint16,
    int64,
)


def dtype_from_numpy(numpy_dtype):
    """Return the Warp dtype corresponding to a NumPy dtype."""
    wp_dtype = np_dtype_to_warp_type.get(numpy_dtype)
    if wp_dtype is not None:
        return wp_dtype
    else:
        raise TypeError(f"Cannot convert {numpy_dtype} to a Warp type")


def dtype_to_numpy(warp_dtype):
    """Return the NumPy dtype corresponding to a Warp dtype."""
    np_dtype = warp_type_to_np_dtype.get(warp_dtype)
    if np_dtype is not None:
        return np_dtype
    else:
        raise TypeError(f"Cannot convert {warp_dtype} to a NumPy type")


# represent a Python range iterator
class range_t:
    def __init__(self):
        pass


# definition just for kernel type (cannot be a parameter), see bvh.h
class bvh_query_t:
    """Object used to track state during BVH traversal."""

    def __init__(self):
        pass


BvhQuery = bvh_query_t


# definition just for kernel type (cannot be a parameter), see mesh.h
class mesh_query_aabb_t:
    """Object used to track state during mesh traversal."""

    def __init__(self):
        pass


MeshQueryAABB = mesh_query_aabb_t


# definition just for kernel type (cannot be a parameter), see hash_grid.h
class hash_grid_query_t:
    """Object used to track state during neighbor traversal."""

    def __init__(self):
        pass


HashGridQuery = hash_grid_query_t


# maximum number of dimensions, must match array.h
ARRAY_MAX_DIMS = 4
LAUNCH_MAX_DIMS = 4

# must match array.h
ARRAY_TYPE_REGULAR = 0
ARRAY_TYPE_INDEXED = 1
ARRAY_TYPE_FABRIC = 2
ARRAY_TYPE_FABRIC_INDEXED = 3


# represents bounds for kernel launch (number of threads across multiple dimensions)
class launch_bounds_t(ctypes.Structure):
    _fields_ = [("shape", ctypes.c_int32 * LAUNCH_MAX_DIMS), ("ndim", ctypes.c_int32), ("size", ctypes.c_size_t)]

    def __init__(self, shape: int | Sequence[int]):
        if isinstance(shape, int):
            # 1d launch
            self.ndim = 1
            self.size = shape
            self.shape[0] = shape

        else:
            # nd launch
            self.ndim = len(shape)
            self.size = 1

            for i in range(self.ndim):
                self.shape[i] = shape[i]
                self.size = self.size * shape[i]

        # initialize the remaining dims to 1
        for i in range(self.ndim, LAUNCH_MAX_DIMS):
            self.shape[i] = 1


class shape_t(ctypes.Structure):
    _fields_ = [("dims", ctypes.c_int32 * ARRAY_MAX_DIMS)]

    def __init__(self):
        pass


class array_t(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_uint64),
        ("grad", ctypes.c_uint64),
        ("shape", ctypes.c_int32 * ARRAY_MAX_DIMS),
        ("strides", ctypes.c_int32 * ARRAY_MAX_DIMS),
        ("ndim", ctypes.c_int32),
    ]

    def __init__(self, data=0, grad=0, ndim=0, shape=(0,), strides=(0,)):
        self.data = data
        self.grad = grad
        self.ndim = ndim
        for i in range(ndim):
            self.shape[i] = shape[i]
            self.strides[i] = strides[i]

    # structured type description used when array_t is packed in a struct and shared via numpy structured array.
    @classmethod
    def numpy_dtype(cls):
        return cls._numpy_dtype_

    # structured value used when array_t is packed in a struct and shared via a numpy structured array
    def numpy_value(self):
        return (self.data, self.grad, list(self.shape), list(self.strides), self.ndim)


# NOTE: must match array_t._fields_
array_t._numpy_dtype_ = {
    "names": ["data", "grad", "shape", "strides", "ndim"],
    "formats": ["u8", "u8", f"{ARRAY_MAX_DIMS}i4", f"{ARRAY_MAX_DIMS}i4", "i4"],
    "offsets": [
        array_t.data.offset,
        array_t.grad.offset,
        array_t.shape.offset,
        array_t.strides.offset,
        array_t.ndim.offset,
    ],
    "itemsize": ctypes.sizeof(array_t),
}


class indexedarray_t(ctypes.Structure):
    _fields_ = [
        ("data", array_t),
        ("indices", ctypes.c_void_p * ARRAY_MAX_DIMS),
        ("shape", ctypes.c_int32 * ARRAY_MAX_DIMS),
    ]

    def __init__(self, data, indices, shape):
        if data is None:
            self.data = array().__ctype__()
            for i in range(ARRAY_MAX_DIMS):
                self.indices[i] = ctypes.c_void_p(None)
                self.shape[i] = 0
        else:
            self.data = data.__ctype__()
            for i in range(data.ndim):
                if indices[i] is not None:
                    self.indices[i] = ctypes.c_void_p(indices[i].ptr)
                else:
                    self.indices[i] = ctypes.c_void_p(None)
                self.shape[i] = shape[i]


def type_ctype(dtype):
    if dtype == float:
        return ctypes.c_float
    elif dtype == int:
        return ctypes.c_int32
    else:
        # scalar type
        return dtype._type_


def type_length(dtype):
    if dtype == float or dtype == int or isinstance(dtype, warp.codegen.Struct):
        return 1
    else:
        return dtype._length_


def type_scalar_type(dtype):
    return getattr(dtype, "_wp_scalar_type_", dtype)


# Cache results of type_size_in_bytes(), because the function is actually quite slow.
_type_size_cache = {
    float: 4,
    int: 4,
}


def type_size_in_bytes(dtype: type) -> int:
    size = _type_size_cache.get(dtype)

    if size is None:
        if dtype.__module__ == "ctypes":
            size = ctypes.sizeof(dtype)
        elif hasattr(dtype, "_type_"):
            size = getattr(dtype, "_length_", 1) * ctypes.sizeof(dtype._type_)
        elif isinstance(dtype, warp.codegen.Struct):
            size = ctypes.sizeof(dtype.ctype)
        elif dtype == Any:
            raise TypeError("A concrete type is required")
        else:
            raise TypeError(f"Invalid data type: {dtype}")
        _type_size_cache[dtype] = size

    return size


def type_to_warp(dtype: type) -> type:
    if dtype == float:
        return float32
    elif dtype == int:
        return int32
    elif dtype == builtins.bool:
        return bool
    else:
        return dtype


def type_typestr(dtype: type) -> str:
    if dtype == bool:
        return "|b1"
    elif dtype == float16:
        return "<f2"
    elif dtype == float32:
        return "<f4"
    elif dtype == float64:
        return "<f8"
    elif dtype == int8:
        return "|i1"
    elif dtype == uint8:
        return "|u1"
    elif dtype == int16:
        return "<i2"
    elif dtype == uint16:
        return "<u2"
    elif dtype == int32:
        return "<i4"
    elif dtype == uint32:
        return "<u4"
    elif dtype == int64:
        return "<i8"
    elif dtype == uint64:
        return "<u8"
    elif isinstance(dtype, warp.codegen.Struct):
        return f"|V{ctypes.sizeof(dtype.ctype)}"
    elif issubclass(dtype, ctypes.Array):
        return type_typestr(dtype._wp_scalar_type_)
    else:
        raise Exception("Unknown ctype")


# converts any known type to a human readable string, good for error messages, reporting etc
def type_repr(t):
    if is_array(t):
        return str(f"array(ndim={t.ndim}, dtype={t.dtype})")
    if is_tile(t):
        return str(f"tile(dtype={t.dtype}, shape={t.shape}")
    if type_is_vector(t):
        return str(f"vector(length={t._shape_[0]}, dtype={t._wp_scalar_type_})")
    if type_is_matrix(t):
        return str(f"matrix(shape=({t._shape_[0]}, {t._shape_[1]}), dtype={t._wp_scalar_type_})")
    if isinstance(t, warp.codegen.Struct):
        return type_repr(t.cls)
    if t in scalar_types:
        return t.__name__

    name = getattr(t, "__qualname__", t.__name__)
    return t.__module__ + "." + name


def type_is_int(t):
    if t == int:
        t = int32

    return t in int_types


def type_is_float(t):
    if t == float:
        t = float32

    return t in float_types


# returns True if the passed *type* is a vector
def type_is_vector(t):
    return getattr(t, "_wp_generic_type_hint_", None) is Vector


# returns True if the passed *type* is a quaternion
def type_is_quaternion(t):
    return getattr(t, "_wp_generic_type_hint_", None) is Quaternion


# returns True if the passed *type* is a matrix
def type_is_matrix(t):
    return getattr(t, "_wp_generic_type_hint_", None) is Matrix


# returns True if the passed *type* is a transformation
def type_is_transformation(t):
    return getattr(t, "_wp_generic_type_hint_", None) is Transformation


value_types = (int, float, builtins.bool) + scalar_and_bool_types


# returns true for all value types (int, float, bool, scalars, vectors, matrices)
def type_is_value(x: Any) -> builtins.bool:
    return x in value_types or hasattr(x, "_wp_scalar_type_")


# equivalent of the above but for values
def is_int(x: Any) -> builtins.bool:
    return type_is_int(type(x))


def is_float(x: Any) -> builtins.bool:
    return type_is_float(type(x))


def is_value(x: Any) -> builtins.bool:
    return type_is_value(type(x))


def is_array(a) -> builtins.bool:
    """Return true if the passed *instance* is one of the array types."""
    return isinstance(a, array_types)


def scalars_equal(a, b, match_generic):
    # convert to canonical types
    if a == float:
        a = float32
    elif a == int:
        a = int32
    elif a == builtins.bool:
        a = bool

    if b == float:
        b = float32
    elif b == int:
        b = int32
    elif b == builtins.bool:
        b = bool

    if match_generic:
        if a == Any or b == Any:
            return True
        if a == Int and b in int_types:
            return True
        if b == Int and a in int_types:
            return True
        if a == Int and b == Int:
            return True
        if a == Scalar and b in scalar_and_bool_types:
            return True
        if b == Scalar and a in scalar_and_bool_types:
            return True
        if a == Scalar and b == Scalar:
            return True
        if a == Float and b in float_types:
            return True
        if b == Float and a in float_types:
            return True
        if a == Float and b == Float:
            return True

    return a == b


def types_equal(a, b, match_generic=False):
    if match_generic:
        # Special cases to interpret the types listed in `int_tuple_type_hints`
        # as generic hints that accept any integer types.
        if a in int_tuple_type_hints and isinstance(b, Sequence):
            a_length = int_tuple_type_hints[a]
            if (a_length == -1 or a_length == len(b)) and all(
                scalars_equal(x, Int, match_generic=match_generic) for x in b
            ):
                return True
        if b in int_tuple_type_hints and isinstance(a, Sequence):
            b_length = int_tuple_type_hints[b]
            if (b_length == -1 or b_length == len(a)) and all(
                scalars_equal(x, Int, match_generic=match_generic) for x in a
            ):
                return True
        if a in int_tuple_type_hints and b in int_tuple_type_hints:
            a_length = int_tuple_type_hints[a]
            b_length = int_tuple_type_hints[b]
            if a_length is None or b_length is None or a_length == b_length:
                return True

    a_origin = get_origin(a)
    b_origin = get_origin(b)
    if a_origin is tuple and b_origin is tuple:
        a_args = get_args(a)
        b_args = get_args(b)
        if len(a_args) == len(b_args) and all(
            scalars_equal(x, y, match_generic=match_generic) for x, y in zip(a_args, b_args)
        ):
            return True
    elif a_origin is tuple and isinstance(b, Sequence):
        a_args = get_args(a)
        if len(a_args) == len(b) and all(scalars_equal(x, y, match_generic=match_generic) for x, y in zip(a_args, b)):
            return True
    elif b_origin is tuple and isinstance(a, Sequence):
        b_args = get_args(b)
        if len(b_args) == len(a) and all(scalars_equal(x, y, match_generic=match_generic) for x, y in zip(b_args, a)):
            return True

    # convert to canonical types
    if a == float:
        a = float32
    elif a == int:
        a = int32
    elif a == builtins.bool:
        a = bool

    if b == float:
        b = float32
    elif b == int:
        b = int32
    elif b == builtins.bool:
        b = bool

    if getattr(a, "_wp_generic_type_hint_", "a") is getattr(b, "_wp_generic_type_hint_", "b"):
        for p1, p2 in zip(a._wp_type_params_, b._wp_type_params_):
            if not scalars_equal(p1, p2, match_generic):
                return False

        return True

    if is_array(a) and type(a) is type(b) and types_equal(a.dtype, b.dtype, match_generic=match_generic):
        return True

    # match NewStructInstance and Struct dtype
    if getattr(a, "cls", "a") is getattr(b, "cls", "b"):
        return True

    if is_tile(a) and is_tile(b):
        return True

    return scalars_equal(a, b, match_generic)


def strides_from_shape(shape: tuple, dtype):
    ndims = len(shape)
    strides = [None] * ndims

    i = ndims - 1
    strides[i] = type_size_in_bytes(dtype)

    while i > 0:
        strides[i - 1] = strides[i] * shape[i]
        i -= 1

    return tuple(strides)


def check_array_shape(shape: tuple):
    """Checks that the size in each dimension is positive and less than 2^31."""

    for dim_index, dim_size in enumerate(shape):
        if dim_size < 0:
            raise ValueError(f"Array shapes must be non-negative, got {dim_size} in dimension {dim_index}")
        if dim_size >= 2**31:
            raise ValueError(
                "Array shapes must not exceed the maximum representable value of a signed 32-bit integer, "
                f"got {dim_size} in dimension {dim_index}."
            )


def array_ctype_from_interface(interface: dict, dtype=None, owner=None):
    """Get native array descriptor (array_t) from __array_interface__ or __cuda_array_interface__ dictionary"""

    ptr = interface.get("data")[0]
    shape = interface.get("shape")
    strides = interface.get("strides")
    typestr = interface.get("typestr")

    element_dtype = dtype_from_numpy(np.dtype(typestr))

    if strides is None:
        strides = strides_from_shape(shape, element_dtype)

    if dtype is None:
        # accept verbatim
        pass
    elif hasattr(dtype, "_shape_"):
        # vector/matrix types, ensure element dtype matches
        if element_dtype != dtype._wp_scalar_type_:
            raise RuntimeError(
                f"Could not convert array interface with typestr='{typestr}' to Warp array with dtype={dtype}"
            )
        dtype_shape = dtype._shape_
        dtype_dims = len(dtype._shape_)
        ctype_size = ctypes.sizeof(dtype._type_)
        # ensure inner shape matches
        if dtype_dims > len(shape) or dtype_shape != shape[-dtype_dims:]:
            raise RuntimeError(
                f"Could not convert array interface with shape {shape} to Warp array with dtype={dtype}, ensure that source inner shape is {dtype_shape}"
            )
        # ensure inner strides are contiguous
        if strides[-1] != ctype_size or (dtype_dims > 1 and strides[-2] != ctype_size * dtype_shape[-1]):
            raise RuntimeError(
                f"Could not convert array interface with shape {shape} to Warp array with dtype={dtype}, because the source inner strides are not contiguous"
            )
        # trim shape and strides
        shape = tuple(shape[:-dtype_dims]) or (1,)
        strides = tuple(strides[:-dtype_dims]) or (ctype_size,)
    else:
        # scalar types, ensure dtype matches
        if element_dtype != dtype:
            raise RuntimeError(
                f"Could not convert array interface with typestr='{typestr}' to Warp array with dtype={dtype}"
            )

    # create array descriptor
    array_ctype = array_t(ptr, 0, len(shape), shape, strides)

    # keep owner alive
    if owner is not None:
        array_ctype._ref = owner

    return array_ctype


class array(Array[DType]):
    """A fixed-size multi-dimensional array containing values of the same type.

    Attributes:
        dtype (DType): The data type of the array.
        ndim (int): The number of array dimensions.
        size (int): The number of items in the array.
        capacity (int): The amount of memory in bytes allocated for this array.
        shape (tuple[int]): Dimensions of the array.
        strides (tuple[int]): Number of bytes in each dimension between successive elements of the array.
        ptr (int): Pointer to underlying memory allocation backing the array.
        device (Device): The device where the array's memory allocation resides.
        pinned (bool): Indicates whether the array was allocated in pinned host memory.
        is_contiguous (bool): Indicates whether this array has a contiguous memory layout.
        deleter (Callable[[int, int], None]): A function to be called when the array is deleted,
            taking two arguments: pointer and size. If ``None``, then no function is called.
    """

    # member attributes available during code-gen (e.g.: d = array.shape[0])
    # (initialized when needed)
    _vars = None

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.deleter = None
        return instance

    def __init__(
        self,
        data: list | tuple | npt.NDArray | None = None,
        dtype: Any = Any,
        shape: int | tuple[int, ...] | list[int] | None = None,
        strides: tuple[int, ...] | None = None,
        length: int | None = None,
        ptr: int | None = None,
        capacity: int | None = None,
        device=None,
        pinned: builtins.bool = False,
        copy: builtins.bool = True,
        owner: builtins.bool = False,  # deprecated - pass deleter instead
        deleter: Callable[[int, int], None] | None = None,
        ndim: int | None = None,
        grad: array | None = None,
        requires_grad: builtins.bool = False,
    ):
        """Constructs a new Warp array object

        When the ``data`` argument is a valid list, tuple, or ndarray the array will be constructed from this object's data.
        For objects that are not stored sequentially in memory (e.g.: a list), then the data will first
        be flattened before being transferred to the memory space given by device.

        The second construction path occurs when the ``ptr`` argument is a non-zero uint64 value representing the
        start address in memory where existing array data resides, e.g.: from an external or C-library. The memory
        allocation should reside on the same device given by the device argument, and the user should set the length
        and dtype parameter appropriately.

        If neither ``data`` nor ``ptr`` are specified, the ``shape`` or ``length`` arguments are checked next.
        This construction path can be used to create new uninitialized arrays, but users are encouraged to call
        ``wp.empty()``, ``wp.zeros()``, or ``wp.full()`` instead to create new arrays.

        If none of the above arguments are specified, a simple type annotation is constructed.  This is used when annotating
        kernel arguments or struct members (e.g.,``arr: wp.array(dtype=float)``).  In this case, only ``dtype`` and ``ndim``
        are taken into account and no memory is allocated for the array.

        Args:
            data: An object to construct the array from, can be a Tuple, List, or generally any type convertible to an np.array
            dtype: One of the available `data types <#data-types>`_, such as :class:`warp.float32`, :class:`warp.mat33`, or a custom `struct <#structs>`_. If dtype is ``Any`` and data is an ndarray, then it will be inferred from the array data type
            shape: Dimensions of the array
            strides: Number of bytes in each dimension between successive elements of the array
            length: Number of elements of the data type (deprecated, users should use ``shape`` argument)
            ptr: Address of an external memory address to alias (``data`` should be ``None``)
            capacity: Maximum size in bytes of the ``ptr`` allocation (``data`` should be ``None``)
            device (Devicelike): Device the array lives on
            copy: Whether the incoming ``data`` will be copied or aliased. Aliasing requires that
                the incoming ``data`` already lives on the ``device`` specified and the data types match.
            owner: Whether the array will try to deallocate the underlying memory when it is deleted
                (deprecated, pass ``deleter`` if you wish to transfer ownership to Warp)
            deleter: Function to be called when the array is deleted, taking two arguments: pointer and size
            requires_grad: Whether or not gradients will be tracked for this array, see :class:`warp.Tape` for details
            grad: The array in which to accumulate gradients in the backward pass. If ``None`` and ``requires_grad`` is ``True``,
                then a gradient array will be allocated automatically.
            pinned: Whether to allocate pinned host memory, which allows asynchronous host–device transfers
                (only applicable with ``device="cpu"``)

        """

        self.ctype = None

        # properties
        self._requires_grad = False
        self._grad = None
        # __array_interface__ or __cuda_array_interface__, evaluated lazily and cached
        self._array_interface = None
        self.is_transposed = False

        # reference to other array
        self._ref = None

        # canonicalize dtype
        if dtype == int:
            dtype = int32
        elif dtype == float:
            dtype = float32
        elif dtype == builtins.bool:
            dtype = bool

        # convert shape to tuple (or leave shape=None if neither shape nor length were specified)
        if shape is not None:
            if isinstance(shape, int):
                shape = (shape,)
            else:
                # The type of shape's elements are eventually passed onto capacity which is used to allocate memory. We
                # explicitly enforce that shape is a tuple of (64-bit) ints to ensure that capacity is 64-bit.
                shape = tuple(int(x) for x in shape)
                if len(shape) > ARRAY_MAX_DIMS:
                    raise RuntimeError(
                        f"Failed to create array with shape {shape}, the maximum number of dimensions is {ARRAY_MAX_DIMS}"
                    )
        elif length is not None:
            # backward compatibility
            warp.utils.warn(
                "The 'length' keyword is deprecated and will be removed in a future version. Use 'shape' instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            shape = (length,)

        if owner:
            warp.utils.warn(
                "The 'owner' keyword in the array initializer is\n"
                "deprecated and will be removed in a future version. It currently has no effect.\n"
                "Pass a function to the 'deleter' keyword instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        # determine the construction path from the given arguments
        if data is not None:
            # data or ptr, not both
            if ptr is not None:
                raise RuntimeError("Can only construct arrays with either `data` or `ptr` arguments, not both")
            self._init_from_data(data, dtype, shape, device, copy, pinned)
        elif ptr is not None:
            self._init_from_ptr(ptr, dtype, shape, strides, capacity, device, pinned, deleter)
        elif shape is not None:
            self._init_new(dtype, shape, strides, device, pinned)
        else:
            self._init_annotation(dtype, ndim or 1)

        # initialize read flag
        self.mark_init()

        # initialize gradient, if needed
        if self.device is not None:
            if grad is not None:
                # this will also check whether the gradient array is compatible
                self.grad = grad
            else:
                # allocate gradient if needed
                self._requires_grad = requires_grad
                if requires_grad:
                    self._alloc_grad()

    def _init_from_data(self, data, dtype, shape, device, copy, pinned):
        if not hasattr(data, "__len__"):
            raise RuntimeError(f"Data must be a sequence or array, got scalar {data}")

        if hasattr(dtype, "_wp_scalar_type_"):
            dtype_shape = dtype._shape_
            dtype_ndim = len(dtype_shape)
            scalar_dtype = dtype._wp_scalar_type_
        else:
            dtype_shape = ()
            dtype_ndim = 0
            scalar_dtype = dtype

        try:
            # Performance note: try first, ask questions later
            device = warp.context.runtime.get_device(device)
        except Exception:
            # Fallback to using the public API for retrieving the device,
            # which takes take of initializing Warp if needed.
            device = warp.context.get_device(device)

        if device.is_cuda and hasattr(data, "__cuda_array_interface__"):
            desc = data.__cuda_array_interface__
            data_shape = desc.get("shape")
            data_strides = desc.get("strides")
            data_dtype = np.dtype(desc.get("typestr"))
            data_ptr = desc.get("data")[0]

            if dtype == Any:
                dtype = np_dtype_to_warp_type[data_dtype]

            if data_strides is None:
                data_strides = strides_from_shape(data_shape, dtype)

            data_ndim = len(data_shape)

            # determine whether the input needs reshaping
            target_npshape = None
            if shape is not None:
                target_npshape = (*shape, *dtype_shape)
            elif dtype_ndim > 0:
                # prune inner dimensions of length 1
                while data_ndim > 1 and data_shape[-1] == 1:
                    data_shape = data_shape[:-1]
                # if the inner dims don't match exactly, check if the innermost dim is a multiple of type length
                if data_ndim < dtype_ndim or data_shape[-dtype_ndim:] != dtype_shape:
                    if data_shape[-1] == dtype._length_:
                        target_npshape = (*data_shape[:-1], *dtype_shape)
                    elif data_shape[-1] % dtype._length_ == 0:
                        target_npshape = (*data_shape[:-1], data_shape[-1] // dtype._length_, *dtype_shape)
                    else:
                        if dtype_ndim == 1:
                            raise RuntimeError(
                                f"The inner dimensions of the input data are not compatible with the requested vector type {warp.context.type_str(dtype)}: expected an inner dimension that is a multiple of {dtype._length_}"
                            )
                        else:
                            raise RuntimeError(
                                f"The inner dimensions of the input data are not compatible with the requested matrix type {warp.context.type_str(dtype)}: expected inner dimensions {dtype._shape_} or a multiple of {dtype._length_}"
                            )

            if target_npshape is None:
                target_npshape = data_shape if shape is None else shape

            # determine final shape and strides
            if dtype_ndim > 0:
                # make sure the inner dims are contiguous for vector/matrix types
                scalar_size = type_size_in_bytes(dtype._wp_scalar_type_)
                inner_contiguous = data_strides[-1] == scalar_size
                if inner_contiguous and dtype_ndim > 1:
                    inner_contiguous = data_strides[-2] == scalar_size * dtype_shape[-1]

                shape = target_npshape[:-dtype_ndim] or (1,)
                strides = data_strides if shape == data_shape else strides_from_shape(shape, dtype)
            else:
                shape = target_npshape or (1,)
                strides = data_strides if shape == data_shape else strides_from_shape(shape, dtype)

            self._init_from_ptr(data_ptr, dtype, shape, strides, None, device, False, None)

            # keep a ref to the source data to keep allocation alive
            self._ref = data
            return

        # convert input data to ndarray (handles lists, tuples, etc.) and determine dtype
        if dtype == Any:
            # infer dtype from data
            try:
                arr = np.asarray(data)
            except Exception as e:
                raise RuntimeError(f"Failed to convert input data to an array: {e}") from e
            dtype = np_dtype_to_warp_type.get(arr.dtype)
            if dtype is None:
                raise RuntimeError(f"Unsupported input data dtype: {arr.dtype}")
        elif isinstance(dtype, warp.codegen.Struct):
            if isinstance(data, np.ndarray):
                # construct from numpy structured array
                if data.dtype != dtype.numpy_dtype():
                    raise RuntimeError(
                        f"Invalid source data type for array of structs, expected {dtype.numpy_dtype()}, got {data.dtype}"
                    )
                arr = data
            elif isinstance(data, (list, tuple)):
                # construct from a sequence of structs
                try:
                    # convert each struct instance to its corresponding ctype
                    ctype_list = [v.__ctype__() for v in data]
                    # convert the list of ctypes to a contiguous ctypes array
                    ctype_arr = (dtype.ctype * len(ctype_list))(*ctype_list)
                    # convert to numpy
                    arr = np.frombuffer(ctype_arr, dtype=dtype.ctype)
                except Exception as e:
                    raise RuntimeError(
                        f"Error while trying to construct Warp array from a sequence of Warp structs: {e}"
                    ) from e
            else:
                raise RuntimeError(
                    "Invalid data argument for array of structs, expected a sequence of structs or a NumPy structured array"
                )
        else:
            # convert input data to the given dtype
            npdtype = warp_type_to_np_dtype.get(scalar_dtype)
            if npdtype is None:
                raise RuntimeError(
                    f"Failed to convert input data to an array with Warp type {warp.context.type_str(dtype)}"
                )
            try:
                arr = np.asarray(data, dtype=npdtype)
            except Exception as e:
                raise RuntimeError(f"Failed to convert input data to an array with type {npdtype}: {e}") from e

        # determine whether the input needs reshaping
        target_npshape = None
        if shape is not None:
            target_npshape = (*shape, *dtype_shape)
        elif dtype_ndim > 0:
            # prune inner dimensions of length 1
            while arr.ndim > 1 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
            # if the inner dims don't match exactly, check if the innermost dim is a multiple of type length
            if arr.ndim < dtype_ndim or arr.shape[-dtype_ndim:] != dtype_shape:
                if arr.shape[-1] == dtype._length_:
                    target_npshape = (*arr.shape[:-1], *dtype_shape)
                elif arr.shape[-1] % dtype._length_ == 0:
                    target_npshape = (*arr.shape[:-1], arr.shape[-1] // dtype._length_, *dtype_shape)
                else:
                    if dtype_ndim == 1:
                        raise RuntimeError(
                            f"The inner dimensions of the input data are not compatible with the requested vector type {warp.context.type_str(dtype)}: expected an inner dimension that is a multiple of {dtype._length_}"
                        )
                    else:
                        raise RuntimeError(
                            f"The inner dimensions of the input data are not compatible with the requested matrix type {warp.context.type_str(dtype)}: expected inner dimensions {dtype._shape_} or a multiple of {dtype._length_}"
                        )

        if target_npshape is not None:
            try:
                arr = arr.reshape(target_npshape)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to reshape the input data to the given shape {shape} and type {warp.context.type_str(dtype)}: {e}"
                ) from e

        # determine final shape and strides
        if dtype_ndim > 0:
            # make sure the inner dims are contiguous for vector/matrix types
            scalar_size = type_size_in_bytes(dtype._wp_scalar_type_)
            inner_contiguous = arr.strides[-1] == scalar_size
            if inner_contiguous and dtype_ndim > 1:
                inner_contiguous = arr.strides[-2] == scalar_size * dtype_shape[-1]

            if not inner_contiguous:
                arr = np.ascontiguousarray(arr)

            shape = arr.shape[:-dtype_ndim] or (1,)
            strides = arr.strides[:-dtype_ndim] or (type_size_in_bytes(dtype),)
        else:
            shape = arr.shape or (1,)
            strides = arr.strides or (type_size_in_bytes(dtype),)

        try:
            # Performance note: try first, ask questions later
            device = warp.context.runtime.get_device(device)
        except Exception:
            # Fallback to using the public API for retrieving the device,
            # which takes take of initializing Warp if needed.
            device = warp.context.get_device(device)

        if device.is_cpu and not copy and not pinned:
            # reference numpy memory directly
            self._init_from_ptr(arr.ctypes.data, dtype, shape, strides, None, device, False, None)
            # keep a ref to the source array to keep allocation alive
            self._ref = arr
        else:
            # copy data into a new array
            self._init_new(dtype, shape, None, device, pinned)
            src = array(
                ptr=arr.ctypes.data,
                dtype=dtype,
                shape=shape,
                strides=strides,
                device="cpu",
                copy=False,
            )
            warp.copy(self, src)

    def _init_from_ptr(self, ptr, dtype, shape, strides, capacity, device, pinned, deleter):
        try:
            # Performance note: try first, ask questions later
            device = warp.context.runtime.get_device(device)
        except Exception:
            # Fallback to using the public API for retrieving the device,
            # which takes take of initializing Warp if needed.
            device = warp.context.get_device(device)

        check_array_shape(shape)
        ndim = len(shape)
        dtype_size = type_size_in_bytes(dtype)

        # compute size and contiguous strides
        # Performance note: we could use strides_from_shape() here, but inlining it is faster.
        contiguous_strides = [None] * ndim
        i = ndim - 1
        contiguous_strides[i] = dtype_size
        size = shape[i]
        while i > 0:
            contiguous_strides[i - 1] = contiguous_strides[i] * shape[i]
            i -= 1
            size *= shape[i]
        contiguous_strides = tuple(contiguous_strides)

        if strides is None:
            strides = contiguous_strides
            is_contiguous = True
            if capacity is None:
                capacity = size * dtype_size
        else:
            strides = tuple(strides)
            is_contiguous = strides == contiguous_strides
            if capacity is None:
                capacity = shape[0] * strides[0]

        self.dtype = dtype
        self.ndim = ndim
        self.size = size
        self.capacity = capacity
        self.shape = shape
        self.strides = strides
        self.ptr = ptr
        self.device = device
        self.pinned = pinned if device.is_cpu else False
        self.is_contiguous = is_contiguous
        self.deleter = deleter

    def _init_new(self, dtype, shape, strides, device, pinned):
        try:
            # Performance note: try first, ask questions later
            device = warp.context.runtime.get_device(device)
        except Exception:
            # Fallback to using the public API for retrieving the device,
            # which takes take of initializing Warp if needed.
            device = warp.context.get_device(device)

        check_array_shape(shape)
        ndim = len(shape)
        dtype_size = type_size_in_bytes(dtype)

        # compute size and contiguous strides
        # Performance note: we could use strides_from_shape() here, but inlining it is faster.
        contiguous_strides = [None] * ndim
        i = ndim - 1
        contiguous_strides[i] = dtype_size
        size = shape[i]
        while i > 0:
            contiguous_strides[i - 1] = contiguous_strides[i] * shape[i]
            i -= 1
            size *= shape[i]
        contiguous_strides = tuple(contiguous_strides)

        if strides is None:
            strides = contiguous_strides
            is_contiguous = True
            capacity = size * dtype_size
        else:
            strides = tuple(strides)
            is_contiguous = strides == contiguous_strides

            # To calculate the required capacity, find the dimension with largest stride.
            # Normally it is the first one, but it could be different (e.g., transposed arrays).
            max_stride = strides[0]
            max_dim = 0
            for i in range(1, ndim):
                if strides[i] > max_stride:
                    max_stride = strides[i]
                    max_dim = i

            if max_stride > 0:
                capacity = shape[max_dim] * strides[max_dim]
            else:
                # single element storage with zero strides
                capacity = dtype_size

        allocator = device.get_allocator(pinned=pinned)
        if capacity > 0:
            ptr = allocator.alloc(capacity)
        else:
            ptr = None

        self.dtype = dtype
        self.ndim = ndim
        self.size = size
        self.capacity = capacity
        self.shape = shape
        self.strides = strides
        self.ptr = ptr
        self.device = device
        self.pinned = pinned if device.is_cpu else False
        self.is_contiguous = is_contiguous
        self.deleter = allocator.deleter
        self._allocator = allocator

    def _init_annotation(self, dtype, ndim):
        self.dtype = dtype
        self.ndim = ndim
        self.size = 0
        self.capacity = 0
        self.shape = (0,) * ndim
        self.strides = (0,) * ndim
        self.ptr = None
        self.device = None
        self.pinned = False
        self.is_contiguous = False

    def __del__(self):
        if self.deleter is None:
            return

        with self.device.context_guard:
            self.deleter(self.ptr, self.capacity)

    @property
    def __array_interface__(self):
        # raising an AttributeError here makes hasattr() return False
        if self.device is None or not self.device.is_cpu:
            raise AttributeError(f"__array_interface__ not supported because device is {self.device}")

        if self._array_interface is None:
            # get flat shape (including type shape)
            if isinstance(self.dtype, warp.codegen.Struct):
                # struct
                arr_shape = self.shape
                arr_strides = self.strides
                descr = self.dtype.numpy_dtype()
            elif issubclass(self.dtype, ctypes.Array):
                # vector type, flatten the dimensions into one tuple
                arr_shape = (*self.shape, *self.dtype._shape_)
                dtype_strides = strides_from_shape(self.dtype._shape_, self.dtype._type_)
                arr_strides = (*self.strides, *dtype_strides)
                descr = None
            else:
                # scalar type
                arr_shape = self.shape
                arr_strides = self.strides
                descr = None

            self._array_interface = {
                "data": (self.ptr if self.ptr is not None else 0, False),
                "shape": tuple(arr_shape),
                "strides": tuple(arr_strides),
                "typestr": type_typestr(self.dtype),
                "descr": descr,  # optional description of structured array layout
                "version": 3,
            }

        return self._array_interface

    @property
    def __cuda_array_interface__(self):
        # raising an AttributeError here makes hasattr() return False
        if self.device is None or not self.device.is_cuda:
            raise AttributeError(f"__cuda_array_interface__ is not supported because device is {self.device}")

        if self._array_interface is None:
            # get flat shape (including type shape)
            if issubclass(self.dtype, ctypes.Array):
                # vector type, flatten the dimensions into one tuple
                arr_shape = (*self.shape, *self.dtype._shape_)
                dtype_strides = strides_from_shape(self.dtype._shape_, self.dtype._type_)
                arr_strides = (*self.strides, *dtype_strides)
            else:
                # scalar or struct type
                arr_shape = self.shape
                arr_strides = self.strides

            self._array_interface = {
                "data": (self.ptr if self.ptr is not None else 0, False),
                "shape": tuple(arr_shape),
                "strides": tuple(arr_strides),
                "typestr": type_typestr(self.dtype),
                "version": 2,
            }

        return self._array_interface

    def __dlpack__(self, stream=None):
        # See https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.__dlpack__.html

        if self.device is None:
            raise RuntimeError("Array has no device assigned")

        # check if synchronization is needed
        if stream != -1:
            if self.device.is_cuda:
                # validate stream argument
                if stream is None:
                    stream = 1  # legacy default stream
                elif not isinstance(stream, int) or stream < -1:
                    raise TypeError("DLPack stream must None or an integer >= -1")

                # assume that the array is being used on its device's current stream
                array_stream = self.device.stream

                # Performance note: avoid wrapping the external stream in a temporary Stream object
                if stream != array_stream.cuda_stream:
                    warp.context.runtime.core.cuda_stream_wait_stream(
                        stream, array_stream.cuda_stream, array_stream.cached_event.cuda_event
                    )
            elif self.device.is_cpu:
                # on CPU, stream must be None or -1
                if stream is not None:
                    raise TypeError("DLPack stream must be None or -1 for CPU device")

        return warp.dlpack.to_dlpack(self)

    def __dlpack_device__(self):
        # See https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.__dlpack_device__.html

        if self.device is None:
            raise RuntimeError("Array has no device assigned")

        if self.device.is_cuda:
            return (warp.dlpack.DLDeviceType.kDLCUDA, self.device.ordinal)
        elif self.pinned:
            return (warp.dlpack.DLDeviceType.kDLCUDAHost, 0)
        else:
            return (warp.dlpack.DLDeviceType.kDLCPU, 0)

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        if self.device is None:
            # for 'empty' arrays we just return the type information, these are used in kernel function signatures
            return f"array{self.dtype}"
        else:
            return str(self.numpy())

    def __getitem__(self, key):
        if isinstance(key, int):
            if self.ndim == 1:
                raise RuntimeError("Item indexing is not supported on wp.array objects")
            key = [key]
        elif isinstance(key, (slice, array)):
            key = [key]
        elif isinstance(key, Tuple):
            contains_slice = False
            contains_indices = False
            for k in key:
                if isinstance(k, slice):
                    contains_slice = True
                if isinstance(k, array):
                    contains_indices = True
            if not contains_slice and not contains_indices and len(key) == self.ndim:
                raise RuntimeError("Item indexing is not supported on wp.array objects")
        else:
            raise RuntimeError(f"Invalid index: {key}")

        new_key = []
        for i in range(0, len(key)):
            new_key.append(key[i])
        for _i in range(len(key), self.ndim):
            new_key.append(slice(None, None, None))
        key = tuple(new_key)

        new_shape = []
        new_strides = []
        ptr_offset = 0
        new_dim = self.ndim

        # maps dimension index to an array of indices, if given
        index_arrays = {}

        for idx, k in enumerate(key):
            if isinstance(k, slice):
                start, stop, step = k.start, k.stop, k.step
                if start is None:
                    start = 0
                if stop is None:
                    stop = self.shape[idx]
                if step is None:
                    step = 1
                if start < 0:
                    start = self.shape[idx] + start
                if stop < 0:
                    stop = self.shape[idx] + stop

                if start < 0 or start >= self.shape[idx]:
                    raise RuntimeError(f"Invalid indexing in slice: {start}:{stop}:{step}")
                if stop < 1 or stop > self.shape[idx]:
                    raise RuntimeError(f"Invalid indexing in slice: {start}:{stop}:{step}")
                if stop <= start:
                    raise RuntimeError(f"Invalid indexing in slice: {start}:{stop}:{step}")

                new_shape.append(-((stop - start) // -step))  # ceil division
                new_strides.append(self.strides[idx] * step)

                ptr_offset += self.strides[idx] * start

            elif isinstance(k, array):
                # note: index array properties will be checked during indexedarray construction
                index_arrays[idx] = k

                # shape and strides are unchanged for this dimension
                new_shape.append(self.shape[idx])
                new_strides.append(self.strides[idx])

            else:  # is int
                start = k
                if start < 0:
                    start = self.shape[idx] + start
                if start < 0 or start >= self.shape[idx]:
                    raise RuntimeError(f"Invalid indexing in slice: {k}")
                new_dim -= 1

                ptr_offset += self.strides[idx] * start

        # handle grad
        if self.grad is not None:
            new_grad = array(
                ptr=self.grad.ptr + ptr_offset if self.grad.ptr is not None else None,
                dtype=self.grad.dtype,
                shape=tuple(new_shape),
                strides=tuple(new_strides),
                device=self.grad.device,
                pinned=self.grad.pinned,
            )
            # store back-ref to stop data being destroyed
            new_grad._ref = self.grad
        else:
            new_grad = None

        a = array(
            ptr=self.ptr + ptr_offset if self.ptr is not None else None,
            dtype=self.dtype,
            shape=tuple(new_shape),
            strides=tuple(new_strides),
            device=self.device,
            pinned=self.pinned,
            grad=new_grad,
        )

        # store back-ref to stop data being destroyed
        a._ref = self

        if index_arrays:
            indices = [None] * self.ndim
            for dim, index_array in index_arrays.items():
                indices[dim] = index_array
            return indexedarray(a, indices)
        else:
            return a

    # construct a C-representation of the array for passing to kernels
    def __ctype__(self):
        if self.ctype is None:
            data = 0 if self.ptr is None else ctypes.c_uint64(self.ptr)
            grad = 0 if self.grad is None or self.grad.ptr is None else ctypes.c_uint64(self.grad.ptr)
            self.ctype = array_t(data=data, grad=grad, ndim=self.ndim, shape=self.shape, strides=self.strides)

        return self.ctype

    def __matmul__(self, other):
        """
        Enables A @ B syntax for matrix multiplication
        """
        if not is_array(other):
            return NotImplemented

        if self.ndim != 2 or other.ndim != 2:
            raise RuntimeError(
                f"A has dim = {self.ndim}, B has dim = {other.ndim}. If multiplying with @, A and B must have dim = 2."
            )

        m = self.shape[0]
        n = other.shape[1]
        c = warp.zeros(shape=(m, n), dtype=self.dtype, device=self.device, requires_grad=True)
        d = warp.zeros(shape=(m, n), dtype=self.dtype, device=self.device, requires_grad=True)
        matmul(self, other, c, d)
        return d

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, grad):
        if grad is None:
            self._grad = None
            self._requires_grad = False
        else:
            # make sure the given gradient array is compatible
            if grad.dtype != self.dtype:
                raise ValueError(
                    f"The given gradient array is incompatible: expected dtype {self.dtype}, got {grad.dtype}"
                )
            if grad.shape != self.shape:
                raise ValueError(
                    f"The given gradient array is incompatible: expected shape {self.shape}, got {grad.shape}"
                )
            if grad.device != self.device:
                raise ValueError(
                    f"The given gradient array is incompatible: expected device {self.device}, got {grad.device}"
                )
            if grad.strides != self.strides:
                raise ValueError(
                    f"The given gradient array is incompatible: expected strides {self.strides}, got {grad.strides}"
                )
            self._grad = grad
            self._requires_grad = True

        # trigger re-creation of C-representation
        self.ctype = None

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: builtins.bool):
        if value and self._grad is None:
            self._alloc_grad()
        elif not value:
            self._grad = None

        self._requires_grad = value

        # trigger re-creation of C-representation
        self.ctype = None

    def _alloc_grad(self):
        self._grad = warp.zeros(
            dtype=self.dtype, shape=self.shape, strides=self.strides, device=self.device, pinned=self.pinned
        )

        # trigger re-creation of C-representation
        self.ctype = None

    @property
    def vars(self):
        # member attributes available during code-gen (e.g.: d = array.shape[0])
        # Note: we use a shared dict for all array instances
        if array._vars is None:
            array._vars = {"shape": warp.codegen.Var("shape", shape_t)}
        return array._vars

    def mark_init(self):
        """Resets this array's read flag"""
        self._is_read = False

    def mark_read(self):
        """Marks this array as having been read from in a kernel or recorded function on the tape."""
        # no additional checks required: it is always safe to set an array to READ
        self._is_read = True

        # recursively update all parent arrays
        parent = self._ref
        while parent is not None:
            parent._is_read = True
            parent = parent._ref

    def mark_write(self, **kwargs):
        """Detect if we are writing to an array that has already been read from"""
        if self._is_read:
            if "arg_name" and "kernel_name" and "filename" and "lineno" in kwargs:
                print(
                    f"Warning: Array {self} passed to argument {kwargs['arg_name']} in kernel {kwargs['kernel_name']} at {kwargs['filename']}:{kwargs['lineno']} is being written to but has already been read from in a previous launch. This may corrupt gradient computation in the backward pass."
                )
            else:
                print(
                    f"Warning: Array {self} is being written to but has already been read from in a previous launch. This may corrupt gradient computation in the backward pass."
                )

    def zero_(self):
        """Zeroes-out the array entries."""
        if self.is_contiguous:
            # simple memset is usually faster than generic fill
            self.device.memset(self.ptr, 0, self.size * type_size_in_bytes(self.dtype))
        else:
            self.fill_(0)
        self.mark_init()

    def fill_(self, value):
        """Set all array entries to `value`

        args:
            value: The value to set every array entry to. Must be convertible to the array's ``dtype``.

        Raises:
            ValueError: If `value` cannot be converted to the array's ``dtype``.

        Examples:
            ``fill_()`` can take lists or other sequences when filling arrays of vectors or matrices.

            >>> arr = wp.zeros(2, dtype=wp.mat22)
            >>> arr.numpy()
            array([[[0., 0.],
                    [0., 0.]],
            <BLANKLINE>
                   [[0., 0.],
                    [0., 0.]]], dtype=float32)
            >>> arr.fill_([[1, 2], [3, 4]])
            >>> arr.numpy()
            array([[[1., 2.],
                    [3., 4.]],
            <BLANKLINE>
                   [[1., 2.],
                    [3., 4.]]], dtype=float32)
        """
        if self.size == 0:
            return

        # try to convert the given value to the array dtype
        try:
            if isinstance(self.dtype, warp.codegen.Struct):
                if isinstance(value, self.dtype.cls):
                    cvalue = value.__ctype__()
                elif value == 0:
                    # allow zero-initializing structs using default constructor
                    cvalue = self.dtype().__ctype__()
                else:
                    raise ValueError(
                        f"Invalid initializer value for struct {self.dtype.cls.__name__}, expected struct instance or 0"
                    )
            elif issubclass(self.dtype, ctypes.Array):
                # vector/matrix
                cvalue = self.dtype(value)
            else:
                # scalar
                if type(value) in warp.types.scalar_types:
                    value = value.value
                if self.dtype == float16:
                    cvalue = self.dtype._type_(float_to_half_bits(value))
                else:
                    cvalue = self.dtype._type_(value)
        except Exception as e:
            raise ValueError(f"Failed to convert the value to the array data type: {e}") from e

        cvalue_ptr = ctypes.pointer(cvalue)
        cvalue_size = ctypes.sizeof(cvalue)

        # prefer using memtile for contiguous arrays, because it should be faster than generic fill
        if self.is_contiguous:
            self.device.memtile(self.ptr, cvalue_ptr, cvalue_size, self.size)
        else:
            carr = self.__ctype__()
            carr_ptr = ctypes.pointer(carr)

            if self.device.is_cuda:
                warp.context.runtime.core.array_fill_device(
                    self.device.context, carr_ptr, ARRAY_TYPE_REGULAR, cvalue_ptr, cvalue_size
                )
            else:
                warp.context.runtime.core.array_fill_host(carr_ptr, ARRAY_TYPE_REGULAR, cvalue_ptr, cvalue_size)

        self.mark_init()

    def assign(self, src):
        """Wraps ``src`` in an :class:`warp.array` if it is not already one and copies the contents to ``self``."""
        if is_array(src):
            warp.copy(self, src)
        else:
            warp.copy(self, array(data=src, dtype=self.dtype, copy=False, device="cpu"))

    def numpy(self):
        """Converts the array to a :class:`numpy.ndarray` (aliasing memory through the array interface protocol)
        If the array is on the GPU, a synchronous device-to-host copy (on the CUDA default stream) will be
        automatically performed to ensure that any outstanding work is completed.
        """
        if self.ptr:
            # use the CUDA default stream for synchronous behaviour with other streams
            with warp.ScopedStream(self.device.null_stream):
                a = self.to("cpu", requires_grad=False)
            # convert through __array_interface__
            # Note: this handles arrays of structs using `descr`, so the result will be a structured NumPy array
            return np.asarray(a)
        else:
            # return an empty numpy array with the correct dtype and shape
            if isinstance(self.dtype, warp.codegen.Struct):
                npdtype = self.dtype.numpy_dtype()
                npshape = self.shape
            elif issubclass(self.dtype, ctypes.Array):
                npdtype = warp_type_to_np_dtype[self.dtype._wp_scalar_type_]
                npshape = (*self.shape, *self.dtype._shape_)
            else:
                npdtype = warp_type_to_np_dtype[self.dtype]
                npshape = self.shape
            return np.empty(npshape, dtype=npdtype)

    def cptr(self):
        """Return a ctypes cast of the array address.

        Notes:

        #. Only CPU arrays support this method.
        #. The array must be contiguous.
        #. Accesses to this object are **not** bounds checked.
        #. For ``float16`` types, a pointer to the internal ``uint16`` representation is returned.
        """
        if not self.ptr:
            return None

        if self.device != "cpu" or not self.is_contiguous:
            raise RuntimeError(
                "Accessing array memory through a ctypes ptr is only supported for contiguous CPU arrays."
            )

        if isinstance(self.dtype, warp.codegen.Struct):
            p = ctypes.cast(self.ptr, ctypes.POINTER(self.dtype.ctype))
        else:
            p = ctypes.cast(self.ptr, ctypes.POINTER(self.dtype._type_))

        # store backref to the underlying array to avoid it being deallocated
        p._ref = self

        return p

    def list(self):
        """Returns a flattened list of items in the array as a Python list."""
        a = self.numpy()

        if isinstance(self.dtype, warp.codegen.Struct):
            # struct
            a = a.flatten()
            data = a.ctypes.data
            stride = a.strides[0]
            return [self.dtype.from_ptr(data + i * stride) for i in range(self.size)]
        elif issubclass(self.dtype, ctypes.Array):
            # vector/matrix - flatten, but preserve inner vector/matrix dimensions
            a = a.reshape((self.size, *self.dtype._shape_))
            data = a.ctypes.data
            stride = a.strides[0]
            return [self.dtype.from_ptr(data + i * stride) for i in range(self.size)]
        else:
            # scalar
            return list(a.flatten())

    def to(self, device, requires_grad=None):
        """Returns a Warp array with this array's data moved to the specified device, no-op if already on device."""
        device = warp.get_device(device)
        if self.device == device:
            return self
        else:
            return warp.clone(self, device=device, requires_grad=requires_grad)

    def flatten(self):
        """Returns a zero-copy view of the array collapsed to 1-D. Only supported for contiguous arrays."""
        if self.ndim == 1:
            return self

        if not self.is_contiguous:
            raise RuntimeError("Flattening non-contiguous arrays is unsupported.")

        a = array(
            ptr=self.ptr,
            dtype=self.dtype,
            shape=(self.size,),
            device=self.device,
            pinned=self.pinned,
            copy=False,
            grad=None if self.grad is None else self.grad.flatten(),
        )

        # transfer read flag
        a._is_read = self._is_read

        # store back-ref to stop data being destroyed
        a._ref = self
        return a

    def reshape(self, shape):
        """Returns a reshaped array. Only supported for contiguous arrays.

        Args:
            shape : An int or tuple of ints specifying the shape of the returned array.
        """
        if not self.is_contiguous:
            raise RuntimeError("Reshaping non-contiguous arrays is unsupported.")

        # convert shape to tuple
        if shape is None:
            raise RuntimeError("shape parameter is required.")
        if isinstance(shape, int):
            shape = (shape,)
        elif not isinstance(shape, tuple):
            shape = tuple(shape)

        if len(shape) > ARRAY_MAX_DIMS:
            raise RuntimeError(
                f"Arrays may only have {ARRAY_MAX_DIMS} dimensions maximum, trying to create array with {len(shape)} dims."
            )

        # check for -1 dimension and reformat
        if -1 in shape:
            idx = self.size
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
            new_shape[idx] = int(self.size / denom)
            shape = tuple(new_shape)

        size = 1
        for d in shape:
            size *= d

        if size != self.size:
            raise RuntimeError("Reshaped array must have the same total size as the original.")

        a = array(
            ptr=self.ptr,
            dtype=self.dtype,
            shape=shape,
            strides=None,
            device=self.device,
            pinned=self.pinned,
            copy=False,
            grad=None if self.grad is None else self.grad.reshape(shape),
        )

        # transfer read flag
        a._is_read = self._is_read

        # store back-ref to stop data being destroyed
        a._ref = self
        return a

    def view(self, dtype):
        """Returns a zero-copy view of this array's memory with a different data type.
        ``dtype`` must have the same byte size of the array's native ``dtype``.
        """
        if type_size_in_bytes(dtype) != type_size_in_bytes(self.dtype):
            raise RuntimeError("Cannot cast dtypes of unequal byte size")

        # return an alias of the array memory with different type information
        a = array(
            ptr=self.ptr,
            dtype=dtype,
            shape=self.shape,
            strides=self.strides,
            device=self.device,
            pinned=self.pinned,
            copy=False,
            grad=None if self.grad is None else self.grad.view(dtype),
        )

        # transfer read flag
        a._is_read = self._is_read

        a._ref = self
        return a

    def contiguous(self):
        """Returns a contiguous array with this array's data. No-op if array is already contiguous."""
        if self.is_contiguous:
            return self

        a = warp.empty_like(self)
        warp.copy(a, self)
        return a

    def transpose(self, axes=None):
        """Returns an zero-copy view of the array with axes transposed.

        Note: The transpose operation will return an array with a non-contiguous access pattern.

        Args:
            axes (optional): Specifies the how the axes are permuted. If not specified, the axes order will be reversed.
        """
        # noop if 1d array
        if self.ndim == 1:
            return self

        if axes is None:
            # reverse the order of the axes
            axes = range(self.ndim)[::-1]
        elif len(axes) != len(self.shape):
            raise RuntimeError("Length of parameter axes must be equal in length to array shape")

        shape = []
        strides = []
        for a in axes:
            if not isinstance(a, int):
                raise RuntimeError(f"axis index {a} is not of type int")
            if a >= len(self.shape):
                raise RuntimeError(f"axis index {a} must be smaller than the number of axes in array")
            shape.append(self.shape[a])
            strides.append(self.strides[a])

        a = array(
            ptr=self.ptr,
            dtype=self.dtype,
            shape=tuple(shape),
            strides=tuple(strides),
            device=self.device,
            pinned=self.pinned,
            copy=False,
            grad=None if self.grad is None else self.grad.transpose(axes=axes),
        )

        a.is_transposed = not self.is_transposed

        # transfer read flag
        a._is_read = self._is_read

        a._ref = self
        return a

    def ipc_handle(self) -> bytes:
        """Return an IPC handle of the array as a 64-byte ``bytes`` object

        :func:`from_ipc_handle` can be used with this handle in another process
        to obtain a :class:`array` that shares the same underlying memory
        allocation.

        IPC is currently only supported on Linux.
        Additionally, IPC is only supported for arrays allocated using
        the default memory allocator.

        :class:`Event` objects created with the ``interprocess=True`` argument
        may similarly be shared between processes to synchronize GPU work.

        Example:
            Temporarily using the default memory allocator to allocate an array
            and get its IPC handle::

                with wp.ScopedMempool("cuda:0", False):
                    test_array = wp.full(1024, value=42.0, dtype=wp.float32, device="cuda:0")
                    ipc_handle = test_array.ipc_handle()

        Raises:
            RuntimeError: The array is not associated with a CUDA device.
            RuntimeError: The CUDA device does not appear to support IPC.
            RuntimeError: The array was allocated using the :ref:`mempool memory allocator <mempool_allocators>`.
        """

        if self.device is None or not self.device.is_cuda:
            raise RuntimeError("IPC requires a CUDA device")
        elif self.device.is_ipc_supported is False:
            raise RuntimeError("IPC does not appear to be supported on this CUDA device")
        elif isinstance(self._allocator, warp.context.CudaMempoolAllocator):
            raise RuntimeError(
                "Currently, IPC is only supported for arrays using the default memory allocator.\n"
                "See https://nvidia.github.io/warp/modules/allocators.html for instructions on how to disable\n"
                f"the mempool allocator on device {self.device}."
            )

        # Allocate a buffer for the data (64-element char array)
        ipc_handle_buffer = (ctypes.c_char * 64)()

        warp.context.runtime.core.cuda_ipc_get_mem_handle(self.ptr, ipc_handle_buffer)

        return ipc_handle_buffer.raw


# aliases for arrays with small dimensions
def array1d(*args, **kwargs):
    kwargs["ndim"] = 1
    return array(*args, **kwargs)


# equivalent to calling array(..., ndim=2)
def array2d(*args, **kwargs):
    kwargs["ndim"] = 2
    return array(*args, **kwargs)


# equivalent to calling array(..., ndim=3)
def array3d(*args, **kwargs):
    kwargs["ndim"] = 3
    return array(*args, **kwargs)


# equivalent to calling array(..., ndim=4)
def array4d(*args, **kwargs):
    kwargs["ndim"] = 4
    return array(*args, **kwargs)


def from_ptr(ptr, length, dtype=None, shape=None, device=None):
    warp.utils.warn(
        """This version of wp.from_ptr() is deprecated. OmniGraph
    applications should use from_omni_graph_ptr() instead. To create an array
    from a C pointer, use the array constructor and pass the ptr argument as a
    uint64 value representing the start address in memory where the existing
    array resides. For example, if using ctypes, pass
    ptr=ctypes.cast(pointer, ctypes.POINTER(ctypes.c_size_t)).contents.value.
    Be sure to also specify the dtype and shape parameters.""",
        category=DeprecationWarning,
    )

    return array(
        dtype=dtype,
        length=length,
        capacity=length * type_size_in_bytes(dtype),
        ptr=0 if ptr == 0 else ctypes.cast(ptr, ctypes.POINTER(ctypes.c_size_t)).contents.value,
        shape=shape,
        device=device,
        requires_grad=False,
    )


def _close_cuda_ipc_handle(ptr, size):
    warp.context.runtime.core.cuda_ipc_close_mem_handle(ptr)


def from_ipc_handle(
    handle: bytes, dtype, shape: tuple[int, ...], strides: tuple[int, ...] | None = None, device=None
) -> array:
    """Create an array from an IPC handle.

    The ``dtype``, ``shape``, and optional ``strides`` arguments should
    match the values from the :class:`array` from which ``handle`` was created.

    Args:
        handle: The interprocess memory handle for an existing device memory allocation.
        dtype: One of the available `data types <#data-types>`_, such as :class:`warp.float32`, :class:`warp.mat33`, or a custom `struct <#structs>`_.
        shape: Dimensions of the array.
        strides: Number of bytes in each dimension between successive elements of the array.
        device (Devicelike): Device to associate with the array.

    Returns:
        An array created from the existing memory allocation described by the interprocess memory handle ``handle``.

        A copy of the underlying data is not made. Modifications to the array's data will be reflected in the
        original process from which ``handle`` was exported.

    Raises:
        RuntimeError: IPC is not supported on ``device``.
    """

    try:
        # Performance note: try first, ask questions later
        device = warp.context.runtime.get_device(device)
    except Exception:
        # Fallback to using the public API for retrieving the device,
        # which takes take of initializing Warp if needed.
        device = warp.context.get_device(device)

    if device.is_ipc_supported is False:
        raise RuntimeError(f"IPC is not supported on device {device}.")

    ptr = warp.context.runtime.core.cuda_ipc_open_mem_handle(device.context, handle)

    return array(ptr=ptr, dtype=dtype, shape=shape, strides=strides, device=device, deleter=_close_cuda_ipc_handle)


# A base class for non-contiguous arrays, providing the implementation of common methods like
# contiguous(), to(), numpy(), list(), assign(), zero_(), and fill_().
class noncontiguous_array_base(Array[T]):
    def __init__(self, array_type_id):
        self.type_id = array_type_id
        self.is_contiguous = False

    # return a contiguous copy
    def contiguous(self):
        a = warp.empty_like(self)
        warp.copy(a, self)
        return a

    # copy data from one device to another, nop if already on device
    def to(self, device):
        device = warp.get_device(device)
        if self.device == device:
            return self
        else:
            return warp.clone(self, device=device)

    # return a contiguous numpy copy
    def numpy(self):
        # use the CUDA default stream for synchronous behaviour with other streams
        with warp.ScopedStream(self.device.null_stream):
            return self.contiguous().numpy()

    # returns a flattened list of items in the array as a Python list
    def list(self):
        # use the CUDA default stream for synchronous behaviour with other streams
        with warp.ScopedStream(self.device.null_stream):
            return self.contiguous().list()

    # equivalent to wrapping src data in an array and copying to self
    def assign(self, src):
        if is_array(src):
            warp.copy(self, src)
        else:
            warp.copy(self, array(data=src, dtype=self.dtype, copy=False, device="cpu"))

    def zero_(self):
        self.fill_(0)

    def fill_(self, value):
        if self.size == 0:
            return

        # try to convert the given value to the array dtype
        try:
            if isinstance(self.dtype, warp.codegen.Struct):
                if isinstance(value, self.dtype.cls):
                    cvalue = value.__ctype__()
                elif value == 0:
                    # allow zero-initializing structs using default constructor
                    cvalue = self.dtype().__ctype__()
                else:
                    raise ValueError(
                        f"Invalid initializer value for struct {self.dtype.cls.__name__}, expected struct instance or 0"
                    )
            elif issubclass(self.dtype, ctypes.Array):
                # vector/matrix
                cvalue = self.dtype(value)
            else:
                # scalar
                if type(value) in warp.types.scalar_types:
                    value = value.value
                if self.dtype == float16:
                    cvalue = self.dtype._type_(float_to_half_bits(value))
                else:
                    cvalue = self.dtype._type_(value)
        except Exception as e:
            raise ValueError(f"Failed to convert the value to the array data type: {e}") from e

        cvalue_ptr = ctypes.pointer(cvalue)
        cvalue_size = ctypes.sizeof(cvalue)

        ctype = self.__ctype__()
        ctype_ptr = ctypes.pointer(ctype)

        if self.device.is_cuda:
            warp.context.runtime.core.array_fill_device(
                self.device.context, ctype_ptr, self.type_id, cvalue_ptr, cvalue_size
            )
        else:
            warp.context.runtime.core.array_fill_host(ctype_ptr, self.type_id, cvalue_ptr, cvalue_size)


# helper to check index array properties
def check_index_array(indices, expected_device):
    if not isinstance(indices, array):
        raise ValueError(f"Indices must be a Warp array, got {type(indices)}")
    if indices.ndim != 1:
        raise ValueError(f"Index array must be one-dimensional, got {indices.ndim}")
    if indices.dtype != int32:
        raise ValueError(f"Index array must use int32, got dtype {indices.dtype}")
    if indices.device != expected_device:
        raise ValueError(f"Index array device ({indices.device} does not match data array device ({expected_device}))")


class indexedarray(noncontiguous_array_base):
    # member attributes available during code-gen (e.g.: d = arr.shape[0])
    # (initialized when needed)
    _vars = None

    def __init__(
        self,
        data: array | None = None,
        indices: array | list[array] | None = None,
        dtype=None,
        ndim: int | None = None,
    ):
        super().__init__(ARRAY_TYPE_INDEXED)

        # canonicalize types
        if dtype is not None:
            if dtype == int:
                dtype = int32
            elif dtype == float:
                dtype = float32

        self.data = data
        self.indices = [None] * ARRAY_MAX_DIMS

        if data is not None:
            if not isinstance(data, array):
                raise ValueError("Indexed array data must be a Warp array")
            if dtype is not None and dtype != data.dtype:
                raise ValueError(f"Requested dtype ({dtype}) does not match dtype of data array ({data.dtype})")
            if ndim is not None and ndim != data.ndim:
                raise ValueError(
                    f"Requested dimensionality ({ndim}) does not match dimensionality of data array ({data.ndim})"
                )

            self.dtype = data.dtype
            self.ndim = data.ndim
            self.device = data.device
            self.pinned = data.pinned

            # determine shape from original data shape and index counts
            shape = list(data.shape)

            if indices is not None:
                if isinstance(indices, (list, tuple)):
                    if len(indices) > self.ndim:
                        raise ValueError(
                            f"Number of indices provided ({len(indices)}) exceeds number of dimensions ({self.ndim})"
                        )

                    for i in range(len(indices)):
                        if indices[i] is not None:
                            check_index_array(indices[i], data.device)
                            self.indices[i] = indices[i]
                            shape[i] = len(indices[i])

                elif isinstance(indices, array):
                    # only a single index array was provided
                    check_index_array(indices, data.device)
                    self.indices[0] = indices
                    shape[0] = len(indices)

                else:
                    raise ValueError("Indices must be a single Warp array or a list of Warp arrays")

            self.shape = tuple(shape)

        else:
            # allow empty indexedarrays in type annotations
            self.dtype = dtype
            self.ndim = ndim or 1
            self.device = None
            self.pinned = False
            self.shape = (0,) * self.ndim

        # update size (num elements)
        self.size = 1
        for d in self.shape:
            self.size *= d

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        if self.device is None:
            # type annotation
            return f"indexedarray{self.dtype}"
        else:
            return str(self.numpy())

    # construct a C-representation of the array for passing to kernels
    def __ctype__(self):
        return indexedarray_t(self.data, self.indices, self.shape)

    @property
    def vars(self):
        # member attributes available during code-gen (e.g.: d = arr.shape[0])
        # Note: we use a shared dict for all indexedarray instances
        if indexedarray._vars is None:
            indexedarray._vars = {"shape": warp.codegen.Var("shape", shape_t)}
        return indexedarray._vars


# aliases for indexedarrays with small dimensions
def indexedarray1d(*args, **kwargs):
    kwargs["ndim"] = 1
    return indexedarray(*args, **kwargs)


# equivalent to calling indexedarray(..., ndim=2)
def indexedarray2d(*args, **kwargs):
    kwargs["ndim"] = 2
    return indexedarray(*args, **kwargs)


# equivalent to calling indexedarray(..., ndim=3)
def indexedarray3d(*args, **kwargs):
    kwargs["ndim"] = 3
    return indexedarray(*args, **kwargs)


# equivalent to calling indexedarray(..., ndim=4)
def indexedarray4d(*args, **kwargs):
    kwargs["ndim"] = 4
    return indexedarray(*args, **kwargs)


from warp.fabric import fabricarray, indexedfabricarray  # noqa: E402

array_types = (array, indexedarray, fabricarray, indexedfabricarray)


def array_type_id(a):
    if isinstance(a, array):
        return ARRAY_TYPE_REGULAR
    elif isinstance(a, indexedarray):
        return ARRAY_TYPE_INDEXED
    elif isinstance(a, fabricarray):
        return ARRAY_TYPE_FABRIC
    elif isinstance(a, indexedfabricarray):
        return ARRAY_TYPE_FABRIC_INDEXED
    else:
        raise ValueError("Invalid array type")


# tile object
class Tile:
    alignment = 16

    def __init__(self, dtype, shape, op=None, storage="register", layout="rowmajor", strides=None, owner=True):
        self.dtype = type_to_warp(dtype)
        self.shape = shape
        self.op = op
        self.storage = storage
        self.layout = layout
        self.strides = strides

        # handle case where shape is concrete (rather than just Any)
        if isinstance(self.shape, (list, tuple)):
            if len(shape) == 0:
                raise RuntimeError("Empty shape specified, must have at least 1 dimension")

            # compute total size
            self.size = 1
            for s in self.shape:
                self.size *= s

            # if strides are not provided compute default strides
            if self.strides is None:
                self.strides = [1] * len(self.shape)

                if layout == "rowmajor":
                    for i in range(len(self.shape) - 2, -1, -1):
                        self.strides[i] = self.strides[i + 1] * self.shape[i + 1]
                else:
                    for i in range(1, len(shape)):
                        self.strides[i] = self.strides[i - 1] * self.shape[i - 1]

        self.owner = owner

    # generates C-type string
    def ctype(self):
        from warp.codegen import Var

        if self.storage == "register":
            return f"wp::tile_register_t<{Var.type_to_ctype(self.dtype)},wp::tile_layout_register_t<wp::tile_shape_t<{','.join(map(str, self.shape))}>>>"
        elif self.storage == "shared":
            return f"wp::tile_shared_t<{Var.type_to_ctype(self.dtype)},wp::tile_layout_strided_t<wp::tile_shape_t<{','.join(map(str, self.shape))}>, wp::tile_stride_t<{','.join(map(str, self.strides))}>>, {'true' if self.owner else 'false'}>"
        else:
            raise RuntimeError(f"Unrecognized tile storage type {self.storage}")

    # generates C-initializer string
    def cinit(self, requires_grad=False):
        from warp.codegen import Var

        if self.storage == "register":
            return self.ctype() + "(0.0)"
        elif self.storage == "shared":
            if self.owner:
                # allocate new shared memory tile
                return f"wp::tile_alloc_empty<{Var.type_to_ctype(self.dtype)},wp::tile_shape_t<{','.join(map(str, self.shape))}>,{'true' if requires_grad else 'false'}>()"
            else:
                # tile will be initialized by another call, e.g.: tile_transpose()
                return "nullptr"

    # return total tile size in bytes
    def size_in_bytes(self):
        num_bytes = self.align(type_size_in_bytes(self.dtype) * self.size)
        return num_bytes

    @staticmethod
    def round_up(bytes):
        return ((bytes + Tile.alignment - 1) // Tile.alignment) * Tile.alignment

    # align tile size to natural boundary, default 16-bytes
    def align(self, bytes):
        return Tile.round_up(bytes)


class TileZeros(Tile):
    def __init__(self, dtype, shape, storage="register"):
        Tile.__init__(self, dtype, shape, op="zeros", storage=storage)


class TileOnes(Tile):
    def __init__(self, dtype, shape, storage="register"):
        Tile.__init__(self, dtype, shape, op="ones", storage=storage)


class TileRange(Tile):
    def __init__(self, dtype, start, stop, step, storage="register"):
        self.start = start
        self.stop = stop
        self.step = step

        n = int((stop - start) / step)

        Tile.__init__(self, dtype, shape=(n,), op="arange", storage=storage)


class TileConstant(Tile):
    def __init__(self, dtype, shape):
        Tile.__init__(self, dtype, shape, op="constant", storage="register")


class TileLoad(Tile):
    def __init__(self, array, shape, storage="register"):
        Tile.__init__(self, array.dtype, shape, op="load", storage=storage)


class TileUnaryMap(Tile):
    def __init__(self, t, dtype=None, storage="register"):
        Tile.__init__(self, dtype, t.shape, op="unary_map", storage=storage)

        # if no output dtype specified then assume it's the same as the first arg
        if self.dtype is None:
            self.dtype = t.dtype

        self.t = t


class TileBinaryMap(Tile):
    def __init__(self, a, b, dtype=None, storage="register"):
        Tile.__init__(self, dtype, a.shape, op="binary_map", storage=storage)

        # if no output dtype specified then assume it's the same as the first arg
        if self.dtype is None:
            self.dtype = a.dtype

        self.a = a
        self.b = b


class TileShared(Tile):
    def __init__(self, t):
        Tile.__init__(self, t.dtype, t.shape, "shared", storage="shared")

        self.t = t


def is_tile(t):
    return isinstance(t, Tile)


bvh_constructor_values = {"sah": 0, "median": 1, "lbvh": 2}


class Bvh:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.id = None
        return instance

    def __init__(self, lowers: array, uppers: array, constructor: str | None = None):
        """Class representing a bounding volume hierarchy.

        Depending on which device the input bounds live, it can be either a CPU tree or a GPU tree.

        Attributes:
            id: Unique identifier for this BVH object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            lowers: Array of lower bounds of data type :class:`warp.vec3`.
            uppers: Array of upper bounds of data type :class:`warp.vec3`.
              ``lowers`` and ``uppers`` must live on the same device.
            constructor: The construction algorithm used to build the tree.
              Valid choices are ``"sah"``, ``"median"``, ``"lbvh"``, or ``None``.
              When ``None``, the default constructor will be used (see the note).

        Note:
            Explanation of BVH constructors:

            - ``"sah"``: A CPU-based top-down constructor where the AABBs are split based on Surface Area
              Heuristics (SAH). Construction takes slightly longer than others but has the best query
              performance.
            - ``"median"``: A CPU-based top-down constructor where the AABBs are split based on the median
              of centroids of primitives in an AABB. This constructor is faster than SAH but offers
              inferior query performance.
            - ``"lbvh"``: A GPU-based bottom-up constructor which maximizes parallelism. Construction is very
              fast, especially for large models. Query performance is slightly slower than ``"sah"``.
            - ``None``: The constructor will be automatically chosen based on the device where the tree
              lives. For a GPU tree, the ``"lbvh"`` constructor will be selected; for a CPU tree, the ``"sah"``
              constructor will be selected.

            All three constructors are supported for GPU trees. When a CPU-based constructor is selected
            for a GPU tree, bounds will be copied back to the CPU to run the CPU-based constructor. After
            construction, the CPU tree will be copied to the GPU.

            Only ``"sah"`` and ``"median"`` are supported for CPU trees. If ``"lbvh"`` is selected for a CPU tree, a
            warning message will be issued, and the constructor will automatically fall back to ``"sah"``.
        """

        if len(lowers) != len(uppers):
            raise RuntimeError("The same number of lower and upper bounds must be provided")

        if lowers.device != uppers.device:
            raise RuntimeError("Lower and upper bounds must live on the same device")

        if lowers.dtype != vec3 or not lowers.is_contiguous:
            raise RuntimeError("lowers should be a contiguous array of type wp.vec3")

        if uppers.dtype != vec3 or not uppers.is_contiguous:
            raise RuntimeError("uppers should be a contiguous array of type wp.vec3")

        self.device = lowers.device
        self.lowers = lowers
        self.uppers = uppers

        def get_data(array):
            if array:
                return ctypes.c_void_p(array.ptr)
            else:
                return ctypes.c_void_p(0)

        self.runtime = warp.context.runtime

        if constructor is None:
            if self.device.is_cpu:
                constructor = "sah"
            else:
                constructor = "lbvh"

        if constructor not in bvh_constructor_values:
            raise ValueError(f"Unrecognized BVH constructor type: {constructor}")

        if self.device.is_cpu:
            if constructor == "lbvh":
                warp.utils.warn(
                    "LBVH constructor is not available for a CPU tree. Falling back to SAH constructor.", stacklevel=2
                )
                constructor = "sah"

            self.id = self.runtime.core.bvh_create_host(
                get_data(lowers), get_data(uppers), int(len(lowers)), bvh_constructor_values[constructor]
            )
        else:
            self.id = self.runtime.core.bvh_create_device(
                self.device.context,
                get_data(lowers),
                get_data(uppers),
                int(len(lowers)),
                bvh_constructor_values[constructor],
            )

    def __del__(self):
        if not self.id:
            return

        if self.device.is_cpu:
            self.runtime.core.bvh_destroy_host(self.id)
        else:
            # use CUDA context guard to avoid side effects during garbage collection
            with self.device.context_guard:
                self.runtime.core.bvh_destroy_device(self.id)

    def refit(self):
        """Refit the BVH.

        This should be called after users modify the ``lowers`` or ``uppers`` arrays.
        """

        if self.device.is_cpu:
            self.runtime.core.bvh_refit_host(self.id)
        else:
            self.runtime.core.bvh_refit_device(self.id)
            self.runtime.verify_cuda_device(self.device)


class Mesh:
    from warp.codegen import Var

    vars = {
        "points": Var("points", array(dtype=vec3)),
        "velocities": Var("velocities", array(dtype=vec3)),
        "indices": Var("indices", array(dtype=int32)),
    }

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.id = None
        return instance

    def __init__(
        self,
        points: array,
        indices: array,
        velocities: array | None = None,
        support_winding_number: builtins.bool = False,
        bvh_constructor: str | None = None,
    ):
        """Class representing a triangle mesh.

        Attributes:
            id: Unique identifier for this mesh object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            points: Array of vertex positions of data type :class:`warp.vec3`.
            indices: Array of triangle indices of data type :class:`warp.int32`.
              Should be a 1D array with shape ``(num_tris * 3)``.
            velocities: Optional array of vertex velocities of data type :class:`warp.vec3`.
            support_winding_number: If ``True``, the mesh will build additional
              data structures to support ``wp.mesh_query_point_sign_winding_number()`` queries.
            bvh_constructor: The construction algorithm for the underlying BVH
              (see the docstring of :class:`Bvh` for explanation).
              Valid choices are ``"sah"``, ``"median"``, ``"lbvh"``, or ``None``.
        """

        if points.device != indices.device:
            raise RuntimeError("Mesh points and indices must live on the same device")

        if points.dtype != vec3 or not points.is_contiguous:
            raise RuntimeError("Mesh points should be a contiguous array of type wp.vec3")

        if velocities and (velocities.dtype != vec3 or not velocities.is_contiguous):
            raise RuntimeError("Mesh velocities should be a contiguous array of type wp.vec3")

        if indices.dtype != int32 or not indices.is_contiguous:
            raise RuntimeError("Mesh indices should be a contiguous array of type wp.int32")

        if indices.ndim > 1:
            raise RuntimeError("Mesh indices should be a flattened 1d array of indices")

        self.device = points.device
        self._points = points
        self._velocities = velocities
        self.indices = indices

        self.runtime = warp.context.runtime

        if bvh_constructor is None:
            if self.device.is_cpu:
                bvh_constructor = "sah"
            else:
                bvh_constructor = "lbvh"

        if bvh_constructor not in bvh_constructor_values:
            raise ValueError(f"Unrecognized BVH constructor type: {bvh_constructor}")

        if self.device.is_cpu:
            if bvh_constructor == "lbvh":
                warp.utils.warn(
                    "LBVH constructor is not available for a CPU tree. Falling back to SAH constructor.", stacklevel=2
                )
                bvh_constructor = "sah"

            self.id = self.runtime.core.mesh_create_host(
                points.__ctype__(),
                velocities.__ctype__() if velocities else array().__ctype__(),
                indices.__ctype__(),
                int(len(points)),
                int(indices.size / 3),
                int(support_winding_number),
                bvh_constructor_values[bvh_constructor],
            )
        else:
            self.id = self.runtime.core.mesh_create_device(
                self.device.context,
                points.__ctype__(),
                velocities.__ctype__() if velocities else array().__ctype__(),
                indices.__ctype__(),
                int(len(points)),
                int(indices.size / 3),
                int(support_winding_number),
                bvh_constructor_values[bvh_constructor],
            )

    def __del__(self):
        if not self.id:
            return

        if self.device.is_cpu:
            self.runtime.core.mesh_destroy_host(self.id)
        else:
            # use CUDA context guard to avoid side effects during garbage collection
            with self.device.context_guard:
                self.runtime.core.mesh_destroy_device(self.id)

    def refit(self):
        """Refit the BVH to points.

        This should be called after users modify the ``points`` data.
        """

        if self.device.is_cpu:
            self.runtime.core.mesh_refit_host(self.id)
        else:
            self.runtime.core.mesh_refit_device(self.id)
            self.runtime.verify_cuda_device(self.device)

    @property
    def points(self):
        """The array of mesh's vertex positions of type :class:`warp.vec3`.

        The `Mesh.points` property has a custom setter method. Users can modify the vertex positions in-place,
        but :meth:`refit` must be called manually after such modifications. Alternatively, assigning a new array
        to this property is also supported. The new array must have the same shape as the original, and once assigned,
        The :class:`Mesh` will automatically perform a refit operation based on the new vertex positions.
        """
        return self._points

    @points.setter
    def points(self, points_new):
        if points_new.device != self._points.device:
            raise RuntimeError(
                "The new points and the original points must live on the same device, the "
                f"new points are on {points_new.device} while the old points are on {self._points.device}."
            )

        if points_new.ndim != 1 or points_new.shape[0] != self._points.shape[0]:
            raise RuntimeError(
                "The new points and the original points must have the same shape, the "
                f"new points' shape is {points_new.shape}, while the old points' shape is {self._points.shape}."
            )

        self._points = points_new
        if self.device.is_cpu:
            self.runtime.core.mesh_set_points_host(self.id, points_new.__ctype__())
        else:
            self.runtime.core.mesh_set_points_device(self.id, points_new.__ctype__())
            self.runtime.verify_cuda_device(self.device)

    @property
    def velocities(self):
        """The array of mesh's velocities of type :class:`warp.vec3`.

        This is a property with a custom setter method. Users can modify the velocities in-place,
        or assign a new array to this property. No refitting is needed for changing velocities.
        """
        return self._velocities

    @velocities.setter
    def velocities(self, velocities_new):
        if velocities_new.device != self._velocities.device:
            raise RuntimeError(
                "The new points and the original points must live on the same device, the "
                f"new points are on {velocities_new.device} while the old points are on {self._velocities.device}."
            )

        if velocities_new.ndim != 1 or velocities_new.shape[0] != self._velocities.shape[0]:
            raise RuntimeError(
                "The new points and the original points must have the same shape, the "
                f"new points' shape is {velocities_new.shape}, while the old points' shape is {self._velocities.shape}."
            )

        self._velocities = velocities_new
        if self.device.is_cpu:
            self.runtime.core.mesh_set_velocities_host(self.id, velocities_new.__ctype__())
        else:
            self.runtime.core.mesh_set_velocities_device(self.id, velocities_new.__ctype__())
            self.runtime.verify_cuda_device(self.device)


class Volume:
    #: Enum value to specify nearest-neighbor interpolation during sampling
    CLOSEST = constant(0)
    #: Enum value to specify trilinear interpolation during sampling
    LINEAR = constant(1)

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.id = None
        return instance

    def __init__(self, data: array, copy: builtins.bool = True):
        """Class representing a sparse grid.

        Args:
            data: Array of bytes representing the volume in NanoVDB format.
            copy: Whether the incoming data will be copied or aliased.
        """

        # keep a runtime reference for orderly destruction
        self.runtime = warp.context.runtime

        if data is None:
            return
        self.device = data.device

        owner = False
        if self.device.is_cpu:
            self.id = self.runtime.core.volume_create_host(
                ctypes.cast(data.ptr, ctypes.c_void_p), data.size, copy, owner
            )
        else:
            self.id = self.runtime.core.volume_create_device(
                self.device.context, ctypes.cast(data.ptr, ctypes.c_void_p), data.size, copy, owner
            )

        if self.id == 0:
            raise RuntimeError("Failed to create volume from input array")

    def __del__(self):
        if not self.id:
            return

        if self.device.is_cpu:
            self.runtime.core.volume_destroy_host(self.id)
        else:
            # use CUDA context guard to avoid side effects during garbage collection
            with self.device.context_guard:
                self.runtime.core.volume_destroy_device(self.id)

    def array(self) -> array:
        """Return the raw memory buffer of the :class:`Volume` as an array."""

        buf = ctypes.c_void_p(0)
        size = ctypes.c_uint64(0)
        self.runtime.core.volume_get_buffer_info(self.id, ctypes.byref(buf), ctypes.byref(size))
        return array(ptr=buf.value, dtype=uint8, shape=size.value, device=self.device, owner=False)

    def get_tile_count(self) -> int:
        """Return the number of tiles (NanoVDB leaf nodes) of the volume."""

        voxel_count, tile_count = (
            ctypes.c_uint64(0),
            ctypes.c_uint32(0),
        )
        self.runtime.core.volume_get_tile_and_voxel_count(self.id, ctypes.byref(tile_count), ctypes.byref(voxel_count))
        return tile_count.value

    def get_tiles(self, out: array | None = None) -> array:
        """Return the integer coordinates of all allocated tiles for this volume.

        Args:
            out: If provided, use the `out` array to store the tile coordinates, otherwise
                a new array will be allocated. ``out`` must be a contiguous array
                of ``tile_count`` ``vec3i`` or ``tile_count x 3`` ``int32``
                on the same device as this volume.
        """

        if self.id == 0:
            raise RuntimeError("Invalid Volume")

        tile_count = self.get_tile_count()
        if out is None:
            out = warp.empty(dtype=int32, shape=(tile_count, 3), device=self.device)
        elif out.device != self.device or out.shape[0] < tile_count:
            raise RuntimeError(f"'out' array must an array with at least {tile_count} rows on device {self.device}")
        elif not _is_contiguous_vec_like_array(out, vec_length=3, scalar_types=(int32,)):
            raise RuntimeError(
                "'out' must be a contiguous 1D array with type vec3i or a 2D array of type int32 with shape (N, 3) "
            )

        if self.device.is_cpu:
            self.runtime.core.volume_get_tiles_host(self.id, out.ptr)
        else:
            self.runtime.core.volume_get_tiles_device(self.id, out.ptr)

        return out

    def get_voxel_count(self) -> int:
        """Return the total number of allocated voxels for this volume"""

        voxel_count, tile_count = (
            ctypes.c_uint64(0),
            ctypes.c_uint32(0),
        )
        self.runtime.core.volume_get_tile_and_voxel_count(self.id, ctypes.byref(tile_count), ctypes.byref(voxel_count))
        return voxel_count.value

    def get_voxels(self, out: array | None = None) -> array:
        """Return the integer coordinates of all allocated voxels for this volume.

        Args:
            out: If provided, use the `out` array to store the voxel coordinates, otherwise
                a new array will be allocated. `out` must be a contiguous array of ``voxel_count`` ``vec3i`` or ``voxel_count x 3`` ``int32``
                on the same device as this volume.
        """

        if self.id == 0:
            raise RuntimeError("Invalid Volume")

        voxel_count = self.get_voxel_count()
        if out is None:
            out = warp.empty(dtype=int32, shape=(voxel_count, 3), device=self.device)
        elif out.device != self.device or out.shape[0] < voxel_count:
            raise RuntimeError(f"'out' array must an array with at least {voxel_count} rows on device {self.device}")
        elif not _is_contiguous_vec_like_array(out, vec_length=3, scalar_types=(int32,)):
            raise RuntimeError(
                "'out' must be a contiguous 1D array with type vec3i or a 2D array of type int32 with shape (N, 3) "
            )

        if self.device.is_cpu:
            self.runtime.core.volume_get_voxels_host(self.id, out.ptr)
        else:
            self.runtime.core.volume_get_voxels_device(self.id, out.ptr)

        return out

    def get_voxel_size(self) -> tuple[float, float, float]:
        """Return the voxel size, i.e, world coordinates of voxel's diagonal vector"""

        if self.id == 0:
            raise RuntimeError("Invalid Volume")

        dx, dy, dz = ctypes.c_float(0), ctypes.c_float(0), ctypes.c_float(0)
        self.runtime.core.volume_get_voxel_size(self.id, ctypes.byref(dx), ctypes.byref(dy), ctypes.byref(dz))
        return (dx.value, dy.value, dz.value)

    class GridInfo(NamedTuple):
        """Grid metadata"""

        name: str
        """Grid name"""
        size_in_bytes: int
        """Size of this grid's data, in bytes"""

        grid_index: int
        """Index of this grid in the data buffer"""
        grid_count: int
        """Total number of grids in the data buffer"""
        type_str: str
        """String describing the type of the grid values"""

        translation: vec3f
        """Index-to-world translation"""
        transform_matrix: mat33f
        """Linear part of the index-to-world transform"""

    def get_grid_info(self) -> Volume.GridInfo:
        """Returns the metadata associated with this Volume"""

        grid_index = ctypes.c_uint32(0)
        grid_count = ctypes.c_uint32(0)
        grid_size = ctypes.c_uint64(0)
        translation_buffer = (ctypes.c_float * 3)()
        transform_buffer = (ctypes.c_float * 9)()
        type_str_buffer = (ctypes.c_char * 16)()

        name = self.runtime.core.volume_get_grid_info(
            self.id,
            ctypes.byref(grid_size),
            ctypes.byref(grid_index),
            ctypes.byref(grid_count),
            translation_buffer,
            transform_buffer,
            type_str_buffer,
        )

        if name is None:
            raise RuntimeError("Invalid volume")

        return Volume.GridInfo(
            name.decode("ascii"),
            grid_size.value,
            grid_index.value,
            grid_count.value,
            type_str_buffer.value.decode("ascii"),
            vec3f.from_buffer_copy(translation_buffer),
            mat33f.from_buffer_copy(transform_buffer),
        )

    _nvdb_type_to_dtype = {
        "float": float32,
        "double": float64,
        "int16": int16,
        "int32": int32,
        "int64": int64,
        "Vec3f": vec3f,
        "Vec3d": vec3d,
        "Half": float16,
        "uint32": uint32,
        "bool": bool,
        "Vec4f": vec4f,
        "Vec4d": vec4d,
        "Vec3u8": vec3ub,
        "Vec3u16": vec3us,
        "uint8": uint8,
    }

    @property
    def dtype(self) -> type:
        """Type of the Volume's values as a Warp type.

        If the grid does not contain values (e.g. index grids) or if the NanoVDB type is not
        representable as a Warp type, returns ``None``.
        """
        return Volume._nvdb_type_to_dtype.get(self.get_grid_info().type_str, None)

    _nvdb_index_types = ("Index", "OnIndex", "IndexMask", "OnIndexMask")

    @property
    def is_index(self) -> bool:
        """Whether this Volume contains an index grid, that is, a type of grid that does
        not explicitly store values but associates each voxel to linearized index.
        """

        return self.get_grid_info().type_str in Volume._nvdb_index_types

    def get_feature_array_count(self) -> int:
        """Return the number of supplemental data arrays stored alongside the grid"""

        return self.runtime.core.volume_get_blind_data_count(self.id)

    class FeatureArrayInfo(NamedTuple):
        """Metadata for a supplemental data array"""

        name: str
        """Name of the data array"""
        ptr: int
        """Memory address of the start of the array"""

        value_size: int
        """Size in bytes of the array values"""
        value_count: int
        """Number of values in the array"""
        type_str: str
        """String describing the type of the array values"""

    def get_feature_array_info(self, feature_index: int) -> Volume.FeatureArrayInfo:
        """Return the metadata associated to the feature array at ``feature_index``."""

        buf = ctypes.c_void_p(0)
        value_count = ctypes.c_uint64(0)
        value_size = ctypes.c_uint32(0)
        type_str_buffer = (ctypes.c_char * 16)()

        name = self.runtime.core.volume_get_blind_data_info(
            self.id,
            feature_index,
            ctypes.byref(buf),
            ctypes.byref(value_count),
            ctypes.byref(value_size),
            type_str_buffer,
        )

        if buf.value is None:
            raise RuntimeError("Invalid feature array")

        return Volume.FeatureArrayInfo(
            name.decode("ascii"),
            buf.value,
            value_size.value,
            value_count.value,
            type_str_buffer.value.decode("ascii"),
        )

    def feature_array(self, feature_index: int, dtype=None) -> array:
        """Return one the grid's feature data arrays as a Warp array.

        Args:
            feature_index: Index of the supplemental data array in the grid
            dtype: Data type for the returned Warp array.
              If not provided, will be deduced from the array metadata.
        """

        info = self.get_feature_array_info(feature_index)

        if dtype is None:
            try:
                dtype = Volume._nvdb_type_to_dtype[info.type_str]
            except KeyError:
                # Unknown type, default to byte array
                dtype = uint8

        value_count = info.value_count
        value_size = info.value_size

        if type_size_in_bytes(dtype) == 1:
            # allow requesting a byte array from any type
            value_count *= value_size
            value_size = 1
        elif value_size == 1 and (value_count % type_size_in_bytes(dtype)) == 0:
            # allow converting a byte array to any type
            value_size = type_size_in_bytes(dtype)
            value_count = value_count // value_size

        if type_size_in_bytes(dtype) != value_size:
            raise RuntimeError(f"Cannot cast feature data of size {value_size} to array dtype {type_repr(dtype)}")

        return array(ptr=info.ptr, dtype=dtype, shape=value_count, device=self.device, owner=False)

    @classmethod
    def load_from_nvdb(cls, file_or_buffer, device=None) -> Volume:
        """Create a :class:`Volume` object from a serialized NanoVDB file or in-memory buffer.

        Returns:

            A ``warp.Volume`` object.
        """
        try:
            data = file_or_buffer.read()
        except AttributeError:
            data = file_or_buffer

        magic, version, grid_count, codec = struct.unpack("<QIHH", data[0:16])
        if magic not in (0x304244566F6E614E, 0x324244566F6E614E):  # NanoVDB0 or NanoVDB2 in hex, little-endian
            raise RuntimeError("NanoVDB signature not found")
        if version >> 21 != 32:  # checking major version
            raise RuntimeError("Unsupported NanoVDB version")

        # Skip over segment metadata, store total payload size
        grid_data_offset = 16  # sizeof(FileHeader)
        tot_file_size = 0
        for _ in range(grid_count):
            grid_file_size = struct.unpack("<Q", data[grid_data_offset + 8 : grid_data_offset + 16])[0]
            tot_file_size += grid_file_size

            grid_name_size = struct.unpack("<I", data[grid_data_offset + 136 : grid_data_offset + 140])[0]
            grid_data_offset += 176 + grid_name_size  # sizeof(FileMetadata) + grid name

        file_end = grid_data_offset + tot_file_size

        if codec == 0:  # no compression
            grid_data = data[grid_data_offset:file_end]
        elif codec == 1:  # zip compression
            grid_data = bytearray()
            while grid_data_offset < file_end:
                chunk_size = struct.unpack("<Q", data[grid_data_offset : grid_data_offset + 8])[0]
                grid_data_offset += 8
                grid_data += zlib.decompress(data[grid_data_offset : grid_data_offset + chunk_size])
                grid_data_offset += chunk_size
        elif codec == 2:  # blosc compression
            try:
                import blosc
            except ImportError as err:
                raise RuntimeError(
                    f"NanoVDB buffer is compressed using blosc, but Python module could not be imported: {err}"
                ) from err

            grid_data = bytearray()
            while grid_data_offset < file_end:
                chunk_size = struct.unpack("<Q", data[grid_data_offset : grid_data_offset + 8])[0]
                grid_data_offset += 8
                grid_data += blosc.decompress(data[grid_data_offset : grid_data_offset + chunk_size])
                grid_data_offset += chunk_size
        else:
            raise RuntimeError(f"Unsupported codec code: {codec}")

        magic = struct.unpack("<Q", grid_data[0:8])[0]
        if magic not in (0x304244566F6E614E, 0x314244566F6E614E):  # NanoVDB0 or NanoVDB1 in hex, little-endian
            raise RuntimeError("NanoVDB signature not found on grid!")

        data_array = array(np.frombuffer(grid_data, dtype=np.byte), device=device)
        return cls(data_array)

    def save_to_nvdb(self, path, codec: Literal["none", "zip", "blosc"] = "none"):
        """Serialize the Volume into a NanoVDB (.nvdb) file.

        Args:
            path: File path to save.
            codec: Compression codec used
                "none" - no compression
                "zip" - ZIP compression
                "blosc" - BLOSC compression, requires the blosc module to be installed
        """

        codec_dict = {"none": 0, "zip": 1, "blosc": 2}

        class FileHeader(ctypes.Structure):
            _fields_ = [
                ("magic", ctypes.c_uint64),
                ("version", ctypes.c_uint32),
                ("gridCount", ctypes.c_uint16),
                ("codec", ctypes.c_uint16),
            ]

        class FileMetaData(ctypes.Structure):
            _fields_ = [
                ("gridSize", ctypes.c_uint64),
                ("fileSize", ctypes.c_uint64),
                ("nameKey", ctypes.c_uint64),
                ("voxelCount", ctypes.c_uint64),
                ("gridType", ctypes.c_uint32),
                ("gridClass", ctypes.c_uint32),
                ("worldBBox", ctypes.c_double * 6),
                ("indexBBox", ctypes.c_uint32 * 6),
                ("voxelSize", ctypes.c_double * 3),
                ("nameSize", ctypes.c_uint32),
                ("nodeCount", ctypes.c_uint32 * 4),
                ("tileCount", ctypes.c_uint32 * 3),
                ("codec", ctypes.c_uint16),
                ("padding", ctypes.c_uint16),
                ("version", ctypes.c_uint32),
            ]

        class GridData(ctypes.Structure):
            _fields_ = [
                ("magic", ctypes.c_uint64),
                ("checksum", ctypes.c_uint64),
                ("version", ctypes.c_uint32),
                ("flags", ctypes.c_uint32),
                ("gridIndex", ctypes.c_uint32),
                ("gridCount", ctypes.c_uint32),
                ("gridSize", ctypes.c_uint64),
                ("gridName", ctypes.c_char * 256),
                ("map", ctypes.c_byte * 264),
                ("worldBBox", ctypes.c_double * 6),
                ("voxelSize", ctypes.c_double * 3),
                ("gridClass", ctypes.c_uint32),
                ("gridType", ctypes.c_uint32),
                ("blindMetadataOffset", ctypes.c_int64),
                ("blindMetadataCount", ctypes.c_uint32),
                ("data0", ctypes.c_uint32),
                ("data1", ctypes.c_uint64),
                ("data2", ctypes.c_uint64),
            ]

        NVDB_MAGIC = 0x304244566F6E614E
        NVDB_VERSION = 32 << 21 | 3 << 10 | 3

        try:
            codec_int = codec_dict[codec]
        except KeyError as err:
            raise RuntimeError(f"Unsupported codec requested: {codec}") from err

        if codec_int == 2:
            try:
                import blosc
            except ImportError as err:
                raise RuntimeError(
                    f"blosc compression was requested, but Python module could not be imported: {err}"
                ) from err

        data = self.array().numpy()
        grid_data = GridData.from_buffer(data)

        if grid_data.gridIndex > 0:
            raise RuntimeError(
                "Saving of aliased Volumes is not supported. Use `save_to_nvdb` on the original volume, before any `load_next_grid` calls."
            )

        file_header = FileHeader(NVDB_MAGIC, NVDB_VERSION, grid_data.gridCount, codec_int)

        grid_data_offset = 0
        all_file_meta_data = []
        for i in range(file_header.gridCount):
            if i > 0:
                grid_data = GridData.from_buffer(data[grid_data_offset : grid_data_offset + 672])
            current_grid_data = data[grid_data_offset : grid_data_offset + grid_data.gridSize]
            if codec_int == 1:  # zip compression
                compressed_data = zlib.compress(current_grid_data)
                compressed_size = len(compressed_data)
            elif codec_int == 2:  # blosc compression
                compressed_data = blosc.compress(current_grid_data)
                compressed_size = len(compressed_data)
            else:  # no compression
                compressed_data = current_grid_data
                compressed_size = grid_data.gridSize

            file_meta_data = FileMetaData()
            file_meta_data.gridSize = grid_data.gridSize
            file_meta_data.fileSize = compressed_size
            file_meta_data.gridType = grid_data.gridType
            file_meta_data.gridClass = grid_data.gridClass
            file_meta_data.worldBBox = grid_data.worldBBox
            file_meta_data.voxelSize = grid_data.voxelSize
            file_meta_data.nameSize = len(grid_data.gridName) + 1  # including the closing 0x0
            file_meta_data.codec = codec_int
            file_meta_data.version = NVDB_VERSION

            grid_data_offset += file_meta_data.gridSize

            all_file_meta_data.append((file_meta_data, grid_data.gridName, compressed_data))

        with open(path, "wb") as nvdb:
            nvdb.write(file_header)
            for file_meta_data, grid_name, _ in all_file_meta_data:
                nvdb.write(file_meta_data)
                nvdb.write(grid_name + b"\x00")

            for file_meta_data, _, compressed_data in all_file_meta_data:
                if codec_int > 0:
                    chunk_size = struct.pack("<Q", file_meta_data.fileSize)
                    nvdb.write(chunk_size)
                nvdb.write(compressed_data)

        return path

    @classmethod
    def load_from_address(cls, grid_ptr: int, buffer_size: int = 0, device=None) -> Volume:
        """
        Creates a new :class:`Volume` aliasing an in-memory grid buffer.

        In contrast to :meth:`load_from_nvdb` which should be used to load serialized NanoVDB grids,
        here the buffer must be uncompressed and must not contain file header information.
        If the passed address does not contain a NanoVDB grid, the behavior of this function is undefined.

        Args:
            grid_ptr: Integer address of the start of the grid buffer
            buffer_size: Size of the buffer, in bytes. If not provided, the size will be assumed to be that of the single grid starting at `grid_ptr`.
            device: Device of the buffer, and of the returned Volume. If not provided, the current Warp device is assumed.

        Returns the newly created Volume.
        """

        if not grid_ptr:
            raise (RuntimeError, "Invalid grid buffer pointer")

        # Check that a Volume has not already been created for this address
        # (to allow this we would need to ref-count the volume descriptor)
        existing_buf = ctypes.c_void_p(0)
        existing_size = ctypes.c_uint64(0)
        warp.context.runtime.core.volume_get_buffer_info(
            grid_ptr, ctypes.byref(existing_buf), ctypes.byref(existing_size)
        )

        if existing_buf.value is not None:
            raise RuntimeError(
                "A warp Volume has already been created for this grid, aliasing it more than once is not possible."
            )

        data_array = array(ptr=grid_ptr, dtype=uint8, shape=buffer_size, owner=False, device=device)

        return cls(data_array, copy=False)

    def load_next_grid(self) -> Volume:
        """
        Tries to create a new warp Volume for the next grid that is linked to by this Volume.

        The existence of a next grid is deduced from the `grid_index` and `grid_count` metadata
        as well as the size of this Volume's in-memory buffer.

        Returns the newly created Volume, or None if there is no next grid.
        """

        grid = self.get_grid_info()

        array = self.array()

        if grid.grid_index + 1 >= grid.grid_count or array.capacity <= grid.size_in_bytes:
            return None

        next_volume = Volume.load_from_address(
            array.ptr + grid.size_in_bytes, buffer_size=array.capacity - grid.size_in_bytes, device=self.device
        )
        # makes the new Volume keep a reference to the current grid, as we're aliasing its buffer
        next_volume._previous_grid = self

        return next_volume

    @classmethod
    def load_from_numpy(
        cls, ndarray: np.array, min_world=(0.0, 0.0, 0.0), voxel_size=1.0, bg_value=0.0, device=None
    ) -> Volume:
        """Creates a Volume object from a dense 3D NumPy array.

        This function is only supported for CUDA devices.

        Args:
            min_world: The 3D coordinate of the lower corner of the volume.
            voxel_size: The size of each voxel in spatial coordinates.
            bg_value: Background value
            device: The CUDA device to create the volume on, e.g.: "cuda" or "cuda:0".

        Returns:

            A ``warp.Volume`` object.
        """

        import math

        target_shape = (
            math.ceil(ndarray.shape[0] / 8) * 8,
            math.ceil(ndarray.shape[1] / 8) * 8,
            math.ceil(ndarray.shape[2] / 8) * 8,
        )
        if hasattr(bg_value, "__len__"):
            # vec3, assuming the numpy array is 4D
            padded_array = np.full(
                shape=(target_shape[0], target_shape[1], target_shape[2], 3), fill_value=bg_value, dtype=np.single
            )
            padded_array[0 : ndarray.shape[0], 0 : ndarray.shape[1], 0 : ndarray.shape[2], :] = ndarray
        else:
            padded_amount = (
                math.ceil(ndarray.shape[0] / 8) * 8 - ndarray.shape[0],
                math.ceil(ndarray.shape[1] / 8) * 8 - ndarray.shape[1],
                math.ceil(ndarray.shape[2] / 8) * 8 - ndarray.shape[2],
            )
            padded_array = np.pad(
                ndarray,
                ((0, padded_amount[0]), (0, padded_amount[1]), (0, padded_amount[2])),
                mode="constant",
                constant_values=bg_value,
            )

        shape = padded_array.shape
        volume = warp.Volume.allocate(
            min_world,
            [
                min_world[0] + (shape[0] - 1) * voxel_size,
                min_world[1] + (shape[1] - 1) * voxel_size,
                min_world[2] + (shape[2] - 1) * voxel_size,
            ],
            voxel_size,
            bg_value=bg_value,
            points_in_world_space=True,
            translation=min_world,
            device=device,
        )

        # Populate volume
        if hasattr(bg_value, "__len__"):
            warp.launch(
                warp.utils.copy_dense_volume_to_nano_vdb_v,
                dim=(shape[0], shape[1], shape[2]),
                inputs=[volume.id, warp.array(padded_array, dtype=warp.vec3, device=device)],
                device=device,
            )
        elif isinstance(bg_value, int):
            warp.launch(
                warp.utils.copy_dense_volume_to_nano_vdb_i,
                dim=shape,
                inputs=[volume.id, warp.array(padded_array, dtype=warp.int32, device=device)],
                device=device,
            )
        else:
            warp.launch(
                warp.utils.copy_dense_volume_to_nano_vdb_f,
                dim=shape,
                inputs=[volume.id, warp.array(padded_array, dtype=warp.float32, device=device)],
                device=device,
            )

        return volume

    @classmethod
    def allocate(
        cls,
        min: list[int],
        max: list[int],
        voxel_size: float,
        bg_value=0.0,
        translation=(0.0, 0.0, 0.0),
        points_in_world_space=False,
        device=None,
    ) -> Volume:
        """Allocate a new Volume based on the bounding box defined by min and max.

        This function is only supported for CUDA devices.

        Allocate a volume that is large enough to contain voxels [min[0], min[1], min[2]] - [max[0], max[1], max[2]], inclusive.
        If points_in_world_space is true, then min and max are first converted to index space with the given voxel size and
        translation, and the volume is allocated with those.

        The smallest unit of allocation is a dense tile of 8x8x8 voxels, the requested bounding box is rounded up to tiles, and
        the resulting tiles will be available in the new volume.

        Args:
            min (array-like): Lower 3D coordinates of the bounding box in index space or world space, inclusive.
            max (array-like): Upper 3D coordinates of the bounding box in index space or world space, inclusive.
            voxel_size (float): Voxel size of the new volume.
            bg_value (float or array-like): Value of unallocated voxels of the volume, also defines the volume's type, a :class:`warp.vec3` volume is created if this is `array-like`, otherwise a float volume is created
            translation (array-like): translation between the index and world spaces.
            device (Devicelike): The CUDA device to create the volume on, e.g.: "cuda" or "cuda:0".

        """
        if points_in_world_space:
            min = np.around((np.array(min, dtype=np.float32) - translation) / voxel_size)
            max = np.around((np.array(max, dtype=np.float32) - translation) / voxel_size)

        tile_min = np.array(min, dtype=np.int32) // 8
        tile_max = np.array(max, dtype=np.int32) // 8
        tiles = np.array(
            [
                [i, j, k]
                for i in range(tile_min[0], tile_max[0] + 1)
                for j in range(tile_min[1], tile_max[1] + 1)
                for k in range(tile_min[2], tile_max[2] + 1)
            ],
            dtype=np.int32,
        )
        tile_points = array(tiles * 8, device=device)

        return cls.allocate_by_tiles(tile_points, voxel_size, bg_value, translation, device)

    @staticmethod
    def _fill_transform_buffers(
        voxel_size: float | list[float],
        translation,
        transform,
    ):
        if transform is None:
            if voxel_size is None:
                raise ValueError("Either 'voxel_size' or 'transform' must be provided")

            if isinstance(voxel_size, float):
                voxel_size = (voxel_size, voxel_size, voxel_size)
            transform = mat33f(voxel_size[0], 0.0, 0.0, 0.0, voxel_size[1], 0.0, 0.0, 0.0, voxel_size[2])
        else:
            if voxel_size is not None:
                raise ValueError("Only one of 'voxel_size' or 'transform' must be provided")

            if not isinstance(transform, mat33f):
                transform = mat33f(transform)

        transform_buf = (ctypes.c_float * 9).from_buffer_copy(transform)
        translation_buf = (ctypes.c_float * 3)(translation[0], translation[1], translation[2])
        return transform_buf, translation_buf

    # nanovdb types for which we instantiate the grid builder
    # Should be in sync with WP_VOLUME_BUILDER_INSTANTIATE_TYPES in volume_builder.h
    _supported_allocation_types = [
        "int32",
        "float",
        "Vec3f",
        "Vec4f",
    ]

    @classmethod
    def allocate_by_tiles(
        cls,
        tile_points: array,
        voxel_size: float | list[float] | None = None,
        bg_value=0.0,
        translation=(0.0, 0.0, 0.0),
        device=None,
        transform=None,
    ) -> Volume:
        """Allocate a new Volume with active tiles for each point tile_points.

        This function is only supported for CUDA devices.

        The smallest unit of allocation is a dense tile of 8x8x8 voxels.
        This is the primary method for allocating sparse volumes. It uses an array of points indicating the tiles that must be allocated.

        Example use cases:
            * `tile_points` can mark tiles directly in index space as in the case this method is called by `allocate`.
            * `tile_points` can be a list of points used in a simulation that needs to transfer data to a volume.

        Args:
            tile_points (:class:`warp.array`): Array of positions that define the tiles to be allocated.
                The array may use an integer scalar type (2D N-by-3 array of :class:`warp.int32` or 1D array of `warp.vec3i` values), indicating index space positions,
                or a floating point scalar type (2D N-by-3 array of :class:`warp.float32` or 1D array of `warp.vec3f` values), indicating world space positions.
                Repeated points per tile are allowed and will be efficiently deduplicated.
            voxel_size (float or array-like): Voxel size(s) of the new volume. Ignored if `transform` is given.
            bg_value (array-like, scalar or None): Value of unallocated voxels of the volume, also defines the volume's type. An index volume will be created if `bg_value` is ``None``.
              Other supported grid types are `int`, `float`, `vec3f`, and `vec4f`.
            translation (array-like): Translation between the index and world spaces.
            transform (array-like): Linear transform between the index and world spaces. If ``None``, deduced from `voxel_size`.
            device (Devicelike): The CUDA device to create the volume on, e.g.: "cuda" or "cuda:0".

        """
        device = warp.get_device(device)

        if not device.is_cuda:
            raise RuntimeError("Only CUDA devices are supported for allocate_by_tiles")
        if not _is_contiguous_vec_like_array(tile_points, vec_length=3, scalar_types=(float32, int32)):
            raise RuntimeError(
                "tile_points must be contiguous and either a 1D warp array of vec3f or vec3i or a 2D n-by-3 array of int32 or float32."
            )
        if not tile_points.device.is_cuda:
            tile_points = tile_points.to(device)

        volume = cls(data=None)
        volume.device = device
        in_world_space = type_scalar_type(tile_points.dtype) == float32

        transform_buf, translation_buf = Volume._fill_transform_buffers(voxel_size, translation, transform)

        if bg_value is None:
            volume.id = volume.runtime.core.volume_index_from_tiles_device(
                volume.device.context,
                ctypes.c_void_p(tile_points.ptr),
                tile_points.shape[0],
                transform_buf,
                translation_buf,
                in_world_space,
            )
        else:
            # normalize background value type
            grid_type = type_to_warp(type(bg_value))
            if not (is_value(bg_value) or type_is_vector(grid_type)) and (
                hasattr(bg_value, "__len__") and is_value(bg_value[0])
            ):
                # non-warp vectors are considered float, for backward compatibility
                grid_type = vector(len(bg_value), dtype=float)

            # look for corresponding nvdb type
            try:
                nvdb_type = next(
                    typ
                    for typ in Volume._supported_allocation_types
                    if types_equal(grid_type, Volume._nvdb_type_to_dtype[typ])
                )
            except StopIteration as err:
                raise TypeError(
                    f"Unsupported bg_value type for volume allocation {type_repr(grid_type)}. Supported volume types are {', '.join(Volume._supported_allocation_types)}."
                ) from err

            # cast to ctype
            # wrap scalar values in length-1 vectors to handle specific ctype conversion
            if not type_is_vector(grid_type):
                grid_type = vector(length=1, dtype=grid_type)

            cvalue = grid_type(bg_value)
            cvalue_ptr = ctypes.pointer(cvalue)
            cvalue_size = ctypes.sizeof(cvalue)
            cvalue_type = nvdb_type.encode("ascii")

            volume.id = volume.runtime.core.volume_from_tiles_device(
                volume.device.context,
                ctypes.c_void_p(tile_points.ptr),
                tile_points.shape[0],
                transform_buf,
                translation_buf,
                in_world_space,
                cvalue_ptr,
                cvalue_size,
                cvalue_type,
            )

        if volume.id == 0:
            raise RuntimeError("Failed to create volume")

        return volume

    @classmethod
    def allocate_by_voxels(
        cls,
        voxel_points: array,
        voxel_size: float | list[float] | None = None,
        translation=(0.0, 0.0, 0.0),
        device=None,
        transform=None,
    ) -> Volume:
        """Allocate a new Volume with active voxel for each point voxel_points.

        This function creates an *index* Volume, a special kind of volume that does not any store any
        explicit payload but encodes a linearized index for each active voxel, allowing to lookup and
        sample data from arbitrary external arrays.

        This function is only supported for CUDA devices.

        Args:
            voxel_points (:class:`warp.array`): Array of positions that define the voxels to be allocated.
                The array may use an integer scalar type (2D N-by-3 array of :class:`warp.int32` or 1D array of `warp.vec3i` values), indicating index space positions,
                or a floating point scalar type (2D N-by-3 array of :class:`warp.float32` or 1D array of `warp.vec3f` values), indicating world space positions.
                Repeated points per tile are allowed and will be efficiently deduplicated.
            voxel_size (float or array-like): Voxel size(s) of the new volume. Ignored if `transform` is given.
            translation (array-like): Translation between the index and world spaces.
            transform (array-like): Linear transform between the index and world spaces. If ``None``, deduced from `voxel_size`.
            device (Devicelike): The CUDA device to create the volume on, e.g.: "cuda" or "cuda:0".

        """
        device = warp.get_device(device)

        if not device.is_cuda:
            raise RuntimeError("Only CUDA devices are supported for allocate_by_tiles")
        if not _is_contiguous_vec_like_array(voxel_points, vec_length=3, scalar_types=(float32, int32)):
            raise RuntimeError(
                "voxel_points must be contiguous and either a 1D warp array of vec3f or vec3i or a 2D n-by-3 array of int32 or float32."
            )
        if not voxel_points.device.is_cuda:
            voxel_points = voxel_points.to(device)

        volume = cls(data=None)
        volume.device = device
        in_world_space = type_scalar_type(voxel_points.dtype) == float32

        transform_buf, translation_buf = Volume._fill_transform_buffers(voxel_size, translation, transform)

        volume.id = volume.runtime.core.volume_from_active_voxels_device(
            volume.device.context,
            ctypes.c_void_p(voxel_points.ptr),
            voxel_points.shape[0],
            transform_buf,
            translation_buf,
            in_world_space,
        )

        if volume.id == 0:
            raise RuntimeError("Failed to create volume")

        return volume


def _is_contiguous_vec_like_array(array, vec_length: int, scalar_types: tuple[type]) -> builtins.bool:
    if not (is_array(array) and array.is_contiguous):
        return False
    if type_scalar_type(array.dtype) not in scalar_types:
        return False
    return (array.ndim == 1 and type_length(array.dtype) == vec_length) or (
        array.ndim == 2 and array.shape[1] == vec_length and type_length(array.dtype) == 1
    )


# definition just for kernel type (cannot be a parameter), see mesh.h
# NOTE: its layout must match the corresponding struct defined in C.
# NOTE: it needs to be defined after `indexedarray` to workaround a circular import issue.
class mesh_query_point_t:
    """Output for the mesh query point functions.

    Attributes:
        result (bool): Whether a point is found within the given constraints.
        sign (float32): A value < 0 if query point is inside the mesh, >=0 otherwise.
                        Note that mesh must be watertight for this to be robust
        face (int32): Index of the closest face.
        u (float32): Barycentric u coordinate of the closest point.
        v (float32): Barycentric v coordinate of the closest point.

    See Also:
        :func:`mesh_query_point`, :func:`mesh_query_point_no_sign`,
        :func:`mesh_query_furthest_point_no_sign`,
        :func:`mesh_query_point_sign_normal`,
        and :func:`mesh_query_point_sign_winding_number`.
    """

    from warp.codegen import Var

    vars = {
        "result": Var("result", bool),
        "sign": Var("sign", float32),
        "face": Var("face", int32),
        "u": Var("u", float32),
        "v": Var("v", float32),
    }


MeshQueryPoint = mesh_query_point_t


# definition just for kernel type (cannot be a parameter), see mesh.h
# NOTE: its layout must match the corresponding struct defined in C.
class mesh_query_ray_t:
    """Output for the mesh query ray functions.

    Attributes:
        result (bool): Whether a hit is found within the given constraints.
        sign (float32): A value > 0 if the ray hit in front of the face, returns < 0 otherwise.
        face (int32): Index of the closest face.
        t (float32): Distance of the closest hit along the ray.
        u (float32): Barycentric u coordinate of the closest hit.
        v (float32): Barycentric v coordinate of the closest hit.
        normal (vec3f): Face normal.

    See Also:
        :func:`mesh_query_ray`.
    """

    from warp.codegen import Var

    vars = {
        "result": Var("result", bool),
        "sign": Var("sign", float32),
        "face": Var("face", int32),
        "t": Var("t", float32),
        "u": Var("u", float32),
        "v": Var("v", float32),
        "normal": Var("normal", vec3),
    }


MeshQueryRay = mesh_query_ray_t


def matmul(
    a: array2d,
    b: array2d,
    c: array2d,
    d: array2d,
    alpha: float = 1.0,
    beta: float = 0.0,
    allow_tf32x3_arith: builtins.bool = False,
):
    """Computes a generic matrix-matrix multiplication (GEMM) of the form: `d = alpha * (a @ b) + beta * c`.

    .. versionremoved:: 1.7

    .. deprecated:: 1.6
        Use :doc:`tile primitives </modules/tiles>` instead.

    Args:
        a (array2d): two-dimensional array containing matrix A
        b (array2d): two-dimensional array containing matrix B
        c (array2d): two-dimensional array containing matrix C
        d (array2d): two-dimensional array to which output D is written
        alpha (float): parameter alpha of GEMM
        beta (float): parameter beta of GEMM
        allow_tf32x3_arith (bool): whether to use CUTLASS's 3xTF32 GEMMs, which enable accuracy similar to FP32
                                   while using Tensor Cores
    """

    raise RuntimeError("This function has been removed. Use tile primitives instead.")


def adj_matmul(
    a: array2d,
    b: array2d,
    c: array2d,
    adj_a: array2d,
    adj_b: array2d,
    adj_c: array2d,
    adj_d: array2d,
    alpha: float = 1.0,
    beta: float = 0.0,
    allow_tf32x3_arith: builtins.bool = False,
):
    """Computes the adjoint of a generic matrix-matrix multiplication (GEMM) of the form: `d = alpha * (a @ b) + beta * c`.
        note: the adjoint of parameter alpha is not included but can be computed as `adj_alpha = np.sum(np.concatenate(np.multiply(a @ b, adj_d)))`.
        note: the adjoint of parameter beta is not included but can be computed as `adj_beta = np.sum(np.concatenate(np.multiply(c, adj_d)))`.

    Args:
        a (array2d): two-dimensional array containing matrix A
        b (array2d): two-dimensional array containing matrix B
        c (array2d): two-dimensional array containing matrix C
        adj_a (array2d): two-dimensional array to which the adjoint of matrix A is written
        adj_b (array2d): two-dimensional array to which the adjoint of matrix B is written
        adj_c (array2d): two-dimensional array to which the adjoint of matrix C is written
        adj_d (array2d): two-dimensional array containing the adjoint of matrix D
        alpha (float): parameter alpha of GEMM
        beta (float): parameter beta of GEMM
        allow_tf32x3_arith (bool): whether to use CUTLASS's 3xTF32 GEMMs, which enable accuracy similar to FP32
                                   while using Tensor Cores
    """

    raise RuntimeError("This function has been removed. Use tile primitives instead.")


def batched_matmul(
    a: array3d,
    b: array3d,
    c: array3d,
    d: array3d,
    alpha: float = 1.0,
    beta: float = 0.0,
    allow_tf32x3_arith: builtins.bool = False,
):
    """Computes a batched generic matrix-matrix multiplication (GEMM) of the form: `d = alpha * (a @ b) + beta * c`.

    .. versionremoved:: 1.7

    .. deprecated:: 1.6
        Use :doc:`tile primitives </modules/tiles>` instead.

    Args:
        a (array3d): three-dimensional array containing A matrices. Overall array dimension is {batch_count, M, K}
        b (array3d): three-dimensional array containing B matrices. Overall array dimension is {batch_count, K, N}
        c (array3d): three-dimensional array containing C matrices. Overall array dimension is {batch_count, M, N}
        d (array3d): three-dimensional array to which output D is written. Overall array dimension is {batch_count, M, N}
        alpha (float): parameter alpha of GEMM
        beta (float): parameter beta of GEMM
        allow_tf32x3_arith (bool): whether to use CUTLASS's 3xTF32 GEMMs, which enable accuracy similar to FP32
                                   while using Tensor Cores
    """

    raise RuntimeError("This function has been removed. Use tile primitives instead.")


def adj_batched_matmul(
    a: array3d,
    b: array3d,
    c: array3d,
    adj_a: array3d,
    adj_b: array3d,
    adj_c: array3d,
    adj_d: array3d,
    alpha: float = 1.0,
    beta: float = 0.0,
    allow_tf32x3_arith: builtins.bool = False,
):
    """Computes the adjoint of a batched generic matrix-matrix multiplication (GEMM) of the form: `d = alpha * (a @ b) + beta * c`.

    Args:
        a (array3d): three-dimensional array containing A matrices. Overall array dimension is {batch_count, M, K}
        b (array3d): three-dimensional array containing B matrices. Overall array dimension is {batch_count, K, N}
        c (array3d): three-dimensional array containing C matrices. Overall array dimension is {batch_count, M, N}
        adj_a (array3d): three-dimensional array to which the adjoints of A matrices are written. Overall array dimension is {batch_count, M, K}
        adj_b (array3d): three-dimensional array to which the adjoints of B matrices are written. Overall array dimension is {batch_count, K, N}
        adj_c (array3d): three-dimensional array to which the adjoints of C matrices are written. Overall array dimension is {batch_count, M, N}
        adj_d (array3d): three-dimensional array containing adjoints of D matrices. Overall array dimension is {batch_count, M, N}
        alpha (float): parameter alpha of GEMM
        beta (float): parameter beta of GEMM
        allow_tf32x3_arith (bool): whether to use CUTLASS's 3xTF32 GEMMs, which enable accuracy similar to FP32
                                   while using Tensor Cores
    """

    raise RuntimeError("This function has been removed. Use tile primitives instead.")


class HashGrid:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.id = None
        return instance

    def __init__(self, dim_x, dim_y, dim_z, device=None):
        """Class representing a hash grid object for accelerated point queries.

        Attributes:
            id: Unique identifier for this mesh object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            dim_x (int): Number of cells in x-axis
            dim_y (int): Number of cells in y-axis
            dim_z (int): Number of cells in z-axis
        """

        self.runtime = warp.context.runtime

        self.device = self.runtime.get_device(device)

        if self.device.is_cpu:
            self.id = self.runtime.core.hash_grid_create_host(dim_x, dim_y, dim_z)
        else:
            self.id = self.runtime.core.hash_grid_create_device(self.device.context, dim_x, dim_y, dim_z)

        # indicates whether the grid data has been reserved for use by a kernel
        self.reserved = False

    def build(self, points, radius):
        """Updates the hash grid data structure.

        This method rebuilds the underlying datastructure and should be called any time the set
        of points changes.

        Args:
            points (:class:`warp.array`): Array of points of type :class:`warp.vec3`
            radius (float): The cell size to use for bucketing points, cells are cubes with edges of this width.
                            For best performance the radius used to construct the grid should match closely to
                            the radius used when performing queries.
        """

        if not warp.types.types_equal(points.dtype, warp.vec3):
            raise TypeError("Hash grid points should have type warp.vec3")

        if points.ndim > 1:
            points = points.contiguous().flatten()

        if self.device.is_cpu:
            self.runtime.core.hash_grid_update_host(self.id, radius, ctypes.byref(points.__ctype__()))
        else:
            self.runtime.core.hash_grid_update_device(self.id, radius, ctypes.byref(points.__ctype__()))
        self.reserved = True

    def reserve(self, num_points):
        if self.device.is_cpu:
            self.runtime.core.hash_grid_reserve_host(self.id, num_points)
        else:
            self.runtime.core.hash_grid_reserve_device(self.id, num_points)
        self.reserved = True

    def __del__(self):
        if not self.id:
            return

        if self.device.is_cpu:
            self.runtime.core.hash_grid_destroy_host(self.id)
        else:
            # use CUDA context guard to avoid side effects during garbage collection
            with self.device.context_guard:
                self.runtime.core.hash_grid_destroy_device(self.id)


class MarchingCubes:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.id = None
        return instance

    def __init__(self, nx: int, ny: int, nz: int, max_verts: int, max_tris: int, device=None):
        """CUDA-based Marching Cubes algorithm to extract a 2D surface mesh from a 3D volume.

        Attributes:
            id: Unique identifier for this object.
            verts (:class:`warp.array`): Array of vertex positions of type :class:`warp.vec3f`
              for the output surface mesh.
              This is populated after running :func:`surface`.
            indices (:class:`warp.array`): Array containing indices of type :class:`warp.int32`
              defining triangles for the output surface mesh.
              This is populated after running :func:`surface`.

              Each set of three consecutive integers in the array represents a single triangle,
              in which each integer is an index referring to a vertex in the :attr:`verts` array.

        Args:
            nx: Number of cubes in the x-direction.
            ny: Number of cubes in the y-direction.
            nz: Number of cubes in the z-direction.
            max_verts: Maximum expected number of vertices (used for array preallocation).
            max_tris: Maximum expected number of triangles (used for array preallocation).
            device (Devicelike): CUDA device on which to run marching cubes and allocate memory.

        Raises:
            RuntimeError: ``device`` not a CUDA device.

        .. note::
            The shape of the marching cubes should match the shape of the scalar field being surfaced.

        """

        self.id = 0

        self.runtime = warp.context.runtime

        self.device = self.runtime.get_device(device)

        if not self.device.is_cuda:
            raise RuntimeError("Only CUDA devices are supported for marching cubes")

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.max_verts = max_verts
        self.max_tris = max_tris

        # bindings to warp.so
        self.alloc = self.runtime.core.marching_cubes_create_device
        self.alloc.argtypes = [ctypes.c_void_p]
        self.alloc.restype = ctypes.c_uint64
        self.free = self.runtime.core.marching_cubes_destroy_device

        from warp.context import zeros

        self.verts = zeros(max_verts, dtype=vec3, device=self.device)
        self.indices = zeros(max_tris * 3, dtype=warp.int32, device=self.device)

        # alloc surfacer
        self.id = ctypes.c_uint64(self.alloc(self.device.context))

    def __del__(self):
        if not self.id:
            return

        # use CUDA context guard to avoid side effects during garbage collection
        with self.device.context_guard:
            # destroy surfacer
            self.free(self.id)

    def resize(self, nx: int, ny: int, nz: int, max_verts: int, max_tris: int) -> None:
        """Update the expected input and maximum output sizes for the marching cubes calculation.

        This function has no immediate effect on the underlying buffers.
        The new values take effect on the next :func:`surface` call.

        Args:
          nx: Number of cubes in the x-direction.
          ny: Number of cubes in the y-direction.
          nz: Number of cubes in the z-direction.
          max_verts: Maximum expected number of vertices (used for array preallocation).
          max_tris: Maximum expected number of triangles (used for array preallocation).
        """
        # actual allocations will be resized on next call to surface()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.max_verts = max_verts
        self.max_tris = max_tris

    def surface(self, field: array(dtype=float, ndim=3), threshold: float) -> None:
        """Compute a 2D surface mesh of a given isosurface from a 3D scalar field.

        The triangles and vertices defining the output mesh are written to the
        :attr:`indices` and :attr:`verts` arrays.

        Args:
          field: Scalar field from which to generate a mesh.
          threshold: Target isosurface value.

        Raises:
          ValueError: ``field`` is not a 3D array.
          ValueError: Marching cubes shape does not match the shape of ``field``.
          RuntimeError: :attr:`max_verts` and/or :attr:`max_tris` might be too small to hold the surface mesh.
        """

        # WP_API int marching_cubes_surface_host(const float* field, int nx, int ny, int nz, float threshold, wp::vec3* verts, int* triangles, int max_verts, int max_tris, int* out_num_verts, int* out_num_tris);
        num_verts = ctypes.c_int(0)
        num_tris = ctypes.c_int(0)

        self.runtime.core.marching_cubes_surface_device.restype = ctypes.c_int

        # For now we require that input field shape matches nx, ny, nz
        if field.ndim != 3:
            raise ValueError(f"Input field must be a three-dimensional array (got {field.ndim}).")
        if field.shape[0] != self.nx or field.shape[1] != self.ny or field.shape[2] != self.nz:
            raise ValueError(
                f"Marching cubes shape ({self.nx}, {self.ny}, {self.nz}) does not match the "
                f"input array shape {field.shape}."
            )

        error = self.runtime.core.marching_cubes_surface_device(
            self.id,
            ctypes.cast(field.ptr, ctypes.c_void_p),
            self.nx,
            self.ny,
            self.nz,
            ctypes.c_float(threshold),
            ctypes.cast(self.verts.ptr, ctypes.c_void_p),
            ctypes.cast(self.indices.ptr, ctypes.c_void_p),
            self.max_verts,
            self.max_tris,
            ctypes.c_void_p(ctypes.addressof(num_verts)),
            ctypes.c_void_p(ctypes.addressof(num_tris)),
        )

        if error:
            raise RuntimeError(
                f"Buffers may not be large enough, marching cubes required at least {num_verts} vertices, and {num_tris} triangles."
            )

        # resize the geometry arrays
        self.verts.shape = (num_verts.value,)
        self.indices.shape = (num_tris.value * 3,)

        self.verts.size = num_verts.value
        self.indices.size = num_tris.value * 3


generic_types = (Any, Scalar, Float, Int)


def type_is_generic(t):
    if t in generic_types:
        return True

    if is_array(t):
        return type_is_generic(t.dtype)

    if hasattr(t, "_wp_scalar_type_"):
        # vector/matrix type, check if dtype is generic
        if type_is_generic(t._wp_scalar_type_):
            return True
        # check if any dimension is generic
        for d in t._shape_:
            if d == 0:
                return True

    return False


def type_is_generic_scalar(t):
    return t in (Scalar, Float, Int)


def type_matches_template(arg_type, template_type):
    """Check if an argument type matches a template.

    This function is used to test whether the arguments passed to a generic @wp.kernel or @wp.func
    match the template type annotations.  The template_type can be generic, but the arg_type must be concrete.
    """

    # canonicalize types
    arg_type = type_to_warp(arg_type)
    template_type = type_to_warp(template_type)

    # arg type must be concrete
    if type_is_generic(arg_type):
        return False

    # if template type is not generic, the argument type must match exactly
    if not type_is_generic(template_type):
        return types_equal(arg_type, template_type)

    # template type is generic, check that the argument type matches
    if template_type == Any:
        return True
    elif is_array(template_type):
        # ensure the argument type is a non-generic array with matching dtype and dimensionality
        if type(arg_type) is not type(template_type):
            return False
        if not type_matches_template(arg_type.dtype, template_type.dtype):
            return False
        if arg_type.ndim != template_type.ndim:
            return False
    elif template_type == Float:
        return arg_type in float_types
    elif template_type == Int:
        return arg_type in int_types
    elif template_type == Scalar:
        return arg_type in scalar_types
    elif hasattr(template_type, "_wp_scalar_type_"):
        # vector/matrix type
        if not hasattr(arg_type, "_wp_scalar_type_"):
            return False
        if not type_matches_template(arg_type._wp_scalar_type_, template_type._wp_scalar_type_):
            return False
        ndim = len(template_type._shape_)
        if len(arg_type._shape_) != ndim:
            return False
        # for any non-generic dimensions, make sure they match
        for i in range(ndim):
            if template_type._shape_[i] != 0 and arg_type._shape_[i] != template_type._shape_[i]:
                return False

    return True


def infer_argument_types(args, template_types, arg_names=None):
    """Resolve argument types with the given list of template types."""

    if len(args) != len(template_types):
        raise RuntimeError("Number of arguments must match number of template types.")

    arg_types = []

    for i in range(len(args)):
        arg = args[i]
        arg_type = type(arg)
        arg_name = arg_names[i] if arg_names else str(i)
        if arg_type in warp.types.array_types:
            arg_types.append(arg_type(dtype=arg.dtype, ndim=arg.ndim))
        elif arg_type in warp.types.scalar_and_bool_types:
            arg_types.append(arg_type)
        elif arg_type in (int, float, builtins.bool):
            # canonicalize type
            arg_types.append(warp.types.type_to_warp(arg_type))
        elif hasattr(arg_type, "_wp_scalar_type_"):
            # vector/matrix type
            arg_types.append(arg_type)
        elif issubclass(arg_type, warp.codegen.StructInstance):
            # a struct
            arg_types.append(arg._cls)
        # elif arg_type in [warp.types.launch_bounds_t, warp.types.shape_t, warp.types.range_t]:
        #     arg_types.append(arg_type)
        # elif arg_type in [warp.hash_grid_query_t, warp.mesh_query_aabb_t, warp.mesh_query_point_t, warp.mesh_query_ray_t, warp.bvh_query_t]:
        #     arg_types.append(arg_type)
        elif arg is None:
            # allow passing None for arrays
            t = template_types[i]
            if warp.types.is_array(t):
                arg_types.append(type(t)(dtype=t.dtype, ndim=t.ndim))
            else:
                raise TypeError(f"Unable to infer the type of argument '{arg_name}', got None")
        else:
            # TODO: attempt to figure out if it's a vector/matrix type given as a numpy array, list, etc.
            raise TypeError(f"Unable to infer the type of argument '{arg_name}', got {arg_type}")

    return arg_types


simple_type_codes = {
    int: "i4",
    float: "f4",
    builtins.bool: "b",
    bool: "b",
    str: "str",  # accepted by print()
    int8: "i1",
    int16: "i2",
    int32: "i4",
    int64: "i8",
    uint8: "u1",
    uint16: "u2",
    uint32: "u4",
    uint64: "u8",
    float16: "f2",
    float32: "f4",
    float64: "f8",
    shape_t: "sh",
    range_t: "rg",
    launch_bounds_t: "lb",
    hash_grid_query_t: "hgq",
    mesh_query_aabb_t: "mqa",
    mesh_query_point_t: "mqp",
    mesh_query_ray_t: "mqr",
    bvh_query_t: "bvhq",
}


def get_type_code(arg_type: type) -> str:
    if arg_type == Any:
        # special case for generics
        # note: since Python 3.11 Any is a type, so we check for it first
        return "?"
    elif isinstance(arg_type, type):
        if hasattr(arg_type, "_wp_scalar_type_"):
            # vector/matrix type
            dtype_code = get_type_code(arg_type._wp_scalar_type_)
            # check for "special" vector/matrix subtypes
            if hasattr(arg_type, "_wp_generic_type_str_"):
                type_str = arg_type._wp_generic_type_str_
                if type_str == "quat_t":
                    return f"q{dtype_code}"
                elif type_str == "transform_t":
                    return f"t{dtype_code}"
                # elif type_str == "spatial_vector_t":
                #     return f"sv{dtype_code}"
                # elif type_str == "spatial_matrix_t":
                #     return f"sm{dtype_code}"
            # generic vector/matrix
            ndim = len(arg_type._shape_)
            if ndim == 1:
                dim_code = "?" if arg_type._shape_[0] == 0 else str(arg_type._shape_[0])
                return f"v{dim_code}{dtype_code}"
            elif ndim == 2:
                dim_code0 = "?" if arg_type._shape_[0] == 0 else str(arg_type._shape_[0])
                dim_code1 = "?" if arg_type._shape_[1] == 0 else str(arg_type._shape_[1])
                return f"m{dim_code0}{dim_code1}{dtype_code}"
            else:
                raise TypeError("Invalid vector/matrix dimensionality")
        else:
            # simple type
            type_code = simple_type_codes.get(arg_type)
            if type_code is not None:
                return type_code
            else:
                raise TypeError(f"Unrecognized type '{arg_type}'")
    elif isinstance(arg_type, array):
        return f"a{arg_type.ndim}{get_type_code(arg_type.dtype)}"
    elif isinstance(arg_type, indexedarray):
        return f"ia{arg_type.ndim}{get_type_code(arg_type.dtype)}"
    elif isinstance(arg_type, fabricarray):
        return f"fa{arg_type.ndim}{get_type_code(arg_type.dtype)}"
    elif isinstance(arg_type, indexedfabricarray):
        return f"ifa{arg_type.ndim}{get_type_code(arg_type.dtype)}"
    elif isinstance(arg_type, warp.codegen.Struct):
        return arg_type.native_name
    elif arg_type == Scalar:
        # generic scalar type
        return "s?"
    elif arg_type == Float:
        # generic float
        return "f?"
    elif arg_type == Int:
        # generic int
        return "i?"
    elif isinstance(arg_type, Callable):
        # TODO: elaborate on Callable type?
        return "c"
    else:
        raise TypeError(f"Unrecognized type '{arg_type}'")


def get_signature(arg_types: list[type], func_name: str | None = None, arg_names: list[str] | None = None) -> str:
    type_codes: list[str] = []
    for i, arg_type in enumerate(arg_types):
        try:
            type_codes.append(get_type_code(arg_type))
        except Exception as e:
            if arg_names is not None:
                arg_str = f"'{arg_names[i]}'"
            else:
                arg_str = str(i + 1)
            if func_name is not None:
                func_str = f" of function {func_name}"
            else:
                func_str = ""
            raise RuntimeError(f"Failed to determine type code for argument {arg_str}{func_str}: {e}") from e

    return "_".join(type_codes)


def is_generic_signature(sig):
    return "?" in sig
