# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import builtins
import ctypes
import hashlib
import inspect
import struct
import zlib
from typing import Any, Callable, Generic, List, Tuple, TypeVar, Union

import numpy as np

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
    pass


# shared hash for all constants
_constant_hash = hashlib.sha256()


def constant(x):
    """Function to declare compile-time constants accessible from Warp kernels

    Args:
        x: Compile-time constant value, can be any of the built-in math types.
    """

    global _constant_hash

    # hash the constant value
    if isinstance(x, builtins.bool):
        # This needs to come before the check for `int` since all boolean
        # values are also instances of `int`.
        _constant_hash.update(struct.pack("?", x))
    elif isinstance(x, int):
        _constant_hash.update(struct.pack("<q", x))
    elif isinstance(x, float):
        _constant_hash.update(struct.pack("<d", x))
    elif isinstance(x, float16):
        # float16 is a special case
        p = ctypes.pointer(ctypes.c_float(x.value))
        _constant_hash.update(p.contents)
    elif isinstance(x, tuple(scalar_types)):
        p = ctypes.pointer(x._type_(x.value))
        _constant_hash.update(p.contents)
    elif isinstance(x, ctypes.Array):
        _constant_hash.update(bytes(x))
    else:
        raise RuntimeError(f"Invalid constant type: {type(x)}")

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
        elif dtype in [Scalar, Float]:
            _type_ = ctypes.c_float
        else:
            _type_ = dtype._type_

        # warp scalar type:
        _wp_scalar_type_ = dtype
        _wp_type_params_ = [length, dtype]
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
                        f"Expected to assign a slice from a sequence of values "
                        f"but got `{type(value).__name__}` instead"
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
        elif dtype in [Scalar, Float]:
            _type_ = ctypes.c_float
        else:
            _type_ = dtype._type_

        # warp scalar type:
        # used in type checking and when writing out c++ code for constructors:
        _wp_scalar_type_ = dtype
        _wp_type_params_ = [shape[0], shape[1], dtype]
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
                    for i in range(self._shape_[1]):
                        super().__setitem__(offset + i, mat_t.scalar_import(row[i]))
            else:
                raise ValueError(
                    f"Invalid number of arguments in matrix constructor, expected {self._length_} elements, got {num_args}"
                )

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
                    f"Expected to assign a slice from a sequence of values " f"but got `{type(v).__name__}` instead"
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


class bool:
    _length_ = 1
    _type_ = ctypes.c_bool

    def __init__(self, x=False):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0

    def __float__(self) -> float:
        return float(self.value != 0)

    def __int__(self) -> int:
        return int(self.value != 0)


class float16:
    _length_ = 1
    _type_ = ctypes.c_uint16

    def __init__(self, x=0.0):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0.0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)


class float32:
    _length_ = 1
    _type_ = ctypes.c_float

    def __init__(self, x=0.0):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0.0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)


class float64:
    _length_ = 1
    _type_ = ctypes.c_double

    def __init__(self, x=0.0):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0.0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)


class int8:
    _length_ = 1
    _type_ = ctypes.c_int8

    def __init__(self, x=0):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __index__(self) -> int:
        return int(self.value)


class uint8:
    _length_ = 1
    _type_ = ctypes.c_uint8

    def __init__(self, x=0):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __index__(self) -> int:
        return int(self.value)


class int16:
    _length_ = 1
    _type_ = ctypes.c_int16

    def __init__(self, x=0):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __index__(self) -> int:
        return int(self.value)


class uint16:
    _length_ = 1
    _type_ = ctypes.c_uint16

    def __init__(self, x=0):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __index__(self) -> int:
        return int(self.value)


class int32:
    _length_ = 1
    _type_ = ctypes.c_int32

    def __init__(self, x=0):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __index__(self) -> int:
        return int(self.value)


class uint32:
    _length_ = 1
    _type_ = ctypes.c_uint32

    def __init__(self, x=0):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __index__(self) -> int:
        return int(self.value)


class int64:
    _length_ = 1
    _type_ = ctypes.c_int64

    def __init__(self, x=0):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __index__(self) -> int:
        return int(self.value)


class uint64:
    _length_ = 1
    _type_ = ctypes.c_uint64

    def __init__(self, x=0):
        self.value = x

    def __bool__(self) -> bool:
        return self.value != 0

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __index__(self) -> int:
        return int(self.value)


def quaternion(dtype=Any):
    class quat_t(vector(length=4, dtype=dtype)):
        pass
        # def __init__(self, *args):
        #     super().__init__(args)

    ret = quat_t
    ret._wp_type_params_ = [dtype]
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
        _wp_generic_type_str_ = "transform_t"
        _wp_generic_type_hint_ = Transformation
        _wp_constructor_ = "transformation"

        def __init__(self, *args, **kwargs):
            if len(args) == 1 and len(kwargs) == 0:
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


# definition just for kernel type (cannot be a parameter), see mesh.h
class mesh_query_aabb_t:
    """Object used to track state during mesh traversal."""

    def __init__(self):
        pass


# definition just for kernel type (cannot be a parameter), see hash_grid.h
class hash_grid_query_t:
    """Object used to track state during neighbor traversal."""

    def __init__(self):
        pass


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

    def __init__(self, shape):
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


def type_size_in_bytes(dtype):
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


def type_to_warp(dtype):
    if dtype == float:
        return float32
    elif dtype == int:
        return int32
    elif dtype == builtins.bool:
        return bool
    else:
        return dtype


def type_typestr(dtype):
    if dtype == bool:
        return "?"
    elif dtype == float16:
        return "<f2"
    elif dtype == float32:
        return "<f4"
    elif dtype == float64:
        return "<f8"
    elif dtype == int8:
        return "b"
    elif dtype == uint8:
        return "B"
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
    if type_is_vector(t):
        return str(f"vector(length={t._shape_[0]}, dtype={t._wp_scalar_type_})")
    if type_is_matrix(t):
        return str(f"matrix(shape=({t._shape_[0]}, {t._shape_[1]}), dtype={t._wp_scalar_type_})")
    if isinstance(t, warp.codegen.Struct):
        return type_repr(t.cls)
    if t in scalar_types:
        return t.__name__

    return t.__module__ + "." + t.__qualname__


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


# returns True if the passed *type* is a matrix
def type_is_matrix(t):
    return getattr(t, "_wp_generic_type_hint_", None) is Matrix


value_types = (int, float, builtins.bool) + scalar_types


# returns true for all value types (int, float, bool, scalars, vectors, matrices)
def type_is_value(x):
    return x in value_types or issubclass(x, ctypes.Array)


# equivalent of the above but for values
def is_int(x):
    return type_is_int(type(x))


def is_float(x):
    return type_is_float(type(x))


def is_value(x):
    return type_is_value(type(x))


# returns true if the passed *instance* is one of the array types
def is_array(a):
    return isinstance(a, array_types)


def scalars_equal(a, b, match_generic):
    if match_generic:
        if a == Any or b == Any:
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

    return a == b


def types_equal(a, b, match_generic=False):
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

    if is_array(a) and type(a) is type(b):
        return True

    return scalars_equal(a, b, match_generic)


def strides_from_shape(shape: Tuple, dtype):
    ndims = len(shape)
    strides = [None] * ndims

    i = ndims - 1
    strides[i] = type_size_in_bytes(dtype)

    while i > 0:
        strides[i - 1] = strides[i] * shape[i]
        i -= 1

    return tuple(strides)


def check_array_shape(shape: Tuple):
    """Checks that the size in each dimension is positive and less than 2^32."""

    for dim_index, dim_size in enumerate(shape):
        if dim_size < 0:
            raise ValueError(f"Array shapes must be non-negative, got {dim_size} in dimension {dim_index}")
        if dim_size >= 2**31:
            raise ValueError(
                "Array shapes must not exceed the maximum representable value of a signed 32-bit integer, "
                f"got {dim_size} in dimension {dim_index}."
            )


class array(Array):
    # member attributes available during code-gen (e.g.: d = array.shape[0])
    # (initialized when needed)
    _vars = None

    def __new__(cls, *args, **kwargs):
        instance = super(array, cls).__new__(cls)
        instance.deleter = None
        return instance

    def __init__(
        self,
        data=None,
        dtype: DType = Any,
        shape=None,
        strides=None,
        length=None,
        ptr=None,
        capacity=None,
        device=None,
        pinned=False,
        copy=True,
        owner=False,  # deprecated - pass deleter instead
        deleter=None,
        ndim=None,
        grad=None,
        requires_grad=False,
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
            data (Union[list, tuple, ndarray]): An object to construct the array from, can be a Tuple, List, or generally any type convertible to an np.array
            dtype (Union): One of the built-in types, e.g.: :class:`warp.mat33`, if dtype is Any and data an ndarray then it will be inferred from the array data type
            shape (tuple): Dimensions of the array
            strides (tuple): Number of bytes in each dimension between successive elements of the array
            length (int): Number of elements of the data type (deprecated, users should use `shape` argument)
            ptr (uint64): Address of an external memory address to alias (data should be None)
            capacity (int): Maximum size in bytes of the ptr allocation (data should be None)
            device (Devicelike): Device the array lives on
            copy (bool): Whether the incoming data will be copied or aliased, this is only possible when the incoming `data` already lives on the device specified and types match
            owner (bool): Should the array object try to deallocate memory when it is deleted (deprecated, pass `deleter` if you wish to transfer ownership to Warp)
            deleter (Callable): Function to be called when deallocating the array, taking two arguments, pointer and size
            requires_grad (bool): Whether or not gradients will be tracked for this array, see :class:`warp.Tape` for details
            grad (array): The gradient array to use
            pinned (bool): Whether to allocate pinned host memory, which allows asynchronous host-device transfers (only applicable with device="cpu")

        """

        self.ctype = None

        # properties
        self._requires_grad = False
        self._grad = None
        # __array_interface__ or __cuda_array_interface__, evaluated lazily and cached
        self._array_interface = None
        self.is_transposed = False

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
                shape = tuple(shape)
                if len(shape) > ARRAY_MAX_DIMS:
                    raise RuntimeError(
                        f"Failed to create array with shape {shape}, the maximum number of dimensions is {ARRAY_MAX_DIMS}"
                    )
        elif length is not None:
            # backward compatibility
            shape = (length,)

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

        if hasattr(data, "__cuda_array_interface__"):
            try:
                # Performance note: try first, ask questions later
                device = warp.context.runtime.get_device(device)
            except:
                warp.context.assert_initialized()
                raise

            if device.is_cuda:
                desc = data.__cuda_array_interface__
                shape = desc.get("shape")
                strides = desc.get("strides")
                dtype = np_dtype_to_warp_type[np.dtype(desc.get("typestr"))]
                ptr = desc.get("data")[0]

                self._init_from_ptr(ptr, dtype, shape, strides, None, device, False, None)

                # keep a ref to the source data to keep allocation alive
                self._ref = data
                return
            else:
                raise RuntimeError(
                    f"Trying to construct a Warp array from data argument's __cuda_array_interface__ but {device} is not CUDA-capable"
                )

        if hasattr(dtype, "_wp_scalar_type_"):
            dtype_shape = dtype._shape_
            dtype_ndim = len(dtype_shape)
            scalar_dtype = dtype._wp_scalar_type_
        else:
            dtype_shape = ()
            dtype_ndim = 0
            scalar_dtype = dtype

        # convert input data to ndarray (handles lists, tuples, etc.) and determine dtype
        if dtype == Any:
            # infer dtype from data
            try:
                arr = np.array(data, copy=False, ndmin=1)
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
                arr = np.array(data, dtype=npdtype, copy=False, ndmin=1)
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
        except:
            warp.context.assert_initialized()
            raise

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
        except:
            warp.context.assert_initialized()
            raise

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
        except:
            warp.context.assert_initialized()
            raise

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
            capacity = shape[0] * strides[0]

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

        if self.device.is_cuda and stream != -1:
            if not isinstance(stream, int):
                raise TypeError("DLPack stream must be an integer or None")

            # assume that the array is being used on its device's current stream
            array_stream = self.device.stream

            # the external stream should wait for outstanding operations to complete
            if stream in (None, 0, 1):
                external_stream = 0
            else:
                external_stream = stream

            # Performance note: avoid wrapping the external stream in a temporary Stream object
            if external_stream != array_stream.cuda_stream:
                warp.context.runtime.core.cuda_stream_wait_stream(
                    external_stream, array_stream.cuda_stream, array_stream.cached_event.cuda_event
                )

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
        if self.ndim != 2 or other.ndim != 2:
            raise RuntimeError(
                "A has dim = {}, B has dim = {}. If multiplying with @, A and B must have dim = 2.".format(
                    self.ndim, other.ndim
                )
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
            if (
                grad.dtype != self.dtype
                or grad.shape != self.shape
                or grad.strides != self.strides
                or grad.device != self.device
            ):
                raise ValueError("The given gradient array is incompatible")
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

    def zero_(self):
        """Zeroes-out the array entries."""
        if self.is_contiguous:
            # simple memset is usually faster than generic fill
            self.device.memset(self.ptr, 0, self.size * type_size_in_bytes(self.dtype))
        else:
            self.fill_(0)

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
            return np.array(a, copy=False)
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

        a._ref = self
        return a


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
        "This version of wp.from_ptr() is deprecated. OmniGraph applications should use from_omni_graph_ptr() instead. In the future, wp.from_ptr() will work only with regular pointers.",
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


# A base class for non-contiguous arrays, providing the implementation of common methods like
# contiguous(), to(), numpy(), list(), assign(), zero_(), and fill_().
class noncontiguous_array_base(Generic[T]):
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


class indexedarray(noncontiguous_array_base[T]):
    # member attributes available during code-gen (e.g.: d = arr.shape[0])
    # (initialized when needed)
    _vars = None

    def __init__(self, data: array = None, indices: Union[array, List[array]] = None, dtype=None, ndim=None):
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


class Bvh:
    def __init__(self, lowers, uppers):
        """Class representing a bounding volume hierarchy.

        Attributes:
            id: Unique identifier for this bvh object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            lowers (:class:`warp.array`): Array of lower bounds :class:`warp.vec3`
            uppers (:class:`warp.array`): Array of upper bounds :class:`warp.vec3`
        """

        self.id = 0

        if len(lowers) != len(uppers):
            raise RuntimeError("Bvh the same number of lower and upper bounds must be provided")

        if lowers.device != uppers.device:
            raise RuntimeError("Bvh lower and upper bounds must live on the same device")

        if lowers.dtype != vec3 or not lowers.is_contiguous:
            raise RuntimeError("Bvh lowers should be a contiguous array of type wp.vec3")

        if uppers.dtype != vec3 or not uppers.is_contiguous:
            raise RuntimeError("Bvh uppers should be a contiguous array of type wp.vec3")

        self.device = lowers.device
        self.lowers = lowers
        self.uppers = uppers

        def get_data(array):
            if array:
                return ctypes.c_void_p(array.ptr)
            else:
                return ctypes.c_void_p(0)

        self.runtime = warp.context.runtime

        if self.device.is_cpu:
            self.id = self.runtime.core.bvh_create_host(get_data(lowers), get_data(uppers), int(len(lowers)))
        else:
            self.id = self.runtime.core.bvh_create_device(
                self.device.context, get_data(lowers), get_data(uppers), int(len(lowers))
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
        """Refit the BVH. This should be called after users modify the `lowers` and `uppers` arrays."""

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

    def __init__(self, points=None, indices=None, velocities=None, support_winding_number=False):
        """Class representing a triangle mesh.

        Attributes:
            id: Unique identifier for this mesh object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            points (:class:`warp.array`): Array of vertex positions of type :class:`warp.vec3`
            indices (:class:`warp.array`): Array of triangle indices of type :class:`warp.int32`, should be a 1d array with shape (num_tris, 3)
            velocities (:class:`warp.array`): Array of vertex velocities of type :class:`warp.vec3` (optional)
            support_winding_number (bool): If true the mesh will build additional datastructures to support `wp.mesh_query_point_sign_winding_number()` queries
        """

        self.id = 0

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
        self.points = points
        self.velocities = velocities
        self.indices = indices

        self.runtime = warp.context.runtime

        if self.device.is_cpu:
            self.id = self.runtime.core.mesh_create_host(
                points.__ctype__(),
                velocities.__ctype__() if velocities else array().__ctype__(),
                indices.__ctype__(),
                int(len(points)),
                int(indices.size / 3),
                int(support_winding_number),
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
        """Refit the BVH to points. This should be called after users modify the `points` data."""

        if self.device.is_cpu:
            self.runtime.core.mesh_refit_host(self.id)
        else:
            self.runtime.core.mesh_refit_device(self.id)
            self.runtime.verify_cuda_device(self.device)


class Volume:
    #: Enum value to specify nearest-neighbor interpolation during sampling
    CLOSEST = constant(0)
    #: Enum value to specify trilinear interpolation during sampling
    LINEAR = constant(1)

    def __init__(self, data: array):
        """Class representing a sparse grid.

        Args:
            data (:class:`warp.array`): Array of bytes representing the volume in NanoVDB format
        """

        self.id = 0

        # keep a runtime reference for orderly destruction
        self.runtime = warp.context.runtime

        if data is None:
            return

        if data.device is None:
            raise RuntimeError("Invalid device")
        self.device = data.device

        if self.device.is_cpu:
            self.id = self.runtime.core.volume_create_host(ctypes.cast(data.ptr, ctypes.c_void_p), data.size)
        else:
            self.id = self.runtime.core.volume_create_device(
                self.device.context, ctypes.cast(data.ptr, ctypes.c_void_p), data.size
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
        """Returns the raw memory buffer of the Volume as an array"""
        buf = ctypes.c_void_p(0)
        size = ctypes.c_uint64(0)
        if self.device.is_cpu:
            self.runtime.core.volume_get_buffer_info_host(self.id, ctypes.byref(buf), ctypes.byref(size))
        else:
            self.runtime.core.volume_get_buffer_info_device(self.id, ctypes.byref(buf), ctypes.byref(size))
        return array(ptr=buf.value, dtype=uint8, shape=size.value, device=self.device)

    def get_tiles(self) -> array:
        if self.id == 0:
            raise RuntimeError("Invalid Volume")

        buf = ctypes.c_void_p(0)
        size = ctypes.c_uint64(0)
        if self.device.is_cpu:
            self.runtime.core.volume_get_tiles_host(self.id, ctypes.byref(buf), ctypes.byref(size))
            deleter = self.device.default_allocator.deleter
        else:
            self.runtime.core.volume_get_tiles_device(self.id, ctypes.byref(buf), ctypes.byref(size))
            if self.device.is_mempool_supported:
                deleter = self.device.mempool_allocator.deleter
            else:
                deleter = self.device.default_allocator.deleter
        num_tiles = size.value // (3 * 4)

        return array(ptr=buf.value, dtype=int32, shape=(num_tiles, 3), device=self.device, deleter=deleter)

    def get_voxel_size(self) -> Tuple[float, float, float]:
        if self.id == 0:
            raise RuntimeError("Invalid Volume")

        dx, dy, dz = ctypes.c_float(0), ctypes.c_float(0), ctypes.c_float(0)
        self.runtime.core.volume_get_voxel_size(self.id, ctypes.byref(dx), ctypes.byref(dy), ctypes.byref(dz))
        return (dx.value, dy.value, dz.value)

    @classmethod
    def load_from_nvdb(cls, file_or_buffer, device=None) -> Volume:
        """Creates a Volume object from a NanoVDB file or in-memory buffer.

        Returns:

            A ``warp.Volume`` object.
        """
        try:
            data = file_or_buffer.read()
        except AttributeError:
            data = file_or_buffer

        magic, version, grid_count, codec = struct.unpack("<QIHH", data[0:16])
        if magic != 0x304244566F6E614E:
            raise RuntimeError("NanoVDB signature not found")
        if version >> 21 != 32:  # checking major version
            raise RuntimeError("Unsupported NanoVDB version")
        if grid_count != 1:
            raise RuntimeError("Only NVDBs with exactly one grid are supported")

        grid_data_offset = 192 + struct.unpack("<I", data[152:156])[0]
        if codec == 0:  # no compression
            grid_data = data[grid_data_offset:]
        elif codec == 1:  # zip compression
            grid_data = zlib.decompress(data[grid_data_offset + 8 :])
        else:
            raise RuntimeError(f"Unsupported codec code: {codec}")

        magic = struct.unpack("<Q", grid_data[0:8])[0]
        if magic != 0x304244566F6E614E:
            raise RuntimeError("NanoVDB signature not found on grid!")

        data_array = array(np.frombuffer(grid_data, dtype=np.byte), device=device)
        return cls(data_array)

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
            padded_array = np.array((target_shape[0], target_shape[1], target_shape[2], 3), dtype=np.single)
            padded_array[:, :, :, :] = np.array(bg_value)
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
        min: List[int],
        max: List[int],
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

    @classmethod
    def allocate_by_tiles(
        cls, tile_points: array, voxel_size: float, bg_value=0.0, translation=(0.0, 0.0, 0.0), device=None
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
                The array can be a 2D, N-by-3 array of :class:`warp.int32` values, indicating index space positions,
                or can be a 1D array of :class:`warp.vec3` values, indicating world space positions.
                Repeated points per tile are allowed and will be efficiently deduplicated.
            voxel_size (float): Voxel size of the new volume.
            bg_value (float or array-like): Value of unallocated voxels of the volume, also defines the volume's type, a :class:`warp.vec3` volume is created if this is `array-like`, otherwise a float volume is created
            translation (array-like): Translation between the index and world spaces.
            device (Devicelike): The CUDA device to create the volume on, e.g.: "cuda" or "cuda:0".

        """
        device = warp.get_device(device)

        if voxel_size <= 0.0:
            raise RuntimeError(f"Voxel size must be positive! Got {voxel_size}")
        if not device.is_cuda:
            raise RuntimeError("Only CUDA devices are supported for allocate_by_tiles")
        if not (
            isinstance(tile_points, array)
            and (tile_points.dtype == int32 and tile_points.ndim == 2)
            or (tile_points.dtype == vec3 and tile_points.ndim == 1)
        ):
            raise RuntimeError("Expected an warp array of vec3s or of n-by-3 int32s as tile_points!")
        if not tile_points.device.is_cuda:
            tile_points = array(tile_points, dtype=tile_points.dtype, device=device)

        volume = cls(data=None)
        volume.device = device
        in_world_space = tile_points.dtype == vec3
        if hasattr(bg_value, "__len__"):
            volume.id = volume.runtime.core.volume_v_from_tiles_device(
                volume.device.context,
                ctypes.c_void_p(tile_points.ptr),
                tile_points.shape[0],
                voxel_size,
                bg_value[0],
                bg_value[1],
                bg_value[2],
                translation[0],
                translation[1],
                translation[2],
                in_world_space,
            )
        elif isinstance(bg_value, int):
            volume.id = volume.runtime.core.volume_i_from_tiles_device(
                volume.device.context,
                ctypes.c_void_p(tile_points.ptr),
                tile_points.shape[0],
                voxel_size,
                bg_value,
                translation[0],
                translation[1],
                translation[2],
                in_world_space,
            )
        else:
            volume.id = volume.runtime.core.volume_f_from_tiles_device(
                volume.device.context,
                ctypes.c_void_p(tile_points.ptr),
                tile_points.shape[0],
                voxel_size,
                float(bg_value),
                translation[0],
                translation[1],
                translation[2],
                in_world_space,
            )

        if volume.id == 0:
            raise RuntimeError("Failed to create volume")

        return volume


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
    from warp.context import runtime

    device = a.device

    if b.device != device or c.device != device or d.device != device:
        raise RuntimeError("Matrices A, B, C, and D must all be on the same device as the runtime device.")

    if a.dtype != b.dtype or a.dtype != c.dtype or a.dtype != d.dtype:
        raise RuntimeError(
            "wp.matmul currently only supports operation between {A, B, C, D} matrices of the same type."
        )

    if (
        (not a.is_contiguous and not a.is_transposed)
        or (not b.is_contiguous and not b.is_transposed)
        or (not c.is_contiguous)
        or (not d.is_contiguous)
    ):
        raise RuntimeError(
            "wp.matmul is only valid for contiguous arrays, with the exception that A and/or B may be transposed."
        )

    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    if b.shape != (k, n) or c.shape != (m, n) or d.shape != (m, n):
        raise RuntimeError(
            "Invalid shapes for matrices: A = {} B = {} C = {} D = {}".format(a.shape, b.shape, c.shape, d.shape)
        )

    if runtime.tape:
        runtime.tape.record_func(
            backward=lambda: adj_matmul(a, b, c, a.grad, b.grad, c.grad, d.grad, alpha, beta, allow_tf32x3_arith),
            arrays=[a, b, c, d],
        )

    # cpu fallback if no cuda devices found
    if device == "cpu":
        d.assign(alpha * (a.numpy() @ b.numpy()) + beta * c.numpy())
        return

    cc = device.arch
    ret = runtime.core.cutlass_gemm(
        device.context,
        cc,
        m,
        n,
        k,
        type_typestr(a.dtype).encode(),
        ctypes.c_void_p(a.ptr),
        ctypes.c_void_p(b.ptr),
        ctypes.c_void_p(c.ptr),
        ctypes.c_void_p(d.ptr),
        alpha,
        beta,
        not a.is_transposed,
        not b.is_transposed,
        allow_tf32x3_arith,
        1,
    )
    if not ret:
        raise RuntimeError("matmul failed.")


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
    from warp.context import runtime

    device = a.device

    if (
        b.device != device
        or c.device != device
        or adj_a.device != device
        or adj_b.device != device
        or adj_c.device != device
        or adj_d.device != device
    ):
        raise RuntimeError(
            "Matrices A, B, C, D, and their adjoints must all be on the same device as the runtime device."
        )

    if (
        a.dtype != b.dtype
        or a.dtype != c.dtype
        or a.dtype != adj_a.dtype
        or a.dtype != adj_b.dtype
        or a.dtype != adj_c.dtype
        or a.dtype != adj_d.dtype
    ):
        raise RuntimeError(
            "wp.adj_matmul currently only supports operation between {A, B, C, adj_D, adj_A, adj_B, adj_C} matrices of the same type."
        )

    if (
        (not a.is_contiguous and not a.is_transposed)
        or (not b.is_contiguous and not b.is_transposed)
        or (not c.is_contiguous)
        or (not adj_a.is_contiguous and not adj_a.is_transposed)
        or (not adj_b.is_contiguous and not adj_b.is_transposed)
        or (not adj_c.is_contiguous)
        or (not adj_d.is_contiguous)
    ):
        raise RuntimeError(
            "wp.matmul is only valid for contiguous arrays, with the exception that A and/or B and their associated adjoints may be transposed."
        )

    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    if (
        a.shape != (m, k)
        or b.shape != (k, n)
        or c.shape != (m, n)
        or adj_d.shape != (m, n)
        or adj_a.shape != (m, k)
        or adj_b.shape != (k, n)
        or adj_c.shape != (m, n)
    ):
        raise RuntimeError(
            "Invalid shapes for matrices: A = {} B = {} C = {} adj_D = {} adj_A = {} adj_B = {} adj_C = {}".format(
                a.shape, b.shape, c.shape, adj_d.shape, adj_a.shape, adj_b.shape, adj_c.shape
            )
        )

    # cpu fallback if no cuda devices found
    if device == "cpu":
        adj_a.assign(alpha * np.matmul(adj_d.numpy(), b.numpy().transpose()) + adj_a.numpy())
        adj_b.assign(alpha * (a.numpy().transpose() @ adj_d.numpy()) + adj_b.numpy())
        adj_c.assign(beta * adj_d.numpy() + adj_c.numpy())
        return

    cc = device.arch

    # adj_a
    if not a.is_transposed:
        ret = runtime.core.cutlass_gemm(
            device.context,
            cc,
            m,
            k,
            n,
            type_typestr(a.dtype).encode(),
            ctypes.c_void_p(adj_d.ptr),
            ctypes.c_void_p(b.ptr),
            ctypes.c_void_p(adj_a.ptr),
            ctypes.c_void_p(adj_a.ptr),
            alpha,
            1.0,
            True,
            b.is_transposed,
            allow_tf32x3_arith,
            1,
        )
        if not ret:
            raise RuntimeError("adj_matmul failed.")
    else:
        ret = runtime.core.cutlass_gemm(
            device.context,
            cc,
            k,
            m,
            n,
            type_typestr(a.dtype).encode(),
            ctypes.c_void_p(b.ptr),
            ctypes.c_void_p(adj_d.ptr),
            ctypes.c_void_p(adj_a.ptr),
            ctypes.c_void_p(adj_a.ptr),
            alpha,
            1.0,
            not b.is_transposed,
            False,
            allow_tf32x3_arith,
            1,
        )
        if not ret:
            raise RuntimeError("adj_matmul failed.")

    # adj_b
    if not b.is_transposed:
        ret = runtime.core.cutlass_gemm(
            device.context,
            cc,
            k,
            n,
            m,
            type_typestr(a.dtype).encode(),
            ctypes.c_void_p(a.ptr),
            ctypes.c_void_p(adj_d.ptr),
            ctypes.c_void_p(adj_b.ptr),
            ctypes.c_void_p(adj_b.ptr),
            alpha,
            1.0,
            a.is_transposed,
            True,
            allow_tf32x3_arith,
            1,
        )
        if not ret:
            raise RuntimeError("adj_matmul failed.")
    else:
        ret = runtime.core.cutlass_gemm(
            device.context,
            cc,
            n,
            k,
            m,
            type_typestr(a.dtype).encode(),
            ctypes.c_void_p(adj_d.ptr),
            ctypes.c_void_p(a.ptr),
            ctypes.c_void_p(adj_b.ptr),
            ctypes.c_void_p(adj_b.ptr),
            alpha,
            1.0,
            False,
            not a.is_transposed,
            allow_tf32x3_arith,
            1,
        )
        if not ret:
            raise RuntimeError("adj_matmul failed.")

    # adj_c
    warp.launch(
        kernel=warp.utils.add_kernel_2d,
        dim=adj_c.shape,
        inputs=[adj_c, adj_d, adj_d.dtype(beta)],
        device=device,
        record_tape=False,
    )


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
    from warp.context import runtime

    device = a.device

    if b.device != device or c.device != device or d.device != device:
        raise RuntimeError("Matrices A, B, C, and D must all be on the same device as the runtime device.")

    if a.dtype != b.dtype or a.dtype != c.dtype or a.dtype != d.dtype:
        raise RuntimeError(
            "wp.batched_matmul currently only supports operation between {A, B, C, D} matrices of the same type."
        )

    if (
        (not a.is_contiguous and not a.is_transposed)
        or (not b.is_contiguous and not b.is_transposed)
        or (not c.is_contiguous)
        or (not d.is_contiguous)
    ):
        raise RuntimeError(
            "wp.matmul is only valid for contiguous arrays, with the exception that A and/or B may be transposed."
        )

    m = a.shape[1]
    n = b.shape[2]
    k = a.shape[2]
    batch_count = a.shape[0]
    if b.shape != (batch_count, k, n) or c.shape != (batch_count, m, n) or d.shape != (batch_count, m, n):
        raise RuntimeError(
            "Invalid shapes for matrices: A = {} B = {} C = {} D = {}".format(a.shape, b.shape, c.shape, d.shape)
        )

    if runtime.tape:
        runtime.tape.record_func(
            backward=lambda: adj_batched_matmul(
                a, b, c, a.grad, b.grad, c.grad, d.grad, alpha, beta, allow_tf32x3_arith
            ),
            arrays=[a, b, c, d],
        )

    # cpu fallback if no cuda devices found
    if device == "cpu":
        d.assign(alpha * np.matmul(a.numpy(), b.numpy()) + beta * c.numpy())
        return

    # handle case in which batch_count exceeds max_batch_count, which is a CUDA array size maximum
    max_batch_count = 65535
    iters = int(batch_count / max_batch_count)
    remainder = batch_count % max_batch_count

    cc = device.arch
    for i in range(iters):
        idx_start = i * max_batch_count
        idx_end = (i + 1) * max_batch_count if i < iters - 1 else batch_count
        ret = runtime.core.cutlass_gemm(
            device.context,
            cc,
            m,
            n,
            k,
            type_typestr(a.dtype).encode(),
            ctypes.c_void_p(a[idx_start:idx_end, :, :].ptr),
            ctypes.c_void_p(b[idx_start:idx_end, :, :].ptr),
            ctypes.c_void_p(c[idx_start:idx_end, :, :].ptr),
            ctypes.c_void_p(d[idx_start:idx_end, :, :].ptr),
            alpha,
            beta,
            not a.is_transposed,
            not b.is_transposed,
            allow_tf32x3_arith,
            max_batch_count,
        )
        if not ret:
            raise RuntimeError("Batched matmul failed.")

    idx_start = iters * max_batch_count
    ret = runtime.core.cutlass_gemm(
        device.context,
        cc,
        m,
        n,
        k,
        type_typestr(a.dtype).encode(),
        ctypes.c_void_p(a[idx_start:, :, :].ptr),
        ctypes.c_void_p(b[idx_start:, :, :].ptr),
        ctypes.c_void_p(c[idx_start:, :, :].ptr),
        ctypes.c_void_p(d[idx_start:, :, :].ptr),
        alpha,
        beta,
        not a.is_transposed,
        not b.is_transposed,
        allow_tf32x3_arith,
        remainder,
    )
    if not ret:
        raise RuntimeError("Batched matmul failed.")


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
    from warp.context import runtime

    device = a.device

    if (
        b.device != device
        or c.device != device
        or adj_a.device != device
        or adj_b.device != device
        or adj_c.device != device
        or adj_d.device != device
    ):
        raise RuntimeError(
            "Matrices A, B, C, D, and their adjoints must all be on the same device as the runtime device."
        )

    if (
        a.dtype != b.dtype
        or a.dtype != c.dtype
        or a.dtype != adj_a.dtype
        or a.dtype != adj_b.dtype
        or a.dtype != adj_c.dtype
        or a.dtype != adj_d.dtype
    ):
        raise RuntimeError(
            "wp.adj_batched_matmul currently only supports operation between {A, B, C, adj_D, adj_A, adj_B, adj_C} matrices of the same type."
        )

    m = a.shape[1]
    n = b.shape[2]
    k = a.shape[2]
    batch_count = a.shape[0]
    if (
        b.shape != (batch_count, k, n)
        or c.shape != (batch_count, m, n)
        or adj_d.shape != (batch_count, m, n)
        or adj_a.shape != (batch_count, m, k)
        or adj_b.shape != (batch_count, k, n)
        or adj_c.shape != (batch_count, m, n)
    ):
        raise RuntimeError(
            "Invalid shapes for matrices: A = {} B = {} C = {} adj_D = {} adj_A = {} adj_B = {} adj_C = {}".format(
                a.shape, b.shape, c.shape, adj_d.shape, adj_a.shape, adj_b.shape, adj_c.shape
            )
        )

    if (
        (not a.is_contiguous and not a.is_transposed)
        or (not b.is_contiguous and not b.is_transposed)
        or (not c.is_contiguous)
        or (not adj_a.is_contiguous and not adj_a.is_transposed)
        or (not adj_b.is_contiguous and not adj_b.is_transposed)
        or (not adj_c.is_contiguous)
        or (not adj_d.is_contiguous)
    ):
        raise RuntimeError(
            "wp.matmul is only valid for contiguous arrays, with the exception that A and/or B and their associated adjoints may be transposed."
        )

    # cpu fallback if no cuda devices found
    if device == "cpu":
        adj_a.assign(alpha * np.matmul(adj_d.numpy(), b.numpy().transpose((0, 2, 1))) + adj_a.numpy())
        adj_b.assign(alpha * np.matmul(a.numpy().transpose((0, 2, 1)), adj_d.numpy()) + adj_b.numpy())
        adj_c.assign(beta * adj_d.numpy() + adj_c.numpy())
        return

    # handle case in which batch_count exceeds max_batch_count, which is a CUDA array size maximum
    max_batch_count = 65535
    iters = int(batch_count / max_batch_count)
    remainder = batch_count % max_batch_count

    cc = device.arch

    for i in range(iters):
        idx_start = i * max_batch_count
        idx_end = (i + 1) * max_batch_count if i < iters - 1 else batch_count

        # adj_a
        if not a.is_transposed:
            ret = runtime.core.cutlass_gemm(
                device.context,
                cc,
                m,
                k,
                n,
                type_typestr(a.dtype).encode(),
                ctypes.c_void_p(adj_d[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(b[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(adj_a[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(adj_a[idx_start:idx_end, :, :].ptr),
                alpha,
                1.0,
                True,
                b.is_transposed,
                allow_tf32x3_arith,
                max_batch_count,
            )
            if not ret:
                raise RuntimeError("adj_matmul failed.")
        else:
            ret = runtime.core.cutlass_gemm(
                device.context,
                cc,
                k,
                m,
                n,
                type_typestr(a.dtype).encode(),
                ctypes.c_void_p(b[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(adj_d[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(adj_a[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(adj_a[idx_start:idx_end, :, :].ptr),
                alpha,
                1.0,
                not b.is_transposed,
                False,
                allow_tf32x3_arith,
                max_batch_count,
            )
            if not ret:
                raise RuntimeError("adj_matmul failed.")

        # adj_b
        if not b.is_transposed:
            ret = runtime.core.cutlass_gemm(
                device.context,
                cc,
                k,
                n,
                m,
                type_typestr(a.dtype).encode(),
                ctypes.c_void_p(a[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(adj_d[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(adj_b[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(adj_b[idx_start:idx_end, :, :].ptr),
                alpha,
                1.0,
                a.is_transposed,
                True,
                allow_tf32x3_arith,
                max_batch_count,
            )
            if not ret:
                raise RuntimeError("adj_matmul failed.")
        else:
            ret = runtime.core.cutlass_gemm(
                device.context,
                cc,
                n,
                k,
                m,
                type_typestr(a.dtype).encode(),
                ctypes.c_void_p(adj_d[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(a[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(adj_b[idx_start:idx_end, :, :].ptr),
                ctypes.c_void_p(adj_b[idx_start:idx_end, :, :].ptr),
                alpha,
                1.0,
                False,
                not a.is_transposed,
                allow_tf32x3_arith,
                max_batch_count,
            )
            if not ret:
                raise RuntimeError("adj_matmul failed.")

    idx_start = iters * max_batch_count

    # adj_a
    if not a.is_transposed:
        ret = runtime.core.cutlass_gemm(
            device.context,
            cc,
            m,
            k,
            n,
            type_typestr(a.dtype).encode(),
            ctypes.c_void_p(adj_d[idx_start:, :, :].ptr),
            ctypes.c_void_p(b[idx_start:, :, :].ptr),
            ctypes.c_void_p(adj_a[idx_start:, :, :].ptr),
            ctypes.c_void_p(adj_a[idx_start:, :, :].ptr),
            alpha,
            1.0,
            True,
            b.is_transposed,
            allow_tf32x3_arith,
            remainder,
        )
        if not ret:
            raise RuntimeError("adj_matmul failed.")
    else:
        ret = runtime.core.cutlass_gemm(
            device.context,
            cc,
            k,
            m,
            n,
            type_typestr(a.dtype).encode(),
            ctypes.c_void_p(b[idx_start:, :, :].ptr),
            ctypes.c_void_p(adj_d[idx_start:, :, :].ptr),
            ctypes.c_void_p(adj_a[idx_start:, :, :].ptr),
            ctypes.c_void_p(adj_a[idx_start:, :, :].ptr),
            alpha,
            1.0,
            not b.is_transposed,
            False,
            allow_tf32x3_arith,
            remainder,
        )
        if not ret:
            raise RuntimeError("adj_matmul failed.")

    # adj_b
    if not b.is_transposed:
        ret = runtime.core.cutlass_gemm(
            device.context,
            cc,
            k,
            n,
            m,
            type_typestr(a.dtype).encode(),
            ctypes.c_void_p(a[idx_start:, :, :].ptr),
            ctypes.c_void_p(adj_d[idx_start:, :, :].ptr),
            ctypes.c_void_p(adj_b[idx_start:, :, :].ptr),
            ctypes.c_void_p(adj_b[idx_start:, :, :].ptr),
            alpha,
            1.0,
            a.is_transposed,
            True,
            allow_tf32x3_arith,
            remainder,
        )
        if not ret:
            raise RuntimeError("adj_matmul failed.")
    else:
        ret = runtime.core.cutlass_gemm(
            device.context,
            cc,
            n,
            k,
            m,
            type_typestr(a.dtype).encode(),
            ctypes.c_void_p(adj_d[idx_start:, :, :].ptr),
            ctypes.c_void_p(a[idx_start:, :, :].ptr),
            ctypes.c_void_p(adj_b[idx_start:, :, :].ptr),
            ctypes.c_void_p(adj_b[idx_start:, :, :].ptr),
            alpha,
            1.0,
            False,
            not a.is_transposed,
            allow_tf32x3_arith,
            remainder,
        )
        if not ret:
            raise RuntimeError("adj_matmul failed.")

    # adj_c
    warp.launch(
        kernel=warp.utils.add_kernel_3d,
        dim=adj_c.shape,
        inputs=[adj_c, adj_d, adj_d.dtype(beta)],
        device=device,
        record_tape=False,
    )


class HashGrid:
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

        self.id = 0

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
    def __init__(self, nx: int, ny: int, nz: int, max_verts: int, max_tris: int, device=None):
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
        self.indices = zeros(max_tris * 3, dtype=int, device=self.device)

        # alloc surfacer
        self.id = ctypes.c_uint64(self.alloc(self.device.context))

    def __del__(self):
        if not self.id:
            return

        # use CUDA context guard to avoid side effects during garbage collection
        with self.device.context_guard:
            # destroy surfacer
            self.free(self.id)

    def resize(self, nx: int, ny: int, nz: int, max_verts: int, max_tris: int):
        # actual allocations will be resized on next call to surface()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.max_verts = max_verts
        self.max_tris = max_tris

    def surface(self, field: array(dtype=float), threshold: float):
        # WP_API int marching_cubes_surface_host(const float* field, int nx, int ny, int nz, float threshold, wp::vec3* verts, int* triangles, int max_verts, int max_tris, int* out_num_verts, int* out_num_tris);
        num_verts = ctypes.c_int(0)
        num_tris = ctypes.c_int(0)

        self.runtime.core.marching_cubes_surface_device.restype = ctypes.c_int

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
        elif arg_type in (int, float):
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


def get_type_code(arg_type):
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
        return warp.codegen.make_full_qualified_name(arg_type.cls)
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


def get_signature(arg_types, func_name=None, arg_names=None):
    type_codes = []
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
