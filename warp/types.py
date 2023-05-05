# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes
import hashlib
import inspect
import struct
import zlib
import numpy as np

from typing import Any
from typing import Tuple
from typing import TypeVar
from typing import Generic
from typing import List
from typing import Callable
from typing import Union

import warp

# type hints
Length = TypeVar("Length", bound=int)
Rows = TypeVar("Rows")
Cols = TypeVar("Cols")
DType = TypeVar("DType")

Int = TypeVar("Int")
Float = TypeVar("Float")
Scalar = TypeVar("Scalar")
Vector = Generic[Length, Scalar]
Matrix = Generic[Rows, Cols, Scalar]
Quaternion = Generic[Float]
Transformation = Generic[Float]

DType = TypeVar("DType")
Array = Generic[DType]

T = TypeVar("T")

# shared hash for all constants
_constant_hash = hashlib.sha256()


def constant(x):
    """Function to declare compile-time constants accessible from Warp kernels

    Args:
        x: Compile-time constant value, can be any of the built-in math types.
    """

    global _constant_hash

    # hash the constant value
    if isinstance(x, int):
        _constant_hash.update(struct.pack("<q", x))
    elif isinstance(x, float):
        _constant_hash.update(struct.pack("<d", x))
    elif isinstance(x, bool):
        _constant_hash.update(struct.pack("?", x))
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


# ----------------------
# built-in types


def vector(length, dtype):
    # canonicalize dtype
    if dtype == int:
        dtype = int32
    elif dtype == float:
        dtype = float32

    class vec_t(ctypes.Array):
        # ctypes.Array data for length, shape and c type:
        _length_ = 0 if length is Any else length
        _shape_ = (_length_,)
        _type_ = ctypes.c_float if dtype in [Scalar, Float] else dtype._type_

        # warp scalar type:
        _wp_scalar_type_ = dtype
        _wp_type_params_ = [length, dtype]
        _wp_generic_type_str_ = "vec_t"
        _wp_constructor_ = "vector"

        def __init__(self, *args):
            if self._wp_scalar_type_ == float16:
                # special case for float16 type: in this case, data is stored
                # as uint16 but it's actually half precision floating point
                # data. This means we need to convert each of the arguments
                # to uint16s containing half float bits before storing them in
                # the array:
                from warp.context import runtime

                scalar_value = runtime.core.float_to_half_bits
            else:
                scalar_value = lambda x: x

            num_args = len(args)
            if num_args == 0:
                super().__init__()
            elif num_args == 1:
                if hasattr(args[0], "__len__"):
                    # try to copy from expanded sequence, e.g. (1, 2, 3)
                    self.__init__(*args[0])
                else:
                    # set all elements to the same value
                    value = scalar_value(args[0])
                    for i in range(self._length_):
                        super().__setitem__(i, value)
            elif num_args == self._length_:
                # set all scalar elements
                for i in range(self._length_):
                    super().__setitem__(i, scalar_value(args[i]))
            else:
                raise ValueError(
                    f"Invalid number of arguments in vector constructor, expected {self._length_} elements, got {num_args}"
                )

        def __add__(self, y):
            return warp.add(self, y)

        def __radd__(self, y):
            return warp.add(self, y)

        def __sub__(self, y):
            return warp.sub(self, y)

        def __rsub__(self, x):
            return warp.sub(x, self)

        def __mul__(self, y):
            return warp.mul(self, y)

        def __rmul__(self, x):
            return warp.mul(x, self)

        def __div__(self, y):
            return warp.div(self, y)

        def __rdiv__(self, x):
            return warp.div(x, self)

        def __pos__(self, y):
            return warp.pos(self, y)

        def __neg__(self, y):
            return warp.neg(self, y)

        def __str__(self):
            return f"[{', '.join(map(str, self))}]"

    return vec_t


def matrix(shape, dtype):
    assert len(shape) == 2

    # canonicalize dtype
    if dtype == int:
        dtype = int32
    elif dtype == float:
        dtype = float32

    class mat_t(ctypes.Array):
        _length_ = 0 if shape[0] == Any or shape[1] == Any else shape[0] * shape[1]
        _shape_ = (0, 0) if _length_ == 0 else shape
        _type_ = ctypes.c_float if dtype in [Scalar, Float] else dtype._type_

        # warp scalar type:
        # used in type checking and when writing out c++ code for constructors:
        _wp_scalar_type_ = dtype
        _wp_type_params_ = [shape[0], shape[1], dtype]
        _wp_generic_type_str_ = "mat_t"
        _wp_constructor_ = "matrix"

        _wp_row_type_ = vector(0 if shape[1] == Any else shape[1], dtype)

        def __init__(self, *args):
            if self._wp_scalar_type_ == float16:
                # special case for float16 type: in this case, data is stored
                # as uint16 but it's actually half precision floating point
                # data. This means we need to convert each of the arguments
                # to uint16s containing half float bits before storing them in
                # the array:
                from warp.context import runtime

                scalar_value = runtime.core.float_to_half_bits
            else:
                scalar_value = lambda x: x

            num_args = len(args)
            if num_args == 0:
                super().__init__()
            elif num_args == 1:
                if hasattr(args[0], "__len__"):
                    # try to copy from expanded sequence, e.g. [[1, 0], [0, 1]]
                    self.__init__(*args[0])
                else:
                    # set all elements to the same value
                    value = scalar_value(args[0])
                    for i in range(self._length_):
                        super().__setitem__(i, value)
            elif num_args == self._length_:
                # set all scalar elements
                for i in range(self._length_):
                    super().__setitem__(i, scalar_value(args[i]))
            elif num_args == self._shape_[0]:
                # row vectors
                for i, row in enumerate(args):
                    if not hasattr(row, "__len__") or len(row) != self._shape_[1]:
                        raise TypeError(
                            f"Invalid argument in matrix constructor, expected row of length {self._shape_[1]}, got {row}"
                        )
                    offset = i * self._shape_[1]
                    for i in range(self._shape_[1]):
                        super().__setitem__(offset + i, scalar_value(row[i]))
            else:
                raise ValueError(
                    f"Invalid number of arguments in matrix constructor, expected {self._length_} elements, got {num_args}"
                )

        def __add__(self, y):
            return warp.add(self, y)

        def __radd__(self, y):
            return warp.add(self, y)

        def __sub__(self, y):
            return warp.sub(self, y)

        def __rsub__(self, x):
            return warp.sub(x, self)

        def __mul__(self, y):
            return warp.mul(self, y)

        def __rmul__(self, x):
            return warp.mul(x, self)

        def __div__(self, y):
            return warp.div(self, y)

        def __rdiv__(self, x):
            return warp.div(x, self)

        def __pos__(self, y):
            return warp.pos(self, y)

        def __neg__(self, y):
            return warp.neg(self, y)

        def __str__(self):
            row_str = []
            for r in range(self._shape_[0]):
                row_val = self.get_row(r)
                row_str.append(f"[{', '.join(map(str, row_val))}]")

            return "[" + ",\n ".join(row_str) + "]"

        def get_row(self, r):
            if r < 0 or r >= self._shape_[0]:
                raise IndexError("Invalid row index")
            row_start = r * self._shape_[1]
            row_end = row_start + self._shape_[1]
            return self._wp_row_type_(*super().__getitem__(slice(row_start, row_end)))

        def set_row(self, r, v):
            if r < 0 or r >= self._shape_[0]:
                raise IndexError("Invalid row index")
            row_start = r * self._shape_[1]
            row_end = row_start + self._shape_[1]
            super().__setitem__(slice(row_start, row_end), v)

        def __getitem__(self, key):
            if isinstance(key, Tuple):
                # element indexing m[i,j]
                return super().__getitem__(key[1] * self._shape_[0] + key[1])
            elif isinstance(key, int):
                # row vector indexing m[r]
                return self.get_row(key)
            else:
                # slice etc.
                return super().__getitem__(key)

        def __setitem__(self, key, value):
            if isinstance(key, Tuple):
                # element indexing m[i,j] = x
                return super().__setitem__(key[1] * self._shape_[0] + key[1], value)
            elif isinstance(key, int):
                # row vector indexing m[r] = v
                self.set_row(key, value)
                return value
            else:
                # slice etc.
                return super().__setitem__(key, value)

    return mat_t


class void:
    def __init__(self):
        pass


class float16:
    _length_ = 1
    _type_ = ctypes.c_uint16

    def __init__(self, x=0.0):
        self.value = x


class float32:
    _length_ = 1
    _type_ = ctypes.c_float

    def __init__(self, x=0.0):
        self.value = x


class float64:
    _length_ = 1
    _type_ = ctypes.c_double

    def __init__(self, x=0.0):
        self.value = x


class int8:
    _length_ = 1
    _type_ = ctypes.c_int8

    def __init__(self, x=0):
        self.value = x


class uint8:
    _length_ = 1
    _type_ = ctypes.c_uint8

    def __init__(self, x=0):
        self.value = x


class int16:
    _length_ = 1
    _type_ = ctypes.c_int16

    def __init__(self, x=0):
        self.value = x


class uint16:
    _length_ = 1
    _type_ = ctypes.c_uint16

    def __init__(self, x=0):
        self.value = x


class int32:
    _length_ = 1
    _type_ = ctypes.c_int32

    def __init__(self, x=0):
        self.value = x


class uint32:
    _length_ = 1
    _type_ = ctypes.c_uint32

    def __init__(self, x=0):
        self.value = x


class int64:
    _length_ = 1
    _type_ = ctypes.c_int64

    def __init__(self, x=0):
        self.value = x


class uint64:
    _length_ = 1
    _type_ = ctypes.c_uint64

    def __init__(self, x=0):
        self.value = x


def quaternion(dtype=Any):
    class quat_t(vector(length=4, dtype=dtype)):
        pass
        # def __init__(self, *args):
        #     super().__init__(args)

    ret = quat_t
    ret._wp_type_params_ = [dtype]
    ret._wp_generic_type_str_ = "quat_t"
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
        _wp_type_params_ = [dtype]
        _wp_generic_type_str_ = "transform_t"
        _wp_constructor_ = "transformation"

        def __init__(self, p=(0.0, 0.0, 0.0), q=(0.0, 0.0, 0.0, 1.0)):
            super().__init__()

            self[0:3] = vector(length=3, dtype=dtype)(*p)
            self[3:7] = quaternion(dtype=dtype)(*q)

        @property
        def p(self):
            return self[0:3]

        @property
        def q(self):
            return self[3:7]

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


int_types = [int8, uint8, int16, uint16, int32, uint32, int64, uint64]
float_types = [float16, float32, float64]
scalar_types = int_types + float_types

vector_types = [
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
]

np_dtype_to_warp_type = {
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


# represent a Python range iterator
class range_t:
    def __init__(self):
        pass


# definition just for kernel type (cannot be a parameter), see bvh.h
class bvh_query_t:
    def __init__(self):
        pass


# definition just for kernel type (cannot be a parameter), see mesh.h
class mesh_query_aabb_t:
    def __init__(self):
        pass


# definition just for kernel type (cannot be a parameter), see hash_grid.h
class hash_grid_query_t:
    def __init__(self):
        pass


# maximum number of dimensions, must match array.h
ARRAY_MAX_DIMS = 4
LAUNCH_MAX_DIMS = 4

# must match array.h
ARRAY_TYPE_REGULAR = 0
ARRAY_TYPE_INDEXED = 1


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

    _fields_ = [("data", ctypes.c_uint64),
                ("grad", ctypes.c_uint64),
                ("shape", ctypes.c_int32*ARRAY_MAX_DIMS),
                ("strides", ctypes.c_int32*ARRAY_MAX_DIMS),
                ("ndim", ctypes.c_int32)]
    
    def __init__(self, data=0, grad=0, ndim=0, shape=(0,), strides=(0,)):
        self.data = data
        self.grad = grad
        self.ndim = ndim
        for i in range(ndim):
            self.shape[i] = shape[i]
            self.strides[i] = strides[i]


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
    if dtype == float or dtype == int:
        return 1
    else:
        return dtype._length_


def type_size_in_bytes(dtype):
    if dtype.__module__ == "ctypes":
        return ctypes.sizeof(dtype)
    elif dtype == float or dtype == int:
        return 4
    elif hasattr(dtype, "_type_"):
        return getattr(dtype, "_length_", 1) * ctypes.sizeof(dtype._type_)
    else:
        return 0


def type_to_warp(dtype):
    if dtype == float:
        return float32
    elif dtype == int:
        return int32
    else:
        return dtype


def type_typestr(dtype):
    if dtype == float16:
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
    elif issubclass(dtype, ctypes.Array):
        return type_typestr(dtype._wp_scalar_type_)
    else:
        raise Exception("Unknown ctype")


def type_is_int(t):
    if t == int:
        t = int32

    return t in int_types


def type_is_float(t):
    if t == float:
        t = float32

    return t in float_types


# returns true for all value types (int, float, bool, scalars, vectors, matrices)
def type_is_value(x):
    if (x == int) or (x == float) or (x == bool) or (x in scalar_types) or issubclass(x, ctypes.Array):
        return True
    else:
        return False


# equivalent of the above but for values
def is_int(x):
    return type_is_int(type(x))


def is_float(x):
    return type_is_float(type(x))


def is_value(x):
    return type_is_value(type(x))


def types_equal(a, b, match_generic=False):
    # convert to canonical types
    if a == float:
        a = float32
    if a == int:
        a = int32

    if b == float:
        b = float32
    if b == int:
        b = int32

    def are_equal(p1, p2):
        if match_generic:
            if p1 == Any or p2 == Any:
                return True
            if p1 == Scalar and p2 in scalar_types:
                return True
            if p2 == Scalar and p1 in scalar_types:
                return True
            if p1 == Scalar and p2 == Scalar:
                return True
            if p1 == Float and p2 in float_types:
                return True
            if p2 == Float and p1 in float_types:
                return True
            if p1 == Float and p2 == Float:
                return True
        return p1 == p2

    if (
        hasattr(a, "_wp_generic_type_str_")
        and hasattr(b, "_wp_generic_type_str_")
        and a._wp_generic_type_str_ == b._wp_generic_type_str_
    ):
        return all([are_equal(p1, p2) for p1, p2 in zip(a._wp_type_params_, b._wp_type_params_)])
    if isinstance(a, array) and isinstance(b, array):
        return True
    if isinstance(a, indexedarray) and isinstance(b, indexedarray):
        return True
    else:
        return are_equal(a, b)


def strides_from_shape(shape: Tuple, dtype):
    ndims = len(shape)
    strides = [None] * ndims

    i = ndims - 1
    strides[i] = type_size_in_bytes(dtype)

    while i > 0:
        strides[i - 1] = strides[i] * shape[i]
        i -= 1

    return tuple(strides)


class array(Array):
    # member attributes available during code-gen (e.g.: d = array.shape[0])
    # (initialized when needed)
    _vars = None

    def __init__(
        self,
        data=None,
        dtype: DType = Any,
        shape=None,
        strides=None,
        length=0,
        ptr=None,
        grad_ptr=None,
        capacity=0,
        device=None,
        copy=True,
        owner=True,
        ndim=None,
        requires_grad=False,
        pinned=False,
    ):
        """Constructs a new Warp array object from existing data.

        When the ``data`` argument is a valid list, tuple, or ndarray the array will be constructed from this object's data.
        For objects that are not stored sequentially in memory (e.g.: a list), then the data will first
        be flattened before being transferred to the memory space given by device.

        The second construction path occurs when the ``ptr`` argument is a non-zero uint64 value representing the
        start address in memory where existing array data resides, e.g.: from an external or C-library. The memory
        allocation should reside on the same device given by the device argument, and the user should set the length
        and dtype parameter appropriately.

        Args:
            data (Union[list, tuple, ndarray]) An object to construct the array from, can be a Tuple, List, or generally any type convertible to an np.array
            dtype (Union): One of the built-in types, e.g.: :class:`warp.mat33`, if dtype is Any and data an ndarray then it will be inferred from the array data type
            shape (tuple): Dimensions of the array
            strides (tuple): Number of bytes in each dimension between successive elements of the array
            length (int): Number of elements (rows) of the data type (deprecated, users should use `shape` argument)
            ptr (uint64): Address of an external memory address to alias (data should be None)
            grad_ptr (uint64): Address of an external memory address to alias for the gradient array
            capacity (int): Maximum size in bytes of the ptr allocation (data should be None)
            device (Devicelike): Device the array lives on
            copy (bool): Whether the incoming data will be copied or aliased, this is only possible when the incoming `data` already lives on the device specified and types match
            owner (bool): Should the array object try to deallocate memory when it is deleted
            requires_grad (bool): Whether or not gradients will be tracked for this array, see :class:`warp.Tape` for details
            pinned (bool): Whether to allocate pinned host memory, which allows asynchronous host-device transfers (only applicable with device="cpu")

        """

        self.owner = False

        # convert shape to Tuple
        if shape is None:
            shape = tuple(length for _ in range(ndim or 1))
        elif isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, List):
            shape = tuple(shape)

        self.shape = shape

        if len(shape) > ARRAY_MAX_DIMS:
            raise RuntimeError(
                f"Arrays may only have {ARRAY_MAX_DIMS} dimensions maximum, trying to create array with {len(shape)} dims."
            )

        # canonicalize dtype
        if dtype == int:
            dtype = int32
        elif dtype == float:
            dtype = float32

        if data is not None or ptr is not None:
            from .context import runtime

            device = runtime.get_device(device)

        if data is not None:
            if device.is_capturing:
                raise RuntimeError(f"Cannot allocate memory on device {device} while graph capture is active")

            if ptr is not None:
                # data or ptr, not both
                raise RuntimeError("Should only construct arrays with either data or ptr arguments, not both")

            try:
                # force convert tuples and lists (or any array type) to ndarray
                arr = np.array(data, copy=False)
            except Exception as e:
                raise RuntimeError(
                    "When constructing an array the data argument must be convertible to ndarray type type. Encountered an error while converting:"
                    + str(e)
                )

            if dtype == Any:
                # infer dtype from the source data array
                dtype = np_dtype_to_warp_type[arr.dtype]

            try:
                # try to convert src array to destination type
                arr = arr.astype(dtype=type_typestr(dtype), copy=False)
            except:
                raise RuntimeError(
                    f"Could not convert input data with type {arr.dtype} to array with type {dtype._type_}"
                )

            # ensure contiguous
            arr = np.ascontiguousarray(arr)

            # remove any trailing dimensions of length 1
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=len(arr.shape) - 1)

            ptr = arr.__array_interface__["data"][0]
            shape = arr.__array_interface__["shape"]
            strides = arr.__array_interface__.get("strides", None)

            # Convert input shape to Warp
            if type_length(dtype) > 1:
                # if we are constructing an array of vectors/matrices, but input
                # is one dimensional (i.e.: flattened) then try and reshape to
                # to match target dtype, inferring the first dimension
                if arr.ndim == 1:
                    arr = arr.reshape((-1, *dtype._shape_))

                # last dimension should match dtype shape when using vector types,
                # e.g.: array of mat22 objects should have shape (n, 2, 2)
                dtype_ndim = len(dtype._shape_)

                trailing_shape = arr.shape[-dtype_ndim:]
                leading_shape = arr.shape[0:-dtype_ndim]

                if dtype._shape_ != trailing_shape:
                    raise RuntimeError(
                        f"Last dimensions of input array should match the specified data type, given shape {arr.shape}, expected last dimensions to match dtype shape {dtype._shape_}"
                    )

                shape = leading_shape

                if strides is not None:
                    strides = strides[0:-dtype_ndim]

            if device.is_cpu and copy == False:
                # ref numpy memory directly
                self.shape = shape
                self.ptr = ptr
                self.grad_ptr = grad_ptr
                self.dtype = dtype
                self.strides = strides
                self.capacity = arr.size * type_size_in_bytes(dtype)
                self.device = device
                self.owner = False
                self.pinned = False

                # keep a ref to source array to keep allocation alive
                self.ref = arr

            else:
                # otherwise, we must transfer to device memory
                # create a host wrapper around the numpy array
                # and a new destination array to copy it to
                src = array(
                    dtype=dtype,
                    shape=shape,
                    strides=strides,
                    capacity=arr.size * type_size_in_bytes(dtype),
                    ptr=ptr,
                    device="cpu",
                    copy=False,
                    owner=False,
                )
                dest = warp.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad, pinned=pinned)
                dest.owner = False

                # copy data using the CUDA default stream for synchronous behaviour with other streams
                warp.copy(dest, src, stream=device.null_stream)

                # object copy to self and transfer data ownership, would probably be cleaner to have _empty, _zero, etc as class methods
                from copy import copy as shallowcopy

                self.__dict__ = shallowcopy(dest.__dict__)
                self.owner = True

        else:
            # explicit construction from ptr to external memory
            self.shape = shape
            self.strides = strides
            self.capacity = capacity
            self.dtype = dtype
            self.ptr = ptr
            self.grad_ptr = grad_ptr
            self.device = device
            self.owner = owner
            if device is not None and device.is_cpu:
                self.pinned = pinned
            else:
                self.pinned = False

            self.__name__ = "array<" + type.__name__ + ">"

        # update ndim
        if ndim is None:
            self.ndim = len(self.shape)
        else:
            self.ndim = ndim

        # update size (num elements)
        self.size = 1
        for d in self.shape:
            self.size *= d

        self._grad = None

        # set up array interface access so we can treat this object as a numpy array
        if self.ptr:
            # update byte strides and contiguous flag
            contiguous_strides = strides_from_shape(self.shape, self.dtype)
            if strides is None:
                self.strides = contiguous_strides
                self.is_contiguous = True
            else:
                self.strides = strides
                self.is_contiguous = strides[:ndim] == contiguous_strides[:ndim]

            # store flat shape (including type shape)
            if self.dtype not in [Any, Scalar, Float, Int] and issubclass(dtype, ctypes.Array):
                # vector type, flatten the dimensions into one tuple
                arr_shape = (*self.shape, *self.dtype._shape_)
                dtype_strides = strides_from_shape(self.dtype._shape_, self.dtype._type_)
                arr_strides = (*self.strides, *dtype_strides)
            else:
                # scalar type
                arr_shape = self.shape
                arr_strides = self.strides

            if device.is_cpu:
                self.__array_interface__ = {
                    "data": (self.ptr, False),
                    "shape": tuple(arr_shape),
                    "strides": tuple(arr_strides),
                    "typestr": type_typestr(self.dtype),
                    "version": 3,
                }

            # set up cuda array interface access so we can treat this object as a Torch tensor
            elif device.is_cuda:
                self.__cuda_array_interface__ = {
                    "data": (self.ptr, False),
                    "shape": tuple(arr_shape),
                    "strides": tuple(arr_strides),
                    "typestr": type_typestr(self.dtype),
                    "version": 2,
                }

            # controls if gradients will be computed by wp.Tape
            # this will trigger allocation of a gradient array if it doesn't exist already
            self.requires_grad = requires_grad

        else:
            # array has no data
            self.strides = (0,) * self.ndim
            self.is_contiguous = False
            self.requires_grad = False

        self.ctype = None

    def __del__(self):
        if self.owner and self.device is not None and self.ptr is not None:
            # TODO: ill-timed gc could trigger superfluous context switches here
            #       Delegate to a separate thread? (e.g., device_free_async)
            if self.device.is_capturing:
                raise RuntimeError(f"Cannot free memory on device {self.device} while graph capture is active")

            # use CUDA context guard to avoid side effects during garbage collection
            with self.device.context_guard:
                self.device.allocator.free(self.ptr, self.capacity, self.pinned)

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        if self.device is None:
            # for 'empty' arrays we just return the type information, these are used in kernel function signatures
            return f"array{self.dtype}"
        else:
            return str(self.to("cpu").numpy())

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
        for i in range(len(key), self.ndim):
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

                if start < 0 or start > self.shape[idx] - 1:
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
                if start < 0 or start > self.shape[idx] - 1:
                    raise RuntimeError(f"Invalid indexing in slice: {k}")
                new_dim -= 1

                ptr_offset += self.strides[idx] * start

        a = array(
            dtype=self.dtype,
            shape=tuple(new_shape),
            strides=tuple(new_strides),
            ptr=self.ptr + ptr_offset,
            grad_ptr=(self.grad_ptr + ptr_offset if self.grad_ptr is not None else None),
            capacity=self.capacity,
            device=self.device,
            owner=False,
            ndim=new_dim,
            requires_grad=self.requires_grad,
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
            grad = 0 if self.grad_ptr is None else ctypes.c_uint64(self.grad_ptr)
            self.ctype = array_t(data=data, grad=grad, ndim=self.ndim, shape=self.shape, strides=self.strides)

        return self.ctype

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        # trigger re-creation of C-representation
        self.ctype = None
        if value is None:
            self.grad_ptr = None
            self._grad = None
            return
        if self._grad is None:
            self.grad_ptr = value.ptr
            self._grad = value
        else:
            self._grad.assign(value)

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        if value and self._grad is None:
            self._alloc_grad()
        elif not value:
            self._grad = None

        self._requires_grad = value

    def _alloc_grad(self):
        if self.grad_ptr is None:
            num_bytes = self.size * type_size_in_bytes(self.dtype)
            self.grad_ptr = self.device.allocator.alloc(num_bytes, pinned=self.pinned)
            if self.grad_ptr is None:
                raise RuntimeError("Memory allocation failed on device: {} for {} bytes".format(self.device, num_bytes))
            with warp.ScopedStream(self.device.null_stream):
                self.device.memset(self.grad_ptr, 0, num_bytes)

        self._grad = array(ptr=self.grad_ptr, shape=self.shape, dtype=self.dtype, device=self.device, requires_grad=False, owner=False)
        # trigger re-creation of C-representation
        self.ctype = None

    @property
    def vars(self):
        # member attributes available during code-gen (e.g.: d = array.shape[0])
        # Note: we use a shared dict for all array instances
        if array._vars is None:
            from warp.codegen import Var

            array._vars = {"shape": Var("shape", shape_t)}
        return array._vars

    def zero_(self):
        if not self.is_contiguous:
            raise RuntimeError(f"Assigning to non-contiguous arrays is unsupported.")

        if self.device is not None and self.ptr is not None:
            self.device.memset(
                ctypes.c_void_p(self.ptr), ctypes.c_int(0), ctypes.c_size_t(self.size * type_size_in_bytes(self.dtype))
            )

    def fill_(self, value):
        if not self.is_contiguous:
            raise RuntimeError(f"Assigning to non-contiguous arrays is unsupported.")

        if self.device is not None and self.ptr is not None:
            if isinstance(value, ctypes.Array):
                # in this case we're filling the array with a vector or
                # something similar, eg arr.fill_(wp.vec3(1.0,2.0,3.0)).

                # check input type:
                value_type_ok = False
                if issubclass(self.dtype, ctypes.Array):
                    value_type_ok = (self.dtype._length_ == value._length_) and (self.dtype._type_ == value._type_)
                if not value_type_ok:
                    raise RuntimeError(
                        f"wp.array has Array type elements (eg vec, mat etc). Value type must match element type in wp.array.fill_() method"
                    )

                src = ctypes.cast(value, ctypes.POINTER(ctypes.c_void_p))

                srcsize = value._length_ * ctypes.sizeof(value._type_)
                dst = ctypes.cast(self.ptr, ctypes.POINTER(ctypes.c_int))
                self.device.memtile(dst, src, srcsize, self.size)

            else:
                # In this case we're just filling the array with a scalar,
                # eg arr.fill_(1.0). If the elements are scalars, we need to
                # set them all to "value", otherwise we need to set all the
                # components of all the vector elements to "value":

                # work out array element type:
                elem_type = self.dtype._type_ if issubclass(self.dtype, ctypes.Array) else type_ctype(self.dtype)
                elem_size = ctypes.sizeof(elem_type)

                # convert value to array type
                # we need a special case for float16 because it's annoying...
                if types_equal(self.dtype, float16) or (
                    hasattr(self.dtype, "_wp_scalar_type_") and types_equal(self.dtype._wp_scalar_type_, float16)
                ):
                    # special case for float16:
                    # If you just do elem_type(value), it'll just convert "value"
                    # to uint16 then interpret the bits as float16, which will
                    # mess the data up. Instead, we use float_to_half_bits() to
                    # convert "value" to a float16 and return its bits in a uint16:

                    from warp.context import runtime

                    src_value = elem_type(runtime.core.float_to_half_bits(ctypes.c_float(value)))
                else:
                    src_value = elem_type(value)

                # use memset for these special cases because it's quicker (probably...):
                total_bytes = self.size * type_size_in_bytes(self.dtype)
                if elem_size in [1, 2, 4] and (total_bytes % 4 == 0):
                    # interpret as a 4 byte integer:
                    dest_value = ctypes.cast(ctypes.pointer(src_value), ctypes.POINTER(ctypes.c_int)).contents
                    if elem_size == 1:
                        # need to repeat the bits, otherwise we'll get an array interleaved with zeros:
                        dest_value.value = dest_value.value & 0x000000FF
                        dest_value.value = (
                            dest_value.value
                            + (dest_value.value << 8)
                            + (dest_value.value << 16)
                            + (dest_value.value << 24)
                        )
                    elif elem_size == 2:
                        # need to repeat the bits, otherwise we'll get an array interleaved with zeros:
                        dest_value.value = dest_value.value & 0x0000FFFF
                        dest_value.value = dest_value.value + (dest_value.value << 16)

                    self.device.memset(
                        ctypes.cast(self.ptr, ctypes.POINTER(ctypes.c_int)), dest_value, ctypes.c_size_t(total_bytes)
                    )
                else:
                    num_elems = self.size * self.dtype._length_ if issubclass(self.dtype, ctypes.Array) else self.size
                    dst = ctypes.cast(self.ptr, ctypes.POINTER(ctypes.c_int))
                    self.device.memtile(dst, ctypes.pointer(src_value), elem_size, num_elems)

    # equivalent to wrapping src data in an array and copying to self
    def assign(self, src):
        if isinstance(src, array):
            warp.copy(self, src)
        else:
            warp.copy(self, array(src, dtype=self.dtype, copy=False, device="cpu"))

    # convert array to ndarray (alias memory through array interface)
    def numpy(self):
        # use the CUDA default stream for synchronous behaviour with other streams
        with warp.ScopedStream(self.device.null_stream):
            if self.ptr is None:
                return np.empty(shape=self.shape, dtype=self.dtype)
            else:
                return np.array(self.to("cpu"), copy=False)

    # convert data from one device to another, nop if already on device
    def to(self, device):
        device = warp.get_device(device)
        if self.device == device:
            return self
        else:
            dest = warp.empty(shape=self.shape, dtype=self.dtype, device=device, requires_grad=self.requires_grad)
            # to copy between devices, array must be contiguous
            warp.copy(dest, self.contiguous())
            return dest

    def flatten(self):
        if not self.is_contiguous:
            raise RuntimeError(f"Flattening non-contiguous arrays is unsupported.")

        a = array(
            dtype=self.dtype,
            shape=(self.size,),
            strides=(type_size_in_bytes(self.dtype),),
            ptr=self.ptr,
            capacity=self.capacity,
            device=self.device,
            copy=False,
            owner=False,
            ndim=1,
            requires_grad=self.requires_grad,
        )

        # store back-ref to stop data being destroyed
        a._ref = self
        return a

    def reshape(self, shape):
        if not self.is_contiguous:
            raise RuntimeError("Reshaping non-contiguous arrays is unsupported.")

        # convert shape to tuple
        if shape is None:
            raise RuntimeError("shape parameter is required.")
        if isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, List):
            shape = tuple(shape)

        if len(shape) > ARRAY_MAX_DIMS:
            raise RuntimeError(
                f"Arrays may only have {ARRAY_MAX_DIMS} dimensions maximum, trying to create array with {len(shape)} dims."
            )

        size = 1
        for d in shape:
            size *= d

        if size != self.size:
            raise RuntimeError("Reshaped array must have the same total size as the original.")

        a = array(
            dtype=self.dtype,
            shape=shape,
            strides=None,
            ptr=self.ptr,
            grad_ptr=self.grad_ptr,
            capacity=self.capacity,
            device=self.device,
            copy=False,
            owner=False,
            ndim=len(shape),
            requires_grad=self.requires_grad,
        )

        # store back-ref to stop data being destroyed
        a._ref = self
        return a

    def view(self, dtype):
        if type_size_in_bytes(dtype) != type_size_in_bytes(self.dtype):
            raise RuntimeError("cannot reinterpret cast dtypes of unequal byte size")
        else:
            # return an alias of the array memory with different type information
            a = array(
                data=None,
                dtype=dtype,
                shape=self.shape,
                strides=self.strides,
                ptr=self.ptr,
                grad_ptr=self.grad_ptr,
                capacity=self.capacity,
                device=self.device,
                copy=False,
                owner=False,
                ndim=self.ndim,
                requires_grad=self.requires_grad,
            )

            a._ref = self
            return a

    def contiguous(self):
        if self.is_contiguous:
            return self

        a = warp.empty_like(self)
        warp.copy(a, self)

        return a

    # note: transpose operation will return an array with a non-contiguous access pattern
    def transpose(self, axes=None):
        # noop if 1d array
        if len(self.shape) == 1:
            return self

        if axes is None:
            # reverse the order of the axes
            axes = range(self.ndim)[::-1]

        if len(axes) != len(self.shape):
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
            data=None,
            dtype=self.dtype,
            shape=tuple(shape),
            strides=tuple(strides),
            ptr=self.ptr,
            grad_ptr=self.grad_ptr,
            capacity=self.capacity,
            device=self.device,
            copy=False,
            owner=False,
            ndim=self.ndim,
            requires_grad=self.requires_grad,
        )

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
    return array(
        dtype=dtype,
        length=length,
        capacity=length * type_size_in_bytes(dtype),
        ptr=ctypes.cast(ptr, ctypes.POINTER(ctypes.c_size_t)).contents.value,
        shape=shape,
        device=device,
        owner=False,
        requires_grad=False,
    )


class indexedarray(Generic[T]):
    # member attributes available during code-gen (e.g.: d = arr.shape[0])
    # (initialized when needed)
    _vars = None

    def __init__(
        self, data: array = None, indices: Union[array, List[array]] = None, dtype=None, ndim=None
    ):
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
                # helper to check index array properties
                def check_index_array(inds, data):
                    if inds.ndim != 1:
                        raise ValueError(f"Index array must be one-dimensional, got {inds.ndim}")
                    if inds.dtype != int32:
                        raise ValueError(f"Index array must use int32, got dtype {inds.dtype}")
                    if inds.device != data.device:
                        raise ValueError(
                            f"Index array device ({inds.device} does not match data array device ({data.device}))"
                        )

                if isinstance(indices, (list, tuple)):
                    if len(indices) > self.ndim:
                        raise ValueError(
                            f"Number of indices provided ({len(indices)}) exceeds number of dimensions ({self.ndim})"
                        )

                    for i in range(len(indices)):
                        if isinstance(indices[i], array):
                            check_index_array(indices[i], data)
                            self.indices[i] = indices[i]
                            shape[i] = len(indices[i])
                        elif indices[i] is not None:
                            raise TypeError(f"Invalid index array type: {type(indices[i])}")

                elif isinstance(indices, array):
                    # only a single index array was provided
                    check_index_array(indices, data)
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

        self.is_contiguous = False

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return f"indexedarray{self.dtype}"

    # construct a C-representation of the array for passing to kernels
    def __ctype__(self):
        return indexedarray_t(self.data, self.indices, self.shape)

    @property
    def vars(self):
        # member attributes available during code-gen (e.g.: d = arr.shape[0])
        # Note: we use a shared dict for all indexedarray instances
        if indexedarray._vars is None:
            from warp.codegen import Var

            indexedarray._vars = {"shape": Var("shape", shape_t)}
        return indexedarray._vars

    def contiguous(self):
        a = warp.empty_like(self)
        warp.copy(a, self)

        return a

    # convert data from one device to another, nop if already on device
    def to(self, device):
        device = warp.get_device(device)
        if self.device == device:
            return self
        else:
            dest = warp.empty(shape=self.shape, dtype=self.dtype, device=device)
            # to copy between devices, array must be contiguous
            warp.copy(dest, self.contiguous())
            return dest

    # convert array to ndarray (alias memory through array interface)
    def numpy(self):
        # use the CUDA default stream for synchronous behaviour with other streams
        with warp.ScopedStream(self.device.null_stream):
            return np.array(self.contiguous().to("cpu"), copy=False)


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


array_types = (array, indexedarray)


def is_array(a):
    return isinstance(a, array_types)


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

        from warp.context import runtime

        if self.device.is_cpu:
            self.id = runtime.core.bvh_create_host(get_data(lowers), get_data(uppers), int(len(lowers)))
        else:
            self.id = runtime.core.bvh_create_device(
                self.device.context, get_data(lowers), get_data(uppers), int(len(lowers))
            )

    def __del__(self):
        try:
            from warp.context import runtime

            if self.device.is_cpu:
                runtime.core.bvh_destroy_host(self.id)
            else:
                # use CUDA context guard to avoid side effects during garbage collection
                with self.device.context_guard:
                    runtime.core.bvh_destroy_device(self.id)

        except:
            pass

    def refit(self):
        """Refit the Bvh. This should be called after users modify the `lowers` and `uppers` arrays."""

        from warp.context import runtime

        if self.device.is_cpu:
            runtime.core.bvh_refit_host(self.id)
        else:
            runtime.core.bvh_refit_device(self.id)
            runtime.verify_cuda_device(self.device)


class Mesh:
    from warp.codegen import Var

    vars = {
        "points": Var("points", array(dtype=vec3)),
        "velocities": Var("velocities", array(dtype=vec3)),
        "indices": Var("indices", array(dtype=int32, ndim=2)),
    }

    def __init__(self, points=None, indices=None, velocities=None):
        """Class representing a triangle mesh.

        Attributes:
            id: Unique identifier for this mesh object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            points (:class:`warp.array`): Array of vertex positions of type :class:`warp.vec3`
            indices (:class:`warp.array`): Array of triangle indices of type :class:`warp.int32`, should be length 3*number of triangles
            velocities (:class:`warp.array`): Array of vertex velocities of type :class:`warp.vec3` (optional)
        """

        if points.device != indices.device:
            raise RuntimeError("Mesh points and indices must live on the same device")

        if points.dtype != vec3 or not points.is_contiguous:
            raise RuntimeError("Mesh points should be a contiguous array of type wp.vec3")

        if velocities and (velocities.dtype != vec3 or not velocities.is_contiguous):
            raise RuntimeError("Mesh velocities should be a contiguous array of type wp.vec3")

        if indices.dtype != int32 or not indices.is_contiguous:
            raise RuntimeError("Mesh indices should be a contiguous array of type wp.int32")

        self.device = points.device
        self.points = points
        self.velocities = velocities
        self.indices = indices

        from warp.context import runtime

        if self.device.is_cpu:
            self.id = runtime.core.mesh_create_host(
                points.__ctype__(),
                velocities.__ctype__() if velocities else array().__ctype__(),
                indices.__ctype__(),
                int(len(points)),
                int(indices.size / 3))
        else:
            self.id = runtime.core.mesh_create_device(
                self.device.context,
                points.__ctype__(),
                velocities.__ctype__() if velocities else array().__ctype__(),
                indices.__ctype__(),
                int(len(points)),
                int(indices.size / 3))

    def __del__(self):
        try:
            from warp.context import runtime

            if self.device.is_cpu:
                runtime.core.mesh_destroy_host(self.id)
            else:
                # use CUDA context guard to avoid side effects during garbage collection
                with self.device.context_guard:
                    runtime.core.mesh_destroy_device(self.id)
        except:
            pass

    def refit(self):
        """Refit the BVH to points. This should be called after users modify the `points` data."""

        from warp.context import runtime

        if self.device.is_cpu:
            runtime.core.mesh_refit_host(self.id)
        else:
            runtime.core.mesh_refit_device(self.id)
            runtime.verify_cuda_device(self.device)


class Volume:
    CLOSEST = constant(0)
    LINEAR = constant(1)

    def __init__(self, data: array):
        """Class representing a sparse grid.

        Attributes:
            CLOSEST (int): Enum value to specify nearest-neighbor interpolation during sampling
            LINEAR (int): Enum value to specify trilinear interpolation during sampling

        Args:
            data (:class:`warp.array`): Array of bytes representing the volume in NanoVDB format
        """

        self.id = 0

        from warp.context import runtime

        self.context = runtime

        if data is None:
            return

        if data.device is None:
            raise RuntimeError("Invalid device")
        self.device = data.device

        if self.device.is_cpu:
            self.id = self.context.core.volume_create_host(ctypes.cast(data.ptr, ctypes.c_void_p), data.size)
        else:
            self.id = self.context.core.volume_create_device(
                self.device.context, ctypes.cast(data.ptr, ctypes.c_void_p), data.size
            )

        if self.id == 0:
            raise RuntimeError("Failed to create volume from input array")

    def __del__(self):
        if self.id == 0:
            return

        try:
            from warp.context import runtime

            if self.device.is_cpu:
                runtime.core.volume_destroy_host(self.id)
            else:
                # use CUDA context guard to avoid side effects during garbage collection
                with self.device.context_guard:
                    runtime.core.volume_destroy_device(self.id)

        except:
            pass

    def array(self):
        buf = ctypes.c_void_p(0)
        size = ctypes.c_uint64(0)
        if self.device.is_cpu:
            self.context.core.volume_get_buffer_info_host(self.id, ctypes.byref(buf), ctypes.byref(size))
        else:
            self.context.core.volume_get_buffer_info_device(self.id, ctypes.byref(buf), ctypes.byref(size))
        return array(ptr=buf.value, dtype=uint8, length=size.value, device=self.device, owner=False)

    def get_tiles(self):
        if self.id == 0:
            raise RuntimeError("Invalid Volume")

        buf = ctypes.c_void_p(0)
        size = ctypes.c_uint64(0)
        if self.device.is_cpu:
            self.context.core.volume_get_tiles_host(self.id, ctypes.byref(buf), ctypes.byref(size))
        else:
            self.context.core.volume_get_tiles_device(self.id, ctypes.byref(buf), ctypes.byref(size))
        num_tiles = size.value // (3 * 4)
        return array(
            ptr=buf.value, dtype=int32, shape=(num_tiles, 3), length=size.value, device=self.device, owner=True
        )

    def get_voxel_size(self):
        if self.id == 0:
            raise RuntimeError("Invalid Volume")

        dx, dy, dz = ctypes.c_float(0), ctypes.c_float(0), ctypes.c_float(0)
        self.context.core.volume_get_voxel_size(self.id, ctypes.byref(dx), ctypes.byref(dy), ctypes.byref(dz))
        return (dx.value, dy.value, dz.value)

    @classmethod
    def load_from_nvdb(cls, file_or_buffer, device=None):
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
    def allocate(
        cls,
        min: List[int],
        max: List[int],
        voxel_size: float,
        bg_value=0.0,
        translation=(0.0, 0.0, 0.0),
        points_in_world_space=False,
        device=None,
    ):
        """Allocate a new Volume based on the bounding box defined by min and max.

        Allocate a volume that is large enough to contain voxels [min[0], min[1], min[2]] - [max[0], max[1], max[2]], inclusive.
        If points_in_world_space is true, then min and max are first converted to index space with the given voxel size and
        translation, and the volume is allocated with those.

        The smallest unit of allocation is a dense tile of 8x8x8 voxels, the requested bounding box is rounded up to tiles, and
        the resulting tiles will be available in the new volume.

        Args:
            min (array-like): Lower 3D-coordinates of the bounding box in index space or world space, inclusive
            max (array-like): Upper 3D-coordinates of the bounding box in index space or world space, inclusive
            voxel_size (float): Voxel size of the new volume
            bg_value (float or array-like): Value of unallocated voxels of the volume, also defines the volume's type, a :class:`warp.vec3` volume is created if this is `array-like`, otherwise a float volume is created
            translation (array-like): translation between the index and world spaces
            device (Devicelike): Device the array lives on

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
    ):
        """Allocate a new Volume with active tiles for each point tile_points.

        The smallest unit of allocation is a dense tile of 8x8x8 voxels.
        This is the primary method for allocating sparse volumes. It uses an array of points indicating the tiles that must be allocated.

        Example use cases:
            * `tile_points` can mark tiles directly in index space as in the case this method is called by `allocate`.
            * `tile_points` can be a list of points used in a simulation that needs to transfer data to a volume.

        Args:
            tile_points (:class:`warp.array`): Array of positions that define the tiles to be allocated.
                The array can be a 2d, N-by-3 array of :class:`warp.int32` values, indicating index space positions,
                or can be a 1D array of :class:`warp.vec3` values, indicating world space positions.
                Repeated points per tile are allowed and will be efficiently deduplicated.
            voxel_size (float): Voxel size of the new volume
            bg_value (float or array-like): Value of unallocated voxels of the volume, also defines the volume's type, a :class:`warp.vec3` volume is created if this is `array-like`, otherwise a float volume is created
            translation (array-like): translation between the index and world spaces
            device (Devicelike): Device the array lives on

        """
        from warp.context import runtime

        device = runtime.get_device(device)

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
            volume.id = volume.context.core.volume_v_from_tiles_device(
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
        elif type(bg_value) == int:
            volume.id = volume.context.core.volume_i_from_tiles_device(
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
            volume.id = volume.context.core.volume_f_from_tiles_device(
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


def matmul(
    a: array2d,
    b: array2d,
    c: array2d,
    d: array2d,
    alpha: float = 1.0,
    beta: float = 0.0,
    allow_tf32x3_arith: bool = False,
    device=None,
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
        device: device we want to use to multiply matrices. Defaults to active runtime device. If "cpu", resorts to using numpy multiplication.
    """
    from warp.context import runtime

    if device is None:
        device = runtime.get_device(device)

    if a.device != device or b.device != device or c.device != device or d.device != device:
        raise RuntimeError("Matrices A, B, C, and D must all be on the same device as the runtime device.")

    if a.dtype != b.dtype or a.dtype != c.dtype or a.dtype != d.dtype:
        raise RuntimeError(
            "wp.matmul currently only supports operation between {A, B, C, D} matrices of the same type."
        )

    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    if b.shape != (k, n) or c.shape != (m, n) or d.shape != (m, n):
        raise RuntimeError(
            "Invalid shapes for matrices: A = {} B = {} C = {} D = {}".format(a.shape, b.shape, c.shape, d.shape)
        )

    # cpu fallback if no cuda devices found
    if device == "cpu":
        d.assign(alpha * (a.numpy() @ b.numpy()) + beta * c.numpy())
        return

    cc = device.arch
    ret = runtime.core.cutlass_gemm(
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
        True,
        True,
        allow_tf32x3_arith,
        1,
    )
    if not ret:
        raise RuntimeError("Matmul failed.")

    if runtime.tape:
        runtime.tape.record_func(
            backward=lambda: adj_matmul(
                a, b, c, a.grad, b.grad, c.grad, d.grad, alpha, beta, allow_tf32x3_arith, device
            ),
            arrays=[a, b, c, d],
        )


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
    allow_tf32x3_arith: bool = False,
    device=None,
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
        device: device we want to use to multiply matrices. Defaults to active runtime device. If "cpu", resorts to using numpy multiplication.
    """
    from warp.context import runtime

    if device is None:
        device = runtime.get_device(device)

    if (
        a.device != device
        or b.device != device
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
        adj_a.assign(alpha * np.matmul(adj_d.numpy(), b.numpy().transpose()))
        adj_b.assign(alpha * (a.numpy().transpose() @ adj_d.numpy()))
        adj_c.assign(beta * adj_d.numpy())
        return

    cc = device.arch

    # adj_a
    ret = runtime.core.cutlass_gemm(
        cc,
        m,
        k,
        n,
        type_typestr(a.dtype).encode(),
        ctypes.c_void_p(adj_d.ptr),
        ctypes.c_void_p(b.ptr),
        ctypes.c_void_p(a.ptr),
        ctypes.c_void_p(adj_a.ptr),
        alpha,
        0.0,
        True,
        False,
        allow_tf32x3_arith,
        1,
    )
    if not ret:
        raise RuntimeError("adj_matmul failed.")

    # adj_b
    ret = runtime.core.cutlass_gemm(
        cc,
        k,
        n,
        m,
        type_typestr(a.dtype).encode(),
        ctypes.c_void_p(a.ptr),
        ctypes.c_void_p(adj_d.ptr),
        ctypes.c_void_p(b.ptr),
        ctypes.c_void_p(adj_b.ptr),
        alpha,
        0.0,
        False,
        True,
        allow_tf32x3_arith,
        1,
    )
    if not ret:
        raise RuntimeError("adj_matmul failed.")

    # adj_c
    ret = runtime.core.cutlass_gemm(
        cc,
        m,
        n,
        k,
        type_typestr(a.dtype).encode(),
        ctypes.c_void_p(a.ptr),
        ctypes.c_void_p(b.ptr),
        ctypes.c_void_p(adj_d.ptr),
        ctypes.c_void_p(adj_c.ptr),
        0.0,
        beta,
        True,
        True,
        allow_tf32x3_arith,
        1,
    )
    if not ret:
        raise RuntimeError("adj_matmul failed.")


def batched_matmul(
    a: array3d,
    b: array3d,
    c: array3d,
    d: array3d,
    alpha: float = 1.0,
    beta: float = 0.0,
    allow_tf32x3_arith: bool = False,
    device=None,
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
        device: device we want to use to multiply matrices. Defaults to active runtime device. If "cpu", resorts to using numpy multiplication.
    """
    from warp.context import runtime

    if device is None:
        device = runtime.get_device(device)

    if a.device != device or b.device != device or c.device != device or d.device != device:
        raise RuntimeError("Matrices A, B, C, and D must all be on the same device as the runtime device.")

    if a.dtype != b.dtype or a.dtype != c.dtype or a.dtype != d.dtype:
        raise RuntimeError(
            "wp.batched_matmul currently only supports operation between {A, B, C, D} matrices of the same type."
        )

    m = a.shape[1]
    n = b.shape[2]
    k = a.shape[2]
    batch_count = a.shape[0]
    if b.shape != (batch_count, k, n) or c.shape != (batch_count, m, n) or d.shape != (batch_count, m, n):
        raise RuntimeError(
            "Invalid shapes for matrices: A = {} B = {} C = {} D = {}".format(a.shape, b.shape, c.shape, d.shape)
        )

    # cpu fallback if no cuda devices found
    if device == "cpu":
        d.assign(alpha * np.matmul(a.numpy(), b.numpy()) + beta * c.numpy())
        return

    cc = device.arch
    ret = runtime.core.cutlass_gemm(
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
        True,
        True,
        allow_tf32x3_arith,
        batch_count,
    )
    if not ret:
        raise RuntimeError("Batched matmul failed.")

    if runtime.tape:
        runtime.tape.record_func(
            backward=lambda: adj_matmul(
                a, b, c, a.grad, b.grad, c.grad, d.grad, alpha, beta, allow_tf32x3_arith, device
            ),
            arrays=[a, b, c, d],
        )


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
    allow_tf32x3_arith: bool = False,
    device=None,
):
    """Computes a batched generic matrix-matrix multiplication (GEMM) of the form: `d = alpha * (a @ b) + beta * c`.

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
        device: device we want to use to multiply matrices. Defaults to active runtime device. If "cpu", resorts to using numpy multiplication.
    """
    from warp.context import runtime

    if device is None:
        device = runtime.get_device(device)

    if (
        a.device != device
        or b.device != device
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

    # cpu fallback if no cuda devices found
    if device == "cpu":
        adj_a.assign(alpha * np.matmul(adj_d.numpy(), b.numpy().transpose((0, 2, 1))))
        adj_b.assign(alpha * np.matmul(a.numpy().transpose((0, 2, 1)), adj_d.numpy()))
        adj_c.assign(beta * adj_d.numpy())
        return

    cc = device.arch

    # adj_a
    ret = runtime.core.cutlass_gemm(
        cc,
        m,
        k,
        n,
        type_typestr(a.dtype).encode(),
        ctypes.c_void_p(adj_d.ptr),
        ctypes.c_void_p(b.ptr),
        ctypes.c_void_p(a.ptr),
        ctypes.c_void_p(adj_a.ptr),
        alpha,
        0.0,
        True,
        False,
        allow_tf32x3_arith,
        batch_count,
    )
    if not ret:
        raise RuntimeError("adj_matmul failed.")

    # adj_b
    ret = runtime.core.cutlass_gemm(
        cc,
        k,
        n,
        m,
        type_typestr(a.dtype).encode(),
        ctypes.c_void_p(a.ptr),
        ctypes.c_void_p(adj_d.ptr),
        ctypes.c_void_p(b.ptr),
        ctypes.c_void_p(adj_b.ptr),
        alpha,
        0.0,
        False,
        True,
        allow_tf32x3_arith,
        batch_count,
    )
    if not ret:
        raise RuntimeError("adj_matmul failed.")

    # adj_c
    ret = runtime.core.cutlass_gemm(
        cc,
        m,
        n,
        k,
        type_typestr(a.dtype).encode(),
        ctypes.c_void_p(a.ptr),
        ctypes.c_void_p(b.ptr),
        ctypes.c_void_p(adj_d.ptr),
        ctypes.c_void_p(adj_c.ptr),
        0.0,
        beta,
        True,
        True,
        allow_tf32x3_arith,
        batch_count,
    )
    if not ret:
        raise RuntimeError("adj_matmul failed.")


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

        from warp.context import runtime

        self.device = runtime.get_device(device)

        if self.device.is_cpu:
            self.id = runtime.core.hash_grid_create_host(dim_x, dim_y, dim_z)
        else:
            self.id = runtime.core.hash_grid_create_device(self.device.context, dim_x, dim_y, dim_z)

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

        from warp.context import runtime

        if self.device.is_cpu:
            runtime.core.hash_grid_update_host(self.id, radius, ctypes.cast(points.ptr, ctypes.c_void_p), len(points))
        else:
            runtime.core.hash_grid_update_device(self.id, radius, ctypes.cast(points.ptr, ctypes.c_void_p), len(points))

    def reserve(self, num_points):
        from warp.context import runtime

        if self.device.is_cpu:
            runtime.core.hash_grid_reserve_host(self.id, num_points)
        else:
            runtime.core.hash_grid_reserve_device(self.id, num_points)

    def __del__(self):
        try:
            from warp.context import runtime

            if self.device.is_cpu:
                runtime.core.hash_grid_destroy_host(self.id)
            else:
                # use CUDA context guard to avoid side effects during garbage collection
                with self.device.context_guard:
                    runtime.core.hash_grid_destroy_device(self.id)

        except:
            pass


class MarchingCubes:
    def __init__(self, nx: int, ny: int, nz: int, max_verts: int, max_tris: int, device=None):
        from warp.context import runtime

        self.device = runtime.get_device(device)

        if not self.device.is_cuda:
            raise RuntimeError("Only CUDA devices are supported for marching cubes")

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.max_verts = max_verts
        self.max_tris = max_tris

        # bindings to warp.so
        self.alloc = runtime.core.marching_cubes_create_device
        self.alloc.argtypes = [ctypes.c_void_p]
        self.alloc.restype = ctypes.c_uint64
        self.free = runtime.core.marching_cubes_destroy_device

        from warp.context import zeros

        self.verts = zeros(max_verts, dtype=vec3, device=self.device)
        self.indices = zeros(max_tris * 3, dtype=int, device=self.device)

        # alloc surfacer
        self.id = ctypes.c_uint64(self.alloc(self.device.context))

    def __del__(self):
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
        from warp.context import runtime

        # WP_API int marching_cubes_surface_host(const float* field, int nx, int ny, int nz, float threshold, wp::vec3* verts, int* triangles, int max_verts, int max_tris, int* out_num_verts, int* out_num_tris);
        num_verts = ctypes.c_int(0)
        num_tris = ctypes.c_int(0)

        runtime.core.marching_cubes_surface_device.restype = ctypes.c_int

        error = runtime.core.marching_cubes_surface_device(
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
                "Error occured buffers may not be large enough, marching cubes required at least {num_verts} vertices, and {num_tris} triangles."
            )

        # resize the geometry arrays
        self.verts.shape = (num_verts.value,)
        self.indices.shape = (num_tris.value * 3,)

        self.verts.size = num_verts.value
        self.indices.size = num_tris.value * 3


def type_is_generic(t):
    if t in (Any, Scalar, Float, Int):
        return True
    elif is_array(t):
        return type_is_generic(t.dtype)
    elif hasattr(t, "_wp_scalar_type_"):
        # vector/matrix type, check if dtype is generic
        if type_is_generic(t._wp_scalar_type_):
            return True
        # check if any dimension is generic
        for d in t._shape_:
            if d == 0:
                return True
    else:
        return False


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
        if type(arg_type) != type(template_type):
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


simple_type_codes = {
    int: "i4",
    float: "f4",
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
                if type_str == "quaternion":
                    return f"q{dtype_code}"
                elif type_str == "transform_t":
                    return f"t{dtype_code}"
                elif type_str == "spatial_vector_t":
                    return f"sv{dtype_code}"
                elif type_str == "spatial_matrix_t":
                    return f"sm{dtype_code}"
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
            raise RuntimeError(f"Failed to determine type code for argument {arg_str}{func_str}: {e}")

    return "_".join(type_codes)


def is_generic_signature(sig):
    return "?" in sig
