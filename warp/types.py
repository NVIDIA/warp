# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes 
import hashlib
import inspect
import itertools
import struct
import zlib
import numpy as np

from typing import Any
from typing import Tuple
from typing import TypeVar
from typing import Generic
from typing import List

import warp

class constant:
    """Class to declare compile-time constants accessible from Warp kernels
    
    Args:
        x: Compile-time constant value, can be any of the built-in math types.    
    """

    def __init__(self, x):

        self.val = x

        # hash the constant value
        if isinstance(x, int):
            constant._hash.update(struct.pack("<q", x))
        elif isinstance(x, float):
            constant._hash.update(struct.pack("<d", x))
        elif isinstance(x, bool):
            constant._hash.update(struct.pack("?", x))
        elif isinstance(x, float16):
            # float16 is a special case
            p = ctypes.pointer(ctypes.c_float(x.value))
            constant._hash.update(p.contents)
        elif isinstance(x, tuple(scalar_types)):
            p = ctypes.pointer(x._type_(x.value))
            constant._hash.update(p.contents)
        elif isinstance(x, tuple(vector_types)):
            constant._hash.update(bytes(x))
        else:
            raise RuntimeError(f"Invalid constant type: {type(x)}")

    def __eq__(self, other):
        return self.val == other

    # shared hash for all constants    
    _hash = hashlib.sha256()

#----------------------
# built-in types
def vector(length, type):
        
    class vector_t(ctypes.Array):

        _length_ = length
        _shape_ = (length, )
        _type_ = type
        
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

        def __neg__(self, y):
            return warp.neg(self, y)

        def __str__(self):
            return f"[{', '.join(map(str, self))}]"

        def __getitem__(self, key):
            # used to terminate iterations
            if isinstance(key, int) and key >= self._length_:
                raise IndexError()
            else:
                return super().__getitem__(key)

    return vector_t


def matrix(shape, type):
        
    assert(len(shape) == 2)

    class matrix_t(ctypes.Array):

        _length_ = shape[0]*shape[1]
        _shape_ = shape
        _type_ = type        
        
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

        def __neg__(self, y):
            return warp.neg(self, y)

        def _row(self, r):
            row_start = r*self._shape_[1]
            row_end = row_start + self._shape_[1]
            row_type = vector(self._shape_[1], self._type_)
            row_val = row_type(*super().__getitem__(slice(row_start,row_end)))

            return row_val

        def __str__(self):
            row_str = []
            for r in range(self._shape_[0]):      
                row_val = self._row(r)
                row_str.append(f"[{', '.join(map(str, row_val))}]")
            
            return "[" + ",\n ".join(row_str) + "]"

        def __getitem__(self, key):
            if isinstance(key, Tuple):
                # element indexing m[i,j]
                return super().__getitem__(key[1]*self._shape_[0] + key[1])
            else:
                # used to terminate iterations
                if key >= self._length_[0]:
                    raise IndexError()
                else:
                    return self._row(key)

    return matrix_t



class vec2(vector(length=2, type=ctypes.c_float)):
    pass
    
class vec3(vector(length=3, type=ctypes.c_float)):
    pass

class vec4(vector(length=4, type=ctypes.c_float)):
    pass

class quat(vector(length=4, type=ctypes.c_float)):
    pass
    
class mat22(matrix(shape=(2,2), type=ctypes.c_float)):
    pass
    
class mat33(matrix(shape=(3,3), type=ctypes.c_float)):
    pass

class mat44(matrix(shape=(4,4), type=ctypes.c_float)):
    pass

class spatial_vector(vector(length=6, type=ctypes.c_float)):
    pass

class spatial_matrix(matrix(shape=(6,6), type=ctypes.c_float)):
    pass

class transform(vector(length=7, type=ctypes.c_float)):
    
    def __init__(self, p=(0.0, 0.0, 0.0), q=(0.0, 0.0, 0.0, 1.0)):
        super().__init__()

        self[0:3] = vec3(*p)
        self[3:7] = quat(*q)

    @property 
    def p(self):
        return self[0:3]

    @property 
    def q(self):
        return self[3:7]

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


compute_types = [int32, float32]
scalar_types = [int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64]
vector_types = [vec2, vec3, vec4, mat22, mat33, mat44, quat, transform, spatial_vector, spatial_matrix]


np_dtype_to_warp_type = {
    np.dtype(np.int8): int8,
    np.dtype(np.uint8): uint8,
    np.dtype(np.int16): int16,
    np.dtype(np.uint16): uint16,
    np.dtype(np.int32): int32,
    np.dtype(np.int64): int64,
    np.dtype(np.uint8): uint8,
    np.dtype(np.uint32): uint32,
    np.dtype(np.uint64): uint64,
    np.dtype(np.byte): int8,
    np.dtype(np.ubyte): uint8,
    np.dtype(np.float16): float16,
    np.dtype(np.float32): float32,
    np.dtype(np.float64): float64
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

# maximum number of dimensions
ARRAY_MAX_DIMS = 4
LAUNCH_MAX_DIMS = 4

# represents bounds for kernel launch (number of threads across multiple dimensions)
class launch_bounds_t(ctypes.Structure):

    _fields_ = [("shape", ctypes.c_int32*LAUNCH_MAX_DIMS),
                ("ndim", ctypes.c_int32),
                ("size", ctypes.c_int32)]
  
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
                self.size = self.size*shape[i]

        # initialize the remaining dims to 1
        for i in range(self.ndim, LAUNCH_MAX_DIMS):
            self.shape[i] = 1


class shape_t(ctypes.Structure): 

    _fields_ = [("dims", ctypes.c_int32*ARRAY_MAX_DIMS)]
    
    def __init__(self):
        pass


class array_t(ctypes.Structure): 

    _fields_ = [("data", ctypes.c_uint64),
                ("shape", ctypes.c_int32*ARRAY_MAX_DIMS),
                ("strides", ctypes.c_int32*ARRAY_MAX_DIMS),
                ("ndim", ctypes.c_int32)]
    
    def __init__(self):
        self.data = 0
        self.shape = (0,)*ARRAY_MAX_DIMS
        self.strides = (0,)*ARRAY_MAX_DIMS
        self.ndim = 0       

        

def type_ctype(dtype):

    if dtype == float:
        return ctypes.c_float
    elif dtype == int:
        return ctypes.c_int32
    else:
        # scalar type
        return dtype._type_

def type_length(dtype):
    if (dtype == float or dtype == int):
        return 1
    else:
        return dtype._length_

def type_size_in_bytes(dtype):
    if (dtype == float or dtype == int or dtype == ctypes.c_float or dtype == ctypes.c_int32):
        return 4
    elif hasattr(dtype, "_type_"):
        return getattr(dtype, "_length_", 1) * ctypes.sizeof(dtype._type_)
    else:   
        return 0

def type_to_warp(dtype):
    if (dtype == float):
        return float32
    elif (dtype == int):
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
        # vector types all currently float type
        return "<f4"
    else:
        raise Exception("Unknown ctype")

def type_is_int(t):
    if (t == int or
        t == int8 or
        t == uint8 or
        t == int16 or
        t == uint16 or
        t == int32 or 
        t == uint32 or 
        t == int64 or         
        t == uint64):
        return True
    else:
        return False

def type_is_float(t):
    if (t == float or t == float32):
        return True
    else:
        return False

def types_equal(a, b):
    
    # convert to canonical types
    if (a == float):
        a = float32
    if (a == int):
        a = int32

    if (b == float):
        b = float32
    if (b == int):
        b = int32
        
    if isinstance(a, array) and isinstance(b, array):
        return True
    else:
        return a == b

def strides_from_shape(shape:Tuple, dtype):

    ndims = len(shape)
    strides = [None] * ndims

    i = ndims - 1
    strides[i] = type_size_in_bytes(dtype)

    while i > 0:
        strides[i - 1] = strides[i] * shape[i]
        i -= 1

    return tuple(strides)

T = TypeVar('T')


class array (Generic[T]):

    # member attributes available during code-gen (e.g.: d = array.shape[0])
    # (initialized when needed)
    _vars = None

    def __init__(self, data=None, dtype: T=None, shape=None, strides=None, length=0, ptr=None, capacity=0, device=None, copy=True, owner=True, ndim=None, requires_grad=False, pinned=False):
        """ Constructs a new Warp array object from existing data.

        When the ``data`` argument is a valid list, tuple, or ndarray the array will be constructed from this object's data.
        For objects that are not stored sequentially in memory (e.g.: a list), then the data will first 
        be flattened before being transferred to the memory space given by device.

        The second construction path occurs when the ``ptr`` argument is a non-zero uint64 value representing the
        start address in memory where existing array data resides, e.g.: from an external or C-library. The memory
        allocation should reside on the same device given by the device argument, and the user should set the length
        and dtype parameter appropriately.

        Args:
            data (Union[list, tuple, ndarray]) An object to construct the array from, can be a Tuple, List, or generally any type convertable to an np.array
            dtype (Union): One of the built-in types, e.g.: :class:`warp.mat33`, if dtype is None and data an ndarray then it will be inferred from the array data type
            shape (Tuple): Dimensions of the array
            strides (Tuple): Number of bytes in each dimension between successive elements of the array
            length (int): Number of elements (rows) of the data type (deprecated, users should use `shape` argument)
            ptr (uint64): Address of an external memory address to alias (data should be None)
            capacity (int): Maximum size in bytes of the ptr allocation (data should be None)
            device (Devicelike): Device the array lives on
            copy (bool): Whether the incoming data will be copied or aliased, this is only possible when the incoming `data` already lives on the device specified and types match
            owner (bool): Should the array object try to deallocate memory when it is deleted
            requires_grad (bool): Whether or not gradients will be tracked for this array, see :class:`warp.Tape` for details
            pinned (bool): Whether to allocate pinned host memory, which allows asynchronous host-device transfers (only applicable with device="cpu")

        """

        self.owner = False

        # convert shape to Tuple
        if shape == None:
            shape = (length,)   
        elif isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, List):
            shape = tuple(shape)

        self.shape = shape

        if len(shape) > ARRAY_MAX_DIMS:
            raise RuntimeError(f"Arrays may only have {ARRAY_MAX_DIMS} dimensions maximum, trying to create array with {len(shape)} dims.")

        # canonicalize dtype
        if (dtype == int):
            dtype = int32
        elif (dtype == float):
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
                raise RuntimeError("When constructing an array the data argument must be convertable to ndarray type type. Encountered an error while converting:" + str(e))
            
            if dtype == None:
                # infer dtype from the source data array
                dtype = np_dtype_to_warp_type[arr.dtype]

            try:
                # try to convert src array to destination type
                arr = arr.astype(dtype=type_typestr(dtype), copy=False)
            except:
                raise RuntimeError(f"Could not convert input data with type {arr.dtype} to array with type {dtype._type_}")
            
            # ensure contiguous
            arr = np.ascontiguousarray(arr)

            # remove any trailing dimensions of length 1
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=len(arr.shape)-1)

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
                    raise RuntimeError(f"Last dimensions of input array should match the specified data type, given shape {arr.shape}, expected last dimensions to match dtype shape {dtype._shape_}")

                shape = leading_shape
            
                if strides is not None:
                    strides = strides[0:-dtype_ndim]
            


            if device.is_cpu and copy == False:

                # ref numpy memory directly
                self.shape=shape
                self.ptr = ptr
                self.dtype=dtype
                self.strides = strides
                self.capacity=arr.size*type_size_in_bytes(dtype)
                self.device = device
                self.owner = False
                self.pinned = False

                # keep a ref to source array to keep allocation alive
                self.ref = arr

            else:

                # otherwise, we must transfer to device memory
                # create a host wrapper around the numpy array
                # and a new destination array to copy it to
                src = array(dtype=dtype, shape=shape, strides=strides, capacity=arr.size*type_size_in_bytes(dtype), ptr=ptr, device='cpu', copy=False, owner=False)
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
            self.device = device
            self.owner = owner
            if device is not None and device.is_cpu:
                self.pinned = pinned
            else:
                self.pinned = False

            self.__name__ = "array<" + type.__name__ + ">"


        # update ndim
        if ndim == None:
            self.ndim = len(self.shape)
        else:
            self.ndim = ndim

        # update size (num elements)
        self.size = 1
        for d in self.shape:
            self.size *= d

        # update byte strides and contiguous flag
        contiguous_strides = strides_from_shape(self.shape, self.dtype)
        if strides is None:
            self.strides = contiguous_strides
            self.is_contiguous = True
        else:
            self.strides = strides
            self.is_contiguous = strides[:ndim] == contiguous_strides[:ndim]

        # store flat shape (including type shape)
        if dtype in vector_types:
            # vector type, flatten the dimensions into one tuple
            arr_shape = (*self.shape, *self.dtype._shape_)
            dtype_strides =  strides_from_shape(self.dtype._shape_, self.dtype._type_) 
            arr_strides = (*self.strides, *dtype_strides)
        else:
            # scalar type
            arr_shape = self.shape
            arr_strides = self.strides

        # set up array interface access so we can treat this object as a numpy array
        if self.ptr:
            if device.is_cpu:

                self.__array_interface__ = { 
                    "data": (self.ptr, False), 
                    "shape": tuple(arr_shape),  
                    "strides": tuple(arr_strides),  
                    "typestr": type_typestr(self.dtype), 
                    "version": 3 
                }

            # set up cuda array interface access so we can treat this object as a Torch tensor
            elif device.is_cuda:

                self.__cuda_array_interface__ = {
                    "data": (self.ptr, False),
                    "shape": tuple(arr_shape),
                    "strides": tuple(arr_strides),
                    "typestr": type_typestr(self.dtype),
                    "version": 2
                }

        self.grad = None

        # controls if gradients will be computed in by wp.Tape
        # this will trigger allocation of a gradient array if it doesn't exist already
        self.requires_grad = requires_grad


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

        if self.device == None:
            # for 'empty' arrays we just return the type information, these are used in kernel function signatures
            return f"array{self.dtype}"
        else:
            return str(self.to("cpu").numpy())

    # construct a C-representation of the array for passing to kernels
    def __ctype__(self):
        a = array_t()
        
        if (self.ptr == None):
            a.data = 0
        else:
            a.data = ctypes.c_uint64(self.ptr)

        a.ndim = ctypes.c_int32(len(self.shape))

        for i in range(a.ndim):
            a.shape[i] = self.shape[i]
            a.strides[i] = self.strides[i]

        return a        

    @property
    def requires_grad(self):

        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value:bool):
        
        if value and self.grad is None:
            self._alloc_grad()
        elif not value:
            self.grad = None

        self._requires_grad = value

    def _alloc_grad(self):

        self.grad = warp.zeros(shape=self.shape, dtype=self.dtype, device=self.device, requires_grad=False)

    @property
    def vars(self):
        # member attributes available during code-gen (e.g.: d = array.shape[0])
        # Note: we use a shared dict for all array instances
        if array._vars is None:
            from warp.codegen import Var
            array._vars = { "shape": Var("shape", shape_t) }
        return array._vars


    def zero_(self):

        if not self.is_contiguous:
            raise RuntimeError(f"Assigning to non-continuguous arrays is unsupported.")

        if self.device is not None and self.ptr is not None:
            self.device.memset(ctypes.c_void_p(self.ptr), ctypes.c_int(0), ctypes.c_size_t(self.size*type_size_in_bytes(self.dtype)))


    def fill_(self, value):

        if not self.is_contiguous:
            raise RuntimeError(f"Assigning to non-continuguous arrays is unsupported.")

        if self.device is not None and self.ptr is not None:

            # convert value to array type
            src_type = type_ctype(self.dtype)
            src_value = src_type(value)

            # cast to a 4-byte integer for memset
            dest_ptr = ctypes.cast(ctypes.pointer(src_value), ctypes.POINTER(ctypes.c_int))
            dest_value = dest_ptr.contents

            self.device.memset(ctypes.cast(self.ptr,ctypes.POINTER(ctypes.c_int)), dest_value, ctypes.c_size_t(self.size*type_size_in_bytes(self.dtype)))

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
            return np.array(self.to("cpu"), copy=False)
        

    # convert data from one device to another, nop if already on device
    def to(self, device):
        
        device = warp.get_device(device)
        if self.device == device:
            return self
        else:
            dest = warp.empty(shape=self.shape, dtype=self.dtype, device=device)
            warp.copy(dest, self)
            return dest


    # def flatten(self):

    #     a = array(ptr=self.ptr,
    #               dtype=self.dtype,
    #               shape=(self.size,),
    #               device=self.device,
    #               owner=False,
    #               ndim=1,
    #               requires_grad=self.requires_grad)

    #     # store back-ref to stop data being destroyed
    #     a._ref = self
    #     return a        

    # def astype(self, dtype):

    #     # return an alias of the array memory with different type information
    #     src_bytes = self.length*type_length(self.dtype)
    #     src_capacity = self.capacity*type_length(self.dtype)

    #     dst_length = src_length/type_length(dtype)
    #     dst_capacity = src_capacity/type_length(dtype)

    #     if ((src_length % type_length(dtype)) > 0):
    #         raise RuntimeError("Dimensions are incompatible for type cast")

    #     arr = array(
    #         ptr=self.ptr, 
    #         dtype=dtype,
    #         length=int(dst_length),
    #         capacity=int(dst_capacity),
    #         device=self.device,
    #         owner=False)

    #     return arr

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


class Bvh:

    def __init__(self, lowers, uppers):
        """ Class representing a bounding volume hierarchy.

        Attributes:
            id: Unique identifier for this bvh object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            lowers (:class:`warp.array`): Array of lower bounds :class:`warp.vec3`
            uppers (:class:`warp.array`): Array of upper bounds :class:`warp.vec3`
        """

        if (len(lowers) != len(uppers)):
            raise RuntimeError("Bvh the same number of lower and upper bounds must be provided")

        if (lowers.device != uppers.device):
            raise RuntimeError("Bvh lower and upper bounds must live on the same device")

        if (lowers.dtype != vec3 or not lowers.is_contiguous):
            raise RuntimeError("Bvh lowers should be a contiguous array of type wp.vec3")

        if (uppers.dtype != vec3 or not uppers.is_contiguous):
            raise RuntimeError("Bvh uppers should be a contiguous array of type wp.vec3")


        self.device = lowers.device
        self.lowers = lowers
        self.upupers = uppers

        def get_data(array):
            if (array):
                return ctypes.c_void_p(array.ptr)
            else:
                return ctypes.c_void_p(0)

        from warp.context import runtime

        if self.device.is_cpu:
            self.id = runtime.core.bvh_create_host(
                get_data(lowers), 
                get_data(uppers), 
                int(len(lowers)))
        else:
            self.id = runtime.core.bvh_create_device(
                self.device.context,
                get_data(lowers), 
                get_data(uppers), 
                int(len(lowers)))


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
        """ Refit the Bvh. This should be called after users modify the `lowers` and `uppers` arrays."""
                
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
        "indices": Var("indices", array(dtype=int32, ndim=2))
    }

    def __init__(self, points=None, indices=None, velocities=None):
        """ Class representing a triangle mesh.

        Attributes:
            id: Unique identifier for this mesh object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            points (:class:`warp.array`): Array of vertex positions of type :class:`warp.vec3`
            indices (:class:`warp.array`): Array of triangle indices of type :class:`warp.int32`, should be length 3*number of triangles
            velocities (:class:`warp.array`): Array of vertex velocities of type :class:`warp.vec3` (optional)
        """

        if (points.device != indices.device):
            raise RuntimeError("Mesh points and indices must live on the same device")

        if (points.dtype != vec3 or not points.is_contiguous):
            raise RuntimeError("Mesh points should be a contiguous array of type wp.vec3")

        if (velocities and (velocities.dtype != vec3 or not velocities.is_contiguous)):
            raise RuntimeError("Mesh velocities should be a contiguous array of type wp.vec3")

        if (indices.dtype != int32 or not indices.is_contiguous):
            raise RuntimeError("Mesh indices should be a contiguous array of type wp.int32")


        self.device = points.device
        self.points = points
        self.velocities = velocities
        self.indices = indices

        def get_data(array):
            if (array):
                return ctypes.c_void_p(array.ptr)
            else:
                return ctypes.c_void_p(0)

        from warp.context import runtime

        if self.device.is_cpu:
            self.id = runtime.core.mesh_create_host(
                get_data(points), 
                get_data(velocities), 
                get_data(indices), 
                int(len(points)), 
                int(indices.size/3))
        else:
            self.id = runtime.core.mesh_create_device(
                self.device.context,
                get_data(points), 
                get_data(velocities), 
                get_data(indices), 
                int(len(points)), 
                int(indices.size/3))


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
        """ Refit the BVH to points. This should be called after users modify the `points` data."""
                
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
        """ Class representing a sparse grid.

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
            raise RuntimeError(f"Invalid device")
        self.device = data.device

        if self.device.is_cpu:
            self.id = self.context.core.volume_create_host(ctypes.cast(data.ptr, ctypes.c_void_p), data.size)
        else:
            self.id = self.context.core.volume_create_device(self.device.context, ctypes.cast(data.ptr, ctypes.c_void_p), data.size)

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
        return array(ptr=buf.value, dtype=int32, shape=(num_tiles, 3), length=size.value, device=self.device, owner=True)

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
        if magic != 0x304244566f6e614e:
            raise RuntimeError("NanoVDB signature not found")
        if version>>21 != 32: # checking major version
            raise RuntimeError("Unsupported NanoVDB version")
        if grid_count != 1:
            raise RuntimeError("Only NVDBs with exactly one grid are supported")

        grid_data_offset = 192 + struct.unpack("<I", data[152:156])[0]
        if codec == 0: # no compression
            grid_data = data[grid_data_offset:]
        elif codec == 1: # zip compression
            grid_data = zlib.decompress(data[grid_data_offset + 8:])
        else:
            raise RuntimeError(f"Unsupported codec code: {codec}")

        magic = struct.unpack("<Q", grid_data[0:8])[0]
        if magic != 0x304244566f6e614e:
            raise RuntimeError("NanoVDB signature not found on grid!")

        data_array = array(np.frombuffer(grid_data, dtype=np.byte), device=device)
        return cls(data_array)

    @classmethod
    def allocate(cls, min: List[int], max: List[int], voxel_size: float, bg_value=0.0, translation=(0.0,0.0,0.0), points_in_world_space=False, device=None):
        """ Allocate a new Volume based on the bounding box defined by min and max.

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
        tiles = np.array([[i, j, k] for i in range(tile_min[0], tile_max[0] + 1)
                                    for j in range(tile_min[1], tile_max[1] + 1)
                                    for k in range(tile_min[2], tile_max[2] + 1)],
                         dtype=np.int32)
        tile_points = array(tiles * 8, device=device)

        return cls.allocate_by_tiles(tile_points, voxel_size, bg_value, translation, device)

    @classmethod
    def allocate_by_tiles(cls, tile_points: array, voxel_size: float, bg_value=0.0, translation=(0.0,0.0,0.0), device=None):
        """ Allocate a new Volume with active tiles for each point tile_points.

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
        if not (isinstance(tile_points, array) and
            (tile_points.dtype == int32 and tile_points.ndim  == 2) or
            (tile_points.dtype == vec3 and tile_points.ndim  == 1)):
            raise RuntimeError(f"Expected an warp array of vec3s or of n-by-3 int32s as tile_points!")
        if not tile_points.device.is_cuda:
            tile_points = array(tile_points, dtype=tile_points.dtype, device=device)

        volume = cls(data=None)
        volume.device = device
        in_world_space = (tile_points.dtype == vec3)
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
                in_world_space)
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
                in_world_space)
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
                in_world_space)

        if volume.id == 0:
            raise RuntimeError("Failed to create volume")

        return volume

class HashGrid:

    def __init__(self, dim_x, dim_y, dim_z, device=None):
        """ Class representing a hash grid object for accelerated point queries.

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
        """ Updates the hash grid data structure.

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
        self.indices = zeros(max_tris*3, dtype=int, device=self.device)

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

        error = runtime.core.marching_cubes_surface_device(self.id,
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
                                                           ctypes.c_void_p(ctypes.addressof(num_tris)))

        if error:
            raise RuntimeError("Error occured buffers may not be large enough, marching cubes required at least {num_verts} vertices, and {num_tris} triangles.")


        # resize the geometry arrays
        self.verts.shape = (num_verts.value, )
        self.indices.shape = (num_tris.value*3, )

        self.verts.size = num_verts.value
        self.indices.size = num_tris.value*3




