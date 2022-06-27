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


class constant:
    """Class to declare compile-time constants accessible from Warp kernels
    
    Args:
        x: Compile-time constant value, can be any of the built-in math types.    
    """

    def __init__(self, x):

        self.val = x
        constant._hash.update(bytes(str(x), 'utf-8'))

    def __eq__(self, other):
        return self.val == other

    # shared hash for all constants    
    _hash = hashlib.sha256()

#----------------------
# built-in types


class vec2(ctypes.Array):
    
    _length_ = 2
    _shape_ = (2,)
    _type_ = ctypes.c_float    
    
class vec3(ctypes.Array):
    
    _length_ = 3
    _shape_ = (3,)
    _type_ = ctypes.c_float
    
class vec4(ctypes.Array):
    
    _length_ = 4
    _shape_ = (4,)
    _type_ = ctypes.c_float

class quat(ctypes.Array):
    
    _length_ = 4
    _shape_ = (4,)
    _type_ = ctypes.c_float
    
class mat22(ctypes.Array):
    
    _length_ = 4
    _shape_ = (2,2)
    _type_ = ctypes.c_float
    
class mat33(ctypes.Array):
    
    _length_ = 9
    _shape_ = (3,3)
    _type_ = ctypes.c_float

class mat44(ctypes.Array):
    
    _length_ = 16
    _shape_ = (4,4)
    _type_ = ctypes.c_float

class spatial_vector(ctypes.Array):
    
    _length_ = 6
    _shape_ = (6,)
    _type_ = ctypes.c_float

class spatial_matrix(ctypes.Array):
    
    _length_ = 36
    _shape_ = (6,6)
    _type_ = ctypes.c_float

class transform(ctypes.Array):
    
    _length_ = 7
    _shape_ = (7,)
    _type_ = ctypes.c_float

    def __init__(self, p=(0.0, 0.0, 0.0), q=(0.0, 0.0, 0.0, 1.0)):
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


class array_t(ctypes.Structure): 

    _fields_ = [("data", ctypes.c_uint64),
                ("shape", ctypes.c_int32*ARRAY_MAX_DIMS),
                ("ndim", ctypes.c_int32)]
    
    def __init__(self):
        self.data = 0
        self.shape = (0,)*ARRAY_MAX_DIMS
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
    if (dtype == float or dtype == int):
        return 4
    else:
        return dtype._length_*ctypes.sizeof(dtype._type_)

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

T = TypeVar('T')

class array (Generic[T]):

    def __init__(self, data=None, dtype: T=None, shape=None, length=0, ptr=None, capacity=0, device=None, copy=True, owner=True, ndim=None, requires_grad=False):
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
            length (int): Number of elements (rows) of the data type (deprecated, users should use `shape` argument)
            ptr (uint64): Address of an external memory address to alias (data should be None)
            capacity (int): Maximum size in bytes of the ptr allocation (data should be None)
            device (str): Device the array lives on
            copy (bool): Whether the incoming data will be copied or aliased, this is only possible when the incoming `data` already lives on the device specified and types match
            owner (bool): Should the array object try to deallocate memory when it is deleted
            requires_grad (bool): Whether or not gradients will be tracked for this array, see :class:`warp.Tape` for details

        """

        self.owner = False
        self.grad = None

        if shape == None:
            shape = (length,)   
        elif isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, Tuple):
            self.shape = shape


        if len(shape) > ARRAY_MAX_DIMS:
            raise RuntimeError(f"Arrays may only have {ARRAY_MAX_DIMS} dimensions maximum, trying to create array with {len(shape)} dims.")


        # canonicalize dtype
        if (dtype == int):
            dtype = int32
        elif (dtype == float):
            dtype = float32

        if data is not None:

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


            if (device == "cpu" and copy == False):

                # ref numpy memory directly
                self.ptr = ptr
                self.dtype=dtype
                self.shape=shape
                self.capacity=arr.size*type_size_in_bytes(dtype)
                self.device = device
                self.owner = False

                # keep a ref to source array to keep allocation alive
                self.ref = arr

            else:

                from warp.context import empty, copy

                # otherwise, we must transfer to device memory
                # create a host wrapper around the numpy array
                # and a new destination array to copy it to
                src = array(dtype=dtype, shape=shape, capacity=arr.size*type_size_in_bytes(dtype), ptr=ptr, device='cpu', copy=False, owner=False)
                dest = empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
                dest.owner = False
                
                # data copy
                copy(dest, src)

                # object copy to self and transfer data ownership, would probably be cleaner to have _empty, _zero, etc as class methods
                from copy import copy as shallowcopy

                self.__dict__ = shallowcopy(dest.__dict__)
                self.owner = True

        else:
            
            # explicit construction from ptr to external memory
            self.shape = shape
            self.capacity = capacity
            self.dtype = dtype
            self.ptr = ptr
            self.device = device
            self.owner = owner

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

        # store flat shape (including type shape)
        if dtype in vector_types:
            # vector type, flatten the dimensions into one tuple
            arr_shape = (*self.shape, *self.dtype._shape_)
        else:
            # scalar type
            arr_shape = self.shape

        # set up array interface access so we can treat this object as a numpy array
        if device == "cpu":

            self.__array_interface__ = { 
                "data": (self.ptr, False), 
                "shape": arr_shape,  
                "typestr": type_typestr(self.dtype), 
                "version": 3 
            }

        # set up cuda array interface access so we can treat this object as a Torch tensor
        if device == "cuda":

            self.__cuda_array_interface__ = {
                "data": (self.ptr, False),
                "shape": arr_shape,
                "typestr": type_typestr(self.dtype),
                "version": 2
            }

        # controls if gradients will be computed in by wp.Tape
        # this will trigger allocation of a gradient array if it doesn't exist already
        self.requires_grad = requires_grad


    def __del__(self):
        
        try:
            if (self.owner):

                # this import can fail during process destruction
                # in this case we allow OS to clean up allocations
                from warp.context import runtime

                if (self.device == "cpu"):
                    runtime.host_allocator.free(self.ptr, self.capacity)
                else:
                    runtime.device_allocator.free(self.ptr, self.capacity)
        
        except Exception as e:
            pass
                

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

        return a        

    def __setattr__(self, __name: str, __value: Any) -> None:

        if __name == "requires_grad" and __value == True and self.grad == None:
            self._alloc_grad()

        return super().__setattr__(__name, __value)

    def _alloc_grad(self):

        from warp.context import zeros
        self.grad = zeros(shape=self.shape, dtype=self.dtype, device=self.device, requires_grad=False)


    def zero_(self):

        from warp.context import runtime

        if (self.device == "cpu"):
            runtime.core.memset_host(ctypes.cast(self.ptr,ctypes.POINTER(ctypes.c_int)), ctypes.c_int(0), ctypes.c_size_t(self.size*type_size_in_bytes(self.dtype)))

        if(self.device == "cuda"):
            runtime.core.memset_device(ctypes.cast(self.ptr,ctypes.POINTER(ctypes.c_int)), ctypes.c_int(0), ctypes.c_size_t(self.size*type_size_in_bytes(self.dtype)))


    def fill_(self, value):

        from warp.context import runtime

        # convert value to array type
        src_type = type_ctype(self.dtype)
        src_value = src_type(value)

        # cast to a 4-byte integer for memset
        dest_ptr = ctypes.cast(ctypes.pointer(src_value), ctypes.POINTER(ctypes.c_int))
        dest_value = dest_ptr.contents

        if (self.device == "cpu"):
            runtime.core.memset_host(ctypes.cast(self.ptr,ctypes.POINTER(ctypes.c_int)), dest_value, ctypes.c_size_t(self.size*type_size_in_bytes(self.dtype)))

        if(self.device == "cuda"):
            runtime.core.memset_device(ctypes.cast(self.ptr,ctypes.POINTER(ctypes.c_int)), dest_value, ctypes.c_size_t(self.size*type_size_in_bytes(self.dtype)))

    # equivalent to wrapping src data in an array and copying to self
    def assign(self, src):
        from warp.context import copy

        if isinstance(src, array):
            copy(self, src)
        else:
            copy(self, array(src, dtype=self.dtype, copy=False, device="cpu"))


    # convert array to ndarray (alias memory through array interface)
    def numpy(self):
        return np.array(self.to("cpu"), copy=False)
        

    # convert data from one device to another, nop if already on device
    def to(self, device):

        if (self.device == device):
            return self
        else:
            from warp.context import empty, copy, synchronize

            dest = empty(shape=self.shape, dtype=self.dtype, device=device)
            copy(dest, self)
            
            # todo: only synchronize when there is a device->host transfer outstanding
            synchronize()

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


class Mesh:

    def __init__(self, points, indices, velocities=None):
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

        if (points.dtype != vec3):
            raise RuntimeError("Mesh points should be an array of type wp.vec3")

        if (velocities and velocities.dtype != vec3):
            raise RuntimeError("Mesh velocities should be an array of type wp.vec3")

        if (indices.dtype != int32):
            raise RuntimeError("Mesh indices should be an array of type wp.int32")


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

        if (self.device == "cpu"):
            self.id = runtime.core.mesh_create_host(
                get_data(points), 
                get_data(velocities), 
                get_data(indices), 
                int(len(points)), 
                int(len(indices)/3))
        else:
            self.id = runtime.core.mesh_create_device(
                get_data(points), 
                get_data(velocities), 
                get_data(indices), 
                int(len(points)), 
                int(len(indices)/3))


    def __del__(self):

        try:
                
            from warp.context import runtime

            if (self.device == "cpu"):
                runtime.core.mesh_destroy_host(self.id)
            else:
                runtime.core.mesh_destroy_device(self.id)
        
        except:
            pass

    def refit(self):
        """ Refit the BVH to points. This should be called after users modify the `points` data."""
                
        from warp.context import runtime

        if (self.device == "cpu"):
            runtime.core.mesh_refit_host(self.id)
        else:
            runtime.core.mesh_refit_device(self.id)
            runtime.verify_device()



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

        if data.device != "cpu" and data.device != "cuda":
            raise RuntimeError(f"Unknown device type '{data.device}'")
        self.device = data.device

        if self.device == "cpu":
            self.id = self.context.core.volume_create_host(ctypes.cast(data.ptr, ctypes.c_void_p), data.size)
        else:
            self.id = self.context.core.volume_create_device(ctypes.cast(data.ptr, ctypes.c_void_p), data.size)

        if self.id == 0:
            raise RuntimeError("Failed to create volume from input array")

    def __del__(self):

        if self.id == 0:
            return
        
        try:
                
            from warp.context import runtime

            runtime.verify_device()
            if self.device == "cpu":
                runtime.core.volume_destroy_host(self.id)
            else:
                runtime.core.volume_destroy_device(self.id)

        except:
            pass


    def array(self):

        buf = ctypes.c_void_p(0)
        size = ctypes.c_uint64(0)
        if self.device == "cpu":
            self.context.core.volume_get_buffer_info_host(self.id, ctypes.byref(buf), ctypes.byref(size))
        else:
            self.context.core.volume_get_buffer_info_device(self.id, ctypes.byref(buf), ctypes.byref(size))
        return array(ptr=buf.value, dtype=uint8, length=size.value, device=self.device, owner=False)

    @classmethod
    def load_from_nvdb(cls, file_or_buffer, device):
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
        if codec == 0: # no compession
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


class HashGrid:

    def __init__(self, dim_x, dim_y, dim_z, device):
        """ Class representing a triangle mesh.

        Attributes:
            id: Unique identifier for this mesh object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            dim_x (int): Number of cells in x-axis
            dim_y (int): Number of cells in y-axis
            dim_z (int): Number of cells in z-axis
        """

        self.device = device

        from warp.context import runtime
       
        if (device == "cpu"):
            self.id = runtime.core.hash_grid_create_host(dim_x, dim_y, dim_z)
        elif (device == "cuda"):
            self.id = runtime.core.hash_grid_create_device(dim_x, dim_y, dim_z)


    def build(self, points, radius):
        """ Updates the hash grid data structure.

        This method rebuilds the underlying datastructure and should be called any time the set
        of points changes.

        Attributes:
            id: Unique identifier for this mesh object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            points (:class:`warp.array`): Array of points of type :class:`warp.vec3`
            radius (float): The cell size to use for bucketing points, cells are cubes with edges of this width.
                            For best performance the radius used to construct the grid should match closely to
                            the radius used when performing queries.                          
        """
        
        from warp.context import runtime

        if (self.device == "cpu"):
            runtime.core.hash_grid_update_host(self.id, radius, ctypes.cast(points.ptr, ctypes.c_void_p), len(points))
        else:
            runtime.core.hash_grid_update_device(self.id, radius, ctypes.cast(points.ptr, ctypes.c_void_p), len(points))


    def reserve(self, num_points):

        from warp.context import runtime

        if (self.device == "cpu"):
            runtime.core.hash_grid_reserve_host(self.id, num_points)
        else:
            runtime.core.hash_grid_reserve_device(self.id, num_points)


    def __del__(self):

        try:

            from warp.context import runtime

            if (self.device == "cpu"):
                runtime.core.hash_grid_destroy_host(self.id)
            else:
                runtime.core.hash_grid_destroy_device(self.id)

        except:
            pass


