
import ctypes 
import numpy as np

#----------------------
# built-in types


class vec2(ctypes.Array):
    
    _length_ = 2
    _type_ = ctypes.c_float
    
class vec3(ctypes.Array):
    
    _length_ = 3
    _type_ = ctypes.c_float
    
class vec4(ctypes.Array):
    
    _length_ = 4
    _type_ = ctypes.c_float

class quat(ctypes.Array):
    
    _length_ = 4
    _type_ = ctypes.c_float
    
class mat22(ctypes.Array):
    
    _length_ = 4
    _type_ = ctypes.c_float
    
class mat33(ctypes.Array):
    
    _length_ = 9
    _type_ = ctypes.c_float

class mat44(ctypes.Array):
    
    _length_ = 16
    _type_ = ctypes.c_float

class spatial_vector(ctypes.Array):
    
    _length_ = 6
    _type_ = ctypes.c_float

class spatial_matrix(ctypes.Array):
    
    _length_ = 36
    _type_ = ctypes.c_float

class transform(ctypes.Array):
    
    _length_ = 7
    _type_ = ctypes.c_float

    def __init__(self, p, q):
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

class float32:

    _length_ = 1
    _type_ = ctypes.c_float


class float64:

    _length_ = 1
    _type_ = ctypes.c_double


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



# definition just for kernel type (cannot be a parameter), see mesh.h
class mesh_query_aabb_t:

    def __init__(self):
        pass

# definition just for kernel type (cannot be a parameter), see hash_grid.h
class hash_grid_query_t:

    def __init__(self):
        pass


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

def type_ctype(dtype):
    if (dtype == float):
        return ctypes.c_float
    elif (dtype == int):
        return ctypes.c_int32
    else:
        return dtype._type_

def type_typestr(ctype):
   
    if (ctype == ctypes.c_float):
        return "<f4"
    elif (ctype == ctypes.c_double):
        return "<f8"
    elif (ctype == ctypes.c_int32):
        return "<i4"
    elif (ctype == ctypes.c_uint32):
        return "<u4"
    elif (ctype == ctypes.c_int64):
        return "<i8"
    elif (ctype == ctypes.c_uint64):
        return "<u8"
    else:
        raise Exception("Unknown ctype")

def type_is_int(t):
    if (t == int or t == int32 or t == int64 or t == uint32 or t == uint64):
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
        
    if (isinstance(a, array) and isinstance(b, array)):
        return a.dtype == b.dtype
    else:
        return a == b

class array:

    def __init__(self, data=None, dtype=float32, length=0, capacity=0, device=None, copy=True, owner=True, requires_grad=False):
        """ Constructs a new Warp array object 

        This method allows manually constructing an array from existing data. 

        Args:
            data (Union[list, tuple, iterable], wp.uint64) 
            dtype (Union): One of the built-in types, e.g.: :class:`warp.mat33`
            length (int): Number of elements (rows) of the data type
            capacity (int): Maximum size in bytes of the `data` allocation
            device (str): Device the allocation lives on
            copy (bool): Whether the incoming data will be copied or aliased, this is only possible when the incoming `data` already lives on the device specified
            owner (bool): Should the array object try to deallocate memory when it is deleted
            requires_grad (bool): Whether or not gradients will be tracked for this array, see :class:`warp.Tape` for details

        """
        
        self.owner = False

        # convert built-in numeric type to wp type
        if (dtype == int):
            dtype = int32

        elif (dtype == float):
            dtype = float32

        # save flag, controls if gradients will be computed in by wp.Tape
        self.requires_grad = requires_grad

        # construct from numpy array, list, tuple
        if (isinstance(data, np.ndarray) or 
            isinstance(data, list) or 
            isinstance(data, tuple)):

            from warp.context import empty, copy, synchronize

            # convert lists / tuples to ndarrays if necessary
            arr = np.array(data, copy=False)

            # try to convert src array to destination shape
            try:
                arr = arr.reshape((-1, type_length(dtype)))
            except:
                raise RuntimeError(f"Could not reshape input data with shape {arr.shape} to array with shape (*, {type_length(dtype)}")

            # try to convert src array to destination type
            try:
                arr = arr.astype(dtype=type_typestr(dtype._type_))
            except:
                raise RuntimeError(f"Could not convert input data with type {arr.dtype} to array with type {dtype._type_}")
            
            # ensure contiguous
            arr = np.ascontiguousarray(arr)

            ptr = arr.__array_interface__["data"][0]
            shape = arr.__array_interface__["shape"]
            length = shape[0]

            if (device == "cpu" and copy == False):

                from warp.context import runtime
                
                # ref numpy memory directly
                self.data = ptr
                self.dtype=dtype
                self.length=length
                self.capacity=length*type_size_in_bytes(dtype)
                self.device = device
                self.owner = False

                # keep a ref to source array to keep allocation alive
                self.ref = arr

            else:

                # otherwise, create a host wrapper around the numpy
                #  array and a new destination array to copy it to
                src = array(dtype=dtype, length=length, capacity=length*type_size_in_bytes(dtype), data=ptr, device='cpu', copy=False, owner=False)
                dest = empty(length, dtype=dtype, device=device, requires_grad=requires_grad)
                dest.owner = False
                
                # data copy
                copy(dest, src)

                # object copy to self and transfer data ownership, would probably be cleaner to have _empty, _zero, etc as class methods
                from copy import copy as shallowcopy

                self.__dict__ = shallowcopy(dest.__dict__)
                self.owner = True
           
            
        else:
            
            # explicit construction, data is interpreted as the address for raw memory 
            self.length = length
            self.capacity = capacity
            self.dtype = dtype
            self.data = data
            self.device = device
            self.owner = owner

            self.__name__ = "array<" + type.__name__ + ">"


        # store 2D shape (useful for interop with tensor frameworks)
        self.shape = (self.length, type_length(self.dtype))
        
        # store a ptr cast of the data address (to avoid accessing ctypes during destruction)
        self.ptr = ctypes.cast(self.data, ctypes.POINTER(ctypes.c_int))

        # set up array interface access so we can treat this object as a read-only numpy array
        if (device == "cpu"):

            self.__array_interface__ = { 
                "data": (self.data, False), 
                "shape": self.shape,  
                "typestr": type_typestr(type_ctype(self.dtype)), 
                "version": 3 
            }


    def __del__(self):
        
        # must be careful to not call any globals here
        # since this may be called during destruction / shutdown
        # even ctypes module may no longer exist

        if (self.owner):

            # todo: check this is OK during shutdown
            from warp.context import runtime

            if (self.device == "cpu"):
                runtime.host_allocator.free(self.ptr, self.capacity)
            else:
                runtime.device_allocator.free(self.ptr, self.capacity)
                

    def __len__(self):

        return self.length

    def __str__(self):

        if self.device == None:
            # for 'empty' arrays we just return the type information, these are used in kernel function signatures
            return f"array{self.dtype}"
        else:
            return str(self.to("cpu").numpy())

    def zero_(self):

        from warp.context import runtime

        if (self.device == "cpu"):
            runtime.core.memset_host(ctypes.cast(self.data,ctypes.POINTER(ctypes.c_int)), ctypes.c_int(0), ctypes.c_size_t(self.length*type_size_in_bytes(self.dtype)))

        if(self.device == "cuda"):
            runtime.core.memset_device(ctypes.cast(self.data,ctypes.POINTER(ctypes.c_int)), ctypes.c_int(0), ctypes.c_size_t(self.length*type_size_in_bytes(self.dtype)))



    # equivalent to wrapping src data in an array and copying to self
    def assign(self, src):
        from warp.context import copy
        copy(self, array(src, dtype=self.dtype, copy=False, device=self.device))

    def numpy(self):

        if (self.device == "cpu"):

            # todo: make each wp type return it's corresponding ctype
            # ptr_type = ctypes.POINTER(type_ctype(self.dtype))
            # ptr = ctypes.cast(self.data, ptr_type)

            # view = np.ctypeslib.as_array(ptr, shape=(self.length, type_length(self.dtype)))
            # return view
            return np.array(self, copy=False)
        
        else:
            # bring back to CPU
            return self.to("cpu").numpy()


    def to(self, device):

        if (self.device == device):
            return self
        else:
            from warp.context import empty, copy, synchronize

            dest = empty(n=self.length, dtype=self.dtype, device=device)
            copy(dest, self)
            
            # todo: only synchronize when there is a device->host transfer outstanding
            synchronize()

            return dest

    def astype(self, dtype):

        # return an alias of the array memory with different type information
        src_length = self.length*type_length(self.dtype)
        src_capacity = self.capacity*type_length(self.dtype)

        dst_length = src_length/type_length(dtype)
        dst_capacity = src_capacity/type_length(dtype)

        if ((src_length % type_length(dtype)) > 0):
            raise RuntimeError("Dimensions are incompatible for type cast")

        arr = array(
            data=self.data, 
            dtype=dtype,
            length=int(dst_length),
            capacity=int(dst_capacity),
            device=self.device,
            owner=False)

        return arr


class Mesh:

    def __init__(self, points, velocities, indices):
        """ Class representing a triangle mesh.

        Attributes:
            id: Unique identifier for this mesh object, can be passed to kernels.
            device: Device this object lives on, all buffers must live on the same device.

        Args:
            points (:class:`warp.array`): Array of vertex positions of type :class:`wp.vec3`
            velocities (:class:`warp.array`): Array of vertex velocities of type :class:`wp.vec3`
            indices (:class:`warp.array`): Array of triangle indices of type :class:`wp.int32`, should be length 3*number of triangles
        """

        self.points = points
        self.velocities = velocities
        self.indices = indices

        if (points.device != indices.device):
            raise RuntimeError("Points and indices must live on the same device")

        # inherit context from points, todo: find this globally
        self.context = points.context
        self.device = points.device

        def get_data(array):
            if (array):
                return ctypes.c_void_p(array.data)
            else:
                return ctypes.c_void_p(0)


        if (self.device == "cpu"):
            self.id = self.context.core.mesh_create_host(
                get_data(points), 
                get_data(velocities), 
                get_data(indices), 
                int(points.length), 
                int(indices.length/3))
        else:
            self.id = self.context.core.mesh_create_device(
                get_data(points), 
                get_data(velocities), 
                get_data(indices), 
                int(points.length), 
                int(indices.length/3))


    def __del__(self):

        # we need to make sure correct CUDA device is set even during GC
        self.context.verify_device()

        if (self.device == "cpu"):
            self.context.core.mesh_destroy_host(self.id)
        else:
            self.context.core.mesh_destroy_device(self.id)

    def refit(self):
        """ Refit the BVH to points. This should be called after users modify the `points` data."""
                
        if (self.device == "cpu"):
            self.context.core.mesh_refit_host(self.id)
        else:
            self.context.core.mesh_refit_device(self.id)
            self.context.verify_device()



class Volume:

    def __init__(self, vdb):
        pass



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
        self.context = runtime
        
        if (device == "cpu"):
            self.id = self.context.core.hash_grid_create_host(dim_x, dim_y, dim_z)
        elif (device == "cuda"):
            self.id = self.context.core.hash_grid_create_device(dim_x, dim_y, dim_z)


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
        

        if (self.device == "cpu"):
            self.context.core.hash_grid_update_host(self.id, radius, ctypes.cast(points.data, ctypes.c_void_p), len(points))
        else:
            self.context.core.hash_grid_update_device(self.id, radius, ctypes.cast(points.data, ctypes.c_void_p), len(points))


    def __del__(self):

        if (self.device == "cpu"):
            self.context.core.hash_grid_destroy_host(self.id)
        else:
            self.context.core.hash_grid_destroy_device(self.id)


