
import ctypes 
import numpy as np
from copy import copy as shallowcopy

#----------------------
# built-in types


class vec3(ctypes.Structure):
    
    _fields_ = [("value", ctypes.c_float*3)]
    
    def __init__(self):
        pass        

    @staticmethod
    def length():
        return 3

    @staticmethod
    def size():
        return 12

    @staticmethod
    def ctype():
        return ctypes.c_float


class vec4(ctypes.Structure):
    
    _fields_ = [("value", ctypes.c_float*4)]

    def __init__(self):
        pass

    @staticmethod
    def length():
        return 4

    @staticmethod
    def size():
        return 16

    @staticmethod
    def ctype():
        return ctypes.c_float

class quat(ctypes.Structure):
    
    _fields_ = [("value", ctypes.c_float*4)]

    def __init__(self):
        pass
    
    @staticmethod
    def length():
        return 4

    @staticmethod
    def size():
        return 16

    @staticmethod
    def ctype():
        return ctypes.c_float        


class mat22(ctypes.Structure):
    
    _fields_ = [("value", ctypes.c_float*4)]
    
    def __init__(self):
        pass
    
    @staticmethod
    def length():
        return 4

    @staticmethod
    def size():
        return 16

    @staticmethod
    def ctype():
        return ctypes.c_float        

class mat33(ctypes.Structure):
    
    _fields_ = [("value", ctypes.c_float*9)]
    
    def __init__(self):
        pass
    
    @staticmethod
    def length():
        return 9

    @staticmethod
    def size():
        return 36

    @staticmethod
    def ctype():
        return ctypes.c_float

class mat44(ctypes.Structure):
    
    _fields_ = [("value", ctypes.c_float*16)]
    
    def __init__(self):
        pass
    
    @staticmethod
    def length():
        return 16

    @staticmethod
    def size():
        return 64

    @staticmethod
    def ctype():
        return ctypes.c_float


class spatial_vector(ctypes.Structure):
    
    _fields_ = [("value", ctypes.c_float*6)]

    def __init__(self):
        pass
    
    @staticmethod
    def length():
        return 6

    @staticmethod
    def size():
        return 36

    @staticmethod
    def ctype():
        return ctypes.c_float        

class spatial_matrix(ctypes.Structure):
    
    _fields_ = [("value", ctypes.c_float*36)]
    
    def __init__(self):
        pass
    
    @staticmethod
    def length():
        return 36

    @staticmethod
    def size():
        return 144

    @staticmethod
    def ctype():
        return ctypes.c_float

class spatial_transform(ctypes.Structure):
    
    _fields_ = [("value", ctypes.c_float*7)]
    
    def __init__(self):
        pass
    
    @staticmethod
    def length():
        return 7
    
    @staticmethod
    def size():
        return 28

    @staticmethod
    def ctype():
        return ctypes.c_float        

class void:

    def __init__(self):
        pass

class float32:

    @staticmethod
    def length():
        return 1
    
    @staticmethod
    def size():
        return 4

    @staticmethod
    def ctype():
        return ctypes.c_float        

class float64:

    @staticmethod
    def length():
        return 1
    
    @staticmethod
    def size():
        return 8

    @staticmethod
    def ctype():
        return ctypes.c_double


class int32:

    def __init__(self, x=0):
        self.value = x

    @staticmethod
    def length():
        return 1
    
    @staticmethod
    def size():
        return 4

    @staticmethod
    def ctype():
        return ctypes.c_int32


class uint32:

    def __init__(self, x=0):
        self.value = x

    @staticmethod
    def length():
        return 1
    
    @staticmethod
    def size():
        return 4

    @staticmethod
    def ctype():
        return ctypes.c_uint32


class int64:

    def __init__(self, x=0):
        self.value = x

    @staticmethod
    def length():
        return 1
    
    @staticmethod
    def size():
        return 8

    @staticmethod
    def ctype():
        return ctypes.c_int64


class uint64:

    def __init__(self, x=0):
        self.value = x

    @staticmethod
    def length():
        return 1
    
    @staticmethod
    def size():
        return 8

    @staticmethod
    def ctype():
        return ctypes.c_uint64


def type_length(dtype):
    if (dtype == float or dtype == int):
        return 1
    else:
        return dtype.length()

def type_size_in_bytes(dtype):
    if (dtype == float or dtype == int):
        return 4
    else:
        return dtype.size()

def type_ctype(dtype):
    if (dtype == float):
        return ctypes.c_float
    elif (dtype == int):
        return ctypes.c_int32
    else:
        return dtype.ctype()

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

class array:

    def __init__(self, data=None, dtype=float32, length=0, capacity=0, device=None, context=None, owner=True):
        
        # convert built-in numeric type to og type
        if (dtype == int):
            dtype = int32

        elif (dtype == float):
            dtype = float32


        # if src is a list, tuple try to convert to numpy array and construct from that (data will be copied)
        if (isinstance(data, np.ndarray) or 
            isinstance(data, list) or 
            isinstance(data, tuple)):

            from oglang.context import empty, copy, synchronize

            arr = np.array(data)

            # attempt to convert from double to float precision
            if (arr.dtype == np.float64):
                arr = arr.astype(np.float32)

            # todo: need a more robust way to convert types
            if (arr.dtype == np.int64 and dtype==int32):
                arr = arr.astype(np.int32)


            # if array is multi-dimensional, but data type is scalar, then flatten
            if (len(arr.shape) > 1 and type_length(dtype) == 1):
                arr = arr.flatten()

            ptr = arr.__array_interface__["data"][0]
            shape = arr.__array_interface__["shape"]
            rows = shape[0]

            #if (arr.__array_interface__["typestr"] != "<i4" and arr.__array_interface__["typestr"] != "<f4"):
            #if (arr.__array_interface__["typestr"] != "<i4" and arr.__array_interface__["typestr"] != "<f4"):
                #raise RuntimeError("Source numpy array must be either 32bit integer or floating point data")

            if (arr.__array_interface__["typestr"] == "<f8"):
                raise RuntimeError("64bit floating point (double) data type not supported")

            src = array(dtype=dtype, length=rows, capacity=rows*type_size_in_bytes(dtype), data=ptr, device='cpu', context=context, owner=False)
            dest = empty(rows, dtype=dtype, device=device)
            dest.owner = False

            # data copy
            copy(dest, src)

            # object copy to self and transfer data ownership, would probably be cleaner to have _empty, _zero, etc as class methods
            self.__dict__ = shallowcopy(dest.__dict__)
            self.owner = True
           
            
        else:
            
            # explicit construction, data is interpreted as the address for raw memory 
            self.length = length
            self.capacity = capacity
            self.dtype = dtype
            self.data = data
            self.device = device
            self.context = context
            self.owner = owner

            self.__name__ = "array<" + type.__name__ + ">"

            # set up array interface access so we can treat this object as a read-only numpy array
            if (device == "cpu"):

                self.__array_interface__ = { 
                    "data": (data, False), 
                    "shape": (self.length, type_length(self.dtype)),  
                    "typestr": type_typestr(type_ctype(dtype)), 
                    "version": 3 
                }


    def __del__(self):
        
        if (self.owner and self.context):

            if (self.device == "cpu"):
                self.context.free_host(self.data)
            else:
                self.context.free_device(self.data)
                

    def __len__(self):

        return self.length

    def __str__(self):

        return str(self.to("cpu").numpy())

    def zero_(self):

        if (self.device == "cpu"):
            self.context.core.memset_host(ctypes.cast(self.data,ctypes.POINTER(ctypes.c_int)), 0, self.capacity)

        if(self.device == "cuda"):
            self.context.core.memset_device(ctypes.cast(self.data,ctypes.POINTER(ctypes.c_int)), 0, self.capacity)

    def numpy(self):

        if (self.device == "cpu"):

            # todo: make each og type return it's corresponding ctype
            # ptr_type = ctypes.POINTER(type_ctype(self.dtype))
            # ptr = ctypes.cast(self.data, ptr_type)

            # view = np.ctypeslib.as_array(ptr, shape=(self.length, type_length(self.dtype)))
            # return view
            return np.array(self, copy=False)
        
        else:
            raise RuntimeError("Cannot convert CUDA array to numpy, copy to a host array first")


    def to(self, device):

        if (self.device == device):
            return self
        else:
            from oglang.context import empty, copy, synchronize

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
            context=self.context,
            owner=False)

        return arr

    #  def __getstate__(self):
    #      # capture what is normally pickled
    #      state = self.__dict__.copy()
    #      # replace the `value` key (now an EnumValue instance), with it's index:
    #      state['value'] = state['value'].index
    #      # what we return here will be stored in the pickle
    #      return state

    #  def __setstate__(self, newstate):
    #      # re-create the EnumState instance based on the stored index
    #      newstate['value'] = self.Values[newstate['value']]
    #      # re-instate our __dict__ state from the pickled state
    #      self.__dict__.update(newstate)

def get_data(array):
    if (array):
        return ctypes.c_void_p(array.data)
    else:
        return ctypes.c_void_p(0)

class Mesh:

    def __init__(self, points, velocities, indices):
        
        self.points = points
        self.velocities = velocities
        self.indices = indices

        if (points.device != indices.device):
            raise RuntimeError("Points and indices must live on the same device")

        # inherit context from points, todo: find this globally
        self.context = points.context
        self.device = points.device

        

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

        if (self.device == "cpu"):
            self.context.core.mesh_destroy_host(self.id)
        else:
            self.context.core.mesh_destroy_device(self.id)

    def refit(self):
        
        if (self.device == "cpu"):
            self.context.core.mesh_refit_host(self.id)
        else:
            self.context.core.mesh_refit_device(self.id)



class Volume:

    def __init__(self, vdb):
        pass



