
import ctypes
import numpy as np

#----------------------
# built-in types


class vec3:
    def __init__(self):
        x = 0.0
        y = 0.0
        z = 0.0

    @staticmethod
    def length():
        return 3

    @staticmethod
    def size():
        return 12

    @staticmethod
    def ctype():
        return ctypes.c_float        



class vec4:
    def __init__(self):
        x = 0.0
        y = 0.0
        z = 0.0
        w = 0.0

    @staticmethod
    def length():
        return 4


    @staticmethod
    def size():
        return 16

    @staticmethod
    def ctype():
        return ctypes.c_float        

class quat:
    def __init__(self):
        x = 0.0
        y = 0.0
        z = 0.0
        w = 1.0

    @staticmethod
    def length():
        return 4

    @staticmethod
    def size():
        return 16

    @staticmethod
    def ctype():
        return ctypes.c_float        


class mat22:
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

class mat33:
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

class spatial_vector:
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

class spatial_matrix:
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

class spatial_transform:
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


class array:

    def __init__(self, dtype, length=0, capacity=0, data=None, device=None, context=None, owner=True):
        self.length = length
        self.capacity = capacity
        self.dtype = dtype
        self.data = data
        self.device = device
        self.context = context
        self.owner = owner

        self.__name__ = "array<" + type.__name__ + ">"

        # set up array interface access so we can treat this object a read-only numpy array
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
            
            # todo: only synchronize  when there is a device->host transfer outstanding
            synchronize()

            return dest



class Mesh:

    def __init__(self, points, indices, device):
        
        self.points = points
        self.indices = indices

        # inherit context from points, todo: find this globally
        self.context = points.context
        self.device = device

        if (self.device == "cpu"):
            self.id = uint64(self.context.core.mesh_create_host(points.data, indices.data, points.length, int(indices.length/3)))
        else:
            self.id = uint64(self.context.core.mesh_create_device(points.data, indices.data, points.length, int(indices.length/3)))


    def __del__(self):

        if (self.device == "cpu"):
            self.context.core.mesh_destroy_host(self.id.value)
        else:
            self.context.core.mesh_destroy_device(self.id.value)

    def refit(self):
        
        if (self.device == "cpu"):
            self.context.core.mesh_refit_host(self.id.value)
        else:
            self.context.core.mesh_refit_device(self.id.value)



class Volume:

    def __init__(self, vdb):
        pass



