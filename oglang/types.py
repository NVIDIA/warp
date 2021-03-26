
import ctypes
import numpy as np

#----------------------
# built-in types


class float3:
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

class float4:
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


class mat22:
    def __init__(self):
        pass

    @staticmethod
    def length():
        return 4

    @staticmethod
    def size():
        return 16

class mat33:
    def __init__(self):
        pass

    @staticmethod
    def length():
        return 9

    @staticmethod
    def size():
        return 36

class spatial_vector:
    def __init__(self):
        pass

    @staticmethod
    def length():
        return 6

    @staticmethod
    def size():
        return 36

class spatial_matrix:
    def __init__(self):
        pass

    @staticmethod
    def length():
        return 36

    @staticmethod
    def size():
        return 144

class spatial_transform:
    def __init__(self):
        pass

    @staticmethod
    def length():
        return 7
    
    @staticmethod
    def size():
        return 28

class void:
    def __init__(self):
        pass

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

class array:

    def __init__(self, type, length=0, capacity=0, data=None, device=None, context=None, owner=True):
        self.length = length
        self.capacity = capacity
        self.type = type
        self.data = data
        self.device = device
        self.context = context
        self.owner = owner

        self.__name__ = "array<" + type.__name__ + ">"

    def __del__(self):
        # todo: free data if owner
        pass


    def numpy(self):
        if (self.device == "cpu"):
            ptr_type = ctypes.POINTER(ctypes.c_float)
            ptr = ctypes.cast(self.data, ptr_type)

            view = np.ctypeslib.as_array(ptr,shape=(self.length, type_length(self.type)))
            return view
        else:
            print("Cannot convert CUDA array to numpy, copy to a device array")


    def to(self, device):
        if (self.device == device):
            return self
        else:
            return self.context.copy(self, device)


