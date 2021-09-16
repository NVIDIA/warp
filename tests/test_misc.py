import numpy as np
import ctypes


class Test(ctypes.Array):

    _length_ = 4
    _type_ = ctypes.c_float
    #__fields__ = [("value", *4) ]

    def __init__(self, x, y, z, w):
        


v = np.ones(4)
print(*v)

t = Test(*v)
print(*t)



