import numpy as np

array = np.zeros((2,3), dtype=np.float32)

#l = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]
l = [[1, 2, 3], [2, 4, 6]]
np.copyto(array, l)
print(l)


array = np.zeros((1,), dtype=np.int32)
np.copyto(array, [3.141], casting='same_kind')
print(array)

array = np.zeros((0,), dtype=np.int32)
np.copyto(array, [], casting='unsafe')
print(array)