# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import warp as wp

wp.init()

@wp.kernel
def add_vec3(dest: wp.array(dtype=wp.vec3),
             c: wp.vec3):

    tid = wp.tid()
    print(c)
    wp.store(dest, tid, c)

@wp.kernel
def transform_vec3(dest: wp.array(dtype=wp.vec3),
    m: wp.mat44,
    v: wp.vec3):

    tid = wp.tid()

    p = wp.transform_point(m, v)
    wp.store(dest, tid, p)

device = "cpu"
n = 32

dest = wp.zeros(n=32, dtype=wp.vec3, device=device)
c = np.array((1.0, 2.0, 3.0))
m = np.array(((1.0, 0.0, 0.0, 1.0),
              (0.0, 1.0, 0.0, 2.0),
              (0.0, 0.0, 1.0, 3.0),
              (0.0, 0.0, 0.0, 1.0)))

print("add_vec3")
wp.launch(add_vec3, dim=n, inputs=[dest, c], device=device)
print(dest)


print("transform_vec3")
wp.launch(transform_vec3, dim=n, inputs=[dest, m, c], device=device)
print(dest)



