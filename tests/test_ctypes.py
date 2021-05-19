# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import oglang as og

@og.kernel
def add_vec3(dest: og.array(dtype=og.vec3),
             c: og.vec3):

    tid = og.tid()

    og.store(dest, tid, c)

@og.kernel
def transform_vec3(dest: og.array(dtype=og.vec3),
    m: og.mat44,
    v: og.vec3):

    tid = og.tid()

    p = og.transform_point(m, v)
    og.store(dest, tid, p)

device = "cpu"
n = 32

dest = og.zeros(n=32, dtype=og.vec3, device=device)
c = np.array((1.0, 2.0, 3.0))
m = np.array(((1.0, 0.0, 0.0, 1.0),
              (0.0, 1.0, 0.0, 2.0),
              (0.0, 0.0, 1.0, 3.0),
              (0.0, 0.0, 0.0, 1.0)))

print("add_vec3")
og.launch(add_vec3, dim=n, inputs=[dest, c], device=device)
print(dest)


print("transform_vec3")
og.launch(transform_vec3, dim=n, inputs=[dest, m, c], device=device)
print(dest)



