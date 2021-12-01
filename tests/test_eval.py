# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import warp as wp

wp.init()

str = """
@wp.kernel
def add_vec3(dest: wp.array(dtype=wp.vec3),
             c: wp.vec3):

    tid = wp.tid()
    print(c)
    wp.store(dest, tid, c)
"""

# fails since we can't getsource() / AST for an exec'd Python function: https://stackoverflow.com/a/67898427
exec(str)

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


