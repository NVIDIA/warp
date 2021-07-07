# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pxr import Usd, UsdGeom, Gf, Sdf

import warp as wp

wp.init()

@wp.kernel
def test_kernel(
    x : wp.array(dtype=float),
    y : wp.array(dtype=float)):

    tid = wp.tid()

    y[tid] = x[tid]*2.0


device = "cpu"
dim = 8
iters = 4
tape = wp.Tape()

# record onto tape
with tape:
    
    # input data
    x0 = wp.array(np.zeros(dim), dtype=wp.float32, device=device, requires_grad=True)
    x = x0

    for i in range(iters):
    
        y = wp.empty_like(x, requires_grad=True)
        wp.launch(kernel=test_kernel, dim=dim, inputs=[x], outputs=[y], device=device)
        x = y

# loss = wp.sum(x)
adj_loss = wp.array(np.ones(dim), device=device, dtype=wp.float32)

# run backward
tape.backward({x: adj_loss})

# look up adjoint of x0
#print(x0)
print(tape.adjoints[x0])

# grad should be 2.0^iters
assert((tape.adjoints[x0].numpy() == np.ones(dim)*16.0).all())