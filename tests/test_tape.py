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

# MM: can use this config to verify device launches succeed
wp.config.verify_cuda = True

device = "cuda"
@wp.kernel
def test(x: wp.array(dtype=float), y: wp.array(dtype=float), output: wp.array(dtype=float), dim: int):
    wp.dense_gemm(int(1), int(1), dim, int(1), int(1), x, y, output)

output_adjoint = wp.array([1], dtype=float, device=device)
output = wp.empty(n=1, dtype=float, device=device, requires_grad=True)

# MM: set inputs to be initialized to some reasonable values
x = wp.array(np.ones(10), dtype=float, device=device, requires_grad=True)

# MM: marked this also as requires_grad, seems Warp is trying to write this adjoint incorrectly
y = wp.array(np.ones(10), dtype=float, device=device, requires_grad=True) 

for i in range(20):

    tape = wp.Tape()
    print(f"Index: {i}")
    
    # MM: disabled the tape reset as this will also clear the output_adjoint incorrectly (Warp bug)
    #tape.reset()
    with tape:
        # MM: dense GEMM uses a fixed block size of 256 
        wp.launch(kernel=test, dim=256, inputs=[x, y, output, 10], device=device)
        print(output.numpy())
    
    tape.backward(adj_user={output: output_adjoint})
    print(tape.adjoints[x].numpy())



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


@wp.kernel
def simple_kernel(a: wp.array(dtype=wp.vec3),
                    b: wp.array(dtype=wp.vec3),
                    c: wp.array(dtype=float)):

    # get thread index
    tid = wp.tid()

    # load two vec3s
    x = a[tid]
    y = b[tid]

    # compute the dot product between vectors
    r = wp.dot(x, y)

    # write result back to memory
    c[tid] = r