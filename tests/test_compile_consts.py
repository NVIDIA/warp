import os
import sys

from warp.types import Volume

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import numpy as np

import test_compile_consts_dummy
# from test_compile_consts_dummy import *

wp.config.mode = "release"
wp.config.verbose = True
wp.config.cache_kernels = False
wp.config.verify_cuda = True

wp.init()

num_points = 10

device = "cuda"

LOCAL_ONE = wp.constant(1)

class Foobar:
    ONE = wp.constant(1)
    TWO = wp.constant(2)

@wp.kernel
def test(x: wp.array(dtype=int)):
    if Foobar.ONE > 0:
        x[wp.tid()] = 123 + Foobar.TWO + test_compile_consts_dummy.MINUS_ONE
    else:
        x[wp.tid()] = 456 + LOCAL_ONE
    

x = wp.array([1] * num_points, dtype=int, device=device)

try:
    wp.launch(kernel=test, dim=num_points, inputs=[x], device=device)
except Exception as e:
    print("FAIL: Kernel compilation and launch failed")
