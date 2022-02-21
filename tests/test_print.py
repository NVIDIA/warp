# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
np.random.seed(42)

wp.init()
wp.verify_cuda = True

@wp.kernel
def test_print():

    wp.print(1.0)
    wp.print("this is a string")
    wp.printf("this is a float %f\n", 457.5)
    wp.printf("this is an int %d\n", 123)


wp.launch(
    kernel=test_print,
    dim=1,
    inputs=[],
    outputs=[],
    device="cuda")

wp.synchronize()
print("finished")

