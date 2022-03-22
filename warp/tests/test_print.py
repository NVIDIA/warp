# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import numpy as np
import math

import warp as wp
from warp.tests.test_base import *

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

