# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Struct
#
# Shows how to define a custom struct and use it during gradient computation
#
###########################################################################

import os
import numpy as np
import warp as wp

wp.init()

@wp.struct
class TestStruct:
    x: wp.vec3
    a: wp.array(dtype=wp.vec3)
    b: wp.array(dtype=wp.vec3)


@wp.kernel
def test_kernel(s: TestStruct):
    tid = wp.tid()

    s.b[tid] = s.a[tid] + s.x


@wp.kernel
def loss_kernel(s: TestStruct, 
                loss: wp.array(dtype=float)):
    
    tid = wp.tid()
    
    v = s.b[tid]
    wp.atomic_add(loss, 0, float(tid + 1) * (v[0] + 2.0 * v[1] + 3.0 * v[2]))

# create struct
ts = TestStruct()

# set members
ts.x = wp.vec3(1.0, 2.0, 3.0)
ts.a = wp.array(
    np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    dtype=wp.vec3,
    requires_grad=True,
)
ts.b = wp.zeros(2, dtype=wp.vec3, requires_grad=True)

loss = wp.zeros(1, dtype=float, requires_grad=True)

tape = wp.Tape()
with tape:
    wp.launch(test_kernel, dim=2, inputs=[ts])
    wp.launch(loss_kernel, dim=2, inputs=[ts, loss])

tape.backward(loss)

print(loss)
print(tape.gradients[ts].a)
