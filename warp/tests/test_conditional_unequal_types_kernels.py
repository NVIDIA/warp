# This file defines a kernel that fails on codegen.py

import warp as wp


@wp.kernel
def unequal_types_kernel():
    x = wp.int32(10)
    y = 10
    z = True

    # Throws a TypeError
    if x == y == z:
        pass
