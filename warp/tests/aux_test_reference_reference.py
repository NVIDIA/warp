# This file is used to test reloading module references.

import warp as wp

wp.init()


@wp.func
def more_magic():
    return 2.0
