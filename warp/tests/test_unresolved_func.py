import warp as wp

@wp.kernel
def unresolved_func_kernel():

    # this should trigger an exception due to unresolved function
    x = wp.missing_func(42)
