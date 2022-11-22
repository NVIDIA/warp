import warp as wp

@wp.kernel
def unresolved_symbol_kernel():

    # this should trigger an exception due to unresolved symbol
    x = missing_symbol
