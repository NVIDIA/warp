import warp as wp

@wp.kernel
def terminal_projection_kernel(grad: wp.array(dtype=wp.float32), 
                               delta: wp.float32):
    """
    Applies the Terminal Operator T to project gradients onto the
    admissible lattice L = {4k}. Prevents signal decay.
    """
    tid = wp.tid()
    s = grad[tid]
    
    # Floor projection to the nearest lock point (L = 4k)
    k_s = wp.floor(s / 4.0) * 4.0
    
    # Delta ensures stability without searching
    if delta > 0.5:
        grad[tid] = k_s + 4.0
    else:
        grad[tid] = k_s
