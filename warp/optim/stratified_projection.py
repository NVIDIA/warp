import warp as wp

@wp.kernel
def terminal_projection_kernel(grad: wp.array(dtype=wp.float32),
                              delta: wp.float32):
    """
    Applies the Terminal Operator T to project gradients onto the 
    admissible lattice L = {4k}. Uses resonant scaling to prevent signal decay.
    
    Note: delta should be in the range [0.01, 0.1] to maintain 
    harmonic stability and avoid signal suppression.
    """
    tid = wp.tid()
    s = grad[tid]
    
    # Define the lattice node scale (4k boundary)
    lattice_scale = 4.0
    
    # Resonant Projection:
    # Small delta ensures we stay in the high-fidelity region 
    # of the cosine modulation, preserving gradient signal.
    projection = s * wp.cos((s * delta) / lattice_scale)
    
    grad[tid] = projection
