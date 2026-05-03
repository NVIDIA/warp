import warp as wp


@wp.kernel
def terminal_projection_kernel(grad: wp.array(dtype=wp.float32), delta: wp.float32):
    """Project gradients using cosine-modulated terminal projection.

    Args:
        grad: In-place gradient array of ``wp.float32`` values.
        delta: Resonance scale. Recommended range is ``[0.01, 0.1]``.
    """
    tid = wp.tid()
    s = grad[tid]

    # Clamp angle to [-pi/2, pi/2] to prevent gradient sign inversion
    # 1.5707963 is approx pi/2
    angle = wp.clamp((s * delta) / 4.0, -1.5707963, 1.5707963)

    grad[tid] = s * wp.cos(angle)
