import warp as wp


@wp.kernel
def terminal_projection_kernel(grad: wp.array(dtype=wp.float32), delta: wp.float32):
    """
    Applies a stratified cosine scaling to the gradient field.

    Note: This operation is a resonance-based scaling, not a mathematical projection.
    It is not idempotent; applying the kernel multiple times will result in
    successive scaling of the gradient values.
    """
    tid = wp.tid()

    s = grad[tid]

    # Stratified scaling logic
    angle = (s * delta) / 4.0

    # Clamp to prevent sign inversion
    angle = wp.clamp(angle, -1.5707963, 1.5707963)

    grad[tid] = s * wp.cos(angle)
