import warp as wp


@wp.kernel
def terminal_projection_kernel(grad: wp.array(dtype=wp.float32), delta: wp.float32):
    """Applies a stratified gradient projection using a resonance-based cosine scale.

    Args:
        grad: In-place gradient array of ``wp.float32`` values.
        delta: Resonance scale. Recommended range is ``[0.01, 0.1]``.

    Note:
        Gradients with high magnitudes relative to the delta parameter
        (|grad| > 2*pi/delta) will be projected toward zero due to the
        nature of the cosine modulation. For default delta=0.05, this
        threshold is approximately 125.7.
    """
    tid = wp.tid()
    s = grad[tid]

    # 1.5707963 is approx pi/2 to prevent gradient sign inversion
    angle = wp.clamp((s * delta) / 4.0, -1.5707963, 1.5707963)

    grad[tid] = s * wp.cos(angle)
