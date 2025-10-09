import warp as wp


def clear_kernel_cache():
    if hasattr(wp, "clear_kernel_cache"):
        return wp.clear_kernel_cache()

    # Fallback when benchmarking older versions of Warp that didn't have
    # `clear_kernel_cache` exposed to the root namespace.
    return wp.build.clear_kernel_cache()
