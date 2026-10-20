Document that `wp.atomic_add()` for `wp.float64` returns zero without updating
the target memory when targeting CUDA GPUs with compute capability below 6.0.
