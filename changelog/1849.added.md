Add the `inline` parameter to `@wp.func` to control whether a Warp function is inlined into its
call sites. `None` leaves the choice to the backend compiler, `True` requires inlining, and
`False` keeps the function out of line. The hint applies to the generated adjoint as well as the
forward function, and is lowered per backend so the decorated function remains valid for CPU and
CUDA.
