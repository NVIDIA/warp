Add the `noinline` and `forceinline` parameters to `@wp.func` to control whether a Warp
function is emitted out of line or inlined into its call sites. The two are mutually
exclusive, apply to the generated adjoint as well as the forward function, and are lowered
per backend so the decorated function remains valid for CPU and CUDA.
