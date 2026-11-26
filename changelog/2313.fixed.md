Fix `wp.config.verify_autograd_array_access` missing reads of an array that is the left operand of a binary
operator, such as `a[tid] * b[tid]`, `(a[tid] - b[tid]) ** 2.0`, or `wp.sin(a[tid] + 1.0)`, so that writing the array
afterwards now emits the overwrite warning. `x[tid] = x[tid] + y` and `x[tid] = x[tid] - y` still don't warn, like
`x[tid] += y`.
