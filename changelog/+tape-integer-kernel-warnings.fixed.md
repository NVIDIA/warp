Fix spurious `wp.Tape.backward()` warnings about kernels configured with `enable_backward=False` when all
of their array arguments hold integers or booleans, such as the kernels that build matrix structure in
`warp.sparse` operations and sparse-output `warp.fem.integrate()` and `warp.fem.interpolate()` calls.
