Add a `"block_jacobi"` preconditioner type to `warp.optim.linear.preconditioner()`. It inverts
the diagonal blocks of a `warp.sparse.BsrMatrix` via Cholesky factorization, generic over block
size, and falls back to standard (scalar) Jacobi for 1x1-block (CSR) matrices.
