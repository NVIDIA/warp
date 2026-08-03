Fix pre-allocated iterative linear solver states to avoid device-memory allocations on every solve, including during
CUDA graph capture.
