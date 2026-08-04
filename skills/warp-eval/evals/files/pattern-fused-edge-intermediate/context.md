# Message-passing production context

Production graphs contain 40–60 million edges with 64 float32 features. Inputs
and outputs remain CUDA-resident. The materialized `[edges, features]` message
tensor peaks near 15 GiB and prevents larger graphs from running.

The public stage accounts for 48% of frame latency and most peak memory.
`torch.compile` preserves the gather, transform and `index_add_` as separate
operations for the dynamic edge list. The project has no other graph backend.

An optional NVIDIA backend and compiled dependency are permitted. Gradients are
not required through this stage. Output accumulation must match the reference
within the predeclared floating-point tolerance; edge ordering itself is not
contractual.
