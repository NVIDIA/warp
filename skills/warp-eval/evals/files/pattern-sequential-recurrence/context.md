# Adaptive-filter production context

One stream contains about 50 million float32 samples and the filter owns 72% of
batch latency. Every output sample is required, and each state update depends
nonlinearly on the exact preceding state. Reordering, approximating or batching
independent streams would change the product contract.

An optional NVIDIA backend and compiled dependency are permitted, so deployment
does not rule out Warp. The input may be staged on either host or device and the
boundary can be widened. There is no maintained CUDA implementation and the
stage is materially large.
