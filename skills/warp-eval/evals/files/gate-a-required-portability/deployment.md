# Deployment contract

The neighbor-search implementation is part of the portable simulation core.
Every accelerated feature in that core must use one source implementation and
must pass the same correctness and performance qualification on NVIDIA CUDA,
AMD ROCm, and Apple Metal devices.

Backend-specific optional paths are not accepted for this component. In
particular, a CUDA-only implementation cannot ship alongside the portable path,
even as an opt-in extra.

The current all-pairs implementation is known to be algorithmically poor. A
portable cell list or spatial hash that preserves the shared-source contract is
within scope.
