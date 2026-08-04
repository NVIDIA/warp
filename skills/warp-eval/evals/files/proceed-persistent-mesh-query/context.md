# Persistent query production context

The public operation processes about two million CUDA-resident float32 points
against a static mesh of 1.5 million triangles on every frame. The mesh is built
once and reused for thousands of frames; results feed another CUDA stage.

A production profile attributes 62% of frame latency and 1.8 GiB of transient
host memory to the current host fallback. No maintained CUDA implementation
exists in the project or its dependencies.

An optional NVIDIA backend and compiled dependency are explicitly permitted.
The existing implementation remains the fallback. The contract requires one
candidate face per point, int32 output, stable shapes and an overflow-free
result; exact tie ordering, float64 and gradients are not required.
