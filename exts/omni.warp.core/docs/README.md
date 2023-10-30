# Warp Core [omni.warp.core]

This extension simply provides the core Python module for NVIDIA Warp.

It can be used by other extensions that want to use the Warp Python module with
minimal additional dependencies, such as for headless execution.

After enabling, use `import warp` from Python to use Warp inside the Omniverse Application's Python environment.

For higher-level components, including Warp-specialized OmniGraph nodes and sample scenes, see the
omni.warp extension.

## About Warp

NVIDIA Warp is a Python framework for writing high-performance simulation and graphics code.
Compute kernels are defined in Python syntax and JIT converted to C++/CUDA and compiled at runtime.

## Documentation

The online Warp documentation is hosted at https://nvidia.github.io/warp.
