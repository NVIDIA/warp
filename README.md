# NVIDIA Warp

Warp is a Python framework for writing high-performance simulation and graphics code. Kernels are defined in Python syntax and JIT converted to C++/CUDA and compiled at runtime.

##  Installation

### Local Python

To install in your local Python environment use:

    pip install -e .


## Requirements

For developers writing their own kernels the following are required:

    * Microsoft Visual Studio 2015 upwards (Windows)
    * GCC 4.0 upwards (Linux)
    * CUDA 11.0 upwards

To run built-in tests you should install the USD Core library to your Python environment using `pip install usd-core`.

## Building

Developers should run `build_lib.py` to build the `warp.dll` / `warp.so` core library. To build with CUDA support ensure that either `CUDA_HOME` or `CUDA_PATH` are set as environment variables pointing to the CUDA installation directory (this is done by default with the NVIDIA CUDA toolkit).

## Documentation

Please refer to the project [documentation](https://mmacklin.gitlab-master-pages.nvidia.com/warp/) for detailed language reference.

## Source

https://gitlab-master.nvidia.com/mmacklin/warp


