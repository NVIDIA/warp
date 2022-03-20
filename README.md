# NVIDIA Warp

Warp is a Python framework for writing high-performance simulation and graphics code. Kernels are defined in Python syntax and JIT converted to C++/CUDA and compiled at runtime.

Please refer to the project [documentation](https://nvidia.github.io/warp/) for detailed API and language reference.


<img src="./docs/img/header.png"/>
<img src="./docs/img/nvdb_flow.png"/>

## Installing

Pre-built packages for Windows and Linux are available on the [releases](https://github.com/NVIDIA/warp) page. To install in your local Python environment use:

    pip install -e .

From the root directory of this repository.

## Building

For developers wanting to build the library themselves the following tools are required:

* Microsoft Visual Studio 2017 upwards (Windows)
* GCC 4.0 upwards (Linux)
* CUDA Toolkit 11.3

After cloning the repository, developers should run `build.bat` or `build.sh` to generate the `warp.dll` / `warp.so` core library respectively.

## Running Examples

The `examples` directory contains a number of scripts that show how to use the API. Most examples will generate USD files containing time-sampled animations in the ``examples/outputs`` directory. Before running examples users should ensure that the ``usd-core`` package is installed using:

    pip install usd-core
    
USD files can be viewed or rendered inside NVIDA Omniverse, Pixar's UsdView, or in Preview on macOS.

## Omniverse

The Warp Omniverse extension is available in the extension registry inside Kit or Create. 

<img src="./docs/img/omniverse.png" width=400px/>

Enabling the extension will automatically install and initialize the Warp Python module inside the Kit Python environment. Please see the [Omniverse Warp Documentation](http://docs.omniverse.nvidia.com/extensions/warp.html) for more details on how to use Warp in Omniverse.

## Discord

We have a **#warp** channel on the public [Omniverse Discord](https://discord.com/invite/XWQNJDNuaC) sever, come chat to us!


