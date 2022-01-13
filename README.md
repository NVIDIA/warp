# NVIDIA Warp

Warp is a Python framework for writing high-performance simulation and graphics code. Kernels are defined in Python syntax and JIT converted to C++/CUDA and compiled at runtime.

## Building

After cloning the repository, developers should run `build.bat` or `build.sh` to generate the `warp.dll` / `warp.so` core library respectively. CUDA dependencies will automatically be fetched.

## Pre-built Binaries

Pre-built binaries for Windows and Linux are available as artifacts on the following TeamCity instance:

https://teamcity.nvidia.com/project/Sandbox_mmacklin_Warp?mode=builds

## Installing

To install in your local Python environment use:

    pip install -e .

To run built-in tests you should install the USD Core library to your Python environment using `pip install usd-core`.

## Omniverse

The Warp Omniverse extension is available in the extension registry inside Kit or Create daily builds. 

![](./docs/img/omniverse.png)

Enabling the extension will automatically install and initialize the Warp Python module inside the Kit Python environment.

If the Warp extension is not visible, (e.g.: for in public or non-daily builds) then you need to add the NVIDIA internal extension registry to your extensions settings: `omniverse://kit-extensions.ov.nvidia.com/exts/kit/default`.

## Documentation

Please refer to the project [documentation](https://mmacklin.gitlab-master-pages.nvidia.com/warp/) for detailed API and language reference.

Please see `#omni-warp` on Slack for discussion and reporting bugs.

## Source

https://gitlab-master.nvidia.com/mmacklin/warp


