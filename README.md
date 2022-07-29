# NVIDIA Warp (Preview)

Warp is a Python framework for writing high-performance simulation and graphics code. Kernels are defined in Python syntax and JIT converted to C++/CUDA and compiled at runtime.

Warp is comes with a rich set of primitives that make it easy to write programs for physics simulation, geometry processing, and procedural animation. In addition, Warp kernels are differentiable, and can be used as part of machine-learning training pipelines with other frameworks such as PyTorch.

Please refer to the project [Documentation](https://nvidia.github.io/warp/) for API and language reference and [CHANGELOG.md](./CHANGELOG.md) for release history.

![](./docs/img/gifs/aldina.gif) ![](./docs/img/gifs/nanovdb.gif)
![](./docs/img/gifs/ocean.gif) ![](./docs/img/gifs/particles.gif)

_A selection of physical simulations computed with Warp_

## Installing 

Warp supports Python versions 3.7.x-3.9.x. The easiest way is to install from PyPi:

    pip install warp-lang

Pre-built binary packages for Windows and Linux are also available on the [Releases](https://github.com/NVIDIA/warp/releases) page. To install in your local Python environment extract the archive and run the following command from the root directory:

    pip install .

## Getting Started

An example first program that computes the lengths of random 3D vectors is given below:

```python
import warp as wp
import numpy as np

wp.init()

num_points = 1024
device = "cuda"

@wp.kernel
def length(points: wp.array(dtype=wp.vec3),
           lengths: wp.array(dtype=float)):

    # thread index
    tid = wp.tid()
    
    # compute distance of each point from origin
    lengths[tid] = wp.length(points[tid])


# allocate an array of 3d points
points = wp.array(np.random.rand(num_points, 3), dtype=wp.vec3, device=device)
lengths = wp.zeros(num_points, dtype=float, device=device)

# launch kernel
wp.launch(kernel=length,
          dim=len(points),
          inputs=[points, lengths],
          device=device)

print(lengths)
```

## Running Examples

The `examples` directory contains a number of scripts that show how to implement different simulation methods using the Warp API. Most examples will generate USD files containing time-sampled animations in the ``examples/outputs`` directory. Before running examples users should ensure that the ``usd-core`` package is installed using:

    pip install usd-core
    
USD files can be viewed or rendered inside NVIDIA [Omniverse](https://developer.nvidia.com/nvidia-omniverse-platform), Pixar's UsdView, and Blender. Note that Preview in macOS is not recommended as it has limited support for time-sampled animations.

Built-in unit tests can be run from the command-line as follows:

    python -m warp.tests

## Building

For developers who want to build the library themselves the following tools are required:

* Microsoft Visual Studio 2017 upwards (Windows)
* GCC 4.0 upwards (Linux)
* CUDA Toolkit 11.3 or higher
* Git LFS installed (https://git-lfs.github.com/) 

After cloning the repository, users should run:

    python build_lib.py

This will generate the `warp.dll` / `warp.so` core library respectively. When building manually users should ensure that their CUDA_PATH environment variable is set and dynamic libraries can be found at runtime. After building the Warp package should be installed using:

    pip install -e .

Which ensures that subsequent modifications to the libary will be reflected in the Python package.

If you are cloning from Windows, please first ensure that you have enabled "Developer Mode" in Windows settings and symlinks in git:

    git config --global core.symlinks true

This will ensure symlinks inside ``exts/omni.warp`` work upon cloning.

## Omniverse

A Warp Omniverse extension is available in the extension registry inside Omniverse Kit or Create:

<img src="./docs/img/omniverse.png" width=550px/>

Enabling the extension will automatically install and initialize the Warp Python module inside the Kit Python environment. Please see the [Omniverse Warp Documentation](http://docs.omniverse.nvidia.com/extensions/warp.html) for more details on how to use Warp in Omniverse.

## Learn More

Please see the following resources for additional background on Warp:

* [GTC 2022 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41599/)
* [GTC 2021 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31838/)
* [SIGGRAPH Asia 2021 Differentiable Simulation Course](https://dl.acm.org/doi/abs/10.1145/3476117.3483433)

The underlying technology in Warp has been used in a number of research projects at NVIDIA including the following publications:

* Accelerated Policy Learning with Parallel Differentiable Simulation - Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A., & Macklin, M. [(2022)](https://short-horizon-actor-critic.github.io/)
* DiSECt: Differentiable Simulator for Robotic Cutting - Heiden, E., Macklin, M., Narang, Y., Fox, D., Garg, A., & Ramos, F [(2021)](https://github.com/NVlabs/DiSECt)
* gradSim: Differentiable Simulation for System Identification and Visuomotor Control - Murthy, J. Krishna, Miles Macklin, Florian Golemo, Vikram Voleti, Linda Petrini, Martin Weiss, Breandan Considine et al. [(2021)](https://krrish94.github.io/publication/2021-gradsim/)


## Citing

If you use Warp in your research please use the following citation:

```bibtex
@misc{warp2022,
title= {Warp: A High-performance Python Framework for GPU Simulation and Graphics},
author = {Miles Macklin},
month = {March},
year = {2022},
note= {NVIDIA GPU Technology Conference (GTC)},
howpublished = {\url{https://github.com/nvidia/warp}}
}
```


## Discord

We have a **#warp** channel on the public [Omniverse Discord](https://discord.com/invite/XWQNJDNuaC) sever, come chat to us!

## License

Warp is provided under the NVIDIA Source Code License (NVSCL), please see [LICENSE.md](./LICENSE.md) for full license text. Note that the license currently allows only non-commercial use of this code.