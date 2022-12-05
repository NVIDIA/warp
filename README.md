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

@wp.kernel
def length(points: wp.array(dtype=wp.vec3),
           lengths: wp.array(dtype=float)):

    # thread index
    tid = wp.tid()
    
    # compute distance of each point from origin
    lengths[tid] = wp.length(points[tid])


# allocate an array of 3d points
points = wp.array(np.random.rand(num_points, 3), dtype=wp.vec3)
lengths = wp.zeros(num_points, dtype=float)

# launch kernel
wp.launch(kernel=length,
          dim=len(points),
          inputs=[points, lengths])

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
* GCC 6.0 upwards (Linux)
* CUDA Toolkit 11.5 or higher
* Git LFS installed (https://git-lfs.github.com/) 

After cloning the repository, users should run:

    python build_lib.py

This will generate the `warp.dll` / `warp.so` core library respectively. When building manually users should ensure that their CUDA_PATH environment variable is set, otherwise Warp will be built without CUDA support. Alternatively, the path to the CUDA toolkit can be passed to the build command as `--cuda_path="..."`. After building the Warp package should be installed using:

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

* [GTC 2022 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41599/).
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

## FAQ

### How does Warp relate to other Python projects for GPU programming, e.g.: Numba, Taichi, cuPy, PyTorch, etc?
-------

Warp is inspired by many of these projects, and is closely related to Numba and Taichi which both expose kernel programming to Python. These frameworks map to traditional GPU programming models, so many of the  high-level concepts are similar, however there are some functionality and implementation differences.

Compared to Numba, Warp supports a smaller subset of Python, but offering auto-differentiation of kernel programs, which is useful for machine learning. Compared to Taichi Warp uses C++/CUDA as an intermediate representation, which makes it convenient to implement and expose low-level routines. In addition, we are building in datastructures to support geometry processing (meshes, sparse volumes, point clouds, USD data) as first-class citizens that are not exposed in other runtimes.

Warp does not offer a full tensor-based programming model like PyTorch and JAX, but is designed to work well with these frameworks through data sharing mechanisms like `__cuda_array_interface__`. For computations that map well to tensors (e.g.: neural-network inference) it makes sense to use these existing tools. For problems with a lot of e.g.: sparsity, conditional logic, hetergenous workloads (like the ones we often find in simulation and graphics), then the kernel-based programming model like the one in Warp are often more convenient since users have control over individual threads.

### Does Warp support all of the Python language?
-------

No, Warp supports a subset of Python that maps well to the GPU. Our goal is to not have any performance cliffs so that users can expect consistently good behavior from kernels that is close to native code. Examples of unsupported concepts that don't map well to the GPU are dynamic types, list comprehensions, exceptions, garbage collection, etc.

### When should I call `wp.synchronize()`?
-------

One of the common sources of confusion for new users is when calls to `wp.synchronize()` are necessary. The answer is "almost never"! Synchronization is quite expensive, and should generally be avoided unless necessary. Warp naturally takes care of synchronization between operations (e.g.: kernel launches, device memory copies). 

For example, the following requires no manual synchronization, as the conversion to NumPy will automatically synchronize:

```python
# run some kernels
wp.launch(kernel_1, dim, [array_x, array_y], device="cuda")
wp.launch(kernel_2, dim, [array_y, array_z], device="cuda")

# bring data back to host (and implicitly synchronize)
x = array_z.numpy()
```

The _only_ case where manual synchronization is needed is when copies are being performed back to CPU asynchronously, e.g.:

```python
# copy data back to cpu from gpu, all copies will happen asynchronously to Python
wp.copy(cpu_array_1, gpu_array_1)
wp.copy(cpu_array_2, gpu_array_2)
wp.copy(cpu_array_3, gpu_array_3)

# ensure that the copies have finished
wp.synchronize()

# return a numpy wrapper around the cpu arrays, note there is no implicit synchronization here
a1 = cpu_array_1.numpy()
a2 = cpu_array_2.numpy()
a3 = cpu_array_3.numpy()
```

### What happens when you differentiate a function like `wp.abs(x)`?
-------

Non-smooth functions such as `y=|x|` do not have a single unique gradient at `x=0`, rather they have what is known as a `subgradient`, which is formally the convex hull of directional derivatives at that point. The way that Warp (and most auto-differentiation framworks) handles these points is to pick an arbitrary gradient from this set, e.g.: for `wp.abs()`, it will arbitrarily choose the gradient to be 1.0 at the origin. You can find the implementation for these functions in `warp/native/builtin.h`.

Most optimizers (particularly ones that exploit stochasticity), are not sensitive to the choice of which gradient to use from the subgradient, although there are exceptions.

### Does Warp support multi-GPU programming?
-------

Yes! Since version `0.4.0` we support allocating, launching, and copying between multiple GPUs in a single process. We follow the naming conventions of PyTorch and use aliases such as `cuda:0`, `cuda:1`, `cpu` to identify individual devices.

### Should I switch to Warp over IsaacGym / PhysX?
-------

Warp is not a replacement for IsaacGym, IsaacSim, or PhysX - while Warp does offer some physical simulation capabilities this is primarily aimed at developers who need differentiable physics, rather than a fully featured physics engine. Warp is also integrated with IsaacGym and is great for performing auxilary tasks such as reward and observation computations for reinforcement learning.



## Discord

We have a **#warp** channel on the public [Omniverse Discord](https://discord.com/invite/XWQNJDNuaC) sever, come chat to us!

## License

Warp is provided under the NVIDIA Source Code License (NVSCL), please see [LICENSE.md](./LICENSE.md) for full license text. Note that the license currently allows only non-commercial use of this code.