# NVIDIA Warp

Warp is a Python framework for writing high-performance simulation and graphics code. Warp takes
regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU.

Warp is designed for spatial computing and comes with a rich set of primitives that make it easy to write
programs for physics simulation, perception, robotics, and geometry processing. In addition, Warp kernels
are differentiable and can be used as part of machine-learning pipelines with frameworks such as PyTorch and JAX.

Please refer to the project [Documentation](https://nvidia.github.io/warp/) for API and language reference and [CHANGELOG.md](./CHANGELOG.md) for release history.

<div align="center">
    <img src="https://github.com/NVIDIA/warp/raw/main/docs/img/header.jpg">
    <p><i>A selection of physical simulations computed with Warp</i></p>
</div>


## Installing

Warp supports Python versions 3.7 onwards. It can run on x86-64 and ARMv8 CPUs on Windows, Linux, and macOS. GPU support requires a CUDA capable NVIDIA GPU and driver (minimum GeForce GTX 9xx).

The easiest way to install Warp is from [PyPI](https://pypi.org/project/warp-lang/):

    pip install warp-lang

Pre-built binary packages are also available on the [Releases](https://github.com/NVIDIA/warp/releases) page. To install in your local Python environment run the following command from the download directory:

    pip install warp_lang-<version and platform>.whl

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

The `examples` directory contains a number of scripts that show how to implement different simulation methods using the Warp API. Most examples will generate USD files containing time-sampled animations (stored in the same directory as the example). Before running examples, users should ensure that the ``usd-core`` and ``matplotlib`` packages are installed using:

    pip install usd-core matplotlib

Examples can be run from the command-line as follows:

    python -m warp.examples.<example_subdir>.<example>

To browse the example source code, you can open the directory where the files are located like this:

    python -m warp.examples.browse

Most examples can be run on either the CPU or a CUDA-capable device, but a handful require a CUDA-capable device. These are marked at the top of the example script.

USD files can be viewed or rendered inside [NVIDIA Omniverse](https://developer.nvidia.com/omniverse), Pixar's UsdView, and Blender. Note that Preview in macOS is not recommended as it has limited support for time-sampled animations.

Built-in unit tests can be run from the command-line as follows:

    python -m warp.tests


### examples/core

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_dem.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/core_dem.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_fluid.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/core_fluid.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_graph_capture.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/core_graph_capture.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_marching_cubes.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/core_marching_cubes.png"></a></td>
        </tr>
        <tr>
            <td align="center">dem</td>
            <td align="center">fluid</td>
            <td align="center">graph capture</td>
            <td align="center">marching cubes</td>
        </tr>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_mesh.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/core_mesh.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_nvdb.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/core_nvdb.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_raycast.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/core_raycast.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_raymarch.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/core_raymarch.png"></a></td>
        </tr>
        <tr>
            <td align="center">mesh</td>
            <td align="center">nvdb</td>
            <td align="center">raycast</td>
            <td align="center">raymarch</td>
        </tr>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_sph.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/core_sph.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_torch.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/core_torch.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_wave.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/core_wave.png"></a></td>
            <td></td>
        </tr>
        <tr>
            <td align="center">sph</td>
            <td align="center">torch</td>
            <td align="center">wave</td>
            <td align="center"></td>
        </tr>
    </tbody>
</table>


### examples/fem

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_apic_fluid.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_apic_fluid.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_convection_diffusion.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_convection_diffusion.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_diffusion_3d.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_diffusion_3d.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_diffusion.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_diffusion.png"></a></td>
        </tr>
        <tr>
            <td align="center">apic fluid</td>
            <td align="center">convection diffusion</td>
            <td align="center">diffusion 3d</td>
            <td align="center">diffusion</td>
        </tr>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_mixed_elasticity.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_mixed_elasticity.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_navier_stokes.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_navier_stokes.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_stokes_transfer.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_stokes_transfer.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_stokes.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_stokes.png"></a></td>
        </tr>
        <tr>
            <td align="center">mixed elasticity</td>
            <td align="center">navier stokes</td>
            <td align="center">stokes transfer</td>
            <td align="center">stokes</td>
        </tr>
    </tbody>
</table>


### examples/optim

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_bounce.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/optim_bounce.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_cloth_throw.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/optim_cloth_throw.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_diffray.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/optim_diffray.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_drone.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/optim_drone.png"></a></td>
        </tr>
        <tr>
            <td align="center">bounce</td>
            <td align="center">cloth throw</td>
            <td align="center">diffray</td>
            <td align="center">drone</td>
        </tr>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_inverse_kinematics.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/optim_inverse_kinematics.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_spring_cage.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/optim_spring_cage.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_trajectory.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/optim_trajectory.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_walker.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/optim_walker.png"></a></td>
        </tr>
        <tr>
            <td align="center">inverse kinematics</td>
            <td align="center">spring cage</td>
            <td align="center">trajectory</td>
            <td align="center">walker</td>
        </tr>
    </tbody>
</table>


### examples/sim

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_cartpole.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_cartpole.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_cloth.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_cloth.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_granular.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_granular.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_granular_collision_sdf.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_granular_collision_sdf.png"></a></td>
        </tr>
        <tr>
            <td align="center">cartpole</td>
            <td align="center">cloth</td>
            <td align="center">granular</td>
            <td align="center">granular collision sdf</td>
        </tr>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_jacobian_ik.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_jacobian_ik.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_quadruped.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_quadruped.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_chain.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_rigid_chain.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_contact.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_rigid_contact.png"></a></td>
        </tr>
        <tr>
            <td align="center">jacobian ik</td>
            <td align="center">quadruped</td>
            <td align="center">rigid chain</td>
            <td align="center">rigid contact</td>
        </tr>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_force.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_rigid_force.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_gyroscopic.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_rigid_gyroscopic.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_soft_contact.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_rigid_soft_contact.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_soft_body.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/sim_soft_body.png"></a></td>
        </tr>
        <tr>
            <td align="center">rigid force</td>
            <td align="center">rigid gyroscopic</td>
            <td align="center">rigid soft contact</td>
            <td align="center">soft body</td>
        </tr>
    </tbody>
</table>


## Building

For developers who want to build the library themselves, the following tools are required:

* Microsoft Visual Studio 2019 upwards (Windows)
* GCC 7.2 upwards (Linux)
* CUDA Toolkit 11.5 or higher
* [Git LFS](https://git-lfs.github.com/) installed

After cloning the repository, users should run:

    python build_lib.py

This will generate the `warp.dll` / `warp.so` core library respectively. It will search for the CUDA Toolkit in the default install directory. This path can be overridden by setting the `CUDA_PATH` environment variable. Alternatively, the path to the CUDA Toolkit can be passed to the build command as `--cuda_path="..."`. After building, the Warp package should be installed using:

    pip install -e .

This ensures that subsequent modifications to the library will be reflected in the Python package.

## Omniverse

A Warp Omniverse extension is available in the extension registry inside Omniverse Kit or USD Composer:

<img src="https://github.com/NVIDIA/warp/raw/main/docs/img/omniverse.png" width=550px/>

Enabling the extension will automatically install and initialize the Warp Python module inside the Kit Python environment.
Please see the [Omniverse Warp Documentation](https://docs.omniverse.nvidia.com/extensions/latest/ext_warp.html) for more details on how to use Warp in Omniverse.

## Learn More

Please see the following resources for additional background on Warp:

* [GTC 2022 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41599)
* [GTC 2021 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31838)
* [SIGGRAPH Asia 2021 Differentiable Simulation Course](https://dl.acm.org/doi/abs/10.1145/3476117.3483433)

The underlying technology in Warp has been used in a number of research projects at NVIDIA including the following publications:

* Accelerated Policy Learning with Parallel Differentiable Simulation - Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A., & Macklin, M. [(2022)](https://short-horizon-actor-critic.github.io)
* DiSECt: Differentiable Simulator for Robotic Cutting - Heiden, E., Macklin, M., Narang, Y., Fox, D., Garg, A., & Ramos, F [(2021)](https://github.com/NVlabs/DiSECt)
* gradSim: Differentiable Simulation for System Identification and Visuomotor Control - Murthy, J. Krishna, Miles Macklin, Florian Golemo, Vikram Voleti, Linda Petrini, Martin Weiss, Breandan Considine et al. [(2021)](https://gradsim.github.io)

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

Warp is inspired by many of these projects and is closely related to Numba and Taichi, which both expose kernel programming to Python. These frameworks map to traditional GPU programming models, so many of the high-level concepts are similar, however there are some functionality and implementation differences.

Compared to Numba, Warp supports a smaller subset of Python, but offers auto-differentiation of kernel programs, which is useful for machine learning. Compared to Taichi, Warp uses C++/CUDA as an intermediate representation, which makes it convenient to implement and expose low-level routines. In addition, we are building in data structures to support geometry processing (meshes, sparse volumes, point clouds, USD data) as first-class citizens that are not exposed in other runtimes.

Warp does not offer a full tensor-based programming model like PyTorch and JAX, but is designed to work well with these frameworks through data sharing mechanisms like `__cuda_array_interface__`. For computations that map well to tensors (e.g.: neural-network inference) it makes sense to use these existing tools. For problems with a lot of e.g.: sparsity, conditional logic, heterogeneous workloads (like the ones we often find in simulation and graphics), then the kernel-based programming model like the one in Warp is often more convenient since users have control over individual threads.

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

For more information about asynchronous operations, please refer to the [concurrency documentation](https://nvidia.github.io/warp/modules/concurrency.html) and [synchronization guidance](https://nvidia.github.io/warp/modules/concurrency.html#synchronization-guidance).

### What happens when you differentiate a function like `wp.abs(x)`?
-------

Non-smooth functions such as `y=|x|` do not have a single unique gradient at `x=0`, rather they have what is known as a `subgradient`, which is formally the convex hull of directional derivatives at that point. The way that Warp (and most auto-differentiation frameworks) handles these points is to pick an arbitrary gradient from this set, e.g.: for `wp.abs()`, it will arbitrarily choose the gradient to be 1.0 at the origin. You can find the implementation for these functions in `warp/native/builtin.h`.

Most optimizers (particularly ones that exploit stochasticity) are not sensitive to the choice of which gradient to use from the subgradient, although there are exceptions.

### Does Warp support multi-GPU programming?
-------

Yes! Since version `0.4.0` we support allocating, launching, and copying between multiple GPUs in a single process. We follow the naming conventions of PyTorch and use aliases such as `cuda:0`, `cuda:1`, `cpu` to identify individual devices.

### Should I switch to Warp over IsaacGym / PhysX?
-------

Warp is not a replacement for IsaacGym, IsaacSim, or PhysX - while Warp does offer some physical simulation capabilities this is primarily aimed at developers who need differentiable physics, rather than a fully featured physics engine. Warp is also integrated with IsaacGym and is great for performing auxiliary tasks such as reward and observation computations for reinforcement learning.

## Discord

We have a **#warp** channel on the public [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) server, come chat to us!

## License

Warp is provided under the NVIDIA Software License, please see [LICENSE.md](./LICENSE.md) for full license text.
