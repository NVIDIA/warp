[![PyPI version](https://badge.fury.io/py/warp-lang.svg)](https://badge.fury.io/py/warp-lang)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/NVIDIA/warp?link=https%3A%2F%2Fgithub.com%2FNVIDIA%2Fwarp%2Fcommits%2Fmain)
[![Downloads](https://static.pepy.tech/badge/warp-lang/month)](https://pepy.tech/project/warp-lang)
[![codecov](https://codecov.io/github/NVIDIA/warp/graph/badge.svg?token=7O1KSM79FG)](https://codecov.io/github/NVIDIA/warp)
![GitHub - CI](https://github.com/NVIDIA/warp/actions/workflows/ci.yml/badge.svg)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.com/invite/nvidiaomniverse)

# NVIDIA Warp

Warp is a Python framework for writing high-performance simulation and graphics code. Warp takes
regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU.

Warp is designed for [spatial computing](https://en.wikipedia.org/wiki/Spatial_computing)
and comes with a rich set of primitives that make it easy to write
programs for physics simulation, perception, robotics, and geometry processing. In addition, Warp kernels
are differentiable and can be used as part of machine-learning pipelines with frameworks such as PyTorch and JAX.

Please refer to the project [Documentation](https://nvidia.github.io/warp/) for API and language reference and [CHANGELOG.md](./CHANGELOG.md) for release history.

<div align="center">
    <img src="https://github.com/NVIDIA/warp/raw/main/docs/img/header.jpg">
    <p><i>A selection of physical simulations computed with Warp</i></p>
</div>

## Installing

Python version 3.9 or newer is recommended. Warp can run on x86-64 and ARMv8 CPUs on Windows, Linux, and macOS.
GPU support requires a CUDA-capable NVIDIA GPU and driver (minimum GeForce GTX 9xx).

The easiest way to install Warp is from [PyPI](https://pypi.org/project/warp-lang/):

```text
pip install warp-lang
```

You can also use `pip install warp-lang[extras]` to install additional dependencies for running examples and USD-related features.

The binaries hosted on PyPI are currently built with the CUDA 11.8 runtime.
We provide binaries built with the CUDA 12.5 runtime on the [GitHub Releases](https://github.com/NVIDIA/warp/releases) page.
Copy the URL of the appropriate wheel file (`warp-lang-{ver}+cu12-py3-none-{platform}.whl`) and pass it to
the `pip install` command, e.g.

```text
pip install https://github.com/NVIDIA/warp/releases/download/v1.2.0/warp_lang-1.2.0+cu12-py3-none-manylinux2014_x86_64.whl
```

The `--force-reinstall` option may need to be used to overwrite a previous installation.

## Getting Started

An example first program that computes the lengths of random 3D vectors is given below:

```python
import warp as wp
import numpy as np

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

The [warp/examples](./warp/examples/) directory contains a number of scripts categorized under subdirectories
that show how to implement various simulation methods using the Warp API.
Most examples will generate USD files containing time-sampled animations in the current working directory.
Before running examples, users should ensure that the ``usd-core``, ``matplotlib``, and ``pyglet`` packages are installed using:

```text
pip install warp-lang[extras]
```

These dependencies can also be manually installed using:

```text
pip install usd-core matplotlib pyglet
```

Examples can be run from the command-line as follows:

    python -m warp.examples.<example_subdir>.<example>

To browse the example source code, you can open the directory where the files are located like this:

    python -m warp.examples.browse

Most examples can be run on either the CPU or a CUDA-capable device, but a handful require a CUDA-capable device. These are marked at the top of the example script.

USD files can be viewed or rendered inside [NVIDIA Omniverse](https://developer.nvidia.com/omniverse), Pixar's UsdView, and Blender. Note that Preview in macOS is not recommended as it has limited support for time-sampled animations.

Built-in unit tests can be run from the command-line as follows:

    python -m warp.tests

### warp/examples/core

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

### warp/examples/fem

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_diffusion_3d.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_diffusion_3d.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_mixed_elasticity.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_mixed_elasticity.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_apic_fluid.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_apic_fluid.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_streamlines.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_streamlines.png"></a></td>
        </tr>
        <tr>
            <td align="center">diffusion 3d</td>
            <td align="center">mixed elasticity</td>
            <td align="center">apic fluid</td>
            <td align="center">streamlines</td>
        </tr>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_convection_diffusion.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_convection_diffusion.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_navier_stokes.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_navier_stokes.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_burgers.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_burgers.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_deformed_geometry.py"><img src="https://github.com/NVIDIA/warp/raw/main/docs/img/examples/fem_deformed_geometry.png"></a></td>
        </tr>
        <tr>
            <td align="center">convection diffusion</td>
            <td align="center">navier stokes</td>
            <td align="center">burgers</td>
            <td align="center">deformed geometry</td>
        </tr>
    </tbody>
</table>

### warp/examples/optim

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

### warp/examples/sim

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
* GCC 9.4 upwards (Linux)
* CUDA Toolkit 11.5 or higher
* [Git LFS](https://git-lfs.github.com/) installed

After cloning the repository, users should run:

```text
python build_lib.py
```

Upon success, the script will output platform-specific binary files in `warp/bin/`.
The build script will look for the CUDA Toolkit in its default installation path.
This path can be overridden by setting the `CUDA_PATH` environment variable. Alternatively,
the path to the CUDA Toolkit can be passed to the build command as
`--cuda_path="..."`. After building, the Warp package should be installed using:

```text
pip install -e .
```

This ensures that subsequent modifications to the library will be reflected in the Python package.

## Learn More

Please see the following resources for additional background on Warp:

* [Product Page](https://developer.nvidia.com/warp-python)
* [GTC 2022 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41599)
* [GTC 2021 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31838)
* [SIGGRAPH Asia 2021 Differentiable Simulation Course](https://dl.acm.org/doi/abs/10.1145/3476117.3483433)
* [GTC 2024 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtc24-s63345/)

The underlying technology in Warp has been used in a number of research projects at NVIDIA including the following publications:

* Accelerated Policy Learning with Parallel Differentiable Simulation - Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A., & Macklin, M. [(2022)](https://short-horizon-actor-critic.github.io)
* DiSECt: Differentiable Simulator for Robotic Cutting - Heiden, E., Macklin, M., Narang, Y., Fox, D., Garg, A., & Ramos, F [(2021)](https://github.com/NVlabs/DiSECt)
* gradSim: Differentiable Simulation for System Identification and Visuomotor Control - Murthy, J. Krishna, Miles Macklin, Florian Golemo, Vikram Voleti, Linda Petrini, Martin Weiss, Breandan Considine et al. [(2021)](https://gradsim.github.io)

## Frequently Asked Questions

See the [FAQ](https://nvidia.github.io/warp/faq.html) in the Warp documentation.

## Support

Problems, questions, and feature requests can be opened on [GitHub Issues](https://github.com/NVIDIA/warp/issues).

The Warp team also monitors the **#warp** channel on the public [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) server, come chat with us!

## Versioning

Versions take the format X.Y.Z, similar to [Python itself](https://devguide.python.org/developer-workflow/development-cycle/#devcycle):

* Increments in X are reserved for major reworks of the project causing disruptive incompatibility (or reaching the 1.0 milestone).
* Increments in Y are for regular releases with a new set of features.
* Increments in Z are for bug fixes. In principle, there are no new features. Can be omitted if 0 or not relevant.

This is similar to [Semantic Versioning](https://semver.org/) but is less strict regarding backward compatibility.
Like with Python, some breaking changes can be present between minor versions if well-documented and gradually introduced.

Note that prior to 0.11.0, this schema was not strictly adhered to.

## License

Warp is provided under the NVIDIA Software License, please see [LICENSE.md](./LICENSE.md) for full license text.

## Contributing

Contributions and pull requests from the community are welcome and are taken under the
terms described in the **Feedback** section of [LICENSE.md](LICENSE.md#9-feedback).
[CONTRIBUTING.md](./CONTRIBUTING.md) provides additional information on how to open a pull request for Warp.

## Citing

If you use Warp in your research, please use the following citation:

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
