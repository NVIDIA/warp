[![PyPI version](https://badge.fury.io/py/warp-lang.svg)](https://badge.fury.io/py/warp-lang)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/NVIDIA/warp?link=https%3A%2F%2Fgithub.com%2FNVIDIA%2Fwarp%2Fcommits%2Fmain)
[![Downloads](https://static.pepy.tech/badge/warp-lang/month)](https://pepy.tech/project/warp-lang)
[![codecov](https://codecov.io/github/NVIDIA/warp/graph/badge.svg?token=7O1KSM79FG)](https://codecov.io/github/NVIDIA/warp)
![GitHub - CI](https://github.com/NVIDIA/warp/actions/workflows/ci.yml/badge.svg)

# NVIDIA Warp

Warp is a Python framework for writing high-performance simulation and graphics code. Warp takes
regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU.

Warp is designed for [spatial computing](https://en.wikipedia.org/wiki/Spatial_computing)
and comes with a rich set of primitives that make it easy to write
programs for physics simulation, perception, robotics, and geometry processing. In addition, Warp kernels
are differentiable and can be used as part of machine-learning pipelines with frameworks such as PyTorch, JAX and Paddle.

Please refer to the project [Documentation](https://nvidia.github.io/warp/) for API and language reference and
[CHANGELOG.md](https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md) for release history.

<div align="center">
    <img src="https://github.com/NVIDIA/warp/raw/main/docs/img/header.jpg">
    <p><i>A selection of physical simulations computed with Warp</i></p>
</div>

## Installing

Python version 3.9 or newer is required. Warp can run on x86-64 and ARMv8 CPUs on Windows, Linux, and macOS.
GPU support requires a CUDA-capable NVIDIA GPU and driver (minimum GeForce GTX 9xx).

The easiest way to install Warp is from [PyPI](https://pypi.org/project/warp-lang/):

```text
pip install warp-lang
```

You can also use `pip install warp-lang[examples]` to install additional dependencies for running examples and USD-related features.

For nightly builds, conda, CUDA 13 builds, building from source, and CUDA driver requirements, see the
[Installation Guide](https://nvidia.github.io/warp/user_guide/installation.html).

## Tutorial Notebooks

The [NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub) contains the current,
actively maintained set of Warp tutorials:

| Notebook | Colab Link |
|----------|------------|
| [Introduction to NVIDIA Warp](https://github.com/NVIDIA/accelerated-computing-hub/blob/32fe3d5a448446fd52c14a6726e1b867cbfed2d9/Accelerated_Python_User_Guide/notebooks/Chapter_12_Intro_to_NVIDIA_Warp.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/32fe3d5a448446fd52c14a6726e1b867cbfed2d9/Accelerated_Python_User_Guide/notebooks/Chapter_12_Intro_to_NVIDIA_Warp.ipynb) |
| [GPU-Accelerated Ising Model Simulation in NVIDIA Warp](https://github.com/NVIDIA/accelerated-computing-hub/blob/32fe3d5a448446fd52c14a6726e1b867cbfed2d9/Accelerated_Python_User_Guide/notebooks/Chapter_12.1_IsingModel_In_Warp.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/32fe3d5a448446fd52c14a6726e1b867cbfed2d9/Accelerated_Python_User_Guide/notebooks/Chapter_12.1_IsingModel_In_Warp.ipynb) |

Additionally, several notebooks in the [notebooks](https://github.com/NVIDIA/warp/tree/main/notebooks) directory
provide additional examples and cover key Warp features:

| Notebook | Colab Link |
|----------|------------|
| [Warp Core Tutorial: Basics](https://github.com/NVIDIA/warp/blob/main/notebooks/core_01_basics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/core_01_basics.ipynb) |
| [Warp Core Tutorial: Generics](https://github.com/NVIDIA/warp/blob/main/notebooks/core_02_generics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/core_02_generics.ipynb) |
| [Warp Core Tutorial: Points](https://github.com/NVIDIA/warp/blob/main/notebooks/core_03_points.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/core_03_points.ipynb) |
| [Warp Core Tutorial: Meshes](https://github.com/NVIDIA/warp/blob/main/notebooks/core_04_meshes.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/core_04_meshes.ipynb) |
| [Warp Core Tutorial: Volumes](https://github.com/NVIDIA/warp/blob/main/notebooks/core_05_volumes.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/core_05_volumes.ipynb) |
| [Warp PyTorch Tutorial: Basics](https://github.com/NVIDIA/warp/blob/main/notebooks/pytorch_01_basics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/pytorch_01_basics.ipynb) |
| [Warp PyTorch Tutorial: Custom Operators](https://github.com/NVIDIA/warp/blob/main/notebooks/pytorch_02_custom_operators.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/pytorch_02_custom_operators.ipynb) |

## Running Examples

The [warp/examples](https://github.com/NVIDIA/warp/tree/main/warp/examples) directory contains a number of scripts categorized under subdirectories
that show how to implement various simulation methods using the Warp API.
Most examples will generate USD files containing time-sampled animations in the current working directory.
Before running examples, install the optional example dependencies using:

```text
pip install warp-lang[examples]
```

On Linux aarch64 systems (e.g., NVIDIA DGX Spark), the `[examples]` extra automatically installs
[`usd-exchange`](https://pypi.org/project/usd-exchange/) instead of `usd-core` as a drop-in replacement,
since `usd-core` wheels are not available for that platform.

Examples can be run from the command-line as follows:

```text
python -m warp.examples.<example_subdir>.<example>
```

To browse the example source code, you can open the directory where the files are located like this:

```text
python -m warp.examples.browse
```

Most examples can be run on either the CPU or a CUDA-capable device, but a handful require a CUDA-capable device. These are marked at the top of the example script.

USD files can be viewed or rendered inside [NVIDIA Omniverse](https://developer.nvidia.com/omniverse), Pixar's UsdView, and Blender. Note that Preview in macOS is not recommended as it has limited support for time-sampled animations.

Built-in unit tests can be run from the command-line as follows:

```text
python -m warp.tests
```

### warp/examples/core

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_dem.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_dem.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_fluid.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_fluid.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_graph_capture.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_graph_capture.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_marching_cubes.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_marching_cubes.png"></a></td>
        </tr>
        <tr>
            <td align="center">dem</td>
            <td align="center">fluid</td>
            <td align="center">graph capture</td>
            <td align="center">marching cubes</td>
        </tr>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_mesh.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_mesh.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_nvdb.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_nvdb.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_raycast.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_raycast.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_raymarch.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_raymarch.png"></a></td>
        </tr>
        <tr>
            <td align="center">mesh</td>
            <td align="center">nvdb</td>
            <td align="center">raycast</td>
            <td align="center">raymarch</td>
        </tr>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_sample_mesh.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_sample_mesh.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_sph.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_sph.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_torch.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_torch.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_wave.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/core_wave.png"></a></td>
        </tr>
        <tr>
            <td align="center">sample mesh</td>
            <td align="center">sph</td>
            <td align="center">torch</td>
            <td align="center">wave</td>
        </tr>
    </tbody>
</table>

### warp/examples/fem

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_diffusion_3d.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_diffusion_3d.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_mixed_elasticity.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_mixed_elasticity.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_apic_fluid.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_apic_fluid.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_streamlines.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_streamlines.png"></a></td>
        </tr>
        <tr>
            <td align="center">diffusion 3d</td>
            <td align="center">mixed elasticity</td>
            <td align="center">apic fluid</td>
            <td align="center">streamlines</td>
        </tr>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_distortion_energy.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_distortion_energy.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_navier_stokes.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_navier_stokes.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_burgers.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_burgers.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_magnetostatics.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_magnetostatics.png"></a></td>
        </tr>
        <tr>
            <td align="center">distortion energy</td>
            <td align="center">navier stokes</td>
            <td align="center">burgers</td>
            <td align="center">magnetostatics</td>
        </tr>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_adaptive_grid.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_adaptive_grid.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_nonconforming_contact.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_nonconforming_contact.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_darcy_ls_optimization.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_darcy_ls_optimization.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_elastic_shape_optimization.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/fem_elastic_shape_optimization.png"></a></td>
        </tr>
        <tr>
            <td align="center">adaptive grid</td>
            <td align="center">nonconforming contact</td>
            <td align="center">darcy level-set optimization</td>
            <td align="center">elastic shape optimization</td>
        </tr>
    </tbody>
</table>

### warp/examples/optim

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_diffray.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/optim_diffray.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_fluid_checkpoint.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/optim_fluid_checkpoint.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_particle_repulsion.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/optim_particle_repulsion.png"></a></td>
            <td></td>
        </tr>
        <tr>
            <td align="center">diffray</td>
            <td align="center">fluid checkpoint</td>
            <td align="center">particle repulsion</td>
            <td align="center"></td>
        </tr>
    </tbody>
</table>

### warp/examples/tile

<table>
    <tbody>
        <tr>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/tile/example_tile_mlp.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/tile_mlp.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/tile/example_tile_nbody.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/tile_nbody.png"></a></td>
            <td><a href="https://github.com/NVIDIA/warp/tree/main/warp/examples/tile/example_tile_mcgp.py"><img src="https://media.githubusercontent.com/media/NVIDIA/warp/refs/heads/main/docs/img/examples/tile_mcgp.png"></a></td>
            <td></td>
        </tr>
        <tr>
            <td align="center">mlp</td>
            <td align="center">nbody</td>
            <td align="center">mcgp</td>
            <td align="center"></td>
        </tr>
    </tbody>
</table>

## Learn More

Please see the following resources for additional background on Warp:

* [Product Page](https://developer.nvidia.com/warp-python)
* [SIGGRAPH 2024 Course Slides](https://dl.acm.org/doi/10.1145/3664475.3664543)
* [GTC 2024 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtc24-s63345/)
* [GTC 2022 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41599)
* [GTC 2021 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31838)
* [SIGGRAPH Asia 2021 Differentiable Simulation Course](https://dl.acm.org/doi/abs/10.1145/3476117.3483433)

The underlying technology in Warp has been used in a number of research projects at NVIDIA including the following publications:

* Accelerated Policy Learning with Parallel Differentiable Simulation - Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A., & Macklin, M. [(2022)](https://short-horizon-actor-critic.github.io)
* DiSECt: Differentiable Simulator for Robotic Cutting - Heiden, E., Macklin, M., Narang, Y., Fox, D., Garg, A., & Ramos, F [(2021)](https://github.com/NVlabs/DiSECt)
* gradSim: Differentiable Simulation for System Identification and Visuomotor Control - Murthy, J. Krishna, Miles Macklin, Florian Golemo, Vikram Voleti, Linda Petrini, Martin Weiss, Breandan Considine et al. [(2021)](https://gradsim.github.io)

## Frequently Asked Questions

See the [FAQ](https://nvidia.github.io/warp/user_guide/faq.html) in the Warp documentation.

## Support

Problems, questions, and feature requests can be opened on [GitHub Issues](https://github.com/NVIDIA/warp/issues).

For inquiries not suited for GitHub Issues, please email <warp-python@nvidia.com>.

## License

Warp is provided under the Apache License, Version 2.0.
Please see [LICENSE.md](https://github.com/NVIDIA/warp/blob/main/LICENSE.md) for full license text.

This project will download and install additional third-party open source software projects.
Review the license terms of these open source projects before use.

## Contributing

Contributions and pull requests from the community are welcome.
Please see the [Contribution Guide](https://nvidia.github.io/warp/user_guide/contribution_guide.html) for more
information on contributing to the development of Warp.

## Publications & Citation

### Research Using Warp

Our [PUBLICATIONS.md](https://github.com/NVIDIA/warp/blob/main/PUBLICATIONS.md) file lists academic and research
publications that leverage the capabilities of Warp.
We encourage you to add your own published work using Warp to this list.

### Citing Warp

If you use Warp in your research, please use the "Cite this repository" button on the
[GitHub repository](https://github.com/NVIDIA/warp) page or refer to the
[CITATION.cff](https://github.com/NVIDIA/warp/blob/main/CITATION.cff) file for citation information.
