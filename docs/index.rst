NVIDIA Warp Documentation
=========================

Warp is a Python framework for writing high-performance simulation and graphics code. Warp takes
regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU.

Warp is designed for `spatial computing <https://en.wikipedia.org/wiki/Spatial_computing>`_
and comes with a rich set of primitives that make it easy to write 
programs for physics simulation, perception, robotics, and geometry processing. In addition, Warp kernels 
are differentiable and can be used as part of machine-learning pipelines with frameworks such as PyTorch, JAX and Paddle.

Below are some examples of simulations implemented using Warp:

.. image:: ./img/header.jpg

Quickstart
----------

The easiest way to install Warp is from `PyPI <https://pypi.org/project/warp-lang>`_:

.. code-block:: sh

    $ pip install warp-lang

You can also use ``pip install warp-lang[examples]`` to install additional dependencies for running examples
and USD-related features.

The binaries hosted on PyPI are currently built with the CUDA 12 runtime and therefore
require a minimum version of the CUDA driver of 525.60.13 (Linux x86-64) or 528.33 (Windows x86-64).

If you require a version of Warp built with the CUDA 13 runtime, you can build Warp from source or
install wheels built with the CUDA 13.0 runtime as described in :ref:`GitHub Installation`.

Tutorial Notebooks
------------------

The `NVIDIA Accelerated Computing Hub <https://github.com/NVIDIA/accelerated-computing-hub>`_ contains the current,
actively maintained set of Warp tutorials:

.. list-table::
   :header-rows: 1

   * - Notebook
     - Colab Link
   * - `Introduction to NVIDIA Warp <https://github.com/NVIDIA/accelerated-computing-hub/blob/9c334fcfcbbaf8d0cff91d012cdb2c11bf0f3dba/Accelerated_Python_User_Guide/notebooks/Chapter_12_Intro_to_NVIDIA_Warp.ipynb>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/9c334fcfcbbaf8d0cff91d012cdb2c11bf0f3dba/Accelerated_Python_User_Guide/notebooks/Chapter_12_Intro_to_NVIDIA_Warp.ipynb
          :alt: Open In Colab
   * - `GPU-Accelerated Ising Model Simulation in NVIDIA Warp <https://github.com/NVIDIA/accelerated-computing-hub/blob/9c334fcfcbbaf8d0cff91d012cdb2c11bf0f3dba/Accelerated_Python_User_Guide/notebooks/Chapter_12.1_IsingModel_In_Warp.ipynb>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/9c334fcfcbbaf8d0cff91d012cdb2c11bf0f3dba/Accelerated_Python_User_Guide/notebooks/Chapter_12.1_IsingModel_In_Warp.ipynb
          :alt: Open In Colab

Additionally, several notebooks in the `notebooks <https://github.com/NVIDIA/warp/tree/main/notebooks>`_ directory
provide additional examples and cover key Warp features:

.. list-table::
   :header-rows: 1

   * - Notebook
     - Colab Link
   * - `Warp Core Tutorial: Basics <https://github.com/NVIDIA/warp/blob/main/notebooks/core_01_basics.ipynb>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/core_01_basics.ipynb
          :alt: Open In Colab
   * - `Warp Core Tutorial: Generics <https://github.com/NVIDIA/warp/blob/main/notebooks/core_02_generics.ipynb>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/core_02_generics.ipynb
          :alt: Open In Colab
   * - `Warp Core Tutorial: Points <https://github.com/NVIDIA/warp/blob/main/notebooks/core_03_points.ipynb>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/core_03_points.ipynb
          :alt: Open In Colab
   * - `Warp Core Tutorial: Meshes <https://github.com/NVIDIA/warp/blob/main/notebooks/core_04_meshes.ipynb>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/core_04_meshes.ipynb
          :alt: Open In Colab
   * - `Warp Core Tutorial: Volumes <https://github.com/NVIDIA/warp/blob/main/notebooks/core_05_volumes.ipynb>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/core_05_volumes.ipynb
          :alt: Open In Colab
   * - `Warp PyTorch Tutorial: Basics <https://github.com/NVIDIA/warp/blob/main/notebooks/pytorch_01_basics.ipynb>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/pytorch_01_basics.ipynb
          :alt: Open In Colab
   * - `Warp PyTorch Tutorial: Custom Operators <https://github.com/NVIDIA/warp/blob/main/notebooks/pytorch_02_custom_operators.ipynb>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/NVIDIA/warp/blob/main/notebooks/pytorch_02_custom_operators.ipynb
          :alt: Open In Colab

Additional Examples
-------------------

The `warp/examples <https://github.com/NVIDIA/warp/tree/main/warp/examples>`_ directory in
the Github repository contains a number of scripts categorized under subdirectories
that show how to implement various simulation methods using the Warp API. Most examples
will generate USD files containing time-sampled animations in the current working directory.
Before running examples, install the optional example dependencies using::

    pip install warp-lang[examples]

On Linux aarch64 systems (e.g., NVIDIA DGX Spark), the ``[examples]`` extra automatically installs
`usd-exchange <https://pypi.org/project/usd-exchange/>`_ instead of ``usd-core`` as a drop-in
replacement, since ``usd-core`` wheels are not available for that platform.

Examples can be run from the command-line as follows::

    python -m warp.examples.<example_subdir>.<example>

Most examples can be run on either the CPU or a CUDA-capable device, but a handful require a CUDA-capable device. These are marked at the top of the example script.

USD files can be viewed or rendered inside NVIDIA
`Omniverse <https://developer.nvidia.com/omniverse>`_,
Pixar's UsdView, and Blender. Note that Preview in macOS is not
recommended as it has limited support for time-sampled animations.

Built-in unit tests can be run from the command-line as follows::

    python -m warp.tests

warp/examples/core
^^^^^^^^^^^^^^^^^^

.. list-table::
    :class: gallery

    * - .. image:: ./img/examples/core_dem.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_dem.py
      - .. image:: ./img/examples/core_fluid.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_fluid.py
      - .. image:: ./img/examples/core_graph_capture.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_graph_capture.py
      - .. image:: ./img/examples/core_marching_cubes.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_marching_cubes.py
    * - dem
      - fluid
      - graph capture
      - marching cubes
    * - .. image:: ./img/examples/core_mesh.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_mesh.py
      - .. image:: ./img/examples/core_nvdb.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_nvdb.py
      - .. image:: ./img/examples/core_raycast.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_raycast.py
      - .. image:: ./img/examples/core_raymarch.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_raymarch.py
    * - mesh
      - nvdb
      - raycast
      - raymarch
    * - .. image:: ./img/examples/core_sample_mesh.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_sample_mesh.py
      - .. image:: ./img/examples/core_sph.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_sph.py
      - .. image:: ./img/examples/core_torch.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_torch.py
      - .. image:: ./img/examples/core_wave.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_wave.py
    * - sample_mesh
      - sph
      - torch
      - wave

warp/examples/fem
^^^^^^^^^^^^^^^^^

.. list-table::
    :class: gallery

    * - .. image:: ./img/examples/fem_diffusion_3d.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_diffusion_3d.py
      - .. image:: ./img/examples/fem_mixed_elasticity.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_mixed_elasticity.py
      - .. image:: ./img/examples/fem_apic_fluid.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_apic_fluid.py
      - .. image:: ./img/examples/fem_streamlines.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_streamlines.py
    * - diffusion 3d
      - mixed elasticity
      - apic fluid
      - streamlines
    * - .. image:: ./img/examples/fem_distortion_energy.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_distortion_energy.py
      - .. image:: ./img/examples/fem_navier_stokes.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_navier_stokes.py
      - .. image:: ./img/examples/fem_burgers.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_burgers.py
      - .. image:: ./img/examples/fem_magnetostatics.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_magnetostatics.py
    * - distortion energy
      - navier stokes
      - burgers
      - magnetostatics
    * - .. image:: ./img/examples/fem_adaptive_grid.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_adaptive_grid.py
      - .. image:: ./img/examples/fem_nonconforming_contact.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_nonconforming_contact.py
      - .. image:: ./img/examples/fem_darcy_ls_optimization.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_darcy_ls_optimization.py
      - .. image:: ./img/examples/fem_elastic_shape_optimization.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_elastic_shape_optimization.py
    * - adaptive grid
      - nonconforming contact
      - darcy level-set optimization
      - elastic shape optimization

warp/examples/optim
^^^^^^^^^^^^^^^^^^^

.. list-table::
    :class: gallery

    * - .. image:: ./img/examples/optim_diffray.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_diffray.py
      - .. image:: ./img/examples/optim_fluid_checkpoint.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_fluid_checkpoint.py
      - .. image:: ./img/examples/optim_particle_repulsion.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_particle_repulsion.py
      -
    * - diffray
      - fluid checkpoint
      - particle repulsion
      -

warp/examples/tile
^^^^^^^^^^^^^^^^^^

.. list-table::
    :class: gallery

    * - .. image:: ./img/examples/tile_mlp.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/tile/example_tile_mlp.py
      - .. image:: ./img/examples/tile_nbody.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/tile/example_tile_nbody.py
      - .. image:: ./img/examples/tile_mcgp.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/tile/example_tile_mcgp.py
      -
    * - mlp
      - nbody
      - mcgp
      -

Omniverse
---------

Omniverse extensions for Warp are available in the extension registry inside
Omniverse Kit or USD Composer.
The ``omni.warp.core`` extension installs Warp into the Omniverse Application's
Python environment, which allows users to import the module in their scripts and nodes. 
The ``omni.warp`` extension provides a collection of OmniGraph nodes and sample
scenes demonstrating uses of Warp in OmniGraph.
Enabling the ``omni.warp`` extension automatically enables the ``omni.warp.core`` extension.

Please see the
`Omniverse Warp Documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_warp.html>`_
for more details on how to use Warp in Omniverse.


Learn More
----------

Please see the following resources for additional background on Warp:

- `Product Page <https://developer.nvidia.com/warp-python>`_
-  `SIGGRAPH 2024 Course Slides <https://dl.acm.org/doi/10.1145/3664475.3664543>`_
-  `GTC 2024 Presentation <https://www.nvidia.com/en-us/on-demand/session/gtc24-s63345>`_
-  `GTC 2022
   Presentation <https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41599>`_
-  `GTC 2021
   Presentation <https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31838>`_
-  `SIGGRAPH Asia 2021 Differentiable Simulation
   Course <https://dl.acm.org/doi/abs/10.1145/3476117.3483433>`_

The underlying technology in Warp has been used in a number of research
projects at NVIDIA including the following publications:

-  Accelerated Policy Learning with Parallel Differentiable Simulation -
   Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg,
   A., & Macklin, M.
   `(2022) <https://short-horizon-actor-critic.github.io>`__
-  DiSECt: Differentiable Simulator for Robotic Cutting - Heiden, E.,
   Macklin, M., Narang, Y., Fox, D., Garg, A., & Ramos, F
   `(2021) <https://github.com/NVlabs/DiSECt>`__
-  gradSim: Differentiable Simulation for System Identification and
   Visuomotor Control - Murthy, J. Krishna, Miles Macklin, Florian
   Golemo, Vikram Voleti, Linda Petrini, Martin Weiss, Breandan
   Considine et
   al. `(2021) <https://gradsim.github.io>`__

Support
-------

Problems, questions, and feature requests can be opened on
`GitHub Issues <https://github.com/NVIDIA/warp/issues>`_.

For inquiries not suited for GitHub Issues, please email warp-python@nvidia.com.

Versioning
----------

Versions take the format X.Y.Z, similar to `Python itself <https://devguide.python.org/developer-workflow/development-cycle/#devcycle>`__:

* Increments in X are reserved for major reworks of the project causing disruptive incompatibility (or reaching the 1.0 milestone).
* Increments in Y are for regular releases with a new set of features.
* Increments in Z are for bug fixes. In principle, there are no new features. Can be omitted if 0 or not relevant.

This is similar to `Semantic Versioning <https://semver.org/>`_ minor versions if well-documented and gradually introduced.

Note that prior to 0.11.0, this schema was not strictly adhered to.

License
-------

Warp is provided under the Apache License, Version 2.0. Please see
`LICENSE.md <https://github.com/NVIDIA/warp/blob/main/LICENSE.md>`__ for the full license text.

This project will download and install additional third-party open source software projects.
Review the license terms of these open source projects before use.

Contributing
------------

Contributions and pull requests from the community are welcome.
Please see the :doc:`user_guide/contribution_guide` for more information on contributing to the development of Warp.

Publications & Citation
-----------------------

Research Using Warp
^^^^^^^^^^^^^^^^^^^
:doc:`/user_guide/publications` lists academic and research publications that leverage the capabilities of Warp.
We encourage you to add your own published work using Warp to this list.

Citing Warp
^^^^^^^^^^^

If you use Warp in your research, please use the following citation:

.. code:: bibtex

    @misc{warp2022,
      title        = {Warp: A High-performance Python Framework for GPU Simulation and Graphics},
      author       = {Miles Macklin},
      month        = {March},
      year         = {2022},
      note         = {NVIDIA GPU Technology Conference (GTC)},
      howpublished = {\url{https://github.com/nvidia/warp}}
    }

Full Table of Contents
----------------------

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    user_guide/installation
    user_guide/basics
    user_guide/runtime
    user_guide/devices
    user_guide/differentiability
    user_guide/generics
    user_guide/tiles
    user_guide/interoperability
    user_guide/configuration
    user_guide/debugging
    user_guide/limitations
    user_guide/contribution_guide
    user_guide/publications
    user_guide/compatibility
    user_guide/faq
    user_guide/changelog

.. toctree::
    :maxdepth: 2
    :caption: Deep Dive

    deep_dive/codegen
    deep_dive/allocators
    deep_dive/concurrency
    deep_dive/profiling

.. toctree::
    :maxdepth: 2
    :caption: Domain Modules

    domain_modules/sparse
    domain_modules/fem
    domain_modules/render

.. toctree::
    :caption: API Reference

    api_reference/warp
    api_reference/warp_autograd
    api_reference/warp_config
    api_reference/warp_fem
    api_reference/warp_jax_experimental
    api_reference/warp_optim
    api_reference/warp_render
    api_reference/warp_sparse
    api_reference/warp_types
    api_reference/warp_utils

.. toctree::
    :caption: Language Reference

    language_reference/builtins

.. toctree::
    :hidden:
    :caption: Project Links

    GitHub <https://github.com/NVIDIA/warp>
    PyPI <https://pypi.org/project/warp-lang>

:ref:`Full Index <genindex>`
