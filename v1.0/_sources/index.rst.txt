NVIDIA Warp Documentation
=========================

Warp is a Python framework for writing high-performance simulation and graphics code. Warp takes
regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU.

Warp is designed for spatial computing and comes with a rich set of primitives that make it easy to write 
programs for physics simulation, perception, robotics, and geometry processing. In addition, Warp kernels 
are differentiable and can be used as part of machine-learning pipelines with frameworks such as PyTorch and JAX.

Below are some examples of simulations implemented using Warp:

.. image:: ./img/header.jpg

Quickstart
----------

Warp supports Python versions 3.7 onwards. It can run on x86-64 and ARMv8 CPUs on Windows, Linux, and macOS. GPU support requires a CUDA-capable NVIDIA GPU and driver (minimum GeForce GTX 9xx).

The easiest way to install Warp is from `PyPI <https://pypi.org/project/warp-lang>`_:

.. code-block:: sh

    $ pip install warp-lang

Pre-built binary packages are also available on the `Releases <https://github.com/NVIDIA/warp/releases>`_ page.
To install in your local Python environment extract the archive and run the following command from the root directory:

.. code-block:: sh

    $ pip install .

Basic Example
-------------

An example first program that computes the lengths of random 3D vectors is given below::

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

Additional Examples
-------------------

The `examples <https://github.com/NVIDIA/warp/tree/main/warp/examples>`_ directory in
the Github repository contains a number of scripts that show how to
implement different simulation methods using the Warp API. Most examples
will generate USD files containing time-sampled animations in the 
same directory as the example. Before running examples users should
ensure that the ``usd-core`` package is installed using::

    pip install usd-core

Examples can be run from the command-line as follows::

    python -m warp.examples.<example_subdir>.<example>

Most examples can be run on either the CPU or a CUDA-capable device, but a handful require a CUDA-capable device. These are marked at the top of the example script.

USD files can be viewed or rendered inside NVIDIA
`Omniverse <https://developer.nvidia.com/omniverse>`_,
Pixar's UsdView, and Blender. Note that Preview in macOS is not
recommended as it has limited support for time-sampled animations.

Built-in unit tests can be run from the command-line as follows::

    python -m warp.tests

examples/core
^^^^^^^^^^^^^

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
    * - .. image:: ./img/examples/core_sph.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_sph.py
      - .. image:: ./img/examples/core_torch.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_torch.py
      - .. image:: ./img/examples/core_wave.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/core/example_wave.py
      -
    * - sph
      - torch
      - wave
      -

examples/fem
^^^^^^^^^^^^

.. list-table::
    :class: gallery

    * - .. image:: ./img/examples/fem_apic_fluid.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_apic_fluid.py
      - .. image:: ./img/examples/fem_convection_diffusion.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_convection_diffusion.py
      - .. image:: ./img/examples/fem_diffusion_3d.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_diffusion_3d.py
      - .. image:: ./img/examples/fem_diffusion.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_diffusion.py
    * - apic fluid
      - convection diffusion
      - diffusion 3d
      - diffusion
    * - .. image:: ./img/examples/fem_mixed_elasticity.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_mixed_elasticity.py
      - .. image:: ./img/examples/fem_navier_stokes.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_navier_stokes.py
      - .. image:: ./img/examples/fem_stokes_transfer.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_stokes_transfer.py
      - .. image:: ./img/examples/fem_stokes.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/fem/example_stokes.py
    * - mixed elasticity
      - navier stokes
      - stokes transfer
      - stokes

examples/optim
^^^^^^^^^^^^^^

.. list-table::
    :class: gallery

    * - .. image:: ./img/examples/optim_bounce.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_bounce.py
      - .. image:: ./img/examples/optim_cloth_throw.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_cloth_throw.py
      - .. image:: ./img/examples/optim_diffray.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_diffray.py
      - .. image:: ./img/examples/optim_drone.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_drone.py
    * - bounce
      - cloth throw
      - diffray
      - drone
    * - .. image:: ./img/examples/optim_inverse_kinematics.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_inverse_kinematics.py
      - .. image:: ./img/examples/optim_spring_cage.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_spring_cage.py
      - .. image:: ./img/examples/optim_trajectory.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_trajectory.py
      - .. image:: ./img/examples/optim_walker.png
           :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/optim/example_walker.py
    * - inverse kinematics
      - spring cage
      - trajectory
      - walker

examples/sim
^^^^^^^^^^^^

.. list-table::
    :class: gallery

    * - .. image:: ./img/examples/sim_cartpole.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_cartpole.py
      - .. image:: ./img/examples/sim_cloth.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_cloth.py
      - .. image:: ./img/examples/sim_granular.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_granular.py
      - .. image:: ./img/examples/sim_granular_collision_sdf.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_granular_collision_sdf.py
    * - cartpole
      - cloth
      - granular
      - granular collision sdf
    * - .. image:: ./img/examples/sim_jacobian_ik.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_jacobian_ik.py
      - .. image:: ./img/examples/sim_quadruped.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_quadruped.py
      - .. image:: ./img/examples/sim_rigid_chain.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_chain.py
      - .. image:: ./img/examples/sim_rigid_contact.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_contact.py
    * - jacobian ik
      - quadruped
      - rigid chain
      - rigid contact
    * - .. image:: ./img/examples/sim_rigid_force.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_force.py
      - .. image:: ./img/examples/sim_rigid_gyroscopic.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_gyroscopic.py
      - .. image:: ./img/examples/sim_rigid_soft_contact.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_rigid_soft_contact.py
      - .. image:: ./img/examples/sim_soft_body.png
            :target: https://github.com/NVIDIA/warp/tree/main/warp/examples/sim/example_soft_body.py
    * - rigid force
      - rigid gyroscopic
      - rigid soft contact
      - soft body

Omniverse
---------

A Warp Omniverse extension is available in the extension registry inside
Omniverse Kit or USD Composer.

Enabling the extension will automatically install and initialize the
Warp Python module inside the Kit Python environment. Please see the
`Omniverse Warp Documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_warp.html>`_
for more details on how to use Warp in Omniverse.


Learn More
----------

Please see the following resources for additional background on Warp:

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
   al. `(2021) <https://gradsim.github.io>`__

Citing
------

If you use Warp in your research please use the following citation:

.. code:: bibtex

    @misc{warp2022,
        title= {Warp: A High-performance Python Framework for GPU Simulation and Graphics},
        author = {Miles Macklin},
        month = {March},
        year = {2022},
        note= {NVIDIA GPU Technology Conference (GTC)},
        howpublished = {\url{https://github.com/nvidia/warp}}
    }

License
-------

Warp is provided under the NVIDIA Software License, please see
`LICENSE.md <https://github.com/NVIDIA/warp/blob/main/LICENSE.md>`_ for the full license text.

Full Table of Contents
----------------------

.. toctree::
    :maxdepth: 2
    :caption: User's Guide

    installation
    basics
    modules/devices
    modules/allocators
    modules/concurrency
    modules/generics
    modules/interoperability
    configuration
    debugging
    limitations
    faq

.. toctree::
    :maxdepth: 2
    :caption: Core Reference
   
    modules/runtime
    modules/functions
   
.. toctree::
    :maxdepth: 2
    :caption: Simulation Reference
   
    modules/sim
    modules/sparse
    modules/fem
    modules/render

.. toctree::
    :hidden:
    :caption: Project Links

    GitHub <https://github.com/NVIDIA/warp>
    PyPI <https://pypi.org/project/warp-lang>
    Discord <https://discord.com/channels/827959428476174346/953756751977648148>

:ref:`Full Index <genindex>`
