NVIDIA Warp Documentation
=========================

Warp is a Python framework for writing high-performance simulation and graphics code. Warp takes
regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU.

Warp is designed for spatial computing and comes with a rich set of primitives that make it easy to write 
programs for physics simulation, perception, robotics, and geometry processing. In addition, Warp kernels 
are differentiable and can be used as part of machine-learning pipelines with frameworks such as PyTorch and JAX.

Below are some examples of simulations implemented using Warp:

.. image:: ./img/header.png

Quickstart
==========

The easiest way is to install Warp is from PyPi:

.. code-block:: sh

    $ pip install warp-lang

Pre-built binary packages for Windows, Linux and macOS are also available on the `Releases <https://github.com/NVIDIA/warp/releases>`__ page. To install in your local Python environment extract the archive and run the following command from the root directory:

.. code-block:: sh

    $ pip install .

Basic example
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

Additional examples
-------------------
The `examples <https://github.com/NVIDIA/warp/tree/main/examples>`__ directory in
the Github repository contains a number of scripts that show how to
implement different simulation methods using the Warp API. Most examples
will generate USD files containing time-sampled animations in the
``examples/outputs`` directory. Before running examples users should
ensure that the ``usd-core`` package is installed using:

::

    pip install usd-core

USD files can be viewed or rendered inside NVIDIA
`Omniverse <https://developer.nvidia.com/omniverse>`__,
Pixar's UsdView, and Blender. Note that Preview in macOS is not
recommended as it has limited support for time-sampled animations.

Built-in unit tests can be run from the command-line as follows:

::

    python -m warp.tests

Omniverse
---------

A Warp Omniverse extension is available in the extension registry inside
Omniverse Kit or USD Composer.

Enabling the extension will automatically install and initialize the
Warp Python module inside the Kit Python environment. Please see the
`Omniverse Warp Documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_warp.html>`__
for more details on how to use Warp in Omniverse.


Learn More
----------

Please see the following resources for additional background on Warp:

-  `GTC 2022
   Presentation <https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41599>`__
-  `GTC 2021
   Presentation <https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31838>`__
-  `SIGGRAPH Asia 2021 Differentiable Simulation
   Course <https://dl.acm.org/doi/abs/10.1145/3476117.3483433>`__

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

Warp is provided under the NVIDIA Source Code License (NVSCL), please see
`LICENSE.md <https://github.com/NVIDIA/warp/blob/main/LICENSE.md>`_ for the full license text.

Please contact `omniverse-license-questions@nvidia.com <mailto:omniverse-license-questions@nvidia.com>`_ for
commercial licensing inquires.

Full Table of Contents
----------------------

.. toctree::
    :maxdepth: 2
    :caption: User's Guide

    installation
    basics
    modules/devices
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
