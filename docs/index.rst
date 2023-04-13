NVIDIA Warp Documentation
=========================

Warp is a Python framework for writing high-performance simulation and graphics code. Kernels are defined in Python syntax and JIT converted to C++/CUDA and compiled at runtime.

Warp is comes with a rich set of primitives that make it easy to write programs for physics simulation, geometry
processing, and procedural animation. In addition, Warp kernels are differentiable, and can be used as part of
machine-learning training pipelines with other frameworks such as PyTorch.

Below are some examples of simulations implemented using Warp:

.. image:: ./img/header.png

License
-------

Warp is provided under the NVIDIA Source Code License (NVSCL), please see
`LICENSE.md <https://github.com/NVIDIA/warp/blob/main/LICENSE.md>`_ for the full license text.
Note that the license currently allows only non-commercial use of this code.

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

User's Guide
------------

.. toctree::
   :maxdepth: 2
   :caption: Basics

   installation
   quickstart
   primer
   debugging
   faq

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules/runtime
   modules/functions
   modules/sim
   modules/interoperability
   genindex

..modindex
