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

User's Guide
------------

.. toctree::
   :maxdepth: 2
   :caption: Basics

   installation
   quickstart
   basics

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules/runtime
   modules/functions
   modules/sim
   modules/interoperability

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Additional Notes
----------------

..
   .. toctree::
   :maxdepth: 2
