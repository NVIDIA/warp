Simulation Reference
====================

.. currentmodule:: warp.sim

.. toctree::
   :maxdepth: 2

Warp includes a simulation module ``warp.sim`` that includes many common physical simulation models, and integrators for explicit and implicit time-stepping.

.. note:: The simulation model is under construction and should be expected to change rapidly, pelase treat this section as work in progress.


Model
--------------

.. autoclass:: ModelBuilder
   :members:

.. autoclass:: Model
   :members:

State
--------------

.. autoclass:: State
   :members:

Integrators
--------------

.. autoclass:: SemiImplicitIntegrator
   :members:

.. autoclass:: XPBDIntegrator
   :members: