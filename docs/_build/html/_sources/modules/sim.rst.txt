warp.sim
====================

.. currentmodule:: warp.sim

..
   .. toctree::
   :maxdepth: 2

Warp includes a simulation module ``warp.sim`` that includes many common physical simulation models, and integrators for explicit and implicit time-stepping.

.. note:: The simulation model is under construction and should be expected to change rapidly, please treat this section as work in progress.


Model
-----

.. autoclass:: ModelBuilder
   :members:

.. autoclass:: Model
   :members:

.. autoclass:: JointAxis
   :members:

.. _Joint types:

Joint types
^^^^^^^^^^^^^^

.. data:: JOINT_PRISMATIC

   Prismatic (slider) joint

.. data:: JOINT_REVOLUTE

   Revolute (hinge) joint

.. data:: JOINT_BALL

   Ball (spherical) joint with quaternion state representation

.. data:: JOINT_FIXED

   Fixed (static) joint

.. data:: JOINT_FREE

   Free (floating) joint

.. data:: JOINT_COMPOUND

   Compound joint with 3 rotational degrees of freedom

.. data:: JOINT_UNIVERSAL

   Universal joint with 2 rotational degrees of freedom

.. data:: JOINT_DISTANCE

   Distance joint that keeps two bodies at a distance within its joint limits (only supported in :class:`XPBDIntegrator` at the moment)

.. data:: JOINT_D6

   Generic D6 joint with up to 3 translational and 3 rotational degrees of freedom

.. _Joint modes:

Joint modes
^^^^^^^^^^^^^^

Joint modes control the behavior of joint axes and can be used to implement joint position or velocity drives.

.. data:: JOINT_MODE_LIMIT

   No target or velocity control is applied, the joint is limited to its joint limits

.. data:: JOINT_MODE_TARGET_POSITION

   The joint is driven to a target position

.. data:: JOINT_MODE_TARGET_VELOCITY
   
   The joint is driven to a target velocity

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

Importers
--------------

Warp sim supports the loading of simulation models from URDF, MuJoCo (MJCF), and USD Physics files.

.. autofunction:: parse_urdf

.. autofunction:: parse_mjcf

.. autofunction:: parse_usd