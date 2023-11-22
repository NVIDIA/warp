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

.. autoclass:: Mesh
   :members:

.. autoclass:: SDF
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

.. _FK-IK:

Forward / Inverse Kinematics
----------------------------

Articulated rigid-body mechanisms are kinematically described by the joints that connect the bodies as well as the relative relative transform from the parent and child body to the respective anchor frames of the joint in the parent and child body:

.. image:: /img/joint_transforms.png
   :width: 400
   :align: center

.. list-table:: Variable names in the kernels from articulation.py
   :widths: 10 90
   :header-rows: 1

   * - Symbol
     - Description
   * - x_wp
     - World transform of the parent body (stored at :attr:`State.body_q`)
   * - x_wc
     - World transform of the child body (stored at :attr:`State.body_q`)
   * - x_pj
     - Transform from the parent body to the joint parent anchor frame (defined by :attr:`Model.joint_X_p`)
   * - x_cj
     - Transform from the child body to the joint child anchor frame (defined by :attr:`Model.joint_X_c`)
   * - x_j
     - Joint transform from the joint parent anchor frame to the joint child anchor frame

In the forward kinematics, the joint transform is determined by the joint coordinates (generalized joint positions :attr:`State.body_q` and velocities :attr:`State.body_qd`).
Given the parent body's world transform :math:`x_{wp}` and the joint transform :math:`x_{j}`, the child body's world transform :math:`x_{wc}` is computed as:

.. math::
   x_{wc} = x_{wp} \cdot x_{pj} \cdot x_{j} \cdot x_{cj}^{-1}.

.. autofunction:: eval_fk

.. autofunction:: eval_ik

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

.. autofunction:: resolve_usd_from_url