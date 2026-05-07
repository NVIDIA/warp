:tocdepth: 3

warp.sim
========

.. currentmodule:: warp.sim

Warp includes a simulation module ``warp.sim`` that includes many common physical simulation models and integrators
for explicit and implicit time-stepping.

Model
-----

.. autoclass:: ModelBuilder
    :members:

.. autoclass:: Model
    :members:

.. autoclass:: ModelShapeMaterials
    :members:

.. autoclass:: ModelShapeGeometry
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

Joint control modes
^^^^^^^^^^^^^^^^^^^

Joint modes control the behavior of how the joint control input :attr:`Control.joint_act` affects the torque applied at a given joint axis.
By default, it behaves as a direct force application via :data:`JOINT_MODE_FORCE`. Other modes can be used to implement joint position or velocity drives:

.. data:: JOINT_MODE_FORCE

    This is the default control mode where the control input is the torque :math:`\tau` applied at the joint axis.

.. data:: JOINT_MODE_TARGET_POSITION

    The control input is the target position :math:`\mathbf{q}_{\text{target}}` which is achieved via PD control of torque :math:`\tau` where the proportional and derivative gains are set by :attr:`Model.joint_target_ke` and :attr:`Model.joint_target_kd`:

    .. math::

        \tau = k_e (\mathbf{q}_{\text{target}} - \mathbf{q}) - k_d \mathbf{\dot{q}}

.. data:: JOINT_MODE_TARGET_VELOCITY
   
    The control input is the target velocity :math:`\mathbf{\dot{q}}_{\text{target}}` which is achieved via a controller of torque :math:`\tau` that brings the velocity at the joint axis to the target through proportional gain :attr:`Model.joint_target_ke`: 
    
    .. math::

        \tau = k_e (\mathbf{\dot{q}}_{\text{target}} - \mathbf{\dot{q}})

State
-----

.. autoclass:: State
    :members:

Control
-------

.. autoclass:: Control
    :members:

.. _FK-IK:

Forward / Inverse Kinematics
----------------------------

Articulated rigid-body mechanisms are kinematically described by the joints that connect the bodies as well as the
relative transform from the parent and child body to the respective anchor frames of the joint in the parent and child body:

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

In the forward kinematics, the joint transform is determined by the joint coordinates (generalized joint positions :attr:`State.joint_q` and velocities :attr:`State.joint_qd`).
Given the parent body's world transform :math:`x_{wp}` and the joint transform :math:`x_{j}`, the child body's world transform :math:`x_{wc}` is computed as:

.. math::
   x_{wc} = x_{wp} \cdot x_{pj} \cdot x_{j} \cdot x_{cj}^{-1}.

.. autofunction:: eval_fk

.. autofunction:: eval_ik

Integrators
-----------

.. autoclass:: Integrator
    :members:

.. autoclass:: SemiImplicitIntegrator
    :members:

.. autoclass:: XPBDIntegrator
    :members:

.. autoclass:: FeatherstoneIntegrator
    :members:

.. autoclass:: VBDIntegrator
    :members:

Collisions
----------


Shapes are used to define the geometry of the rigid bodies in the simulation. Shapes without a parent body are considered static shapes.
Use the ``add_shape_*`` methods in :class:`ModelBuilder` to add shapes to the model, such as :meth:`ModelBuilder.add_shape_sphere`, :meth:`ModelBuilder.add_shape_box`, :meth:`ModelBuilder.add_shape_capsule`, :meth:`ModelBuilder.add_shape_cylinder`, :meth:`ModelBuilder.add_shape_cone`, :meth:`ModelBuilder.add_shape_mesh`, :meth:`ModelBuilder.add_shape_sdf`, and :meth:`ModelBuilder.add_shape_plane`.
The following shape types are supported:

.. data:: GEO_SPHERE

    Sphere shape

.. data:: GEO_BOX
    
    Box shape

.. data:: GEO_CAPSULE

    Capsule shape

.. data:: GEO_CYLINDER

    Cylinder shape

.. data:: GEO_CONE

    Cone shape

.. data:: GEO_MESH

    Triangular mesh shape, see :class:`Mesh`

.. data:: GEO_SDF

    Signed distance field (volume) shape, see :class:`SDF`

.. data:: GEO_PLANE

    Rectangular plane shape (unlimited if the width or length is set to 0)


Besides particle collisions with these shapes, the following rigid-body shape collisions are supported:

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :align: center
   :widths: 10 10 10 10 10 10 10 10 10 10

   * - 
     - Particle
     - Sphere
     - Box
     - Capsule
     - Cylinder
     - Cone
     - Mesh
     - SDF
     - Plane
   * - Particle
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - Sphere
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - 
     - ✅
     - 
     - ✅
   * - Box
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - 
     - ✅
     - 
     - ✅
   * - Capsule
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - 
     - ✅
     - 
     - ✅
   * - Cylinder
     - ✅
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
   * - Cone
     - ✅
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
   * - Mesh
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - 
     - ✅
     - 
     - ✅
   * - SDF
     - ✅
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
   * - Plane
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - 
     - ✅
     - 
     - 

.. autofunction:: collide

Importers
---------

Warp sim supports the loading of simulation models from URDF, MuJoCo (MJCF), and USD Physics files.

.. autofunction:: parse_urdf

.. autofunction:: parse_mjcf

.. autofunction:: parse_usd

.. autofunction:: resolve_usd_from_url

Utility Functions
-----------------

Common utility functions used in simulators.

.. autofunction:: velocity_at_point

.. autofunction:: quat_to_euler

.. autofunction:: quat_from_euler

.. autofunction:: load_mesh
