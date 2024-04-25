# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""A module for building simulation models and state."""

import copy
import math
from typing import List, Optional, Tuple

import numpy as np

import warp as wp

from .inertia import (
    compute_box_inertia,
    compute_capsule_inertia,
    compute_cone_inertia,
    compute_cylinder_inertia,
    compute_mesh_inertia,
    compute_sphere_inertia,
    transform_inertia,
)

Vec3 = List[float]
Vec4 = List[float]
Quat = List[float]
Mat33 = List[float]
Transform = Tuple[Vec3, Quat]

# Particle flags
PARTICLE_FLAG_ACTIVE = wp.constant(wp.uint32(1 << 0))

# Shape geometry types
GEO_SPHERE = wp.constant(0)
GEO_BOX = wp.constant(1)
GEO_CAPSULE = wp.constant(2)
GEO_CYLINDER = wp.constant(3)
GEO_CONE = wp.constant(4)
GEO_MESH = wp.constant(5)
GEO_SDF = wp.constant(6)
GEO_PLANE = wp.constant(7)
GEO_NONE = wp.constant(8)

# Types of joints linking rigid bodies
JOINT_PRISMATIC = wp.constant(0)
JOINT_REVOLUTE = wp.constant(1)
JOINT_BALL = wp.constant(2)
JOINT_FIXED = wp.constant(3)
JOINT_FREE = wp.constant(4)
JOINT_COMPOUND = wp.constant(5)
JOINT_UNIVERSAL = wp.constant(6)
JOINT_DISTANCE = wp.constant(7)
JOINT_D6 = wp.constant(8)

# Joint axis control mode types
JOINT_MODE_FORCE = wp.constant(0)
JOINT_MODE_TARGET_POSITION = wp.constant(1)
JOINT_MODE_TARGET_VELOCITY = wp.constant(2)


def flag_to_int(flag):
    """Converts a flag to an integer."""
    if type(flag) in wp.types.int_types:
        return flag.value
    return int(flag)


# Material properties pertaining to rigid shape contact dynamics
@wp.struct
class ModelShapeMaterials:
    ke: wp.array(dtype=float)  # The contact elastic stiffness (only used by the Euler integrators)
    kd: wp.array(dtype=float)  # The contact damping stiffness (only used by the Euler integrators)
    kf: wp.array(dtype=float)  # The contact friction stiffness (only used by the Euler integrators)
    ka: wp.array(
        dtype=float
    )  # The contact adhesion distance (values greater than 0 mean adhesive contact; only used by the Euler integrators)
    mu: wp.array(dtype=float)  # The coefficient of friction
    restitution: wp.array(dtype=float)  # The coefficient of restitution (only used by XPBD integrator)


# Shape properties of geometry
@wp.struct
class ModelShapeGeometry:
    type: wp.array(dtype=wp.int32)  # The type of geometry (GEO_SPHERE, GEO_BOX, etc.)
    is_solid: wp.array(dtype=wp.uint8)  # Indicates whether the shape is solid or hollow
    thickness: wp.array(
        dtype=float
    )  # The thickness of the shape (used for collision detection, and inertia computation of hollow shapes)
    source: wp.array(dtype=wp.uint64)  # Pointer to the source geometry (in case of a mesh, zero otherwise)
    scale: wp.array(dtype=wp.vec3)  # The 3D scale of the shape


# Axis (linear or angular) of a joint that can have bounds and be driven towards a target
class JointAxis:
    """
    Describes a joint axis that can have limits and be driven towards a target.

    Attributes:

        axis (3D vector or JointAxis): The 3D axis that this JointAxis object describes, or alternatively another JointAxis object to copy from
        limit_lower (float): The lower position limit of the joint axis
        limit_upper (float): The upper position limit of the joint axis
        limit_ke (float): The elastic stiffness of the joint axis limits, only respected by :class:`SemiImplicitIntegrator` and :class:`FeatherstoneIntegrator`
        limit_kd (float): The damping stiffness of the joint axis limits, only respected by :class:`SemiImplicitIntegrator` and :class:`FeatherstoneIntegrator`
        action (float): The force applied by default to this joint axis, or the target position or velocity (depending on the mode, see `Joint modes`_) of the joint axis
        target_ke (float): The proportional gain of the joint axis target drive PD controller
        target_kd (float): The derivative gain of the joint axis target drive PD controller
        mode (int): The mode of the joint axis, see `Joint modes`_
    """

    def __init__(
        self,
        axis,
        limit_lower=-math.inf,
        limit_upper=math.inf,
        limit_ke=100.0,
        limit_kd=10.0,
        action=None,
        target_ke=0.0,
        target_kd=0.0,
        mode=JOINT_MODE_FORCE,
    ):
        if isinstance(axis, JointAxis):
            self.axis = axis.axis
            self.limit_lower = axis.limit_lower
            self.limit_upper = axis.limit_upper
            self.limit_ke = axis.limit_ke
            self.limit_kd = axis.limit_kd
            self.action = axis.action
            self.target_ke = axis.target_ke
            self.target_kd = axis.target_kd
            self.mode = axis.mode
        else:
            self.axis = wp.normalize(wp.vec3(axis))
            self.limit_lower = limit_lower
            self.limit_upper = limit_upper
            self.limit_ke = limit_ke
            self.limit_kd = limit_kd
            if action is not None:
                self.action = action
            elif mode == JOINT_MODE_TARGET_POSITION and (limit_lower > 0.0 or limit_upper < 0.0):
                self.action = 0.5 * (limit_lower + limit_upper)
            else:
                self.action = 0.0
            self.target_ke = target_ke
            self.target_kd = target_kd
            self.mode = mode


class SDF:
    """Describes a signed distance field for simulation

    Attributes:

        volume (Volume): The volume defining the SDF
        I (Mat33): 3x3 inertia matrix of the SDF
        mass (float): The total mass of the SDF
        com (Vec3): The center of mass of the SDF
    """

    def __init__(self, volume=None, I=None, mass=1.0, com=None):
        self.volume = volume
        self.I = I if I is not None else wp.mat33(np.eye(3))
        self.mass = mass
        self.com = com if com is not None else wp.vec3()

        # Need to specify these for now
        self.has_inertia = True
        self.is_solid = True

    def finalize(self, device=None):
        return self.volume.id

    def __hash__(self):
        return hash((self.volume.id))


class Mesh:
    """Describes a triangle collision mesh for simulation

    Example mesh creation from a triangle OBJ mesh file:
    ====================================================

    See :func:`load_mesh` which is provided as a utility function.

    .. code-block:: python

        import numpy as np
        import warp as wp
        import warp.sim
        import openmesh

        m = openmesh.read_trimesh("mesh.obj")
        mesh_points = np.array(m.points())
        mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32).flatten()
        mesh = wp.sim.Mesh(mesh_points, mesh_indices)

    Attributes:

        vertices (List[Vec3]): Mesh 3D vertices points
        indices (List[int]): Mesh indices as a flattened list of vertex indices describing triangles
        I (Mat33): 3x3 inertia matrix of the mesh assuming density of 1.0 (around the center of mass)
        mass (float): The total mass of the body assuming density of 1.0
        com (Vec3): The center of mass of the body
    """

    def __init__(self, vertices: List[Vec3], indices: List[int], compute_inertia=True, is_solid=True):
        """Construct a Mesh object from a triangle mesh

        The mesh center of mass and inertia tensor will automatically be
        calculated using a density of 1.0. This computation is only valid
        if the mesh is closed (two-manifold).

        Args:
            vertices: List of vertices in the mesh
            indices: List of triangle indices, 3 per-element
            compute_inertia: If True, the mass, inertia tensor and center of mass will be computed assuming density of 1.0
            is_solid: If True, the mesh is assumed to be a solid during inertia computation, otherwise it is assumed to be a hollow surface
        """

        self.vertices = np.array(vertices).reshape(-1, 3)
        self.indices = np.array(indices, dtype=np.int32).flatten()
        self.is_solid = is_solid
        self.has_inertia = compute_inertia

        if compute_inertia:
            self.mass, self.com, self.I, _ = compute_mesh_inertia(1.0, vertices, indices, is_solid=is_solid)
        else:
            self.I = wp.mat33(np.eye(3))
            self.mass = 1.0
            self.com = wp.vec3()

    # construct simulation ready buffers from points
    def finalize(self, device=None):
        """
        Constructs a simulation-ready :class:`Mesh` object from the mesh data and returns its ID.

        Args:
            device: The device on which to allocate the mesh buffers

        Returns:
            The ID of the simulation-ready :class:`Mesh`
        """
        with wp.ScopedDevice(device):
            pos = wp.array(self.vertices, dtype=wp.vec3)
            vel = wp.zeros_like(pos)
            indices = wp.array(self.indices, dtype=wp.int32)

            self.mesh = wp.Mesh(points=pos, velocities=vel, indices=indices)
            return self.mesh.id

    def __hash__(self):
        """
        Computes a hash of the mesh data for use in caching. The hash considers the mesh vertices, indices, and whether the mesh is solid or not.
        """
        return hash((tuple(np.array(self.vertices).flatten()), tuple(np.array(self.indices).flatten()), self.is_solid))


class State:
    """The State object holds all *time-varying* data for a model.

    Time-varying data includes particle positions, velocities, rigid body states, and
    anything that is output from the integrator as derived data, e.g.: forces.

    The exact attributes depend on the contents of the model. State objects should
    generally be created using the :func:`Model.state()` function.

    Attributes:

        particle_q (array): Array of 3D particle positions, shape [particle_count], :class:`vec3`
        particle_qd (array): Array of 3D particle velocities, shape [particle_count], :class:`vec3`
        particle_f (array): Array of 3D particle forces, shape [particle_count], :class:`vec3`

        body_q (array): Array of body coordinates (7-dof transforms) in maximal coordinates, shape [body_count], :class:`transform`
        body_qd (array): Array of body velocities in maximal coordinates (first 3 entries represent angular velocity, last 3 entries represent linear velocity), shape [body_count], :class:`spatial_vector`
        body_f (array): Array of body forces in maximal coordinates (first 3 entries represent torque, last 3 entries represent linear force), shape [body_count], :class:`spatial_vector`

            Note:

                :attr:`body_f` represents external wrenches in world frame and denotes wrenches measured w.r.t. to the body's center of mass for all integrators except :class:`FeatherstoneIntegrator` which assumes the wrenches are measured w.r.t. world origin.

        joint_q (array): Array of generalized joint coordinates, shape [joint_coord_count], float
        joint_qd (array): Array of generalized joint velocities, shape [joint_dof_count], float

    """

    def __init__(self):
        self.particle_q = None
        self.particle_qd = None
        self.particle_f = None

        self.body_q = None
        self.body_qd = None
        self.body_f = None

        self.joint_q = None
        self.joint_qd = None

    def clear_forces(self):
        """Clears all forces (for particles and bodies) in the state object."""
        with wp.ScopedTimer("clear_forces", False):
            if self.particle_count:
                self.particle_f.zero_()

            if self.body_count:
                self.body_f.zero_()

    @property
    def requires_grad(self):
        """Indicates whether the state arrays have gradient computation enabled."""
        if self.particle_q:
            return self.particle_q.requires_grad
        if self.body_q:
            return self.body_q.requires_grad
        return False

    @property
    def body_count(self):
        """The number of bodies represented in the state."""
        if self.body_q is None:
            return 0
        return len(self.body_q)

    @property
    def particle_count(self):
        """The number of particles represented in the state."""
        if self.particle_q is None:
            return 0
        return len(self.particle_q)

    @property
    def joint_coord_count(self):
        """The number of generalized joint position coordinates represented in the state."""
        if self.joint_q is None:
            return 0
        return len(self.joint_q)

    @property
    def joint_dof_count(self):
        """The number of generalized joint velocity coordinates represented in the state."""
        if self.joint_qd is None:
            return 0
        return len(self.joint_qd)


class Control:
    """
    The Control object holds all *time-varying* control data for a model.

    Time-varying control data includes joint control inputs, muscle activations, and activation forces for triangle and tetrahedral elements.

    The exact attributes depend on the contents of the model. Control objects should generally be created using the :func:`Model.control()` function.

    Attributes:

        joint_act (array): Array of joint control inputs, shape [joint_axis_count], float
        tri_activations (array): Array of triangle element activations, shape [tri_count], float
        tet_activations (array): Array of tetrahedral element activations, shape [tet_count], float
        muscle_activations (array): Array of muscle activations, shape [muscle_count], float

    """

    def __init__(self, model):
        """
        Args:
            model (Model): The model to use as a reference for the control inputs
        """
        self.model = model
        self.joint_act = None
        self.tri_activations = None
        self.tet_activations = None
        self.muscle_activations = None

    def reset(self):
        """
        Resets the control inputs to their initial state defined in :class:`Model`.
        """
        if self.joint_act is not None:
            self.joint_act.assign(self.model.joint_act)
        if self.tri_activations is not None:
            self.tri_activations.assign(self.model.tri_activations)
        if self.tet_activations is not None:
            self.tet_activations.assign(self.model.tet_activations)
        if self.muscle_activations is not None:
            self.muscle_activations.assign(self.model.muscle_activations)


def compute_shape_mass(type, scale, src, density, is_solid, thickness):
    """Computes the mass, center of mass and 3x3 inertia tensor of a shape

    Args:
        type: The type of shape (GEO_SPHERE, GEO_BOX, etc.)
        scale: The scale of the shape
        src: The source shape (Mesh or SDF)
        density: The density of the shape
        is_solid: Whether the shape is solid or hollow
        thickness: The thickness of the shape (used for collision detection, and inertia computation of hollow shapes)

    Returns:
        The mass, center of mass and 3x3 inertia tensor of the shape
    """
    if density == 0.0 or type == GEO_PLANE:  # zero density means fixed
        return 0.0, wp.vec3(), wp.mat33()

    if type == GEO_SPHERE:
        solid = compute_sphere_inertia(density, scale[0])
        if is_solid:
            return solid
        else:
            hollow = compute_sphere_inertia(density, scale[0] - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GEO_BOX:
        w, h, d = scale * 2.0
        solid = compute_box_inertia(density, w, h, d)
        if is_solid:
            return solid
        else:
            hollow = compute_box_inertia(density, w - thickness, h - thickness, d - thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GEO_CAPSULE:
        r, h = scale[0], scale[1] * 2.0
        solid = compute_capsule_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            hollow = compute_capsule_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GEO_CYLINDER:
        r, h = scale[0], scale[1] * 2.0
        solid = compute_cylinder_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            hollow = compute_cylinder_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GEO_CONE:
        r, h = scale[0], scale[1] * 2.0
        solid = compute_cone_inertia(density, r, h)
        if is_solid:
            return solid
        else:
            hollow = compute_cone_inertia(density, r - thickness, h - 2.0 * thickness)
            return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]
    elif type == GEO_MESH or type == GEO_SDF:
        if src.has_inertia and src.mass > 0.0 and src.is_solid == is_solid:
            m, c, I = src.mass, src.com, src.I

            sx, sy, sz = scale

            mass_ratio = sx * sy * sz * density
            m_new = m * mass_ratio

            c_new = wp.cw_mul(c, scale)

            Ixx = I[0, 0] * (sy**2 + sz**2) / 2 * mass_ratio
            Iyy = I[1, 1] * (sx**2 + sz**2) / 2 * mass_ratio
            Izz = I[2, 2] * (sx**2 + sy**2) / 2 * mass_ratio
            Ixy = I[0, 1] * sx * sy * mass_ratio
            Ixz = I[0, 2] * sx * sz * mass_ratio
            Iyz = I[1, 2] * sy * sz * mass_ratio

            I_new = wp.mat33([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

            return m_new, c_new, I_new
        elif type == GEO_MESH:
            # fall back to computing inertia from mesh geometry
            vertices = np.array(src.vertices) * np.array(scale)
            m, c, I, vol = compute_mesh_inertia(density, vertices, src.indices, is_solid, thickness)
            return m, c, I
    raise ValueError("Unsupported shape type: {}".format(type))


class Model:
    """Holds the definition of the simulation model

    This class holds the non-time varying description of the system, i.e.:
    all geometry, constraints, and parameters used to describe the simulation.

    Attributes:
        requires_grad (float): Indicates whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled
        num_envs (int): Number of articulation environments that were added to the ModelBuilder via `add_builder`

        particle_q (array): Particle positions, shape [particle_count, 3], float
        particle_qd (array): Particle velocities, shape [particle_count, 3], float
        particle_mass (array): Particle mass, shape [particle_count], float
        particle_inv_mass (array): Particle inverse mass, shape [particle_count], float
        particle_radius (array): Particle radius, shape [particle_count], float
        particle_max_radius (float): Maximum particle radius (useful for HashGrid construction)
        particle_ke (array): Particle normal contact stiffness (used by :class:`SemiImplicitIntegrator`), shape [particle_count], float
        particle_kd (array): Particle normal contact damping (used by :class:`SemiImplicitIntegrator`), shape [particle_count], float
        particle_kf (array): Particle friction force stiffness (used by :class:`SemiImplicitIntegrator`), shape [particle_count], float
        particle_mu (array): Particle friction coefficient, shape [particle_count], float
        particle_cohesion (array): Particle cohesion strength, shape [particle_count], float
        particle_adhesion (array): Particle adhesion strength, shape [particle_count], float
        particle_grid (HashGrid): HashGrid instance used for accelerated simulation of particle interactions
        particle_flags (array): Particle enabled state, shape [particle_count], bool
        particle_max_velocity (float): Maximum particle velocity (to prevent instability)

        shape_transform (array): Rigid shape transforms, shape [shape_count, 7], float
        shape_visible (array): Rigid shape visibility, shape [shape_count], bool
        shape_body (array): Rigid shape body index, shape [shape_count], int
        body_shapes (dict): Mapping from body index to list of attached shape indices
        shape_materials (ModelShapeMaterials): Rigid shape contact materials, shape [shape_count], float
        shape_shape_geo (ModelShapeGeometry): Shape geometry properties (geo type, scale, thickness, etc.), shape [shape_count, 3], float
        shape_geo_src (list): List of `wp.Mesh` instances used for rendering of mesh geometry

        shape_collision_group (list): Collision group of each shape, shape [shape_count], int
        shape_collision_group_map (dict): Mapping from collision group to list of shape indices
        shape_collision_filter_pairs (set): Pairs of shape indices that should not collide
        shape_collision_radius (array): Collision radius of each shape used for bounding sphere broadphase collision checking, shape [shape_count], float
        shape_ground_collision (list): Indicates whether each shape should collide with the ground, shape [shape_count], bool
        shape_shape_collision (list): Indicates whether each shape should collide with any other shape, shape [shape_count], bool
        shape_contact_pairs (array): Pairs of shape indices that may collide, shape [contact_pair_count, 2], int
        shape_ground_contact_pairs (array): Pairs of shape, ground indices that may collide, shape [ground_contact_pair_count, 2], int

        spring_indices (array): Particle spring indices, shape [spring_count*2], int
        spring_rest_length (array): Particle spring rest length, shape [spring_count], float
        spring_stiffness (array): Particle spring stiffness, shape [spring_count], float
        spring_damping (array): Particle spring damping, shape [spring_count], float
        spring_control (array): Particle spring activation, shape [spring_count], float

        tri_indices (array): Triangle element indices, shape [tri_count*3], int
        tri_poses (array): Triangle element rest pose, shape [tri_count, 2, 2], float
        tri_activations (array): Triangle element activations, shape [tri_count], float
        tri_materials (array): Triangle element materials, shape [tri_count, 5], float

        edge_indices (array): Bending edge indices, shape [edge_count*4], int
        edge_rest_angle (array): Bending edge rest angle, shape [edge_count], float
        edge_bending_properties (array): Bending edge stiffness and damping parameters, shape [edge_count, 2], float

        tet_indices (array): Tetrahedral element indices, shape [tet_count*4], int
        tet_poses (array): Tetrahedral rest poses, shape [tet_count, 3, 3], float
        tet_activations (array): Tetrahedral volumetric activations, shape [tet_count], float
        tet_materials (array): Tetrahedral elastic parameters in form :math:`k_{mu}, k_{lambda}, k_{damp}`, shape [tet_count, 3]

        muscle_start (array): Start index of the first muscle point per muscle, shape [muscle_count], int
        muscle_params (array): Muscle parameters, shape [muscle_count, 5], float
        muscle_bodies (array): Body indices of the muscle waypoints, int
        muscle_points (array): Local body offset of the muscle waypoints, float
        muscle_activations (array): Muscle activations, shape [muscle_count], float

        body_q (array): Poses of rigid bodies used for state initialization, shape [body_count, 7], float
        body_qd (array): Velocities of rigid bodies used for state initialization, shape [body_count, 6], float
        body_com (array): Rigid body center of mass (in local frame), shape [body_count, 7], float
        body_inertia (array): Rigid body inertia tensor (relative to COM), shape [body_count, 3, 3], float
        body_inv_inertia (array): Rigid body inverse inertia tensor (relative to COM), shape [body_count, 3, 3], float
        body_mass (array): Rigid body mass, shape [body_count], float
        body_inv_mass (array): Rigid body inverse mass, shape [body_count], float
        body_name (list): Rigid body names, shape [body_count], str

        joint_q (array): Generalized joint positions used for state initialization, shape [joint_coord_count], float
        joint_qd (array): Generalized joint velocities used for state initialization, shape [joint_dof_count], float
        joint_act (array): Generalized joint control inputs, shape [joint_axis_count], float
        joint_type (array): Joint type, shape [joint_count], int
        joint_parent (array): Joint parent body indices, shape [joint_count], int
        joint_child (array): Joint child body indices, shape [joint_count], int
        joint_X_p (array): Joint transform in parent frame, shape [joint_count, 7], float
        joint_X_c (array): Joint mass frame in child frame, shape [joint_count, 7], float
        joint_axis (array): Joint axis in child frame, shape [joint_axis_count, 3], float
        joint_armature (array): Armature for each joint axis (only used by :class:`FeatherstoneIntegrator`), shape [joint_count], float
        joint_target_ke (array): Joint stiffness, shape [joint_axis_count], float
        joint_target_kd (array): Joint damping, shape [joint_axis_count], float
        joint_axis_start (array): Start index of the first axis per joint, shape [joint_count], int
        joint_axis_dim (array): Number of linear and angular axes per joint, shape [joint_count, 2], int
        joint_axis_mode (array): Joint axis mode, shape [joint_axis_count], int. See `Joint modes`_.
        joint_linear_compliance (array): Joint linear compliance, shape [joint_count], float
        joint_angular_compliance (array): Joint linear compliance, shape [joint_count], float
        joint_enabled (array): Controls which joint is simulated (bodies become disconnected if False), shape [joint_count], int

            Note:

               This setting is not supported by :class:`FeatherstoneIntegrator`.

        joint_limit_lower (array): Joint lower position limits, shape [joint_count], float
        joint_limit_upper (array): Joint upper position limits, shape [joint_count], float
        joint_limit_ke (array): Joint position limit stiffness (used by the Euler integrators), shape [joint_count], float
        joint_limit_kd (array): Joint position limit damping (used by the Euler integrators), shape [joint_count], float
        joint_twist_lower (array): Joint lower twist limit, shape [joint_count], float
        joint_twist_upper (array): Joint upper twist limit, shape [joint_count], float
        joint_q_start (array): Start index of the first position coordinate per joint, shape [joint_count], int
        joint_qd_start (array): Start index of the first velocity coordinate per joint, shape [joint_count], int
        articulation_start (array): Articulation start index, shape [articulation_count], int
        joint_name (list): Joint names, shape [joint_count], str
        joint_attach_ke (float): Joint attachment force stiffness (used by :class:`SemiImplicitIntegrator`)
        joint_attach_kd (float): Joint attachment force damping (used by :class:`SemiImplicitIntegrator`)

        soft_contact_margin (float): Contact margin for generation of soft contacts
        soft_contact_ke (float): Stiffness of soft contacts (used by the Euler integrators)
        soft_contact_kd (float): Damping of soft contacts (used by the Euler integrators)
        soft_contact_kf (float): Stiffness of friction force in soft contacts (used by the Euler integrators)
        soft_contact_mu (float): Friction coefficient of soft contacts
        soft_contact_restitution (float): Restitution coefficient of soft contacts (used by :class:`XPBDIntegrator`)

        soft_contact_count (array): Number of active particle-shape contacts, shape [1], int
        soft_contact_particle (array), Index of particle per soft contact point, shape [soft_contact_max], int
        soft_contact_shape (array), Index of shape per soft contact point, shape [soft_contact_max], int
        soft_contact_body_pos (array), Positional offset of soft contact point in body frame, shape [soft_contact_max], vec3
        soft_contact_body_vel (array), Linear velocity of soft contact point in body frame, shape [soft_contact_max], vec3
        soft_contact_normal (array), Contact surface normal of soft contact point in world space, shape [soft_contact_max], vec3

        rigid_contact_max (int): Maximum number of potential rigid body contact points to generate ignoring the `rigid_mesh_contact_max` limit.
        rigid_contact_max_limited (int): Maximum number of potential rigid body contact points to generate respecting the `rigid_mesh_contact_max` limit.
        rigid_mesh_contact_max (int): Maximum number of rigid body contact points to generate per mesh (0 = unlimited, default)
        rigid_contact_margin (float): Contact margin for generation of rigid body contacts
        rigid_contact_torsional_friction (float): Torsional friction coefficient for rigid body contacts (used by :class:`XPBDIntegrator`)
        rigid_contact_rolling_friction (float): Rolling friction coefficient for rigid body contacts (used by :class:`XPBDIntegrator`)

        rigid_contact_count (array): Number of active shape-shape contacts, shape [1], int
        rigid_contact_point0 (array): Contact point relative to frame of body 0, shape [rigid_contact_max], vec3
        rigid_contact_point1 (array): Contact point relative to frame of body 1, shape [rigid_contact_max], vec3
        rigid_contact_offset0 (array): Contact offset due to contact thickness relative to body 0, shape [rigid_contact_max], vec3
        rigid_contact_offset1 (array): Contact offset due to contact thickness relative to body 1, shape [rigid_contact_max], vec3
        rigid_contact_normal (array): Contact normal in world space, shape [rigid_contact_max], vec3
        rigid_contact_thickness (array): Total contact thickness, shape [rigid_contact_max], float
        rigid_contact_shape0 (array): Index of shape 0 per contact, shape [rigid_contact_max], int
        rigid_contact_shape1 (array): Index of shape 1 per contact, shape [rigid_contact_max], int

        ground (bool): Whether the ground plane and ground contacts are enabled
        ground_plane (array): Ground plane 3D normal and offset, shape [4], float
        up_vector (np.ndarray): Up vector of the world, shape [3], float
        up_axis (int): Up axis, 0 for x, 1 for y, 2 for z
        gravity (np.ndarray): Gravity vector, shape [3], float

        particle_count (int): Total number of particles in the system
        body_count (int): Total number of bodies in the system
        shape_count (int): Total number of shapes in the system
        joint_count (int): Total number of joints in the system
        tri_count (int): Total number of triangles in the system
        tet_count (int): Total number of tetrahedra in the system
        edge_count (int): Total number of edges in the system
        spring_count (int): Total number of springs in the system
        contact_count (int): Total number of contacts in the system
        muscle_count (int): Total number of muscles in the system
        articulation_count (int): Total number of articulations in the system
        joint_dof_count (int): Total number of velocity degrees of freedom of all joints in the system
        joint_coord_count (int): Total number of position degrees of freedom of all joints in the system

        device (wp.Device): Device on which the Model was allocated

    Note:
        It is strongly recommended to use the ModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if
        desired.
    """

    def __init__(self, device=None):
        self.requires_grad = False
        self.num_envs = 0

        self.particle_q = None
        self.particle_qd = None
        self.particle_mass = None
        self.particle_inv_mass = None
        self.particle_radius = None
        self.particle_max_radius = 0.0
        self.particle_ke = 1.0e3
        self.particle_kd = 1.0e2
        self.particle_kf = 1.0e2
        self.particle_mu = 0.5
        self.particle_cohesion = 0.0
        self.particle_adhesion = 0.0
        self.particle_grid = None
        self.particle_flags = None
        self.particle_max_velocity = 1e5

        self.shape_transform = None
        self.shape_body = None
        self.shape_visible = None
        self.body_shapes = {}
        self.shape_materials = ModelShapeMaterials()
        self.shape_geo = ModelShapeGeometry()
        self.shape_geo_src = None

        self.shape_collision_group = None
        self.shape_collision_group_map = None
        self.shape_collision_filter_pairs = None
        self.shape_collision_radius = None
        self.shape_ground_collision = None
        self.shape_shape_collision = None
        self.shape_contact_pairs = None
        self.shape_ground_contact_pairs = None

        self.spring_indices = None
        self.spring_rest_length = None
        self.spring_stiffness = None
        self.spring_damping = None
        self.spring_control = None
        self.spring_constraint_lambdas = None

        self.tri_indices = None
        self.tri_poses = None
        self.tri_activations = None
        self.tri_materials = None

        self.edge_indices = None
        self.edge_rest_angle = None
        self.edge_bending_properties = None
        self.edge_constraint_lambdas = None

        self.tet_indices = None
        self.tet_poses = None
        self.tet_activations = None
        self.tet_materials = None

        self.muscle_start = None
        self.muscle_params = None
        self.muscle_bodies = None
        self.muscle_points = None
        self.muscle_activations = None

        self.body_q = None
        self.body_qd = None
        self.body_com = None
        self.body_inertia = None
        self.body_inv_inertia = None
        self.body_mass = None
        self.body_inv_mass = None
        self.body_name = None

        self.joint_q = None
        self.joint_qd = None
        self.joint_act = None
        self.joint_type = None
        self.joint_parent = None
        self.joint_child = None
        self.joint_X_p = None
        self.joint_X_c = None
        self.joint_axis = None
        self.joint_armature = None
        self.joint_target_ke = None
        self.joint_target_kd = None
        self.joint_axis_start = None
        self.joint_axis_dim = None
        self.joint_axis_mode = None
        self.joint_linear_compliance = None
        self.joint_angular_compliance = None
        self.joint_enabled = None
        self.joint_limit_lower = None
        self.joint_limit_upper = None
        self.joint_limit_ke = None
        self.joint_limit_kd = None
        self.joint_twist_lower = None
        self.joint_twist_upper = None
        self.joint_q_start = None
        self.joint_qd_start = None
        self.articulation_start = None
        self.joint_name = None

        # todo: per-joint values?
        self.joint_attach_ke = 1.0e3
        self.joint_attach_kd = 1.0e2

        self.soft_contact_margin = 0.2
        self.soft_contact_ke = 1.0e3
        self.soft_contact_kd = 10.0
        self.soft_contact_kf = 1.0e3
        self.soft_contact_mu = 0.5
        self.soft_contact_restitution = 0.0

        self.soft_contact_count = 0
        self.soft_contact_particle = None
        self.soft_contact_shape = None
        self.soft_contact_body_pos = None
        self.soft_contact_body_vel = None
        self.soft_contact_normal = None

        self.rigid_contact_max = 0
        self.rigid_contact_max_limited = 0
        self.rigid_mesh_contact_max = 0
        self.rigid_contact_margin = None
        self.rigid_contact_torsional_friction = None
        self.rigid_contact_rolling_friction = None

        self.rigid_contact_count = None
        self.rigid_contact_point0 = None
        self.rigid_contact_point1 = None
        self.rigid_contact_offset0 = None
        self.rigid_contact_offset1 = None
        self.rigid_contact_normal = None
        self.rigid_contact_thickness = None
        self.rigid_contact_shape0 = None
        self.rigid_contact_shape1 = None

        # toggles ground contact for all shapes
        self.ground = True
        self.ground_plane = None
        self.up_vector = np.array((0.0, 1.0, 0.0))
        self.up_axis = 1
        self.gravity = np.array((0.0, -9.80665, 0.0))

        self.particle_count = 0
        self.body_count = 0
        self.shape_count = 0
        self.joint_count = 0
        self.joint_axis_count = 0
        self.tri_count = 0
        self.tet_count = 0
        self.edge_count = 0
        self.spring_count = 0
        self.muscle_count = 0
        self.articulation_count = 0
        self.joint_dof_count = 0
        self.joint_coord_count = 0

        self.device = wp.get_device(device)

    def state(self, requires_grad=None) -> State:
        """Returns a state object for the model

        The returned state will be initialized with the initial configuration given in
        the model description.

        Args:
            requires_grad (bool): Manual overwrite whether the state variables should have `requires_grad` enabled (defaults to `None` to use the model's setting :attr:`requires_grad`)

        Returns:
            State: The state object
        """

        s = State()
        if requires_grad is None:
            requires_grad = self.requires_grad

        # particles
        if self.particle_count:
            s.particle_q = wp.clone(self.particle_q, requires_grad=requires_grad)
            s.particle_qd = wp.clone(self.particle_qd, requires_grad=requires_grad)
            s.particle_f = wp.zeros_like(self.particle_qd, requires_grad=requires_grad)

        # articulations
        if self.body_count:
            s.body_q = wp.clone(self.body_q, requires_grad=requires_grad)
            s.body_qd = wp.clone(self.body_qd, requires_grad=requires_grad)
            s.body_f = wp.zeros_like(self.body_qd, requires_grad=requires_grad)

        if self.joint_count:
            s.joint_q = wp.clone(self.joint_q, requires_grad=requires_grad)
            s.joint_qd = wp.clone(self.joint_qd, requires_grad=requires_grad)

        return s

    def control(self, requires_grad=None, clone_variables=True) -> Control:
        """
        Returns a control object for the model.

        The returned control object will be initialized with the control inputs given in the model description.

        Args:
            requires_grad (bool): Manual overwrite whether the control variables should have `requires_grad` enabled (defaults to `None` to use the model's setting :attr:`requires_grad`)
            clone_variables (bool): Whether to clone the control inputs or use the original data

        Returns:
            Control: The control object
        """
        c = Control(self)
        if requires_grad is None:
            requires_grad = self.requires_grad
        if clone_variables:
            if self.joint_count:
                c.joint_act = wp.clone(self.joint_act, requires_grad=requires_grad)
            if self.tri_count:
                c.tri_activations = wp.clone(self.tri_activations, requires_grad=requires_grad)
            if self.tet_count:
                c.tet_activations = wp.clone(self.tet_activations, requires_grad=requires_grad)
            if self.muscle_count:
                c.muscle_activations = wp.clone(self.muscle_activations, requires_grad=requires_grad)
        else:
            c.joint_act = self.joint_act
            c.tri_activations = self.tri_activations
            c.tet_activations = self.tet_activations
            c.muscle_activations = self.muscle_activations
        return c

    def _allocate_soft_contacts(self, target, count, requires_grad=False):
        with wp.ScopedDevice(self.device):
            target.soft_contact_count = wp.zeros(1, dtype=wp.int32)
            target.soft_contact_particle = wp.zeros(count, dtype=int)
            target.soft_contact_shape = wp.zeros(count, dtype=int)
            target.soft_contact_body_pos = wp.zeros(count, dtype=wp.vec3, requires_grad=requires_grad)
            target.soft_contact_body_vel = wp.zeros(count, dtype=wp.vec3, requires_grad=requires_grad)
            target.soft_contact_normal = wp.zeros(count, dtype=wp.vec3, requires_grad=requires_grad)
            target.soft_contact_tids = wp.zeros(count, dtype=int)

    def allocate_soft_contacts(self, count, requires_grad=False):
        self._allocate_soft_contacts(self, count, requires_grad)

    def find_shape_contact_pairs(self):
        # find potential contact pairs based on collision groups and collision mask (pairwise filtering)
        import copy
        import itertools

        filters = copy.copy(self.shape_collision_filter_pairs)
        for a, b in self.shape_collision_filter_pairs:
            filters.add((b, a))
        contact_pairs = []
        # iterate over collision groups (islands)
        for group, shapes in self.shape_collision_group_map.items():
            for shape_a, shape_b in itertools.product(shapes, shapes):
                if not self.shape_shape_collision[shape_a]:
                    continue
                if not self.shape_shape_collision[shape_b]:
                    continue
                if shape_a < shape_b and (shape_a, shape_b) not in filters:
                    contact_pairs.append((shape_a, shape_b))
            if group != -1 and -1 in self.shape_collision_group_map:
                # shapes with collision group -1 collide with all other shapes
                for shape_a, shape_b in itertools.product(shapes, self.shape_collision_group_map[-1]):
                    if shape_a < shape_b and (shape_a, shape_b) not in filters:
                        contact_pairs.append((shape_a, shape_b))
        self.shape_contact_pairs = wp.array(np.array(contact_pairs), dtype=wp.int32, device=self.device)
        self.shape_contact_pair_count = len(contact_pairs)
        # find ground contact pairs
        ground_contact_pairs = []
        ground_id = self.shape_count - 1
        for i in range(ground_id):
            if self.shape_ground_collision[i]:
                ground_contact_pairs.append((i, ground_id))
        self.shape_ground_contact_pairs = wp.array(np.array(ground_contact_pairs), dtype=wp.int32, device=self.device)
        self.shape_ground_contact_pair_count = len(ground_contact_pairs)

    def count_contact_points(self):
        """
        Counts the maximum number of rigid contact points that need to be allocated.
        This function returns two values corresponding to the maximum number of potential contacts
        excluding the limiting from `Model.rigid_mesh_contact_max` and the maximum number of
        contact points that may be generated when considering the `Model.rigid_mesh_contact_max` limit.

        :returns:
            - potential_count (int): Potential number of contact points
            - actual_count (int): Actual number of contact points
        """
        from .collide import count_contact_points

        # calculate the potential number of shape pair contact points
        contact_count = wp.zeros(2, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=count_contact_points,
            dim=self.shape_contact_pair_count,
            inputs=[
                self.shape_contact_pairs,
                self.shape_geo,
                self.rigid_mesh_contact_max,
            ],
            outputs=[contact_count],
            device=self.device,
            record_tape=False,
        )
        # count ground contacts
        wp.launch(
            kernel=count_contact_points,
            dim=self.shape_ground_contact_pair_count,
            inputs=[
                self.shape_ground_contact_pairs,
                self.shape_geo,
                self.rigid_mesh_contact_max,
            ],
            outputs=[contact_count],
            device=self.device,
            record_tape=False,
        )
        counts = contact_count.numpy()
        potential_count = int(counts[0])
        actual_count = int(counts[1])
        return potential_count, actual_count

    def allocate_rigid_contacts(self, target=None, count=None, limited_contact_count=None, requires_grad=False):
        if count is not None:
            # potential number of contact points to consider
            self.rigid_contact_max = count
        if limited_contact_count is not None:
            self.rigid_contact_max_limited = limited_contact_count
        if target is None:
            target = self

        with wp.ScopedDevice(self.device):
            # serves as counter of the number of active contact points
            target.rigid_contact_count = wp.zeros(1, dtype=wp.int32)
            # contact point ID within the (shape_a, shape_b) contact pair
            target.rigid_contact_point_id = wp.zeros(self.rigid_contact_max, dtype=wp.int32)
            # position of contact point in body 0's frame before the integration step
            target.rigid_contact_point0 = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.vec3, requires_grad=requires_grad
            )
            # position of contact point in body 1's frame before the integration step
            target.rigid_contact_point1 = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.vec3, requires_grad=requires_grad
            )
            # moment arm before the integration step resulting from thickness displacement added to contact point 0 in body 0's frame (used in XPBD contact friction handling)
            target.rigid_contact_offset0 = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.vec3, requires_grad=requires_grad
            )
            # moment arm before the integration step resulting from thickness displacement added to contact point 1 in body 1's frame (used in XPBD contact friction handling)
            target.rigid_contact_offset1 = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.vec3, requires_grad=requires_grad
            )
            # contact normal in world frame
            target.rigid_contact_normal = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.vec3, requires_grad=requires_grad
            )
            # combined thickness of both shapes
            target.rigid_contact_thickness = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.float32, requires_grad=requires_grad
            )
            # ID of the first shape in the contact pair
            target.rigid_contact_shape0 = wp.zeros(self.rigid_contact_max_limited, dtype=wp.int32)
            # ID of the second shape in the contact pair
            target.rigid_contact_shape1 = wp.zeros(self.rigid_contact_max_limited, dtype=wp.int32)

            # shape IDs of potential contact pairs found during broadphase
            target.rigid_contact_broad_shape0 = wp.zeros(self.rigid_contact_max, dtype=wp.int32)
            target.rigid_contact_broad_shape1 = wp.zeros(self.rigid_contact_max, dtype=wp.int32)

            max_pair_count = self.shape_count * self.shape_count
            # maximum number of contact points per contact pair
            target.rigid_contact_point_limit = wp.zeros(max_pair_count, dtype=wp.int32)
            # currently found contacts per contact pair
            target.rigid_contact_pairwise_counter = wp.zeros(max_pair_count, dtype=wp.int32)
            # ID of thread that found the current contact point
            target.rigid_contact_tids = wp.zeros(self.rigid_contact_max, dtype=wp.int32)

    @property
    def soft_contact_max(self):
        """Maximum number of soft contacts that can be registered"""
        return len(self.soft_contact_particle)


class ModelBuilder:
    """A helper class for building simulation models at runtime.

    Use the ModelBuilder to construct a simulation scene. The ModelBuilder
    and builds the scene representation using standard Python data structures (lists),
    this means it is not differentiable. Once :func:`finalize()`
    has been called the ModelBuilder transfers all data to Warp tensors and returns
    an object that may be used for simulation.

    Example
    -------

    .. code-block:: python

        import warp as wp
        import warp.sim

        builder = wp.sim.ModelBuilder()

        # anchor point (zero mass)
        builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

        # build chain
        for i in range(1, 10):
            builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
            builder.add_spring(i - 1, i, 1.0e3, 0.0, 0)

        # create model
        model = builder.finalize("cuda")

        state = model.state()
        control = model.control()  # optional, to support time-varying control inputs
        integrator = wp.sim.SemiImplicitIntegrator()

        for i in range(100):
            state.clear_forces()
            integrator.simulate(model, state, state, dt=1.0 / 60.0, control=control)

    Note:
        It is strongly recommended to use the ModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if
        desired.
    """

    # Default particle settings
    default_particle_radius = 0.1

    # Default triangle soft mesh settings
    default_tri_ke = 100.0
    default_tri_ka = 100.0
    default_tri_kd = 10.0
    default_tri_drag = 0.0
    default_tri_lift = 0.0

    # Default distance constraint properties
    default_spring_ke = 100.0
    default_spring_kd = 0.0

    # Default edge bending properties
    default_edge_ke = 100.0
    default_edge_kd = 0.0

    # Default rigid shape contact material properties
    default_shape_ke = 1.0e5
    default_shape_kd = 1000.0
    default_shape_kf = 1000.0
    default_shape_ka = 0.0
    default_shape_mu = 0.5
    default_shape_restitution = 0.0
    default_shape_density = 1000.0
    default_shape_thickness = 1e-5

    # Default joint settings
    default_joint_limit_ke = 100.0
    default_joint_limit_kd = 1.0

    def __init__(self, up_vector=(0.0, 1.0, 0.0), gravity=-9.80665):
        self.num_envs = 0

        # particles
        self.particle_q = []
        self.particle_qd = []
        self.particle_mass = []
        self.particle_radius = []
        self.particle_flags = []
        self.particle_max_velocity = 1e5

        # shapes (each shape has an entry in these arrays)
        # transform from shape to body
        self.shape_transform = []
        # maps from shape index to body index
        self.shape_body = []
        self.shape_visible = []
        self.shape_geo_type = []
        self.shape_geo_scale = []
        self.shape_geo_src = []
        self.shape_geo_is_solid = []
        self.shape_geo_thickness = []
        self.shape_material_ke = []
        self.shape_material_kd = []
        self.shape_material_kf = []
        self.shape_material_ka = []
        self.shape_material_mu = []
        self.shape_material_restitution = []
        # collision groups within collisions are handled
        self.shape_collision_group = []
        self.shape_collision_group_map = {}
        self.last_collision_group = 0
        # radius to use for broadphase collision checking
        self.shape_collision_radius = []
        # whether the shape collides with the ground
        self.shape_ground_collision = []
        # whether the shape collides with any other shape
        self.shape_shape_collision = []

        # filtering to ignore certain collision pairs
        self.shape_collision_filter_pairs = set()

        # geometry
        self.geo_meshes = []
        self.geo_sdfs = []

        # springs
        self.spring_indices = []
        self.spring_rest_length = []
        self.spring_stiffness = []
        self.spring_damping = []
        self.spring_control = []

        # triangles
        self.tri_indices = []
        self.tri_poses = []
        self.tri_activations = []
        self.tri_materials = []

        # edges (bending)
        self.edge_indices = []
        self.edge_rest_angle = []
        self.edge_bending_properties = []

        # tetrahedra
        self.tet_indices = []
        self.tet_poses = []
        self.tet_activations = []
        self.tet_materials = []

        # muscles
        self.muscle_start = []
        self.muscle_params = []
        self.muscle_activations = []
        self.muscle_bodies = []
        self.muscle_points = []

        # rigid bodies
        self.body_mass = []
        self.body_inertia = []
        self.body_inv_mass = []
        self.body_inv_inertia = []
        self.body_com = []
        self.body_q = []
        self.body_qd = []
        self.body_name = []
        self.body_shapes = {}  # mapping from body to shapes

        # rigid joints
        self.joint = {}
        self.joint_parent = []  # index of the parent body                      (constant)
        self.joint_parents = {}  # mapping from joint to parent bodies
        self.joint_child = []  # index of the child body                       (constant)
        self.joint_axis = []  # joint axis in child joint frame               (constant)
        self.joint_X_p = []  # frame of joint in parent                      (constant)
        self.joint_X_c = []  # frame of child com (in child coordinates)     (constant)
        self.joint_q = []
        self.joint_qd = []

        self.joint_type = []
        self.joint_name = []
        self.joint_armature = []
        self.joint_target_ke = []
        self.joint_target_kd = []
        self.joint_axis_mode = []
        self.joint_limit_lower = []
        self.joint_limit_upper = []
        self.joint_limit_ke = []
        self.joint_limit_kd = []
        self.joint_act = []

        self.joint_twist_lower = []
        self.joint_twist_upper = []

        self.joint_linear_compliance = []
        self.joint_angular_compliance = []
        self.joint_enabled = []

        self.joint_q_start = []
        self.joint_qd_start = []
        self.joint_axis_start = []
        self.joint_axis_dim = []
        self.articulation_start = []

        self.joint_dof_count = 0
        self.joint_coord_count = 0
        self.joint_axis_total_count = 0

        self.up_vector = wp.vec3(up_vector)
        self.up_axis = wp.vec3(np.argmax(np.abs(up_vector)))
        self.gravity = gravity
        # indicates whether a ground plane has been created
        self._ground_created = False
        # constructor parameters for ground plane shape
        self._ground_params = {
            "plane": (*up_vector, 0.0),
            "width": 0.0,
            "length": 0.0,
            "ke": self.default_shape_ke,
            "kd": self.default_shape_kd,
            "kf": self.default_shape_kf,
            "mu": self.default_shape_mu,
            "restitution": self.default_shape_restitution,
        }

        # Maximum number of soft contacts that can be registered
        self.soft_contact_max = 64 * 1024

        # maximum number of contact points to generate per mesh shape
        self.rigid_mesh_contact_max = 0  # 0 = unlimited

        # contacts to be generated within the given distance margin to be generated at
        # every simulation substep (can be 0 if only one PBD solver iteration is used)
        self.rigid_contact_margin = 0.1
        # torsional friction coefficient (only considered by XPBD so far)
        self.rigid_contact_torsional_friction = 0.5
        # rolling friction coefficient (only considered by XPBD so far)
        self.rigid_contact_rolling_friction = 0.001

        # number of rigid contact points to allocate in the model during self.finalize() per environment
        # if setting is None, the number of worst-case number of contacts will be calculated in self.finalize()
        self.num_rigid_contacts_per_env = None

    @property
    def shape_count(self):
        return len(self.shape_geo_type)

    @property
    def body_count(self):
        return len(self.body_q)

    @property
    def joint_count(self):
        return len(self.joint_type)

    @property
    def joint_axis_count(self):
        return len(self.joint_axis)

    @property
    def particle_count(self):
        return len(self.particle_q)

    @property
    def tri_count(self):
        return len(self.tri_poses)

    @property
    def tet_count(self):
        return len(self.tet_poses)

    @property
    def edge_count(self):
        return len(self.edge_rest_angle)

    @property
    def spring_count(self):
        return len(self.spring_rest_length)

    @property
    def muscle_count(self):
        return len(self.muscle_start)

    @property
    def articulation_count(self):
        return len(self.articulation_start)

    # an articulation is a set of contiguous bodies bodies from articulation_start[i] to articulation_start[i+1]
    # these are used for computing forward kinematics e.g.:
    #
    # model.eval_articulation_fk()
    # model.eval_articulation_j()
    # model.eval_articulation_m()
    #
    # articulations are automatically 'closed' when calling finalize

    def add_articulation(self):
        self.articulation_start.append(self.joint_count)

    def add_builder(self, builder, xform=None, update_num_env_count=True, separate_collision_group=True):
        """Copies the data from `builder`, another `ModelBuilder` to this `ModelBuilder`.

        Args:
            builder (ModelBuilder): a model builder to add model data from.
            xform (:ref:`transform <transform>`): offset transform applied to root bodies.
            update_num_env_count (bool): if True, the number of environments is incremented by 1.
            separate_collision_group (bool): if True, the shapes from the articulations in `builder` will all be put into a single new collision group, otherwise, only the shapes in collision group > -1 will be moved to a new group.
        """

        start_particle_idx = self.particle_count
        if builder.particle_count:
            self.particle_max_velocity = builder.particle_max_velocity
            if xform is not None:
                pos_offset = wp.transform_get_translation(xform)
            else:
                pos_offset = np.zeros(3)
            self.particle_q.extend((np.array(builder.particle_q) + pos_offset).tolist())
            # other particle attributes are added below

        if builder.spring_count:
            self.spring_indices.extend((np.array(builder.spring_indices, dtype=np.int32) + start_particle_idx).tolist())
        if builder.edge_count:
            self.edge_indices.extend((np.array(builder.edge_indices, dtype=np.int32) + start_particle_idx).tolist())
        if builder.tri_count:
            self.tri_indices.extend((np.array(builder.tri_indices, dtype=np.int32) + start_particle_idx).tolist())
        if builder.tet_count:
            self.tet_indices.extend((np.array(builder.tet_indices, dtype=np.int32) + start_particle_idx).tolist())

        start_body_idx = self.body_count
        start_shape_idx = self.shape_count
        for s, b in enumerate(builder.shape_body):
            if b > -1:
                new_b = b + start_body_idx
                self.shape_body.append(new_b)
                self.shape_transform.append(builder.shape_transform[s])
            else:
                self.shape_body.append(-1)
                # apply offset transform to root bodies
                if xform is not None:
                    self.shape_transform.append(xform * builder.shape_transform[s])

        for b, shapes in builder.body_shapes.items():
            self.body_shapes[b + start_body_idx] = [s + start_shape_idx for s in shapes]

        if builder.joint_count:
            joint_X_p = copy.deepcopy(builder.joint_X_p)
            joint_q = copy.deepcopy(builder.joint_q)
            if xform is not None:
                for i in range(len(joint_X_p)):
                    if builder.joint_type[i] == wp.sim.JOINT_FREE:
                        qi = builder.joint_q_start[i]
                        xform_prev = wp.transform(joint_q[qi : qi + 3], joint_q[qi + 3 : qi + 7])
                        tf = xform * xform_prev
                        joint_q[qi : qi + 3] = tf.p
                        joint_q[qi + 3 : qi + 7] = tf.q
                    elif builder.joint_parent[i] == -1:
                        joint_X_p[i] = xform * joint_X_p[i]
            self.joint_X_p.extend(joint_X_p)
            self.joint_q.extend(joint_q)

            self.add_articulation()

            # offset the indices
            self.joint_parent.extend([p + self.joint_count if p != -1 else -1 for p in builder.joint_parent])
            self.joint_child.extend([c + self.joint_count for c in builder.joint_child])

            self.joint_q_start.extend([c + self.joint_coord_count for c in builder.joint_q_start])
            self.joint_qd_start.extend([c + self.joint_dof_count for c in builder.joint_qd_start])

            self.joint_axis_start.extend([a + self.joint_axis_total_count for a in builder.joint_axis_start])

        joint_children = set(builder.joint_child)
        for i in range(builder.body_count):
            if xform is not None and i not in joint_children:
                # rigid body is not attached to a joint, so apply input transform directly
                self.body_q.append(xform * builder.body_q[i])
            else:
                self.body_q.append(builder.body_q[i])

        # apply collision group
        if separate_collision_group:
            self.shape_collision_group.extend([self.last_collision_group + 1 for _ in builder.shape_collision_group])
        else:
            self.shape_collision_group.extend(
                [(g + self.last_collision_group if g > -1 else -1) for g in builder.shape_collision_group]
            )
        shape_count = self.shape_count
        for i, j in builder.shape_collision_filter_pairs:
            self.shape_collision_filter_pairs.add((i + shape_count, j + shape_count))
        for group, shapes in builder.shape_collision_group_map.items():
            if separate_collision_group:
                group = self.last_collision_group + 1
            else:
                group = group + self.last_collision_group if group > -1 else -1
            if group not in self.shape_collision_group_map:
                self.shape_collision_group_map[group] = []
            self.shape_collision_group_map[group].extend([s + shape_count for s in shapes])

        # update last collision group counter
        if separate_collision_group:
            self.last_collision_group += 1
        elif builder.last_collision_group > -1:
            self.last_collision_group += builder.last_collision_group

        more_builder_attrs = [
            "body_inertia",
            "body_mass",
            "body_inv_inertia",
            "body_inv_mass",
            "body_com",
            "body_qd",
            "body_name",
            "joint_type",
            "joint_enabled",
            "joint_X_c",
            "joint_armature",
            "joint_axis",
            "joint_axis_dim",
            "joint_axis_mode",
            "joint_name",
            "joint_qd",
            "joint_act",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_target_ke",
            "joint_target_kd",
            "joint_linear_compliance",
            "joint_angular_compliance",
            "shape_visible",
            "shape_geo_type",
            "shape_geo_scale",
            "shape_geo_src",
            "shape_geo_is_solid",
            "shape_geo_thickness",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_kf",
            "shape_material_ka",
            "shape_material_mu",
            "shape_material_restitution",
            "shape_collision_radius",
            "shape_ground_collision",
            "shape_shape_collision",
            "particle_qd",
            "particle_mass",
            "particle_radius",
            "particle_flags",
            "edge_rest_angle",
            "edge_bending_properties",
            "spring_rest_length",
            "spring_stiffness",
            "spring_damping",
            "spring_control",
            "tri_poses",
            "tri_activations",
            "tri_materials",
            "tet_poses",
            "tet_activations",
            "tet_materials",
        ]

        for attr in more_builder_attrs:
            getattr(self, attr).extend(getattr(builder, attr))

        self.joint_dof_count += builder.joint_dof_count
        self.joint_coord_count += builder.joint_coord_count
        self.joint_axis_total_count += builder.joint_axis_total_count

        self.up_vector = builder.up_vector
        self.gravity = builder.gravity
        self._ground_params = builder._ground_params

        if update_num_env_count:
            self.num_envs += 1

    # register a rigid body and return its index.
    def add_body(
        self,
        origin: Optional[Transform] = None,
        armature: float = 0.0,
        com: Optional[Vec3] = None,
        I_m: Optional[Mat33] = None,
        m: float = 0.0,
        name: str = None,
    ) -> int:
        """Adds a rigid body to the model.

        Args:
            origin: The location of the body in the world frame
            armature: Artificial inertia added to the body
            com: The center of mass of the body w.r.t its origin
            I_m: The 3x3 inertia tensor of the body (specified relative to the center of mass)
            m: Mass of the body
            name: Name of the body (optional)

        Returns:
            The index of the body in the model

        Note:
            If the mass (m) is zero then the body is treated as kinematic with no dynamics

        """

        if origin is None:
            origin = wp.transform()

        if com is None:
            com = wp.vec3()

        if I_m is None:
            I_m = wp.mat33()

        body_id = len(self.body_mass)

        # body data
        inertia = I_m + wp.mat33(np.eye(3)) * armature
        self.body_inertia.append(inertia)
        self.body_mass.append(m)
        self.body_com.append(com)

        if m > 0.0:
            self.body_inv_mass.append(1.0 / m)
        else:
            self.body_inv_mass.append(0.0)

        if any(x for x in inertia):
            self.body_inv_inertia.append(wp.inverse(inertia))
        else:
            self.body_inv_inertia.append(inertia)

        self.body_q.append(origin)
        self.body_qd.append(wp.spatial_vector())

        self.body_name.append(name or f"body {body_id}")
        self.body_shapes[body_id] = []
        return body_id

    def add_joint(
        self,
        joint_type: wp.constant,
        parent: int,
        child: int,
        linear_axes: Optional[List[JointAxis]] = None,
        angular_axes: Optional[List[JointAxis]] = None,
        name: str = None,
        parent_xform: Optional[wp.transform] = None,
        child_xform: Optional[wp.transform] = None,
        linear_compliance: float = 0.0,
        angular_compliance: float = 0.0,
        armature: float = 1e-2,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """
        Generic method to add any type of joint to this ModelBuilder.

        Args:
            joint_type (constant): The type of joint to add (see `Joint types`_)
            parent (int): The index of the parent body (-1 is the world)
            child (int): The index of the child body
            linear_axes (list(:class:`JointAxis`)): The linear axes (see :class:`JointAxis`) of the joint
            angular_axes (list(:class:`JointAxis`)): The angular axes (see :class:`JointAxis`) of the joint
            name (str): The name of the joint (optional)
            parent_xform (:ref:`transform <transform>`): The transform of the joint in the parent body's local frame
            child_xform (:ref:`transform <transform>`): The transform of the joint in the child body's local frame
            linear_compliance (float): The linear compliance of the joint
            angular_compliance (float): The angular compliance of the joint
            armature (float): Artificial inertia added around the joint axes (only considered by :class:`FeatherstoneIntegrator`)
            collision_filter_parent (bool): Whether to filter collisions between shapes of the parent and child bodies
            enabled (bool): Whether the joint is enabled (not considered by :class:`FeatherstoneIntegrator`)

        Returns:
            The index of the added joint
        """
        if linear_axes is None:
            linear_axes = []

        if angular_axes is None:
            angular_axes = []

        if parent_xform is None:
            parent_xform = wp.transform()

        if child_xform is None:
            child_xform = wp.transform()

        if len(self.articulation_start) == 0:
            # automatically add an articulation if none exists
            self.add_articulation()
        self.joint_type.append(joint_type)
        self.joint_parent.append(parent)
        if child not in self.joint_parents:
            self.joint_parents[child] = [parent]
        else:
            self.joint_parents[child].append(parent)
        self.joint_child.append(child)
        self.joint_X_p.append(wp.transform(parent_xform))
        self.joint_X_c.append(wp.transform(child_xform))
        self.joint_name.append(name or f"joint {self.joint_count}")
        self.joint_axis_start.append(len(self.joint_axis))
        self.joint_axis_dim.append((len(linear_axes), len(angular_axes)))
        self.joint_axis_total_count += len(linear_axes) + len(angular_axes)

        self.joint_linear_compliance.append(linear_compliance)
        self.joint_angular_compliance.append(angular_compliance)
        self.joint_enabled.append(enabled)

        def add_axis_dim(dim: JointAxis):
            self.joint_axis.append(dim.axis)
            self.joint_axis_mode.append(dim.mode)
            self.joint_act.append(dim.action)
            self.joint_target_ke.append(dim.target_ke)
            self.joint_target_kd.append(dim.target_kd)
            self.joint_limit_ke.append(dim.limit_ke)
            self.joint_limit_kd.append(dim.limit_kd)
            if np.isfinite(dim.limit_lower):
                self.joint_limit_lower.append(dim.limit_lower)
            else:
                self.joint_limit_lower.append(-1e6)
            if np.isfinite(dim.limit_upper):
                self.joint_limit_upper.append(dim.limit_upper)
            else:
                self.joint_limit_upper.append(1e6)

        for dim in linear_axes:
            add_axis_dim(dim)
        for dim in angular_axes:
            add_axis_dim(dim)

        if joint_type == JOINT_PRISMATIC:
            dof_count = 1
            coord_count = 1
        elif joint_type == JOINT_REVOLUTE:
            dof_count = 1
            coord_count = 1
        elif joint_type == JOINT_BALL:
            dof_count = 3
            coord_count = 4
        elif joint_type == JOINT_FREE or joint_type == JOINT_DISTANCE:
            dof_count = 6
            coord_count = 7
        elif joint_type == JOINT_FIXED:
            dof_count = 0
            coord_count = 0
        elif joint_type == JOINT_UNIVERSAL:
            dof_count = 2
            coord_count = 2
        elif joint_type == JOINT_COMPOUND:
            dof_count = 3
            coord_count = 3
        elif joint_type == JOINT_D6:
            dof_count = len(linear_axes) + len(angular_axes)
            coord_count = dof_count

        for _i in range(coord_count):
            self.joint_q.append(0.0)

        for _i in range(dof_count):
            self.joint_qd.append(0.0)
            self.joint_armature.append(armature)

        if joint_type == JOINT_FREE or joint_type == JOINT_DISTANCE or joint_type == JOINT_BALL:
            # ensure that a valid quaternion is used for the angular dofs
            self.joint_q[-1] = 1.0

        self.joint_q_start.append(self.joint_coord_count)
        self.joint_qd_start.append(self.joint_dof_count)

        self.joint_dof_count += dof_count
        self.joint_coord_count += coord_count

        if collision_filter_parent and parent > -1:
            for child_shape in self.body_shapes[child]:
                for parent_shape in self.body_shapes[parent]:
                    self.shape_collision_filter_pairs.add((parent_shape, child_shape))

        return self.joint_count - 1

    def add_joint_revolute(
        self,
        parent: int,
        child: int,
        parent_xform: Optional[wp.transform] = None,
        child_xform: Optional[wp.transform] = None,
        axis: Vec3 = (1.0, 0.0, 0.0),
        target: float = None,
        target_ke: float = 0.0,
        target_kd: float = 0.0,
        mode: int = JOINT_MODE_FORCE,
        limit_lower: float = -2 * math.pi,
        limit_upper: float = 2 * math.pi,
        limit_ke: float = default_joint_limit_ke,
        limit_kd: float = default_joint_limit_kd,
        linear_compliance: float = 0.0,
        angular_compliance: float = 0.0,
        armature: float = 1e-2,
        name: str = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a revolute (hinge) joint to the model. It has one degree of freedom.

        Args:
            parent: The index of the parent body
            child: The index of the child body
            parent_xform (:ref:`transform <transform>`): The transform of the joint in the parent body's local frame
            child_xform (:ref:`transform <transform>`): The transform of the joint in the child body's local frame
            axis (3D vector or JointAxis): The axis of rotation in the parent body's local frame, can be a JointAxis object whose settings will be used instead of the other arguments
            target: The target angle (in radians) or target velocity of the joint (if None, the joint is considered to be in force control mode)
            target_ke: The stiffness of the joint target
            target_kd: The damping of the joint target
            limit_lower: The lower limit of the joint
            limit_upper: The upper limit of the joint
            limit_ke: The stiffness of the joint limit
            limit_kd: The damping of the joint limit
            linear_compliance: The linear compliance of the joint
            angular_compliance: The angular compliance of the joint
            armature: Artificial inertia added around the joint axis
            name: The name of the joint
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies
            enabled: Whether the joint is enabled

        Returns:
            The index of the added joint

        """
        if parent_xform is None:
            parent_xform = wp.transform()

        if child_xform is None:
            child_xform = wp.transform()

        action = 0.0
        if target is None and mode == JOINT_MODE_TARGET_POSITION:
            action = 0.5 * (limit_lower + limit_upper)
        elif target is not None:
            action = target
            if mode == JOINT_MODE_FORCE:
                mode = JOINT_MODE_TARGET_POSITION
        ax = JointAxis(
            axis=axis,
            limit_lower=limit_lower,
            limit_upper=limit_upper,
            action=action,
            target_ke=target_ke,
            target_kd=target_kd,
            mode=mode,
            limit_ke=limit_ke,
            limit_kd=limit_kd,
        )
        return self.add_joint(
            JOINT_REVOLUTE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            angular_axes=[ax],
            linear_compliance=linear_compliance,
            angular_compliance=angular_compliance,
            armature=armature,
            name=name,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_prismatic(
        self,
        parent: int,
        child: int,
        parent_xform: Optional[wp.transform] = None,
        child_xform: Optional[wp.transform] = None,
        axis: Vec3 = (1.0, 0.0, 0.0),
        target: float = None,
        target_ke: float = 0.0,
        target_kd: float = 0.0,
        mode: int = JOINT_MODE_FORCE,
        limit_lower: float = -1e4,
        limit_upper: float = 1e4,
        limit_ke: float = default_joint_limit_ke,
        limit_kd: float = default_joint_limit_kd,
        linear_compliance: float = 0.0,
        angular_compliance: float = 0.0,
        armature: float = 1e-2,
        name: str = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a prismatic (sliding) joint to the model. It has one degree of freedom.

        Args:
            parent: The index of the parent body
            child: The index of the child body
            parent_xform (:ref:`transform <transform>`): The transform of the joint in the parent body's local frame
            child_xform (:ref:`transform <transform>`): The transform of the joint in the child body's local frame
            axis (3D vector or JointAxis): The axis of rotation in the parent body's local frame, can be a JointAxis object whose settings will be used instead of the other arguments
            target: The target position or velocity of the joint (if None, the joint is considered to be in force control mode)
            target_ke: The stiffness of the joint target
            target_kd: The damping of the joint target
            limit_lower: The lower limit of the joint
            limit_upper: The upper limit of the joint
            limit_ke: The stiffness of the joint limit
            limit_kd: The damping of the joint limit
            linear_compliance: The linear compliance of the joint
            angular_compliance: The angular compliance of the joint
            armature: Artificial inertia added around the joint axis
            name: The name of the joint
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies
            enabled: Whether the joint is enabled

        Returns:
            The index of the added joint

        """
        if parent_xform is None:
            parent_xform = wp.transform()

        if child_xform is None:
            child_xform = wp.transform()

        action = 0.0
        if target is None and mode == JOINT_MODE_TARGET_POSITION:
            action = 0.5 * (limit_lower + limit_upper)
        elif target is not None:
            action = target
            if mode == JOINT_MODE_FORCE:
                mode = JOINT_MODE_TARGET_POSITION
        ax = JointAxis(
            axis=axis,
            limit_lower=limit_lower,
            limit_upper=limit_upper,
            action=action,
            target_ke=target_ke,
            target_kd=target_kd,
            mode=mode,
            limit_ke=limit_ke,
            limit_kd=limit_kd,
        )
        return self.add_joint(
            JOINT_PRISMATIC,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[ax],
            linear_compliance=linear_compliance,
            angular_compliance=angular_compliance,
            armature=armature,
            name=name,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_ball(
        self,
        parent: int,
        child: int,
        parent_xform: Optional[wp.transform] = None,
        child_xform: Optional[wp.transform] = None,
        linear_compliance: float = 0.0,
        angular_compliance: float = 0.0,
        armature: float = 1e-2,
        name: str = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a ball (spherical) joint to the model. Its position is defined by a 4D quaternion (xyzw) and its velocity is a 3D vector.

        Args:
            parent: The index of the parent body
            child: The index of the child body
            parent_xform (:ref:`transform <transform>`): The transform of the joint in the parent body's local frame
            child_xform (:ref:`transform <transform>`): The transform of the joint in the child body's local frame
            linear_compliance: The linear compliance of the joint
            angular_compliance: The angular compliance of the joint
            armature (float): Artificial inertia added around the joint axis (only considered by FeatherstoneIntegrator)
            name: The name of the joint
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies
            enabled: Whether the joint is enabled

        Returns:
            The index of the added joint

        """
        if parent_xform is None:
            parent_xform = wp.transform()

        if child_xform is None:
            child_xform = wp.transform()

        return self.add_joint(
            JOINT_BALL,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_compliance=linear_compliance,
            angular_compliance=angular_compliance,
            armature=armature,
            name=name,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_fixed(
        self,
        parent: int,
        child: int,
        parent_xform: Optional[wp.transform] = None,
        child_xform: Optional[wp.transform] = None,
        linear_compliance: float = 0.0,
        angular_compliance: float = 0.0,
        armature: float = 1e-2,
        name: str = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a fixed (static) joint to the model. It has no degrees of freedom.
        See :meth:`collapse_fixed_joints` for a helper function that removes these fixed joints and merges the connecting bodies to simplify the model and improve stability.

        Args:
            parent: The index of the parent body
            child: The index of the child body
            parent_xform (:ref:`transform <transform>`): The transform of the joint in the parent body's local frame
            child_xform (:ref:`transform <transform>`): The transform of the joint in the child body's local frame
            linear_compliance: The linear compliance of the joint
            angular_compliance: The angular compliance of the joint
            armature (float): Artificial inertia added around the joint axis (only considered by FeatherstoneIntegrator)
            name: The name of the joint
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies
            enabled: Whether the joint is enabled

        Returns:
            The index of the added joint

        """
        if parent_xform is None:
            parent_xform = wp.transform()

        if child_xform is None:
            child_xform = wp.transform()

        return self.add_joint(
            JOINT_FIXED,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_compliance=linear_compliance,
            angular_compliance=angular_compliance,
            armature=armature,
            name=name,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_free(
        self,
        child: int,
        parent_xform: Optional[wp.transform] = None,
        child_xform: Optional[wp.transform] = None,
        armature: float = 0.0,
        parent: int = -1,
        name: str = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a free joint to the model.
        It has 7 positional degrees of freedom (first 3 linear and then 4 angular dimensions for the orientation quaternion in `xyzw` notation) and 6 velocity degrees of freedom (first 3 angular and then 3 linear velocity dimensions).

        Args:
            child: The index of the child body
            parent_xform (:ref:`transform <transform>`): The transform of the joint in the parent body's local frame
            child_xform (:ref:`transform <transform>`): The transform of the joint in the child body's local frame
            armature (float): Artificial inertia added around the joint axis (only considered by FeatherstoneIntegrator)
            parent: The index of the parent body (-1 by default to use the world frame, e.g. to make the child body and its children a floating-base mechanism)
            name: The name of the joint
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies
            enabled: Whether the joint is enabled

        Returns:
            The index of the added joint

        """
        if parent_xform is None:
            parent_xform = wp.transform()

        if child_xform is None:
            child_xform = wp.transform()

        return self.add_joint(
            JOINT_FREE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            armature=armature,
            name=name,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_distance(
        self,
        parent: int,
        child: int,
        parent_xform: Optional[wp.transform] = None,
        child_xform: Optional[wp.transform] = None,
        min_distance: float = -1.0,
        max_distance: float = 1.0,
        compliance: float = 0.0,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a distance joint to the model. The distance joint constraints the distance between the joint anchor points on the two bodies (see :ref:`FK-IK`) it connects to the interval [`min_distance`, `max_distance`].
        It has 7 positional degrees of freedom (first 3 linear and then 4 angular dimensions for the orientation quaternion in `xyzw` notation) and 6 velocity degrees of freedom (first 3 angular and then 3 linear velocity dimensions).

        Args:
            parent: The index of the parent body
            child: The index of the child body
            parent_xform (:ref:`transform <transform>`): The transform of the joint in the parent body's local frame
            child_xform (:ref:`transform <transform>`): The transform of the joint in the child body's local frame
            min_distance: The minimum distance between the bodies (no limit if negative)
            max_distance: The maximum distance between the bodies (no limit if negative)
            compliance: The compliance of the joint
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies
            enabled: Whether the joint is enabled

        Returns:
            The index of the added joint

        .. note:: Distance joints are currently only supported in the :class:`XPBDIntegrator` at the moment.

        """
        if parent_xform is None:
            parent_xform = wp.transform()

        if child_xform is None:
            child_xform = wp.transform()

        ax = JointAxis(
            axis=(1.0, 0.0, 0.0),
            limit_lower=min_distance,
            limit_upper=max_distance,
        )
        return self.add_joint(
            JOINT_DISTANCE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[ax],
            linear_compliance=compliance,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_universal(
        self,
        parent: int,
        child: int,
        axis_0: JointAxis,
        axis_1: JointAxis,
        parent_xform: Optional[wp.transform] = None,
        child_xform: Optional[wp.transform] = None,
        linear_compliance: float = 0.0,
        angular_compliance: float = 0.0,
        armature: float = 1e-2,
        name: str = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a universal joint to the model. U-joints have two degrees of freedom, one for each axis.

        Args:
            parent: The index of the parent body
            child: The index of the child body
            axis_0 (3D vector or JointAxis): The first axis of the joint, can be a JointAxis object whose settings will be used instead of the other arguments
            axis_1 (3D vector or JointAxis): The second axis of the joint, can be a JointAxis object whose settings will be used instead of the other arguments
            parent_xform (:ref:`transform <transform>`): The transform of the joint in the parent body's local frame
            child_xform (:ref:`transform <transform>`): The transform of the joint in the child body's local frame
            linear_compliance: The linear compliance of the joint
            angular_compliance: The angular compliance of the joint
            armature: Artificial inertia added around the joint axes
            name: The name of the joint
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies
            enabled: Whether the joint is enabled

        Returns:
            The index of the added joint

        """
        if parent_xform is None:
            parent_xform = wp.transform()

        if child_xform is None:
            child_xform = wp.transform()

        return self.add_joint(
            JOINT_UNIVERSAL,
            parent,
            child,
            angular_axes=[JointAxis(axis_0), JointAxis(axis_1)],
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_compliance=linear_compliance,
            angular_compliance=angular_compliance,
            armature=armature,
            name=name,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_compound(
        self,
        parent: int,
        child: int,
        axis_0: JointAxis,
        axis_1: JointAxis,
        axis_2: JointAxis,
        parent_xform: Optional[wp.transform] = None,
        child_xform: Optional[wp.transform] = None,
        linear_compliance: float = 0.0,
        angular_compliance: float = 0.0,
        armature: float = 1e-2,
        name: str = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a compound joint to the model, which has 3 degrees of freedom, one for each axis.
        Similar to the ball joint (see :meth:`add_ball_joint`), the compound joint allows bodies to move in a 3D rotation relative to each other,
        except that the rotation is defined by 3 axes instead of a quaternion.
        Depending on the choice of axes, the orientation can be specified through Euler angles, e.g. `z-x-z` or `x-y-x`, or through a Tait-Bryan angle sequence, e.g. `z-y-x` or `x-y-z`.

        Args:
            parent: The index of the parent body
            child: The index of the child body
            axis_0 (3D vector or JointAxis): The first axis of the joint, can be a JointAxis object whose settings will be used instead of the other arguments
            axis_1 (3D vector or JointAxis): The second axis of the joint, can be a JointAxis object whose settings will be used instead of the other arguments
            axis_2 (3D vector or JointAxis): The third axis of the joint, can be a JointAxis object whose settings will be used instead of the other arguments
            parent_xform (:ref:`transform <transform>`): The transform of the joint in the parent body's local frame
            child_xform (:ref:`transform <transform>`): The transform of the joint in the child body's local frame
            linear_compliance: The linear compliance of the joint
            angular_compliance: The angular compliance of the joint
            armature: Artificial inertia added around the joint axes
            name: The name of the joint
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies
            enabled: Whether the joint is enabled

        Returns:
            The index of the added joint

        """
        if parent_xform is None:
            parent_xform = wp.transform()

        if child_xform is None:
            child_xform = wp.transform()

        return self.add_joint(
            JOINT_COMPOUND,
            parent,
            child,
            angular_axes=[JointAxis(axis_0), JointAxis(axis_1), JointAxis(axis_2)],
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_compliance=linear_compliance,
            angular_compliance=angular_compliance,
            armature=armature,
            name=name,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_d6(
        self,
        parent: int,
        child: int,
        linear_axes: Optional[List[JointAxis]] = None,
        angular_axes: Optional[List[JointAxis]] = None,
        name: str = None,
        parent_xform: Optional[wp.transform] = None,
        child_xform: Optional[wp.transform] = None,
        linear_compliance: float = 0.0,
        angular_compliance: float = 0.0,
        armature: float = 1e-2,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ):
        """Adds a generic joint with custom linear and angular axes. The number of axes determines the number of degrees of freedom of the joint.

        Args:
            parent: The index of the parent body
            child: The index of the child body
            linear_axes: A list of linear axes
            angular_axes: A list of angular axes
            name: The name of the joint
            parent_xform (:ref:`transform <transform>`): The transform of the joint in the parent body's local frame
            child_xform (:ref:`transform <transform>`): The transform of the joint in the child body's local frame
            linear_compliance: The linear compliance of the joint
            angular_compliance: The angular compliance of the joint
            armature: Artificial inertia added around the joint axes
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies
            enabled: Whether the joint is enabled

        Returns:
            The index of the added joint

        """
        if linear_axes is None:
            linear_axes = []

        if angular_axes is None:
            angular_axes = []

        if parent_xform is None:
            parent_xform = wp.transform()

        if child_xform is None:
            child_xform = wp.transform()

        return self.add_joint(
            JOINT_D6,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[JointAxis(a) for a in linear_axes],
            angular_axes=[JointAxis(a) for a in angular_axes],
            linear_compliance=linear_compliance,
            angular_compliance=angular_compliance,
            armature=armature,
            name=name,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def plot_articulation(self, plot_shapes=True):
        """Plots the model's articulation."""

        def joint_type_str(type):
            if type == JOINT_FREE:
                return "free"
            elif type == JOINT_BALL:
                return "ball"
            elif type == JOINT_PRISMATIC:
                return "prismatic"
            elif type == JOINT_REVOLUTE:
                return "revolute"
            elif type == JOINT_D6:
                return "D6"
            elif type == JOINT_UNIVERSAL:
                return "universal"
            elif type == JOINT_COMPOUND:
                return "compound"
            elif type == JOINT_FIXED:
                return "fixed"
            elif type == JOINT_DISTANCE:
                return "distance"
            return "unknown"

        vertices = ["world"] + self.body_name
        if plot_shapes:
            vertices += [f"shape_{i}" for i in range(self.shape_count)]
        edges = []
        edge_labels = []
        for i in range(self.joint_count):
            edges.append((self.joint_child[i] + 1, self.joint_parent[i] + 1))
            edge_labels.append(f"{self.joint_name[i]}\n({joint_type_str(self.joint_type[i])})")
        if plot_shapes:
            for i in range(self.shape_count):
                edges.append((len(self.body_name) + i + 1, self.shape_body[i] + 1))
        wp.plot_graph(vertices, edges, edge_labels=edge_labels)

    def collapse_fixed_joints(self, verbose=wp.config.verbose):
        """Removes fixed joints from the model and merges the bodies they connect. This is useful for simplifying the model for faster and more stable simulation."""

        body_data = {}
        body_children = {-1: []}
        visited = {}
        for i in range(self.body_count):
            name = self.body_name[i]
            body_data[i] = {
                "shapes": self.body_shapes[i],
                "q": self.body_q[i],
                "qd": self.body_qd[i],
                "mass": self.body_mass[i],
                "inertia": self.body_inertia[i],
                "inv_mass": self.body_inv_mass[i],
                "inv_inertia": self.body_inv_inertia[i],
                "com": self.body_com[i],
                "name": name,
                "original_id": i,
            }
            visited[i] = False
            body_children[i] = []

        joint_data = {}
        for i in range(self.joint_count):
            name = self.joint_name[i]
            parent = self.joint_parent[i]
            child = self.joint_child[i]
            body_children[parent].append(child)

            q_start = self.joint_q_start[i]
            qd_start = self.joint_qd_start[i]
            if i < self.joint_count - 1:
                q_dim = self.joint_q_start[i + 1] - q_start
                qd_dim = self.joint_qd_start[i + 1] - qd_start
            else:
                q_dim = len(self.joint_q) - q_start
                qd_dim = len(self.joint_qd) - qd_start

            data = {
                "type": self.joint_type[i],
                "q": self.joint_q[q_start : q_start + q_dim],
                "qd": self.joint_qd[qd_start : qd_start + qd_dim],
                "act": self.joint_act[qd_start : qd_start + qd_dim],
                "armature": self.joint_armature[qd_start : qd_start + qd_dim],
                "q_start": q_start,
                "qd_start": qd_start,
                "linear_compliance": self.joint_linear_compliance[i],
                "angular_compliance": self.joint_angular_compliance[i],
                "name": name,
                "parent_xform": wp.transform_expand(self.joint_X_p[i]),
                "child_xform": wp.transform_expand(self.joint_X_c[i]),
                "enabled": self.joint_enabled[i],
                "axes": [],
                "axis_dim": self.joint_axis_dim[i],
                "parent": parent,
                "child": child,
                "original_id": i,
            }
            num_lin_axes, num_ang_axes = self.joint_axis_dim[i]
            start_ax = self.joint_axis_start[i]
            for j in range(start_ax, start_ax + num_lin_axes + num_ang_axes):
                data["axes"].append(
                    {
                        "axis": self.joint_axis[j],
                        "axis_mode": self.joint_axis_mode[j],
                        "target_ke": self.joint_target_ke[j],
                        "target_kd": self.joint_target_kd[j],
                        "limit_ke": self.joint_limit_ke[j],
                        "limit_kd": self.joint_limit_kd[j],
                        "limit_lower": self.joint_limit_lower[j],
                        "limit_upper": self.joint_limit_upper[j],
                    }
                )

            joint_data[(parent, child)] = data

        # sort body children so we traverse the tree in the same order as the bodies are listed
        for children in body_children.values():
            children.sort(key=lambda x: body_data[x]["original_id"])

        retained_joints = []
        retained_bodies = []
        body_remap = {-1: -1}

        # depth first search over the joint graph
        def dfs(parent_body: int, child_body: int, incoming_xform: wp.transform, last_dynamic_body: int):
            nonlocal visited
            nonlocal retained_joints
            nonlocal retained_bodies
            nonlocal body_data
            nonlocal body_remap

            joint = joint_data[(parent_body, child_body)]
            if joint["type"] == JOINT_FIXED:
                joint_xform = joint["parent_xform"] * wp.transform_inverse(joint["child_xform"])
                incoming_xform = incoming_xform * joint_xform
                parent_name = self.body_name[parent_body] if parent_body > -1 else "world"
                child_name = self.body_name[child_body]
                last_dynamic_body_name = self.body_name[last_dynamic_body] if last_dynamic_body > -1 else "world"
                if verbose:
                    print(
                        f'Remove fixed joint {joint["name"]} between {parent_name} and {child_name}, '
                        f"merging {child_name} into {last_dynamic_body_name}"
                    )
                child_id = body_data[child_body]["original_id"]
                for shape in self.body_shapes[child_id]:
                    self.shape_transform[shape] = incoming_xform * self.shape_transform[shape]
                    if verbose:
                        print(
                            f"  Shape {shape} moved to body {last_dynamic_body_name} with transform {self.shape_transform[shape]}"
                        )
                    if last_dynamic_body > -1:
                        self.shape_body[shape] = body_data[last_dynamic_body]["id"]
                        # add inertia to last_dynamic_body
                        m = body_data[child_body]["mass"]
                        com = body_data[child_body]["com"]
                        inertia = body_data[child_body]["inertia"]
                        body_data[last_dynamic_body]["inertia"] += wp.sim.transform_inertia(
                            m, inertia, incoming_xform.p, incoming_xform.q
                        )
                        body_data[last_dynamic_body]["mass"] += m
                        source_m = body_data[last_dynamic_body]["mass"]
                        source_com = body_data[last_dynamic_body]["com"]
                        body_data[last_dynamic_body]["com"] = (m * com + source_m * source_com) / (m + source_m)
                        body_data[last_dynamic_body]["shapes"].append(shape)
                        # indicate to recompute inverse mass, inertia for this body
                        body_data[last_dynamic_body]["inv_mass"] = None
                    else:
                        self.shape_body[shape] = -1
            else:
                joint["parent_xform"] = incoming_xform * joint["parent_xform"]
                joint["parent"] = last_dynamic_body
                last_dynamic_body = child_body
                incoming_xform = wp.transform()
                retained_joints.append(joint)
                new_id = len(retained_bodies)
                body_data[child_body]["id"] = new_id
                retained_bodies.append(child_body)
                for shape in body_data[child_body]["shapes"]:
                    self.shape_body[shape] = new_id

            visited[parent_body] = True
            if visited[child_body] or child_body not in body_children:
                return
            for child in body_children[child_body]:
                if not visited[child]:
                    dfs(child_body, child, incoming_xform, last_dynamic_body)

        for body in body_children[-1]:
            if not visited[body]:
                dfs(-1, body, wp.transform(), -1)

        # repopulate the model
        self.body_name.clear()
        self.body_q.clear()
        self.body_qd.clear()
        self.body_mass.clear()
        self.body_inertia.clear()
        self.body_com.clear()
        self.body_inv_mass.clear()
        self.body_inv_inertia.clear()
        self.body_shapes.clear()
        for i in retained_bodies:
            body = body_data[i]
            new_id = len(self.body_name)
            body_remap[body["original_id"]] = new_id
            self.body_name.append(body["name"])
            self.body_q.append(list(body["q"]))
            self.body_qd.append(list(body["qd"]))
            m = body["mass"]
            inertia = body["inertia"]
            self.body_mass.append(m)
            self.body_inertia.append(inertia)
            self.body_com.append(body["com"])
            if body["inv_mass"] is None:
                # recompute inverse mass and inertia
                if m > 0.0:
                    self.body_inv_mass.append(1.0 / m)
                    self.body_inv_inertia.append(wp.inverse(inertia))
                else:
                    self.body_inv_mass.append(0.0)
                    self.body_inv_inertia.append(wp.mat33(0.0))
            else:
                self.body_inv_mass.append(body["inv_mass"])
                self.body_inv_inertia.append(body["inv_inertia"])
            self.body_shapes[new_id] = body["shapes"]
            body_remap[body["original_id"]] = new_id

        # sort joints so they appear in the same order as before
        retained_joints.sort(key=lambda x: x["original_id"])

        self.joint_name.clear()
        self.joint_type.clear()
        self.joint_parent.clear()
        self.joint_child.clear()
        self.joint_q.clear()
        self.joint_qd.clear()
        self.joint_q_start.clear()
        self.joint_qd_start.clear()
        self.joint_enabled.clear()
        self.joint_linear_compliance.clear()
        self.joint_angular_compliance.clear()
        self.joint_armature.clear()
        self.joint_X_p.clear()
        self.joint_X_c.clear()
        self.joint_axis.clear()
        self.joint_axis_mode.clear()
        self.joint_target_ke.clear()
        self.joint_target_kd.clear()
        self.joint_limit_lower.clear()
        self.joint_limit_upper.clear()
        self.joint_limit_ke.clear()
        self.joint_limit_kd.clear()
        self.joint_axis_dim.clear()
        self.joint_axis_start.clear()
        self.joint_act.clear()
        for joint in retained_joints:
            self.joint_name.append(joint["name"])
            self.joint_type.append(joint["type"])
            self.joint_parent.append(body_remap[joint["parent"]])
            self.joint_child.append(body_remap[joint["child"]])
            self.joint_q_start.append(len(self.joint_q))
            self.joint_qd_start.append(len(self.joint_qd))
            self.joint_q.extend(joint["q"])
            self.joint_qd.extend(joint["qd"])
            self.joint_act.extend(joint["act"])
            self.joint_armature.extend(joint["armature"])
            self.joint_enabled.append(joint["enabled"])
            self.joint_linear_compliance.append(joint["linear_compliance"])
            self.joint_angular_compliance.append(joint["angular_compliance"])
            self.joint_X_p.append(list(joint["parent_xform"]))
            self.joint_X_c.append(list(joint["child_xform"]))
            self.joint_axis_dim.append(joint["axis_dim"])
            self.joint_axis_start.append(len(self.joint_axis))
            for axis in joint["axes"]:
                self.joint_axis.append(axis["axis"])
                self.joint_axis_mode.append(axis["axis_mode"])
                self.joint_target_ke.append(axis["target_ke"])
                self.joint_target_kd.append(axis["target_kd"])
                self.joint_limit_lower.append(axis["limit_lower"])
                self.joint_limit_upper.append(axis["limit_upper"])
                self.joint_limit_ke.append(axis["limit_ke"])
                self.joint_limit_kd.append(axis["limit_kd"])

    # muscles
    def add_muscle(
        self, bodies: List[int], positions: List[Vec3], f0: float, lm: float, lt: float, lmax: float, pen: float
    ) -> float:
        """Adds a muscle-tendon activation unit.

        Args:
            bodies: A list of body indices for each waypoint
            positions: A list of positions of each waypoint in the body's local frame
            f0: Force scaling
            lm: Muscle length
            lt: Tendon length
            lmax: Maximally efficient muscle length

        Returns:
            The index of the muscle in the model

        .. note:: The simulation support for muscles is in progress and not yet fully functional.

        """

        n = len(bodies)

        self.muscle_start.append(len(self.muscle_bodies))
        self.muscle_params.append((f0, lm, lt, lmax, pen))
        self.muscle_activations.append(0.0)

        for i in range(n):
            self.muscle_bodies.append(bodies[i])
            self.muscle_points.append(positions[i])

        # return the index of the muscle
        return len(self.muscle_start) - 1

    # shapes
    def add_shape_plane(
        self,
        plane: Vec4 = (0.0, 1.0, 0.0, 0.0),
        pos: Vec3 = None,
        rot: Quat = None,
        width: float = 10.0,
        length: float = 10.0,
        body: int = -1,
        ke: float = None,
        kd: float = None,
        kf: float = None,
        ka: float = None,
        mu: float = None,
        restitution: float = None,
        thickness: float = None,
        has_ground_collision: bool = False,
        has_shape_collision: bool = True,
        is_visible: bool = True,
        collision_group: int = -1,
    ):
        """
        Adds a plane collision shape.
        If pos and rot are defined, the plane is assumed to have its normal as (0, 1, 0).
        Otherwise, the plane equation defined through the `plane` argument is used.

        Args:
            plane: The plane equation in form a*x + b*y + c*z + d = 0
            pos: The position of the plane in world coordinates
            rot: The rotation of the plane in world coordinates
            width: The extent along x of the plane (infinite if 0)
            length: The extent along z of the plane (infinite if 0)
            body: The body index to attach the shape to (-1 by default to keep the plane static)
            ke: The contact elastic stiffness (None to use the default value :attr:`default_shape_ke`)
            kd: The contact damping stiffness (None to use the default value :attr:`default_shape_kd`)
            kf: The contact friction stiffness (None to use the default value :attr:`default_shape_kf`)
            ka: The contact adhesion distance (None to use the default value :attr:`default_shape_ka`)
            mu: The coefficient of friction (None to use the default value :attr:`default_shape_mu`)
            restitution: The coefficient of restitution (None to use the default value :attr:`default_shape_restitution`)
            thickness: The thickness of the plane (0 by default) for collision handling (None to use the default value :attr:`default_shape_thickness`)
            has_ground_collision: If True, the shape will collide with the ground plane if `Model.ground` is True
            has_shape_collision: If True, the shape will collide with other shapes
            is_visible: Whether the plane is visible
            collision_group: The collision group of the shape

        Returns:
            The index of the added shape

        """
        if pos is None or rot is None:
            # compute position and rotation from plane equation
            normal = np.array(plane[:3])
            normal /= np.linalg.norm(normal)
            pos = plane[3] * normal
            if np.allclose(normal, (0.0, 1.0, 0.0)):
                # no rotation necessary
                rot = (0.0, 0.0, 0.0, 1.0)
            else:
                c = np.cross(normal, (0.0, 1.0, 0.0))
                angle = np.arcsin(np.linalg.norm(c))
                axis = np.abs(c) / np.linalg.norm(c)
                rot = wp.quat_from_axis_angle(axis, angle)
        scale = wp.vec3(width, length, 0.0)

        return self._add_shape(
            body,
            pos,
            rot,
            GEO_PLANE,
            scale,
            None,
            0.0,
            ke,
            kd,
            kf,
            ka,
            mu,
            restitution,
            thickness,
            has_ground_collision=has_ground_collision,
            has_shape_collision=has_shape_collision,
            is_visible=is_visible,
            collision_group=collision_group,
        )

    def add_shape_sphere(
        self,
        body,
        pos: Vec3 = (0.0, 0.0, 0.0),
        rot: Quat = (0.0, 0.0, 0.0, 1.0),
        radius: float = 1.0,
        density: float = None,
        ke: float = None,
        kd: float = None,
        kf: float = None,
        ka: float = None,
        mu: float = None,
        restitution: float = None,
        is_solid: bool = True,
        thickness: float = None,
        has_ground_collision: bool = True,
        has_shape_collision: bool = True,
        collision_group: int = -1,
        is_visible: bool = True,
    ):
        """Adds a sphere collision shape to a body.

        Args:
            body: The index of the parent body this shape belongs to (use -1 for static shapes)
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            radius: The radius of the sphere
            density: The density of the shape (None to use the default value :attr:`default_shape_density`)
            ke: The contact elastic stiffness (None to use the default value :attr:`default_shape_ke`)
            kd: The contact damping stiffness (None to use the default value :attr:`default_shape_kd`)
            kf: The contact friction stiffness (None to use the default value :attr:`default_shape_kf`)
            ka: The contact adhesion distance (None to use the default value :attr:`default_shape_ka`)
            mu: The coefficient of friction (None to use the default value :attr:`default_shape_mu`)
            restitution: The coefficient of restitution (None to use the default value :attr:`default_shape_restitution`)
            is_solid: Whether the sphere is solid or hollow
            thickness: Thickness to use for computing inertia of a hollow sphere, and for collision handling (None to use the default value :attr:`default_shape_thickness`)
            has_ground_collision: If True, the shape will collide with the ground plane if `Model.ground` is True
            has_shape_collision: If True, the shape will collide with other shapes
            collision_group: The collision group of the shape
            is_visible: Whether the sphere is visible

        Returns:
            The index of the added shape

        """

        thickness = self.default_shape_thickness if thickness is None else thickness
        return self._add_shape(
            body,
            wp.vec3(pos),
            wp.quat(rot),
            GEO_SPHERE,
            wp.vec3(radius, 0.0, 0.0),
            None,
            density,
            ke,
            kd,
            kf,
            ka,
            mu,
            restitution,
            thickness + radius,
            is_solid,
            has_ground_collision=has_ground_collision,
            has_shape_collision=has_shape_collision,
            collision_group=collision_group,
            is_visible=is_visible,
        )

    def add_shape_box(
        self,
        body: int,
        pos: Vec3 = (0.0, 0.0, 0.0),
        rot: Quat = (0.0, 0.0, 0.0, 1.0),
        hx: float = 0.5,
        hy: float = 0.5,
        hz: float = 0.5,
        density: float = None,
        ke: float = None,
        kd: float = None,
        kf: float = None,
        ka: float = None,
        mu: float = None,
        restitution: float = None,
        is_solid: bool = True,
        thickness: float = None,
        has_ground_collision: bool = True,
        has_shape_collision: bool = True,
        collision_group: int = -1,
        is_visible: bool = True,
    ):
        """Adds a box collision shape to a body.

        Args:
            body: The index of the parent body this shape belongs to (use -1 for static shapes)
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            hx: The half-extent along the x-axis
            hy: The half-extent along the y-axis
            hz: The half-extent along the z-axis
            density: The density of the shape (None to use the default value :attr:`default_shape_density`)
            ke: The contact elastic stiffness (None to use the default value :attr:`default_shape_ke`)
            kd: The contact damping stiffness (None to use the default value :attr:`default_shape_kd`)
            kf: The contact friction stiffness (None to use the default value :attr:`default_shape_kf`)
            ka: The contact adhesion distance (None to use the default value :attr:`default_shape_ka`)
            mu: The coefficient of friction (None to use the default value :attr:`default_shape_mu`)
            restitution: The coefficient of restitution (None to use the default value :attr:`default_shape_restitution`)
            is_solid: Whether the box is solid or hollow
            thickness: Thickness to use for computing inertia of a hollow box, and for collision handling (None to use the default value :attr:`default_shape_thickness`)
            has_ground_collision: If True, the shape will collide with the ground plane if `Model.ground` is True
            has_shape_collision: If True, the shape will collide with other shapes
            collision_group: The collision group of the shape
            is_visible: Whether the box is visible

        Returns:
            The index of the added shape

        """

        return self._add_shape(
            body,
            wp.vec3(pos),
            wp.quat(rot),
            GEO_BOX,
            wp.vec3(hx, hy, hz),
            None,
            density,
            ke,
            kd,
            kf,
            ka,
            mu,
            restitution,
            thickness,
            is_solid,
            has_ground_collision=has_ground_collision,
            has_shape_collision=has_shape_collision,
            collision_group=collision_group,
            is_visible=is_visible,
        )

    def add_shape_capsule(
        self,
        body: int,
        pos: Vec3 = (0.0, 0.0, 0.0),
        rot: Quat = (0.0, 0.0, 0.0, 1.0),
        radius: float = 1.0,
        half_height: float = 0.5,
        up_axis: int = 1,
        density: float = None,
        ke: float = None,
        kd: float = None,
        kf: float = None,
        ka: float = None,
        mu: float = None,
        restitution: float = None,
        is_solid: bool = True,
        thickness: float = None,
        has_ground_collision: bool = True,
        has_shape_collision: bool = True,
        collision_group: int = -1,
        is_visible: bool = True,
    ):
        """Adds a capsule collision shape to a body.

        Args:
            body: The index of the parent body this shape belongs to (use -1 for static shapes)
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            radius: The radius of the capsule
            half_height: The half length of the center cylinder along the up axis
            up_axis: The axis along which the capsule is aligned (0=x, 1=y, 2=z)
            density: The density of the shape (None to use the default value :attr:`default_shape_density`)
            ke: The contact elastic stiffness (None to use the default value :attr:`default_shape_ke`)
            kd: The contact damping stiffness (None to use the default value :attr:`default_shape_kd`)
            kf: The contact friction stiffness (None to use the default value :attr:`default_shape_kf`)
            ka: The contact adhesion distance (None to use the default value :attr:`default_shape_ka`)
            mu: The coefficient of friction (None to use the default value :attr:`default_shape_mu`)
            restitution: The coefficient of restitution (None to use the default value :attr:`default_shape_restitution`)
            is_solid: Whether the capsule is solid or hollow
            thickness: Thickness to use for computing inertia of a hollow capsule, and for collision handling (None to use the default value :attr:`default_shape_thickness`)
            has_ground_collision: If True, the shape will collide with the ground plane if `Model.ground` is True
            has_shape_collision: If True, the shape will collide with other shapes
            collision_group: The collision group of the shape
            is_visible: Whether the capsule is visible

        Returns:
            The index of the added shape

        """

        q = wp.quat(rot)
        sqh = math.sqrt(0.5)
        if up_axis == 0:
            q = wp.mul(q, wp.quat(0.0, 0.0, -sqh, sqh))
        elif up_axis == 2:
            q = wp.mul(q, wp.quat(sqh, 0.0, 0.0, sqh))

        thickness = self.default_shape_thickness if thickness is None else thickness
        return self._add_shape(
            body,
            wp.vec3(pos),
            wp.quat(q),
            GEO_CAPSULE,
            wp.vec3(radius, half_height, 0.0),
            None,
            density,
            ke,
            kd,
            kf,
            ka,
            mu,
            restitution,
            thickness + radius,
            is_solid,
            has_ground_collision=has_ground_collision,
            has_shape_collision=has_shape_collision,
            collision_group=collision_group,
            is_visible=is_visible,
        )

    def add_shape_cylinder(
        self,
        body: int,
        pos: Vec3 = (0.0, 0.0, 0.0),
        rot: Quat = (0.0, 0.0, 0.0, 1.0),
        radius: float = 1.0,
        half_height: float = 0.5,
        up_axis: int = 1,
        density: float = None,
        ke: float = None,
        kd: float = None,
        kf: float = None,
        ka: float = None,
        mu: float = None,
        restitution: float = None,
        is_solid: bool = True,
        thickness: float = None,
        has_ground_collision: bool = True,
        has_shape_collision: bool = True,
        collision_group: int = -1,
        is_visible: bool = True,
    ):
        """Adds a cylinder collision shape to a body.

        Args:
            body: The index of the parent body this shape belongs to (use -1 for static shapes)
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            radius: The radius of the cylinder
            half_height: The half length of the cylinder along the up axis
            up_axis: The axis along which the cylinder is aligned (0=x, 1=y, 2=z)
            density: The density of the shape (None to use the default value :attr:`default_shape_density`)
            ke: The contact elastic stiffness (None to use the default value :attr:`default_shape_ke`)
            kd: The contact damping stiffness (None to use the default value :attr:`default_shape_kd`)
            kf: The contact friction stiffness (None to use the default value :attr:`default_shape_kf`)
            ka: The contact adhesion distance (None to use the default value :attr:`default_shape_ka`)
            mu: The coefficient of friction (None to use the default value :attr:`default_shape_mu`)
            restitution: The coefficient of restitution (None to use the default value :attr:`default_shape_restitution`)
            is_solid: Whether the cylinder is solid or hollow
            thickness: Thickness to use for computing inertia of a hollow cylinder, and for collision handling (None to use the default value :attr:`default_shape_thickness`)
            has_ground_collision: If True, the shape will collide with the ground plane if `Model.ground` is True
            has_shape_collision: If True, the shape will collide with other shapes
            collision_group: The collision group of the shape
            is_visible: Whether the cylinder is visible

        Note:
            Cylinders are currently not supported in rigid body collision handling.

        Returns:
            The index of the added shape

        """

        q = rot
        sqh = math.sqrt(0.5)
        if up_axis == 0:
            q = wp.mul(rot, wp.quat(0.0, 0.0, -sqh, sqh))
        elif up_axis == 2:
            q = wp.mul(rot, wp.quat(sqh, 0.0, 0.0, sqh))

        return self._add_shape(
            body,
            wp.vec3(pos),
            wp.quat(q),
            GEO_CYLINDER,
            wp.vec3(radius, half_height, 0.0),
            None,
            density,
            ke,
            kd,
            kf,
            ka,
            mu,
            restitution,
            thickness,
            is_solid,
            has_ground_collision=has_ground_collision,
            has_shape_collision=has_shape_collision,
            collision_group=collision_group,
            is_visible=is_visible,
        )

    def add_shape_cone(
        self,
        body: int,
        pos: Vec3 = (0.0, 0.0, 0.0),
        rot: Quat = (0.0, 0.0, 0.0, 1.0),
        radius: float = 1.0,
        half_height: float = 0.5,
        up_axis: int = 1,
        density: float = None,
        ke: float = None,
        kd: float = None,
        kf: float = None,
        ka: float = None,
        mu: float = None,
        restitution: float = None,
        is_solid: bool = True,
        thickness: float = None,
        has_ground_collision: bool = True,
        has_shape_collision: bool = True,
        collision_group: int = -1,
        is_visible: bool = True,
    ):
        """Adds a cone collision shape to a body.

        Args:
            body: The index of the parent body this shape belongs to (use -1 for static shapes)
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            radius: The radius of the cone
            half_height: The half length of the cone along the up axis
            up_axis: The axis along which the cone is aligned (0=x, 1=y, 2=z)
            density: The density of the shape (None to use the default value :attr:`default_shape_density`)
            ke: The contact elastic stiffness (None to use the default value :attr:`default_shape_ke`)
            kd: The contact damping stiffness (None to use the default value :attr:`default_shape_kd`)
            kf: The contact friction stiffness (None to use the default value :attr:`default_shape_kf`)
            ka: The contact adhesion distance (None to use the default value :attr:`default_shape_ka`)
            mu: The coefficient of friction (None to use the default value :attr:`default_shape_mu`)
            restitution: The coefficient of restitution (None to use the default value :attr:`default_shape_restitution`)
            is_solid: Whether the cone is solid or hollow
            thickness: Thickness to use for computing inertia of a hollow cone, and for collision handling (None to use the default value :attr:`default_shape_thickness`)
            has_ground_collision: If True, the shape will collide with the ground plane if `Model.ground` is True
            has_shape_collision: If True, the shape will collide with other shapes
            collision_group: The collision group of the shape
            is_visible: Whether the cone is visible

        Note:
            Cones are currently not supported in rigid body collision handling.

        Returns:
            The index of the added shape

        """

        q = rot
        sqh = math.sqrt(0.5)
        if up_axis == 0:
            q = wp.mul(rot, wp.quat(0.0, 0.0, -sqh, sqh))
        elif up_axis == 2:
            q = wp.mul(rot, wp.quat(sqh, 0.0, 0.0, sqh))

        return self._add_shape(
            body,
            wp.vec3(pos),
            wp.quat(q),
            GEO_CONE,
            wp.vec3(radius, half_height, 0.0),
            None,
            density,
            ke,
            kd,
            kf,
            ka,
            mu,
            restitution,
            thickness,
            is_solid,
            has_ground_collision=has_ground_collision,
            has_shape_collision=has_shape_collision,
            collision_group=collision_group,
            is_visible=is_visible,
        )

    def add_shape_mesh(
        self,
        body: int,
        pos: Optional[Vec3] = None,
        rot: Optional[Quat] = None,
        mesh: Optional[Mesh] = None,
        scale: Optional[Vec3] = None,
        density: float = None,
        ke: float = None,
        kd: float = None,
        kf: float = None,
        ka: float = None,
        mu: float = None,
        restitution: float = None,
        is_solid: bool = True,
        thickness: float = None,
        has_ground_collision: bool = True,
        has_shape_collision: bool = True,
        collision_group: int = -1,
        is_visible: bool = True,
    ):
        """Adds a triangle mesh collision shape to a body.

        Args:
            body: The index of the parent body this shape belongs to (use -1 for static shapes)
            pos: The location of the shape with respect to the parent frame
              (None to use the default value ``wp.vec3(0.0, 0.0, 0.0)``)
            rot: The rotation of the shape with respect to the parent frame
              (None to use the default value ``wp.quat(0.0, 0.0, 0.0, 1.0)``)
            mesh: The mesh object
            scale: Scale to use for the collider. (None to use the default value ``wp.vec3(1.0, 1.0, 1.0)``)
            density: The density of the shape (None to use the default value :attr:`default_shape_density`)
            ke: The contact elastic stiffness (None to use the default value :attr:`default_shape_ke`)
            kd: The contact damping stiffness (None to use the default value :attr:`default_shape_kd`)
            kf: The contact friction stiffness (None to use the default value :attr:`default_shape_kf`)
            ka: The contact adhesion distance (None to use the default value :attr:`default_shape_ka`)
            mu: The coefficient of friction (None to use the default value :attr:`default_shape_mu`)
            restitution: The coefficient of restitution (None to use the default value :attr:`default_shape_restitution`)
            is_solid: If True, the mesh is solid, otherwise it is a hollow surface with the given wall thickness
            thickness: Thickness to use for computing inertia of a hollow mesh, and for collision handling (None to use the default value :attr:`default_shape_thickness`)
            has_ground_collision: If True, the shape will collide with the ground plane if `Model.ground` is True
            has_shape_collision: If True, the shape will collide with other shapes
            collision_group: The collision group of the shape
            is_visible: Whether the mesh is visible

        Returns:
            The index of the added shape

        """

        if pos is None:
            pos = wp.vec3(0.0, 0.0, 0.0)

        if rot is None:
            rot = wp.quat(0.0, 0.0, 0.0, 1.0)

        if scale is None:
            scale = wp.vec3(1.0, 1.0, 1.0)

        return self._add_shape(
            body,
            pos,
            rot,
            GEO_MESH,
            wp.vec3(scale[0], scale[1], scale[2]),
            mesh,
            density,
            ke,
            kd,
            kf,
            ka,
            mu,
            restitution,
            thickness,
            is_solid,
            has_ground_collision=has_ground_collision,
            has_shape_collision=has_shape_collision,
            collision_group=collision_group,
            is_visible=is_visible,
        )

    def add_shape_sdf(
        self,
        body: int,
        pos: Vec3 = (0.0, 0.0, 0.0),
        rot: Quat = (0.0, 0.0, 0.0, 1.0),
        sdf: SDF = None,
        scale: Vec3 = (1.0, 1.0, 1.0),
        density: float = None,
        ke: float = None,
        kd: float = None,
        kf: float = None,
        ka: float = None,
        mu: float = None,
        restitution: float = None,
        is_solid: bool = True,
        thickness: float = None,
        has_ground_collision: bool = True,
        has_shape_collision: bool = True,
        collision_group: int = -1,
        is_visible: bool = True,
    ):
        """Adds SDF collision shape to a body.

        Args:
            body: The index of the parent body this shape belongs to (use -1 for static shapes)
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            sdf: The sdf object
            scale: Scale to use for the collider
            density: The density of the shape (None to use the default value :attr:`default_shape_density`)
            ke: The contact elastic stiffness (None to use the default value :attr:`default_shape_ke`)
            kd: The contact damping stiffness (None to use the default value :attr:`default_shape_kd`)
            kf: The contact friction stiffness (None to use the default value :attr:`default_shape_kf`)
            ka: The contact adhesion distance (None to use the default value :attr:`default_shape_ka`)
            mu: The coefficient of friction (None to use the default value :attr:`default_shape_mu`)
            restitution: The coefficient of restitution (None to use the default value :attr:`default_shape_restitution`)
            is_solid: If True, the SDF is solid, otherwise it is a hollow surface with the given wall thickness
            thickness: Thickness to use for collision handling (None to use the default value :attr:`default_shape_thickness`)
            has_ground_collision: If True, the shape will collide with the ground plane if `Model.ground` is True
            has_shape_collision: If True, the shape will collide with other shapes
            collision_group: The collision group of the shape
            is_visible: Whether the shape is visible

        Returns:
            The index of the added shape

        """
        return self._add_shape(
            body,
            wp.vec3(pos),
            wp.quat(rot),
            GEO_SDF,
            wp.vec3(scale[0], scale[1], scale[2]),
            sdf,
            density,
            ke,
            kd,
            kf,
            ka,
            mu,
            restitution,
            thickness,
            is_solid,
            has_ground_collision=has_ground_collision,
            has_shape_collision=has_shape_collision,
            collision_group=collision_group,
            is_visible=is_visible,
        )

    def _shape_radius(self, type, scale, src):
        """
        Calculates the radius of a sphere that encloses the shape, used for broadphase collision detection.
        """
        if type == GEO_SPHERE:
            return scale[0]
        elif type == GEO_BOX:
            return np.linalg.norm(scale)
        elif type == GEO_CAPSULE or type == GEO_CYLINDER or type == GEO_CONE:
            return scale[0] + scale[1]
        elif type == GEO_MESH:
            vmax = np.max(np.abs(src.vertices), axis=0) * np.max(scale)
            return np.linalg.norm(vmax)
        elif type == GEO_PLANE:
            if scale[0] > 0.0 and scale[1] > 0.0:
                # finite plane
                return np.linalg.norm(scale)
            else:
                return 1.0e6
        else:
            return 10.0

    def _add_shape(
        self,
        body,
        pos,
        rot,
        type,
        scale,
        src=None,
        density=None,
        ke=None,
        kd=None,
        kf=None,
        ka=None,
        mu=None,
        restitution=None,
        thickness=None,
        is_solid=True,
        collision_group=-1,
        collision_filter_parent=True,
        has_ground_collision=True,
        has_shape_collision=True,
        is_visible=True,
    ):
        self.shape_body.append(body)
        shape = self.shape_count
        if body in self.body_shapes:
            # no contacts between shapes of the same body
            for same_body_shape in self.body_shapes[body]:
                self.shape_collision_filter_pairs.add((same_body_shape, shape))
            self.body_shapes[body].append(shape)
        else:
            self.body_shapes[body] = [shape]
        ke = ke if ke is not None else self.default_shape_ke
        kd = kd if kd is not None else self.default_shape_kd
        kf = kf if kf is not None else self.default_shape_kf
        ka = ka if ka is not None else self.default_shape_ka
        mu = mu if mu is not None else self.default_shape_mu
        restitution = restitution if restitution is not None else self.default_shape_restitution
        thickness = thickness if thickness is not None else self.default_shape_thickness
        density = density if density is not None else self.default_shape_density
        self.shape_transform.append(wp.transform(pos, rot))
        self.shape_visible.append(is_visible)
        self.shape_geo_type.append(type)
        self.shape_geo_scale.append((scale[0], scale[1], scale[2]))
        self.shape_geo_src.append(src)
        self.shape_geo_thickness.append(thickness)
        self.shape_geo_is_solid.append(is_solid)
        self.shape_material_ke.append(ke)
        self.shape_material_kd.append(kd)
        self.shape_material_kf.append(kf)
        self.shape_material_ka.append(ka)
        self.shape_material_mu.append(mu)
        self.shape_material_restitution.append(restitution)
        self.shape_collision_group.append(collision_group)
        if collision_group not in self.shape_collision_group_map:
            self.shape_collision_group_map[collision_group] = []
        self.last_collision_group = max(self.last_collision_group, collision_group)
        self.shape_collision_group_map[collision_group].append(shape)
        self.shape_collision_radius.append(self._shape_radius(type, scale, src))
        if collision_filter_parent and body > -1 and body in self.joint_parents:
            for parent_body in self.joint_parents[body]:
                if parent_body > -1:
                    for parent_shape in self.body_shapes[parent_body]:
                        self.shape_collision_filter_pairs.add((parent_shape, shape))
        if body == -1:
            has_ground_collision = False
        self.shape_ground_collision.append(has_ground_collision)
        self.shape_shape_collision.append(has_shape_collision)

        (m, c, I) = compute_shape_mass(type, scale, src, density, is_solid, thickness)

        self._update_body_mass(body, m, I, pos + c, rot)
        return shape

    # particles
    def add_particle(
        self, pos: Vec3, vel: Vec3, mass: float, radius: float = None, flags: wp.uint32 = PARTICLE_FLAG_ACTIVE
    ) -> int:
        """Adds a single particle to the model

        Args:
            pos: The initial position of the particle
            vel: The initial velocity of the particle
            mass: The mass of the particle
            radius: The radius of the particle used in collision handling. If None, the radius is set to the default value (:attr:`default_particle_radius`).
            flags: The flags that control the dynamical behavior of the particle, see PARTICLE_FLAG_* constants

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that does is not subject to dynamics.

        Returns:
            The index of the particle in the system
        """
        self.particle_q.append(pos)
        self.particle_qd.append(vel)
        self.particle_mass.append(mass)
        if radius is None:
            radius = self.default_particle_radius
        self.particle_radius.append(radius)
        self.particle_flags.append(flags)

        return len(self.particle_q) - 1

    def add_spring(self, i: int, j, ke: float, kd: float, control: float):
        """Adds a spring between two particles in the system

        Args:
            i: The index of the first particle
            j: The index of the second particle
            ke: The elastic stiffness of the spring
            kd: The damping stiffness of the spring
            control: The actuation level of the spring

        Note:
            The spring is created with a rest-length based on the distance
            between the particles in their initial configuration.

        """
        self.spring_indices.append(i)
        self.spring_indices.append(j)
        self.spring_stiffness.append(ke)
        self.spring_damping.append(kd)
        self.spring_control.append(control)

        # compute rest length
        p = self.particle_q[i]
        q = self.particle_q[j]

        delta = np.subtract(p, q)
        l = np.sqrt(np.dot(delta, delta))

        self.spring_rest_length.append(l)

    def add_triangle(
        self,
        i: int,
        j: int,
        k: int,
        tri_ke: float = default_tri_ke,
        tri_ka: float = default_tri_ka,
        tri_kd: float = default_tri_kd,
        tri_drag: float = default_tri_drag,
        tri_lift: float = default_tri_lift,
    ) -> float:
        """Adds a triangular FEM element between three particles in the system.

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        parameters specified on the model. See model.tri_ke, model.tri_kd.

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle

        Return:
            The area of the triangle

        Note:
            The triangle is created with a rest-length based on the distance
            between the particles in their initial configuration.
        """
        # TODO: Expose elastic parameters on a per-element basis

        # compute basis for 2D rest pose
        p = self.particle_q[i]
        q = self.particle_q[j]
        r = self.particle_q[k]

        qp = q - p
        rp = r - p

        # construct basis aligned with the triangle
        n = wp.normalize(wp.cross(qp, rp))
        e1 = wp.normalize(qp)
        e2 = wp.normalize(wp.cross(n, e1))

        R = np.array((e1, e2))
        M = np.array((qp, rp))

        D = R @ M.T

        area = np.linalg.det(D) / 2.0

        if area <= 0.0:
            print("inverted or degenerate triangle element")
            return 0.0
        else:
            inv_D = np.linalg.inv(D)

            self.tri_indices.append((i, j, k))
            self.tri_poses.append(inv_D.tolist())
            self.tri_activations.append(0.0)
            self.tri_materials.append((tri_ke, tri_ka, tri_kd, tri_drag, tri_lift))
            return area

    def add_triangles(
        self,
        i: List[int],
        j: List[int],
        k: List[int],
        tri_ke: Optional[List[float]] = None,
        tri_ka: Optional[List[float]] = None,
        tri_kd: Optional[List[float]] = None,
        tri_drag: Optional[List[float]] = None,
        tri_lift: Optional[List[float]] = None,
    ) -> List[float]:
        """Adds triangular FEM elements between groups of three particles in the system.

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        Parameters specified on the model. See model.tri_ke, model.tri_kd.

        Args:
            i: The indices of the first particle
            j: The indices of the second particle
            k: The indices of the third particle

        Return:
            The areas of the triangles

        Note:
            A triangle is created with a rest-length based on the distance
            between the particles in their initial configuration.

        """
        # compute basis for 2D rest pose
        p = np.array(self.particle_q)[i]
        q = np.array(self.particle_q)[j]
        r = np.array(self.particle_q)[k]

        qp = q - p
        rp = r - p

        def normalized(a):
            l = np.linalg.norm(a, axis=-1, keepdims=True)
            l[l == 0] = 1.0
            return a / l

        n = normalized(np.cross(qp, rp))
        e1 = normalized(qp)
        e2 = normalized(np.cross(n, e1))

        R = np.concatenate((e1[..., None], e2[..., None]), axis=-1)
        M = np.concatenate((qp[..., None], rp[..., None]), axis=-1)

        D = np.matmul(R.transpose(0, 2, 1), M)

        areas = np.linalg.det(D) / 2.0
        areas[areas < 0.0] = 0.0
        valid_inds = (areas > 0.0).nonzero()[0]
        if len(valid_inds) < len(areas):
            print("inverted or degenerate triangle elements")

        D[areas == 0.0] = np.eye(2)[None, ...]
        inv_D = np.linalg.inv(D)

        inds = np.concatenate((i[valid_inds, None], j[valid_inds, None], k[valid_inds, None]), axis=-1)

        self.tri_indices.extend(inds.tolist())
        self.tri_poses.extend(inv_D[valid_inds].tolist())
        self.tri_activations.extend([0.0] * len(valid_inds))

        def init_if_none(arr, defaultValue):
            if arr is None:
                return [defaultValue] * len(areas)
            return arr

        tri_ke = init_if_none(tri_ke, self.default_tri_ke)
        tri_ka = init_if_none(tri_ka, self.default_tri_ka)
        tri_kd = init_if_none(tri_kd, self.default_tri_kd)
        tri_drag = init_if_none(tri_drag, self.default_tri_drag)
        tri_lift = init_if_none(tri_lift, self.default_tri_lift)

        self.tri_materials.extend(
            zip(
                np.array(tri_ke)[valid_inds],
                np.array(tri_ka)[valid_inds],
                np.array(tri_kd)[valid_inds],
                np.array(tri_drag)[valid_inds],
                np.array(tri_lift)[valid_inds],
            )
        )
        return areas.tolist()

    def add_tetrahedron(
        self, i: int, j: int, k: int, l: int, k_mu: float = 1.0e3, k_lambda: float = 1.0e3, k_damp: float = 0.0
    ) -> float:
        """Adds a tetrahedral FEM element between four particles in the system.

        Tetrahedra are modeled as viscoelastic elements with a NeoHookean energy
        density based on [Smith et al. 2018].

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle
            l: The index of the fourth particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The element's damping stiffness

        Return:
            The volume of the tetrahedron

        Note:
            The tetrahedron is created with a rest-pose based on the particle's initial configuration

        """
        # compute basis for 2D rest pose
        p = np.array(self.particle_q[i])
        q = np.array(self.particle_q[j])
        r = np.array(self.particle_q[k])
        s = np.array(self.particle_q[l])

        qp = q - p
        rp = r - p
        sp = s - p

        Dm = np.array((qp, rp, sp)).T
        volume = np.linalg.det(Dm) / 6.0

        if volume <= 0.0:
            print("inverted tetrahedral element")
        else:
            inv_Dm = np.linalg.inv(Dm)

            self.tet_indices.append((i, j, k, l))
            self.tet_poses.append(inv_Dm.tolist())
            self.tet_activations.append(0.0)
            self.tet_materials.append((k_mu, k_lambda, k_damp))

        return volume

    def add_edge(
        self,
        i: int,
        j: int,
        k: int,
        l: int,
        rest: float = None,
        edge_ke: float = default_edge_ke,
        edge_kd: float = default_edge_kd,
    ):
        """Adds a bending edge element between four particles in the system.

        Bending elements are designed to be between two connected triangles. Then
        bending energy is based of [Bridson et al. 2002]. Bending stiffness is controlled
        by the `model.tri_kb` parameter.

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle
            l: The index of the fourth particle
            rest: The rest angle across the edge in radians, if not specified it will be computed

        Note:
            The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
            vertices indexed by 'i' and 'j'. This defines two connected triangles with counter clockwise
            winding: (i, k, l), (j, l, k).

        """
        # compute rest angle
        if rest is None:
            x1 = self.particle_q[i]
            x2 = self.particle_q[j]
            x3 = self.particle_q[k]
            x4 = self.particle_q[l]

            n1 = wp.normalize(wp.cross(x3 - x1, x4 - x1))
            n2 = wp.normalize(wp.cross(x4 - x2, x3 - x2))
            e = wp.normalize(x4 - x3)

            d = np.clip(np.dot(n2, n1), -1.0, 1.0)

            angle = math.acos(d)
            sign = np.sign(np.dot(np.cross(n2, n1), e))

            rest = angle * sign

        self.edge_indices.append((i, j, k, l))
        self.edge_rest_angle.append(rest)
        self.edge_bending_properties.append((edge_ke, edge_kd))

    def add_edges(
        self,
        i,
        j,
        k,
        l,
        rest: Optional[List[float]] = None,
        edge_ke: Optional[List[float]] = None,
        edge_kd: Optional[List[float]] = None,
    ):
        """Adds bending edge elements between groups of four particles in the system.

        Bending elements are designed to be between two connected triangles. Then
        bending energy is based of [Bridson et al. 2002]. Bending stiffness is controlled
        by the `model.tri_kb` parameter.

        Args:
            i: The indices of the first particle
            j: The indices of the second particle
            k: The indices of the third particle
            l: The indices of the fourth particle
            rest: The rest angles across the edges in radians, if not specified they will be computed

        Note:
            The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
            vertices indexed by 'i' and 'j'. This defines two connected triangles with counter clockwise
            winding: (i, k, l), (j, l, k).

        """
        if rest is None:
            # compute rest angle
            x1 = np.array(self.particle_q)[i]
            x2 = np.array(self.particle_q)[j]
            x3 = np.array(self.particle_q)[k]
            x4 = np.array(self.particle_q)[l]

            def normalized(a):
                l = np.linalg.norm(a, axis=-1, keepdims=True)
                l[l == 0] = 1.0
                return a / l

            n1 = normalized(np.cross(x3 - x1, x4 - x1))
            n2 = normalized(np.cross(x4 - x2, x3 - x2))
            e = normalized(x4 - x3)

            def dot(a, b):
                return (a * b).sum(axis=-1)

            d = np.clip(dot(n2, n1), -1.0, 1.0)

            angle = np.arccos(d)
            sign = np.sign(dot(np.cross(n2, n1), e))

            rest = angle * sign

        inds = np.concatenate((i[:, None], j[:, None], k[:, None], l[:, None]), axis=-1)

        self.edge_indices.extend(inds.tolist())
        self.edge_rest_angle.extend(rest.tolist())

        def init_if_none(arr, defaultValue):
            if arr is None:
                return [defaultValue] * len(i)
            return arr

        edge_ke = init_if_none(edge_ke, self.default_edge_ke)
        edge_kd = init_if_none(edge_kd, self.default_edge_kd)

        self.edge_bending_properties.extend(zip(edge_ke, edge_kd))

    def add_cloth_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        cell_x: float,
        cell_y: float,
        mass: float,
        reverse_winding: bool = False,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
        tri_ke: float = default_tri_ke,
        tri_ka: float = default_tri_ka,
        tri_kd: float = default_tri_kd,
        tri_drag: float = default_tri_drag,
        tri_lift: float = default_tri_lift,
        edge_ke: float = default_edge_ke,
        edge_kd: float = default_edge_kd,
        add_springs: bool = False,
        spring_ke: float = default_spring_ke,
        spring_kd: float = default_spring_kd,
    ):
        """Helper to create a regular planar cloth grid

        Creates a rectangular grid of particles with FEM triangles and bending elements
        automatically.

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            mass: The mass of each particle
            reverse_winding: Flip the winding of the mesh
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic

        """

        def grid_index(x, y, dim_x):
            return y * dim_x + x

        start_vertex = len(self.particle_q)
        start_tri = len(self.tri_indices)

        for y in range(0, dim_y + 1):
            for x in range(0, dim_x + 1):
                g = wp.vec3(x * cell_x, y * cell_y, 0.0)
                p = wp.quat_rotate(rot, g) + pos
                m = mass

                if x == 0 and fix_left:
                    m = 0.0
                elif x == dim_x and fix_right:
                    m = 0.0
                elif y == 0 and fix_bottom:
                    m = 0.0
                elif y == dim_y and fix_top:
                    m = 0.0

                self.add_particle(p, vel, m)

                if x > 0 and y > 0:
                    if reverse_winding:
                        tri1 = (
                            start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y, dim_x + 1),
                        )

                        tri2 = (
                            start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y, dim_x + 1),
                            start_vertex + grid_index(x - 1, y, dim_x + 1),
                        )

                        self.add_triangle(*tri1, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
                        self.add_triangle(*tri2, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

                    else:
                        tri1 = (
                            start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y - 1, dim_x + 1),
                            start_vertex + grid_index(x - 1, y, dim_x + 1),
                        )

                        tri2 = (
                            start_vertex + grid_index(x, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y, dim_x + 1),
                            start_vertex + grid_index(x - 1, y, dim_x + 1),
                        )

                        self.add_triangle(*tri1, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
                        self.add_triangle(*tri2, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

        end_tri = len(self.tri_indices)

        # bending constraints, could create these explicitly for a grid but this
        # is a good test of the adjacency structure
        adj = wp.utils.MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        spring_indices = set()

        for _k, e in adj.edges.items():
            # skip open edges
            if e.f0 == -1 or e.f1 == -1:
                continue

            self.add_edge(
                e.o0, e.o1, e.v0, e.v1, edge_ke=edge_ke, edge_kd=edge_kd
            )  # opposite 0, opposite 1, vertex 0, vertex 1

            spring_indices.add((min(e.o0, e.o1), max(e.o0, e.o1)))
            spring_indices.add((min(e.o0, e.v0), max(e.o0, e.v0)))
            spring_indices.add((min(e.o0, e.v1), max(e.o0, e.v1)))

            spring_indices.add((min(e.o1, e.v0), max(e.o1, e.v0)))
            spring_indices.add((min(e.o1, e.v1), max(e.o1, e.v1)))

            spring_indices.add((min(e.v0, e.v1), max(e.v0, e.v1)))

        if add_springs:
            for i, j in spring_indices:
                self.add_spring(i, j, spring_ke, spring_kd, control=0.0)

    def add_cloth_mesh(
        self,
        pos: Vec3,
        rot: Quat,
        scale: float,
        vel: Vec3,
        vertices: List[Vec3],
        indices: List[int],
        density: float,
        edge_callback=None,
        face_callback=None,
        tri_ke: float = default_tri_ke,
        tri_ka: float = default_tri_ka,
        tri_kd: float = default_tri_kd,
        tri_drag: float = default_tri_drag,
        tri_lift: float = default_tri_lift,
        edge_ke: float = default_edge_ke,
        edge_kd: float = default_edge_kd,
        add_springs: bool = False,
        spring_ke: float = default_spring_ke,
        spring_kd: float = default_spring_kd,
    ):
        """Helper to create a cloth model from a regular triangle mesh

        Creates one FEM triangle element and one bending element for every face
        and edge in the input triangle mesh

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            vertices: A list of vertex positions
            indices: A list of triangle indices, 3 entries per-face
            density: The density per-area of the mesh
            edge_callback: A user callback when an edge is created
            face_callback: A user callback when a face is created

        Note:

            The mesh should be two manifold.
        """
        num_tris = int(len(indices) / 3)

        start_vertex = len(self.particle_q)
        start_tri = len(self.tri_indices)

        # particles
        for v in vertices:
            p = wp.quat_rotate(rot, v * scale) + pos

            self.add_particle(p, vel, 0.0)

        # triangles
        inds = start_vertex + np.array(indices)
        inds = inds.reshape(-1, 3)
        areas = self.add_triangles(
            inds[:, 0],
            inds[:, 1],
            inds[:, 2],
            [tri_ke] * num_tris,
            [tri_ka] * num_tris,
            [tri_kd] * num_tris,
            [tri_drag] * num_tris,
            [tri_lift] * num_tris,
        )

        for t in range(num_tris):
            area = areas[t]

            self.particle_mass[inds[t, 0]] += density * area / 3.0
            self.particle_mass[inds[t, 1]] += density * area / 3.0
            self.particle_mass[inds[t, 2]] += density * area / 3.0

        end_tri = len(self.tri_indices)

        adj = wp.utils.MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        edgeinds = np.fromiter(
            (x for e in adj.edges.values() if e.f0 != -1 and e.f1 != -1 for x in (e.o0, e.o1, e.v0, e.v1)),
            int,
        ).reshape(-1, 4)
        self.add_edges(
            edgeinds[:, 0],
            edgeinds[:, 1],
            edgeinds[:, 2],
            edgeinds[:, 0],
            edge_ke=[edge_ke] * len(edgeinds),
            edge_kd=[edge_kd] * len(edgeinds),
        )

        if add_springs:
            spring_indices = set()
            for i, j, k, l in edgeinds:
                spring_indices.add((min(i, j), max(i, j)))
                spring_indices.add((min(i, k), max(i, k)))
                spring_indices.add((min(i, l), max(i, l)))

                spring_indices.add((min(j, k), max(j, k)))
                spring_indices.add((min(j, l), max(j, l)))

                spring_indices.add((min(k, l), max(k, l)))

            for i, j in spring_indices:
                self.add_spring(i, j, spring_ke, spring_kd, control=0.0)

    def add_particle_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        mass: float,
        jitter: float,
        radius_mean: float = default_particle_radius,
        radius_std: float = 0.0,
    ):
        rng = np.random.default_rng()
        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v = wp.vec3(x * cell_x, y * cell_y, z * cell_z)
                    m = mass

                    p = wp.quat_rotate(rot, v) + pos + wp.vec3(rng.random(3) * jitter)

                    if radius_std > 0.0:
                        r = radius_mean + np.random.randn() * radius_std
                    else:
                        r = radius_mean
                    self.add_particle(p, vel, m, r)

    def add_soft_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
        tri_ke: float = default_tri_ke,
        tri_ka: float = default_tri_ka,
        tri_kd: float = default_tri_kd,
        tri_drag: float = default_tri_drag,
        tri_lift: float = default_tri_lift,
    ):
        """Helper to create a rectangular tetrahedral FEM grid

        Creates a regular grid of FEM tetrahedra and surface triangles. Useful for example
        to create beams and sheets. Each hexahedral cell is decomposed into 5
        tetrahedral elements.

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            dim_z: The number of rectangular cells along the z-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            cell_z: The width of each cell in the z-direction
            density: The density of each particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic
        """

        start_vertex = len(self.particle_q)

        mass = cell_x * cell_y * cell_z * density

        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):
                    v = wp.vec3(x * cell_x, y * cell_y, z * cell_z)
                    m = mass

                    if fix_left and x == 0:
                        m = 0.0

                    if fix_right and x == dim_x:
                        m = 0.0

                    if fix_top and y == dim_y:
                        m = 0.0

                    if fix_bottom and y == 0:
                        m = 0.0

                    p = wp.quat_rotate(rot, v) + pos

                    self.add_particle(p, vel, m)

        # dict of open faces
        faces = {}

        def add_face(i: int, j: int, k: int):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        def add_tet(i: int, j: int, k: int, l: int):
            self.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)

            add_face(i, k, j)
            add_face(j, k, l)
            add_face(i, j, l)
            add_face(i, l, k)

        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v0 = grid_index(x, y, z) + start_vertex
                    v1 = grid_index(x + 1, y, z) + start_vertex
                    v2 = grid_index(x + 1, y, z + 1) + start_vertex
                    v3 = grid_index(x, y, z + 1) + start_vertex
                    v4 = grid_index(x, y + 1, z) + start_vertex
                    v5 = grid_index(x + 1, y + 1, z) + start_vertex
                    v6 = grid_index(x + 1, y + 1, z + 1) + start_vertex
                    v7 = grid_index(x, y + 1, z + 1) + start_vertex

                    if (x & 1) ^ (y & 1) ^ (z & 1):
                        add_tet(v0, v1, v4, v3)
                        add_tet(v2, v3, v6, v1)
                        add_tet(v5, v4, v1, v6)
                        add_tet(v7, v6, v3, v4)
                        add_tet(v4, v1, v6, v3)

                    else:
                        add_tet(v1, v2, v5, v0)
                        add_tet(v3, v0, v7, v2)
                        add_tet(v4, v7, v0, v5)
                        add_tet(v6, v5, v2, v7)
                        add_tet(v5, v2, v7, v0)

        # add triangles
        for _k, v in faces.items():
            self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

    def add_soft_mesh(
        self,
        pos: Vec3,
        rot: Quat,
        scale: float,
        vel: Vec3,
        vertices: List[Vec3],
        indices: List[int],
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        tri_ke: float = default_tri_ke,
        tri_ka: float = default_tri_ka,
        tri_kd: float = default_tri_kd,
        tri_drag: float = default_tri_drag,
        tri_lift: float = default_tri_lift,
    ):
        """Helper to create a tetrahedral model from an input tetrahedral mesh

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            vertices: A list of vertex positions, array of 3D points
            indices: A list of tetrahedron indices, 4 entries per-element, flattened array
            density: The density per-area of the mesh
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
        """
        num_tets = int(len(indices) / 4)

        start_vertex = len(self.particle_q)

        # dict of open faces
        faces = {}

        def add_face(i, j, k):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        pos = wp.vec3(pos[0], pos[1], pos[2])
        # add particles
        for v in vertices:
            v = wp.vec3(v[0], v[1], v[2])
            p = wp.quat_rotate(rot, v * scale) + pos

            self.add_particle(p, vel, 0.0)

        # add tetrahedra
        for t in range(num_tets):
            v0 = start_vertex + indices[t * 4 + 0]
            v1 = start_vertex + indices[t * 4 + 1]
            v2 = start_vertex + indices[t * 4 + 2]
            v3 = start_vertex + indices[t * 4 + 3]

            volume = self.add_tetrahedron(v0, v1, v2, v3, k_mu, k_lambda, k_damp)

            # distribute volume fraction to particles
            if volume > 0.0:
                self.particle_mass[v0] += density * volume / 4.0
                self.particle_mass[v1] += density * volume / 4.0
                self.particle_mass[v2] += density * volume / 4.0
                self.particle_mass[v3] += density * volume / 4.0

                # build open faces
                add_face(v0, v2, v1)
                add_face(v1, v2, v3)
                add_face(v0, v1, v3)
                add_face(v0, v3, v2)

        # add triangles
        for _k, v in faces.items():
            try:
                self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
            except np.linalg.LinAlgError:
                continue

    # incrementally updates rigid body mass with additional mass and inertia expressed at a local to the body
    def _update_body_mass(self, i, m, I, p, q):
        if i == -1:
            return

        # find new COM
        new_mass = self.body_mass[i] + m

        if new_mass == 0.0:  # no mass
            return

        new_com = (self.body_com[i] * self.body_mass[i] + p * m) / new_mass

        # shift inertia to new COM
        com_offset = new_com - self.body_com[i]
        shape_offset = new_com - p

        new_inertia = transform_inertia(
            self.body_mass[i], self.body_inertia[i], com_offset, wp.quat_identity()
        ) + transform_inertia(m, I, shape_offset, q)

        self.body_mass[i] = new_mass
        self.body_inertia[i] = new_inertia
        self.body_com[i] = new_com

        if new_mass > 0.0:
            self.body_inv_mass[i] = 1.0 / new_mass
        else:
            self.body_inv_mass[i] = 0.0

        if any(x for x in new_inertia):
            self.body_inv_inertia[i] = wp.inverse(new_inertia)
        else:
            self.body_inv_inertia[i] = new_inertia

    def set_ground_plane(
        self,
        normal=None,
        offset=0.0,
        ke: float = default_shape_ke,
        kd: float = default_shape_kd,
        kf: float = default_shape_kf,
        mu: float = default_shape_mu,
        restitution: float = default_shape_restitution,
    ):
        """
        Creates a ground plane for the world. If the normal is not specified,
        the up_vector of the ModelBuilder is used.
        """
        if normal is None:
            normal = self.up_vector
        self._ground_params = {
            "plane": (*normal, offset),
            "width": 0.0,
            "length": 0.0,
            "ke": ke,
            "kd": kd,
            "kf": kf,
            "mu": mu,
            "restitution": restitution,
        }

    def _create_ground_plane(self):
        ground_id = self.add_shape_plane(**self._ground_params)
        self._ground_created = True
        # disable ground collisions as they will be treated separately
        for i in range(self.shape_count - 1):
            self.shape_collision_filter_pairs.add((i, ground_id))

    def finalize(self, device=None, requires_grad=False) -> Model:
        """Convert this builder object to a concrete model for simulation.

        After building simulation elements this method should be called to transfer
        all data to device memory ready for simulation.

        Args:
            device: The simulation device to use, e.g.: 'cpu', 'cuda'
            requires_grad: Whether to enable gradient computation for the model

        Returns:

            A model object.
        """

        # ensure the env count is set correctly
        self.num_envs = max(1, self.num_envs)

        # add ground plane if not already created
        if not self._ground_created:
            self._create_ground_plane()

        # construct particle inv masses
        ms = np.array(self.particle_mass, dtype=np.float32)
        # static particles (with zero mass) have zero inverse mass
        particle_inv_mass = np.divide(1.0, ms, out=np.zeros_like(ms), where=ms != 0.0)

        with wp.ScopedDevice(device):
            # -------------------------------------
            # construct Model (non-time varying) data

            m = Model(device)
            m.requires_grad = requires_grad

            m.ground_plane_params = self._ground_params["plane"]

            m.num_envs = self.num_envs

            # ---------------------
            # particles

            # state (initial)
            m.particle_q = wp.array(self.particle_q, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_qd = wp.array(self.particle_qd, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_mass = wp.array(self.particle_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_inv_mass = wp.array(particle_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_radius = wp.array(self.particle_radius, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_flags = wp.array([flag_to_int(f) for f in self.particle_flags], dtype=wp.uint32)
            m.particle_max_radius = np.max(self.particle_radius) if len(self.particle_radius) > 0 else 0.0
            m.particle_max_velocity = self.particle_max_velocity

            # hash-grid for particle interactions
            m.particle_grid = wp.HashGrid(128, 128, 128)

            # ---------------------
            # collision geometry

            m.shape_transform = wp.array(self.shape_transform, dtype=wp.transform, requires_grad=requires_grad)
            m.shape_body = wp.array(self.shape_body, dtype=wp.int32)
            m.shape_visible = wp.array(self.shape_visible, dtype=wp.bool)
            m.body_shapes = self.body_shapes

            # build list of ids for geometry sources (meshes, sdfs)
            geo_sources = []
            finalized_meshes = {}  # do not duplicate meshes
            for geo in self.shape_geo_src:
                geo_hash = hash(geo)  # avoid repeated hash computations
                if geo:
                    if geo_hash not in finalized_meshes:
                        finalized_meshes[geo_hash] = geo.finalize(device=device)
                    geo_sources.append(finalized_meshes[geo_hash])
                else:
                    # add null pointer
                    geo_sources.append(0)

            m.shape_geo.type = wp.array(self.shape_geo_type, dtype=wp.int32)
            m.shape_geo.source = wp.array(geo_sources, dtype=wp.uint64)
            m.shape_geo.scale = wp.array(self.shape_geo_scale, dtype=wp.vec3, requires_grad=requires_grad)
            m.shape_geo.is_solid = wp.array(self.shape_geo_is_solid, dtype=wp.uint8)
            m.shape_geo.thickness = wp.array(self.shape_geo_thickness, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_geo_src = self.shape_geo_src  # used for rendering
            # store refs to geometry
            m.geo_meshes = self.geo_meshes
            m.geo_sdfs = self.geo_sdfs

            m.shape_materials.ke = wp.array(self.shape_material_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.kd = wp.array(self.shape_material_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.kf = wp.array(self.shape_material_kf, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.ka = wp.array(self.shape_material_ka, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.mu = wp.array(self.shape_material_mu, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.restitution = wp.array(
                self.shape_material_restitution, dtype=wp.float32, requires_grad=requires_grad
            )

            m.shape_collision_filter_pairs = self.shape_collision_filter_pairs
            m.shape_collision_group = self.shape_collision_group
            m.shape_collision_group_map = self.shape_collision_group_map
            m.shape_collision_radius = wp.array(
                self.shape_collision_radius, dtype=wp.float32, requires_grad=requires_grad
            )
            m.shape_ground_collision = self.shape_ground_collision
            m.shape_shape_collision = self.shape_shape_collision

            # ---------------------
            # springs

            m.spring_indices = wp.array(self.spring_indices, dtype=wp.int32)
            m.spring_rest_length = wp.array(self.spring_rest_length, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_stiffness = wp.array(self.spring_stiffness, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_damping = wp.array(self.spring_damping, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_control = wp.array(self.spring_control, dtype=wp.float32, requires_grad=requires_grad)

            # ---------------------
            # triangles

            m.tri_indices = wp.array(self.tri_indices, dtype=wp.int32)
            m.tri_poses = wp.array(self.tri_poses, dtype=wp.mat22, requires_grad=requires_grad)
            m.tri_activations = wp.array(self.tri_activations, dtype=wp.float32, requires_grad=requires_grad)
            m.tri_materials = wp.array(self.tri_materials, dtype=wp.float32, requires_grad=requires_grad)

            # ---------------------
            # edges

            m.edge_indices = wp.array(self.edge_indices, dtype=wp.int32)
            m.edge_rest_angle = wp.array(self.edge_rest_angle, dtype=wp.float32, requires_grad=requires_grad)
            m.edge_bending_properties = wp.array(
                self.edge_bending_properties, dtype=wp.float32, requires_grad=requires_grad
            )

            # ---------------------
            # tetrahedra

            m.tet_indices = wp.array(self.tet_indices, dtype=wp.int32)
            m.tet_poses = wp.array(self.tet_poses, dtype=wp.mat33, requires_grad=requires_grad)
            m.tet_activations = wp.array(self.tet_activations, dtype=wp.float32, requires_grad=requires_grad)
            m.tet_materials = wp.array(self.tet_materials, dtype=wp.float32, requires_grad=requires_grad)

            # -----------------------
            # muscles

            # close the muscle waypoint indices
            muscle_start = copy.copy(self.muscle_start)
            muscle_start.append(len(self.muscle_bodies))

            m.muscle_start = wp.array(muscle_start, dtype=wp.int32)
            m.muscle_params = wp.array(self.muscle_params, dtype=wp.float32, requires_grad=requires_grad)
            m.muscle_bodies = wp.array(self.muscle_bodies, dtype=wp.int32)
            m.muscle_points = wp.array(self.muscle_points, dtype=wp.vec3, requires_grad=requires_grad)
            m.muscle_activations = wp.array(self.muscle_activations, dtype=wp.float32, requires_grad=requires_grad)

            # --------------------------------------
            # rigid bodies

            m.body_q = wp.array(self.body_q, dtype=wp.transform, requires_grad=requires_grad)
            m.body_qd = wp.array(self.body_qd, dtype=wp.spatial_vector, requires_grad=requires_grad)
            m.body_inertia = wp.array(self.body_inertia, dtype=wp.mat33, requires_grad=requires_grad)
            m.body_inv_inertia = wp.array(self.body_inv_inertia, dtype=wp.mat33, requires_grad=requires_grad)
            m.body_mass = wp.array(self.body_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.body_inv_mass = wp.array(self.body_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.body_com = wp.array(self.body_com, dtype=wp.vec3, requires_grad=requires_grad)
            m.body_name = self.body_name

            # joints
            m.joint_type = wp.array(self.joint_type, dtype=wp.int32)
            m.joint_parent = wp.array(self.joint_parent, dtype=wp.int32)
            m.joint_child = wp.array(self.joint_child, dtype=wp.int32)
            m.joint_X_p = wp.array(self.joint_X_p, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_X_c = wp.array(self.joint_X_c, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_axis_start = wp.array(self.joint_axis_start, dtype=wp.int32)
            m.joint_axis_dim = wp.array(np.array(self.joint_axis_dim), dtype=wp.int32, ndim=2)
            m.joint_axis = wp.array(self.joint_axis, dtype=wp.vec3, requires_grad=requires_grad)
            m.joint_q = wp.array(self.joint_q, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_qd = wp.array(self.joint_qd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_name = self.joint_name

            # dynamics properties
            m.joint_armature = wp.array(self.joint_armature, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_ke = wp.array(self.joint_target_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_kd = wp.array(self.joint_target_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_axis_mode = wp.array(self.joint_axis_mode, dtype=wp.int32)
            m.joint_act = wp.array(self.joint_act, dtype=wp.float32, requires_grad=requires_grad)

            m.joint_limit_lower = wp.array(self.joint_limit_lower, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_upper = wp.array(self.joint_limit_upper, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_ke = wp.array(self.joint_limit_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_kd = wp.array(self.joint_limit_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_linear_compliance = wp.array(
                self.joint_linear_compliance, dtype=wp.float32, requires_grad=requires_grad
            )
            m.joint_angular_compliance = wp.array(
                self.joint_angular_compliance, dtype=wp.float32, requires_grad=requires_grad
            )
            m.joint_enabled = wp.array(self.joint_enabled, dtype=wp.int32)

            # 'close' the start index arrays with a sentinel value
            joint_q_start = copy.copy(self.joint_q_start)
            joint_q_start.append(self.joint_coord_count)
            joint_qd_start = copy.copy(self.joint_qd_start)
            joint_qd_start.append(self.joint_dof_count)
            articulation_start = copy.copy(self.articulation_start)
            articulation_start.append(self.joint_count)

            m.joint_q_start = wp.array(joint_q_start, dtype=wp.int32)
            m.joint_qd_start = wp.array(joint_qd_start, dtype=wp.int32)
            m.articulation_start = wp.array(articulation_start, dtype=wp.int32)

            # counts
            m.joint_count = self.joint_count
            m.joint_axis_count = self.joint_axis_count
            m.joint_dof_count = self.joint_dof_count
            m.joint_coord_count = self.joint_coord_count
            m.particle_count = len(self.particle_q)
            m.body_count = len(self.body_q)
            m.shape_count = len(self.shape_geo_type)
            m.tri_count = len(self.tri_poses)
            m.tet_count = len(self.tet_poses)
            m.edge_count = len(self.edge_rest_angle)
            m.spring_count = len(self.spring_rest_length)
            m.muscle_count = len(self.muscle_start)
            m.articulation_count = len(self.articulation_start)

            # contacts
            if m.particle_count:
                m.allocate_soft_contacts(self.soft_contact_max, requires_grad=requires_grad)
            m.find_shape_contact_pairs()
            if self.num_rigid_contacts_per_env is None:
                contact_count, limited_contact_count = m.count_contact_points()
            else:
                contact_count = limited_contact_count = self.num_rigid_contacts_per_env * self.num_envs
            if contact_count:
                if wp.config.verbose:
                    print(f"Allocating {contact_count} rigid contacts.")
                m.allocate_rigid_contacts(
                    count=contact_count, limited_contact_count=limited_contact_count, requires_grad=requires_grad
                )
            m.rigid_mesh_contact_max = self.rigid_mesh_contact_max
            m.rigid_contact_margin = self.rigid_contact_margin
            m.rigid_contact_torsional_friction = self.rigid_contact_torsional_friction
            m.rigid_contact_rolling_friction = self.rigid_contact_rolling_friction

            # enable ground plane
            m.ground_plane = wp.array(self._ground_params["plane"], dtype=wp.float32, requires_grad=requires_grad)
            m.gravity = np.array(self.up_vector) * self.gravity

            m.enable_tri_collisions = False

            return m
