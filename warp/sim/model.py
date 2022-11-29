# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""A module for building simulation models and state.
"""

import warp as wp

import math
import numpy as np
import copy

from typing import Tuple
from typing import List

from warp.types import Volume
Vec3 = List[float]
Vec4 = List[float]
Quat = List[float]
Mat33 = List[float]
Transform = Tuple[Vec3, Quat]
from typing import Optional

from warp.sim.collide import count_contact_points


# shape geometry types
GEO_SPHERE = wp.constant(0)
GEO_BOX = wp.constant(1)
GEO_CAPSULE = wp.constant(2)
GEO_MESH = wp.constant(3)
GEO_SDF = wp.constant(4)
GEO_PLANE = wp.constant(5)
GEO_NONE = wp.constant(6)

# body joint types
JOINT_PRISMATIC = wp.constant(0)
JOINT_REVOLUTE = wp.constant(1)
JOINT_BALL = wp.constant(2)
JOINT_FIXED = wp.constant(3)
JOINT_FREE = wp.constant(4)
JOINT_COMPOUND = wp.constant(5)
JOINT_UNIVERSAL = wp.constant(6)


# Material properties pertaining to rigid shape contact dynamics
@wp.struct
class ShapeContactMaterial:
    ke: wp.array(dtype=float)  # The contact elastic stiffness
    kd: wp.array(dtype=float)  # The contact damping stiffness
    kf: wp.array(dtype=float)  # The contact friction stiffness
    mu: wp.array(dtype=float)  # The coefficient of friction
    restitution: wp.array(dtype=float)  # The coefficient of restitution


# Calculates the mass and inertia of a body given mesh data.
@wp.kernel
def compute_mass_inertia(
    #inputs
    com: wp.vec3,
    alpha: float,
    weight: float,
    indices: wp.array(dtype=int, ndim=1),
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    quads: wp.array(dtype=wp.vec3, ndim=2),
    #outputs
    mass: wp.array(dtype=float, ndim=1),
    inertia: wp.array(dtype=wp.mat33, ndim=1)):

    i = wp.tid()

    p = vertices[indices[i * 3 + 0]]
    q = vertices[indices[i * 3 + 1]]
    r = vertices[indices[i * 3 + 2]]

    mid = (com + p + q + r) / 4.

    pcom = p - com
    qcom = q - com
    rcom = r - com

    Dm = wp.mat33(pcom[0], qcom[0], rcom[0],
                  pcom[1], qcom[1], rcom[1],
                  pcom[2], qcom[2], rcom[2])

    volume = wp.determinant(Dm) / 6.0

    # quadrature points lie on the line between the
    # centroid and each vertex of the tetrahedron
    quads[i, 0] = alpha * (p - mid) + mid
    quads[i, 1] = alpha * (q - mid) + mid
    quads[i, 2] = alpha * (r - mid) + mid
    quads[i, 3] = alpha * (com - mid) + mid
    
    for j in range(4):
        # displacement of quadrature point from COM
        d = quads[i,j] - com

        # accumulate mass
        wp.atomic_add(mass, 0, weight * volume)

        # accumulate inertia
        identity = wp.mat33(1., 0., 0., 0., 1., 0., 0., 0., 1.)
        I = weight * volume * (wp.dot(d, d) * identity - wp.outer(d, d))
        wp.atomic_add(inertia, 0, I)


class Mesh:
    """Describes a triangle collision mesh for simulation

    Attributes:

        vertices (List[Vec3]): Mesh vertices
        indices (List[int]): Mesh indices
        I (Mat33): Inertia tensor of the mesh assuming density of 1.0 (around the center of mass)
        mass (float): The total mass of the body assuming density of 1.0
        com (Vec3): The center of mass of the body
    """

    def __init__(self, vertices: List[Vec3], indices: List[int], compute_inertia=True):
        """Construct a Mesh object from a triangle mesh

        The mesh center of mass and inertia tensor will automatically be 
        calculated using a density of 1.0. This computation is only valid
        if the mesh is closed (two-manifold).

        Args:
            vertices: List of vertices in the mesh
            indices: List of triangle indices, 3 per-element       
        """

        self.vertices = vertices
        self.indices = indices

        if compute_inertia:
            # compute com and inertia (using density=1.0)
            com = np.mean(vertices, 0)
            com_warp = wp.vec3(com[0], com[1], com[2])

            num_tris = int(len(indices) / 3)

            # compute signed inertia for each tetrahedron
            # formed with the interior point, using an order-2
            # quadrature: https://www.sciencedirect.com/science/article/pii/S0377042712001604#br000040

            weight = 0.25
            alpha = math.sqrt(5.0) / 5.0

            # Allocating for mass and inertia.
            I_warp = wp.zeros(1, dtype=wp.mat33)
            mass_warp = wp.zeros(1, dtype=float)

            # Quadrature points
            quads_warp = wp.zeros(shape=(num_tris, 4), dtype=wp.vec3)

            # Launch warp kernel for calculating mass and inertia of body given mesh data.
            wp.launch(kernel=compute_mass_inertia,
                      dim=num_tris,
                      inputs=[
                          com_warp,
                          alpha,
                          weight,
                          wp.array(indices, dtype=int),
                          wp.array(vertices, dtype=wp.vec3),
                          quads_warp
                          ],
                      outputs=[
                          mass_warp,
                          I_warp])

            # Extract mass and inertia and save to class attributes.
            mass = mass_warp.numpy()[0]
            I = I_warp.numpy()[0]

            self.I = I
            self.mass = mass
            self.com = com

        else:
            
            self.I = np.eye(3, dtype=np.float32)
            self.mass = 1.0
            self.com = np.array((0.0, 0.0, 0.0))



    # construct simulation ready buffers from points
    def finalize(self, device=None):

        with wp.ScopedDevice(device):

            pos = wp.array(self.vertices, dtype=wp.vec3)
            vel = wp.zeros_like(pos)
            indices = wp.array(self.indices, dtype=wp.int32)

            self.mesh = wp.Mesh(points=pos, velocities=vel, indices=indices)
            return self.mesh.id

    def __hash__(self):
        return hash((tuple(np.array(self.vertices).flatten()), tuple(np.array(self.indices).flatten())))


class State:
    """The State object holds all *time-varying* data for a model.
    
    Time-varying data includes particle positions, velocities, rigid body states, and
    anything that is output from the integrator as derived data, e.g.: forces. 
    
    The exact attributes depend on the contents of the model. State objects should
    generally be created using the :func:`Model.state()` function.

    Attributes:

        particle_q (wp.array): Tensor of particle positions
        particle_qd (wp.array): Tensor of particle velocities

        body_q (wp.array): Tensor of body coordinates
        body_qd (wp.array): Tensor of body velocities

    """

    def __init__(self):
        
        self.particle_count = 0
        self.body_count = 0

    def clear_forces(self):

        if self.particle_count:
            self.particle_f.zero_()

        if self.body_count:
            self.body_f.zero_()

    def flatten(self):
        """Returns a list of Tensors stored by the state

        This function is intended to be used internal-only but can be used to obtain
        a set of all tensors owned by the state.
        """

        tensors = []

        # build a list of all tensor attributes
        for attr, value in self.__dict__.items():
            if (wp.is_tensor(value)):
                tensors.append(value)

        return tensors


class Model:
    """Holds the definition of the simulation model

    This class holds the non-time varying description of the system, i.e.:
    all geometry, constraints, and parameters used to describe the simulation.

    Attributes:
        particle_q (wp.array): Particle positions, shape [particle_count, 3], float
        particle_qd (wp.array): Particle velocities, shape [particle_count, 3], float
        particle_mass (wp.array): Particle mass, shape [particle_count], float
        particle_inv_mass (wp.array): Particle inverse mass, shape [particle_count], float

        shape_transform (wp.array): Rigid shape transforms, shape [shape_count, 7], float
        shape_body (wp.array): Rigid shape body index, shape [shape_count], int
        shape_geo_type (wp.array): Rigid shape geometry type, [shape_count], int
        shape_geo_src (wp.array): Rigid shape geometry source, shape [shape_count], int
        shape_geo_scale (wp.array): Rigid shape geometry scale, shape [shape_count, 3], float
        shape_materials (wp.array): Rigid shape contact materials, shape [shape_count, 4], float

        spring_indices (wp.array): Particle spring indices, shape [spring_count*2], int
        spring_rest_length (wp.array): Particle spring rest length, shape [spring_count], float
        spring_stiffness (wp.array): Particle spring stiffness, shape [spring_count], float
        spring_damping (wp.array): Particle spring damping, shape [spring_count], float
        spring_control (wp.array): Particle spring activation, shape [spring_count], float

        tri_indices (wp.array): Triangle element indices, shape [tri_count*3], int
        tri_poses (wp.array): Triangle element rest pose, shape [tri_count, 2, 2], float
        tri_activations (wp.array): Triangle element activations, shape [tri_count], float

        edge_indices (wp.array): Bending edge indices, shape [edge_count*2], int
        edge_rest_angle (wp.array): Bending edge rest angle, shape [edge_count], float

        tet_indices (wp.array): Tetrahedral element indices, shape [tet_count*4], int
        tet_poses (wp.array): Tetrahedral rest poses, shape [tet_count, 3, 3], float
        tet_activations (wp.array): Tetrahedral volumetric activations, shape [tet_count], float
        tet_materials (wp.array): Tetrahedral elastic parameters in form :math:`k_{mu}, k_{lambda}, k_{damp}`, shape [tet_count, 3]
        
        body_com (wp.array): Rigid body center of mass (in local frame), shape [body_count, 7], float
        body_inertia (wp.array): Rigid body inertia tensor (relative to COM), shape [body_count, 3, 3], float

        TODO: update this
        joint_type (wp.array): Joint type, shape [joint_count], int
        joint_parent (wp.array): Joint parent, shape [joint_count], int
        joint_X_pj (wp.array): Joint transform in parent frame, shape [joint_count, 7], float
        joint_X_cm (wp.array): Joint mass frame in child frame, shape [joint_count, 7], float
        joint_axis (wp.array): Joint axis in child frame, shape [joint_count, 3], float
        joint_armature (wp.array): Armature for each joint, shape [joint_count], float
        joint_target_ke (wp.array): Joint stiffness, shape [joint_count], float
        joint_target_kd (wp.array): Joint damping, shape [joint_count], float
        joint_target (wp.array): Joint target, shape [joint_count], float
        joint_linear_compliance (wp.array): Joint linear compliance, shape [joint_count], float
        joint_angular_compliance (wp.array): Joint linear compliance, shape [joint_count], float
        joint_limit_lower (wp.array): Joint lower position limits, shape [joint_count], float
        joint_limit_upper (wp.array): Joint upper position limits, shape [joint_count], float
        joint_twist_lower (wp.array): Joint lower twist limit, shape [joint_count], float
        joint_twist_upper (wp.array): Joint upper twist limit, shape [joint_count], float

        particle_count (int): Total number of particles in the system
        body_count (int): Total number of bodies in the system
        shape_count (int): Total number of shapes in the system
        tri_count (int): Total number of triangles in the system
        tet_count (int): Total number of tetrahedra in the system
        edge_count (int): Total number of edges in the system
        spring_count (int): Total number of springs in the system
        contact_count (int): Total number of contacts in the system
        
    Note:
        It is strongly recommended to use the ModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if 
        desired.
    """

    def __init__(self, device=None):

        self.requires_grad = False

        self.particle_q = None
        self.particle_qd = None
        self.particle_mass = None
        self.particle_inv_mass = None

        self.shape_transform = None
        self.shape_body = None
        self.shape_geo_type = None
        self.shape_geo_src = None
        self.shape_geo_scale = None
        self.shape_geo_id = None
        self.shape_materials = None
        self.shape_contact_thickness = None

        self.spring_indices = None
        self.spring_rest_length = None
        self.spring_stiffness = None
        self.spring_damping = None
        self.spring_control = None

        self.tri_indices = None
        self.tri_poses = None
        self.tri_activations = None
        self.tri_materials = None

        self.edge_indices = None
        self.edge_rest_angle = None
        self.edge_bending_properties = None

        self.tet_indices = None
        self.tet_poses = None
        self.tet_activations = None
        self.tet_materials = None
        
        self.body_com = None
        self.body_inertia = None

        self.shape_collision_group = None
        self.shape_collision_group_map = None
        self.shape_collision_filter_pairs = None
        self.shape_collision_radius = None
        self.shape_ground_collision = None
        self.shape_contact_pairs = None
        self.shape_ground_contact_pairs = None

        self.joint_type = None
        self.joint_parent = None
        self.joint_child = None
        self.joint_X_p = None
        self.joint_X_c = None
        self.joint_axis = None
        self.joint_armature = None
        self.joint_target_ke = None
        self.joint_target_kd = None
        self.joint_target = None
        self.joint_linear_compliance = None
        self.joint_angular_compliance = None
        self.joint_limit_lower = None
        self.joint_limit_upper = None
        self.joint_twist_lower = None
        self.joint_twist_upper = None

        # todo: per-joint values?
        self.joint_attach_ke = 1.e+3
        self.joint_attach_kd = 1.e+2

        self.particle_count = 0
        self.body_count = 0
        self.shape_count = 0
        self.tri_count = 0
        self.tet_count = 0
        self.edge_count = 0
        self.spring_count = 0
        self.ground_contact_dim = 0

        self.soft_contact_distance = 0.1
        self.soft_contact_margin = 0.2
        self.soft_contact_ke = 1.e+3
        self.soft_contact_kd = 10.0
        self.soft_contact_kf = 1.e+3
        self.soft_contact_mu = 0.5

        self.edge_bending_properties = None

        self.particle_radius = 0.0
        self.particle_ke = 1.e+3
        self.particle_kd = 1.e+2
        self.particle_kf = 1.e+2
        self.particle_mu = 0.5
        self.particle_cohesion = 0.0
        self.particle_adhesion = 0.0
        self.particle_grid = None

        self.device = wp.get_device(device)

        # toggles ground contact for all shapes
        self.ground = True

    def state(self, requires_grad=False) -> State:
        """Returns a state object for the model

        The returned state will be initialized with the initial configuration given in
        the model description.
        """

        s = State()

        s.particle_count = self.particle_count
        s.body_count = self.body_count

        #--------------------------------
        # dynamic state (input, output)

        s.particle_q = None
        s.particle_qd = None
        s.particle_f = None

        s.body_q = None
        s.body_q_prev = None
        s.body_qd = None
        s.body_qd_prev = None
        s.body_f = None
        s.body_deltas = None

        # particles
        if (self.particle_count):
            s.particle_q = wp.clone(self.particle_q)
            s.particle_qd = wp.clone(self.particle_qd)
            s.particle_f = wp.zeros_like(self.particle_qd)

            s.particle_q.requires_grad = requires_grad
            s.particle_qd.requires_grad = requires_grad
            s.particle_f.requires_grad = requires_grad

        # articulations
        if (self.body_count):
            s.body_q = wp.clone(self.body_q)
            s.body_q_prev = wp.clone(self.body_q)
            s.body_qd = wp.clone(self.body_qd)
            s.body_qd_prev = wp.clone(self.body_qd)
            s.body_f = wp.zeros_like(self.body_qd)

            s.body_deltas = wp.zeros(
                self.body_count,
                dtype=wp.spatial_vector,
                device=s.body_q.device,
                requires_grad=requires_grad)

            s.body_q.requires_grad = requires_grad
            s.body_q_prev.requires_grad = requires_grad
            s.body_qd.requires_grad = requires_grad
            s.body_qd_prev.requires_grad = requires_grad
            s.body_f.requires_grad = requires_grad

        return s

    def allocate_soft_contacts(self, count=None, requires_grad=False):        
        if count is not None:
            self.soft_contact_max = count
        self.soft_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.soft_contact_particle = wp.zeros(self.soft_contact_max, dtype=int, device=self.device)
        self.soft_contact_body = wp.zeros(self.soft_contact_max, dtype=int, device=self.device)
        self.soft_contact_body_pos = wp.zeros(self.soft_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        self.soft_contact_body_vel = wp.zeros(self.soft_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        self.soft_contact_normal = wp.zeros(self.soft_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)

    def find_shape_contact_pairs(self):
        # find potential contact pairs based on collision groups and collision mask (pairwise filtering)
        import itertools, copy
        filters = copy.copy(self.shape_collision_filter_pairs)
        for a, b in self.shape_collision_filter_pairs:
            filters.add((b, a))
        contact_pairs = []
        # iterate over collision groups (islands)
        for group, shapes in self.shape_collision_group_map.items():
            for shape_a, shape_b in itertools.product(shapes, shapes):
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
        ground_id = self.shape_count-1
        for i in range(ground_id):
            if self.shape_ground_collision[i]:
                ground_contact_pairs.append((i, ground_id))
        self.shape_ground_contact_pairs = wp.array(np.array(ground_contact_pairs), dtype=wp.int32, device=self.device)
        self.shape_ground_contact_pair_count = len(ground_contact_pairs)

    def count_contact_points(self):
        """
        Counts the maximum number of contact points that need to be allocated.
        """
        # calculate the potential number of shape pair contact points
        contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=count_contact_points,
            dim=self.shape_contact_pair_count,
            inputs=[
                self.shape_contact_pairs,
                self.shape_geo_type,
                self.shape_geo_scale,
                self.shape_geo_id,
            ],
            outputs=[
                contact_count
            ],
            device=self.device,
            record_tape=False)
        wp.launch(
            kernel=count_contact_points,
            dim=self.shape_ground_contact_pair_count,
            inputs=[
                self.shape_ground_contact_pairs,
                self.shape_geo_type,
                self.shape_geo_scale,
                self.shape_geo_id,
            ],
            outputs=[
                contact_count
            ],
            device=self.device,
            record_tape=False)
        count = contact_count.numpy()[0]
        return int(count)

    def allocate_rigid_contacts(self, count=None, requires_grad=False):
        if count is not None:
            self.rigid_contact_max = count
        # serves as counter of the number of active contact points
        self.rigid_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        # contact point ID within the (shape_a, shape_b) contact pair
        self.rigid_contact_point_id = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=self.device)
        # ID of first rigid body
        self.rigid_contact_body0 = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=self.device)
        # ID of second rigid body
        self.rigid_contact_body1 = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=self.device)
        # position of contact point in body 0's frame before the integration step
        self.rigid_contact_point0 = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        # position of contact point in body 1's frame before the integration step
        self.rigid_contact_point1 = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        # moment arm before the integration step resulting from thickness displacement added to contact point 0 in body 0's frame (used in XPBD contact friction handling)
        self.rigid_contact_offset0 = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        # moment arm before the integration step resulting from thickness displacement added to contact point 1 in body 1's frame (used in XPBD contact friction handling)
        self.rigid_contact_offset1 = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        # contact normal in world frame
        self.rigid_contact_normal = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        # combined thickness of both shapes
        self.rigid_contact_thickness = wp.zeros(self.rigid_contact_max, dtype=wp.float32, device=self.device, requires_grad=requires_grad)
        # ID of the first shape in the contact pair
        self.rigid_contact_shape0 = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=self.device)
        # ID of the second shape in the contact pair
        self.rigid_contact_shape1 = wp.zeros(self.rigid_contact_max, dtype=wp.int32, device=self.device)

        # temporary variables used during the XPBD solver iterations:
        # world space position of contact point resulting from applying current body 0 transform to its point0
        self.rigid_active_contact_point0 = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        # world space position of contact point resulting from applying current body 1 transform to its point1
        self.rigid_active_contact_point1 = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        # current contact distance (negative penetration depth)
        self.rigid_active_contact_distance = wp.zeros(self.rigid_contact_max, dtype=wp.float32, device=self.device, requires_grad=requires_grad)
        # contact distance before the solver iterations
        self.rigid_active_contact_distance_prev = wp.zeros(self.rigid_contact_max, dtype=wp.float32, device=self.device, requires_grad=requires_grad)
        # world space position of point0 before the solver iterations
        self.rigid_active_contact_point0_prev = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        # world space position of point1 before the solver iterations
        self.rigid_active_contact_point1_prev = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        # number of contact constraints per rigid body (used for scaling the constraint contributions, a basic version of mass splitting)
        self.rigid_contact_inv_weight = wp.zeros(len(self.body_q), dtype=wp.float32, device=self.device, requires_grad=requires_grad)
        # number of contact constraints before the solver iterations
        self.rigid_contact_inv_weight_prev = wp.zeros(len(self.body_q), dtype=wp.float32, device=self.device, requires_grad=requires_grad)

    def flatten(self):
        """Returns a list of Tensors stored by the model

        This function is intended to be used internal-only but can be used to obtain
        a set of all tensors owned by the model.
        """

        tensors = []

        # build a list of all tensor attributes
        for attr, value in self.__dict__.items():
            if (wp.is_tensor(value)):
                tensors.append(value)

        return tensors

    def collide(self, state: State):
        import warnings
        warnings.warn("Model.collide() is not needed anymore and will be removed in a future Warp version.", DeprecationWarning, stacklevel=2)


class ModelBuilder:
    """A helper class for building simulation models at runtime.

    Use the ModelBuilder to construct a simulation scene. The ModelBuilder
    and builds the scene representation using standard Python data structures (lists), 
    this means it is not differentiable. Once :func:`finalize()` 
    has been called the ModelBuilder transfers all data to Warp tensors and returns 
    an object that may be used for simulation.

    Example:

        >>> import warp as wp
        >>> import warp.sim
        >>>
        >>> builder = wp.sim.ModelBuilder()
        >>>
        >>> # anchor point (zero mass)
        >>> builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        >>>
        >>> # build chain
        >>> for i in range(1,10):
        >>>     builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
        >>>     builder.add_spring(i-1, i, 1.e+3, 0.0, 0)
        >>>
        >>> # create model
        >>> model = builder.finalize("cuda")
        >>> 
        >>> state = model.state()
        >>> integrator = wp.sim.SemiImplicitIntegrator()
        >>>
        >>> for i in range(100):
        >>>
        >>>    state.clear_forces()
        >>>    integrator.simulate(model, state, state, dt=1.0/60.0)

    Note:
        It is strongly recommended to use the ModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if 
        desired.
    """

    default_tri_ke = 100.0
    default_tri_ka = 100.0
    default_tri_kd = 10.0
    default_tri_drag = 0.0
    default_tri_lift = 0.0

    # Default edge bending properties
    default_edge_ke = 100.0
    default_edge_kd = 0.0

    # Default shape contact material properties
    default_shape_ke = 1.e+5
    default_shape_kd = 1000.0
    default_shape_kf = 1000.0
    default_shape_mu = 0.5
    default_shape_restitution = 0.0
    default_shape_density = 1000.0

    
    def __init__(self, upvector=(0.0, 1.0, 0.0), gravity=-9.80665):
        self.num_envs = 0

        # particles
        self.particle_q = []
        self.particle_qd = []
        self.particle_mass = []

        # shapes (each shape has an entry in these arrays)
        # transform from shape to body
        self.shape_transform = []
        # maps from shape index to body index
        self.shape_body = []
        self.shape_geo_type = []
        self.shape_geo_scale = []
        self.shape_geo_src = []
        self.shape_material_ke = []
        self.shape_material_kd = []
        self.shape_material_kf = []
        self.shape_material_mu = []
        self.shape_material_restitution = []
        self.shape_contact_thickness = []
        # collision groups within collisions are handled
        self.shape_collision_group = []
        self.shape_collision_group_map = {}
        self.last_collision_group = 0
        # radius to use for broadphase collision checking
        self.shape_collision_radius = []
        # whether the shape collides with the ground
        self.shape_ground_collision = []

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
        self.muscle_activation = []
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
        self.joint_parent = []         # index of the parent body                      (constant)
        self.joint_child = []          # index of the child body                       (constant)
        self.joint_axis = []           # joint axis in child joint frame               (constant)
        self.joint_X_p = []            # frame of joint in parent                      (constant)
        self.joint_X_c = []            # frame of child com (in child coordinates)     (constant)
        self.joint_q = []
        self.joint_qd = []

        self.joint_type = []
        self.joint_name = []
        self.joint_armature = []
        self.joint_target_ke = []
        self.joint_target_kd = []
        self.joint_target = []
        self.joint_limit_lower = []
        self.joint_limit_upper = []
        self.joint_limit_ke = []
        self.joint_limit_kd = []
        self.joint_act = []

        self.joint_twist_lower = []
        self.joint_twist_upper = []

        self.joint_linear_compliance = []
        self.joint_angular_compliance = []

        self.joint_q_start = []
        self.joint_qd_start = []
        self.articulation_start = []

        self.joint_count = 0
        self.joint_dof_count = 0
        self.joint_coord_count = 0

        self.upvector = upvector
        self.gravity = gravity
        # indicates whether a ground plane has been created
        self._ground_created = False
        # constructor parameters for ground plane shape
        self._ground_params = dict(
            plane=(*upvector, 0.0),
            width=0.0,
            length=0.0,
            ke=self.default_shape_ke,
            kd=self.default_shape_kd,
            kf=self.default_shape_kf,
            mu=self.default_shape_mu,
            restitution=self.default_shape_restitution)

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
    
    def add_rigid_articulation(self, articulation, xform=None, update_num_env_count=True, separate_collision_group=True):
        """Copies a rigid articulation from `articulation`, another `ModelBuilder`.
        
        Args:
            articulation: a model builder to add rigid articulation from.
            xform: offset transform applied to root bodies.
            update_num_env_count: if True, the number of environments is incremented by 1.
            separate_collision_group: if True, the shapes from the articulation will all be put into a single new collision group, otherwise, only the shapes in collision group > -1 will be moved to a new group.
        """

        joint_X_p = copy.deepcopy(articulation.joint_X_p)
        joint_q = copy.deepcopy(articulation.joint_q)
        if xform is not None:
            for i in range(len(joint_X_p)):
                if articulation.joint_type[i] == wp.sim.JOINT_FREE:
                    qi = articulation.joint_q_start[i]
                    xform_prev = wp.transform(joint_q[qi:qi+3], joint_q[qi+3:qi+7])
                    tf = xform * xform_prev
                    joint_q[qi:qi+3] = tf.p
                    joint_q[qi+3:qi+7] = tf.q
                elif articulation.joint_parent[i] == -1:
                    joint_X_p[i] = xform * joint_X_p[i]
        self.joint_X_p.extend(joint_X_p)
        self.joint_q.extend(joint_q)

        self.add_articulation() 

        start_body_idx = len(self.body_mass)

        # offset the indices
        self.joint_parent.extend([p + self.joint_count if p != -1 else -1 for p in articulation.joint_parent])
        self.joint_child.extend([c + self.joint_count for c in articulation.joint_child])

        self.joint_q_start.extend([c + self.joint_coord_count for c in articulation.joint_q_start])
        self.joint_qd_start.extend([c + self.joint_dof_count for c in articulation.joint_qd_start])

        self.shape_body.extend([b + start_body_idx for b in articulation.shape_body])

        # apply collision group
        if separate_collision_group:
            self.shape_collision_group.extend([self.last_collision_group + 1 for _ in articulation.shape_collision_group])
        else:
            self.shape_collision_group.extend([(g + self.last_collision_group if g > -1 else -1) for g in articulation.shape_collision_group])
        shape_count = len(self.shape_geo_type)
        for i, j in articulation.shape_collision_filter_pairs:
            self.shape_collision_filter_pairs.add((i + shape_count, j + shape_count))
        for group, shapes in articulation.shape_collision_group_map.items():
            if separate_collision_group:
                group = self.last_collision_group + 1
            else:
                group = (group + self.last_collision_group if group > -1 else -1)
            if group not in articulation.shape_collision_group_map:
                self.shape_collision_group_map[group] = []
            self.shape_collision_group_map[group].extend([s + shape_count for s in shapes])

        # update last collision group counter
        if separate_collision_group:
            self.last_collision_group += 1
        elif articulation.last_collision_group > -1:
            self.last_collision_group += articulation.last_collision_group

        rigid_articulation_attrs = [
            "body_inertia",
            "body_mass",
            "body_inv_inertia",
            "body_inv_mass",
            "body_com",
            "body_q",
            "body_qd",
            "body_name",
            "joint_type",
            "joint_X_c",
            "joint_armature",
            "joint_axis",
            "joint_name",
            "joint_qd",
            "joint_act",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_target_ke",
            "joint_target_kd",
            "joint_target",
            "joint_linear_compliance",
            "joint_angular_compliance",
            "shape_transform",
            "shape_geo_type",
            "shape_geo_scale",
            "shape_geo_src",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_kf",
            "shape_material_mu",
            "shape_material_restitution",
            "shape_contact_thickness",
            "shape_collision_radius",
            "shape_ground_collision",
        ]

        for attr in rigid_articulation_attrs:
            getattr(self, attr).extend(getattr(articulation, attr))
        
        self.joint_count += articulation.joint_count
        self.joint_dof_count += articulation.joint_dof_count
        self.joint_coord_count += articulation.joint_coord_count

        if update_num_env_count:
            self.num_envs += 1


    # register a rigid body and return its index.
    def add_body(
        self, 
        origin: Transform,
        parent: int=-1,
        joint_xform: Transform=wp.transform(),    # transform of joint in parent space
        joint_xform_child: Transform=wp.transform(),
        joint_axis: Vec3=(0.0, 0.0, 0.0),
        joint_type: wp.constant=JOINT_FREE,
        joint_target_ke: float=0.0,
        joint_target_kd: float=0.0,
        joint_target: float=None,
        joint_limit_ke: float=100.0,
        joint_limit_kd: float=10.0,
        joint_limit_lower: float=-1.e+3,
        joint_limit_upper: float=1.e+3,
        joint_armature: float=0.0,
        joint_twist_lower: float=-1.e+3,
        joint_twist_upper: float=1.e+3,
        joint_linear_compliance: float=0.0,
        joint_angular_compliance: float=0.0,
        com: Vec3=np.zeros(3),
        I_m: Mat33=np.zeros((3, 3)), 
        m: float=0.0,
        body_name: str=None,
        joint_name: str=None) -> int:

        """Adds a rigid body to the model.

        Args:
            parent: The index of the parent body
            origin: The location of the joint in the parent's local frame connecting this body
            joint_xform: The transform of the body's joint in parent space
            joint_xform_child: Transform body's joint in local space
            joint_axis : Joint axis in local body space
            joint_type : Type of the joint, e.g.: JOINT_PRISMATIC, JOINT_REVOLUTE, etc.
            joint_target_ke: Stiffness of the joint PD controller
            joint_target_kd: Damping of the joint PD controller
            joint_limit_ke: Stiffness of the joint limits
            joint_limit_kd: Damping of the joint limits
            joint_limit_lower: Lower limit of the joint coordinate
            joint_limit_upper: Upper limit of the joint coordinate
            joint_armature: Artificial inertia added around the joint axis
            com: The center of mass of the body w.r.t its origin
            I_m: The 3x3 inertia tensor of the body (specified relative to the center of mass)
            m: Mass of the body

        Returns:
            The index of the body in the model

        Note:
            If the mass (m) is zero then the body is treated as kinematic with no dynamics

        """

        child = len(self.body_mass)

        # body data
        inertia = I_m + np.eye(3)*joint_armature
        self.body_inertia.append(inertia)
        self.body_mass.append(m)
        self.body_com.append(com)
        
        if (m > 0.0):
            self.body_inv_mass.append(1.0/m)
        else:
            self.body_inv_mass.append(0.0)
    
        if inertia.any():
            self.body_inv_inertia.append(np.linalg.inv(inertia))
        else:
            self.body_inv_inertia.append(inertia)
        
        self.body_q.append(origin)
        self.body_qd.append(wp.spatial_vector())

        self.body_name.append(body_name or f"body {child}")
        self.body_shapes[child] = []

        # joint data
        self.joint_type.append(joint_type.val)
        self.joint_parent.append(parent)
        self.joint_child.append(child)
        self.joint_X_p.append(joint_xform)
        self.joint_X_c.append(joint_xform_child)
        self.joint_name.append(joint_name or f"joint {self.joint_count}")

        self.joint_armature.append(joint_armature)
        self.joint_axis.append(np.array(joint_axis))

        if (joint_type == JOINT_PRISMATIC):
            dof_count = 1
            coord_count = 1
        elif (joint_type == JOINT_REVOLUTE):
            dof_count = 1
            coord_count = 1
        elif (joint_type == JOINT_BALL):
            dof_count = 3
            coord_count = 4
        elif (joint_type == JOINT_FREE):
            dof_count = 6
            coord_count = 7
        elif (joint_type == JOINT_FIXED):
            dof_count = 0
            coord_count = 0
        elif (joint_type == JOINT_COMPOUND):
            dof_count = 3
            coord_count = 3
        elif (joint_type == JOINT_UNIVERSAL):
            dof_count = 2
            coord_count = 2

        # convert coefficients to np.arrays() so we can index into them for 
        # compound joints, this just allows user to pass scalars or arrays
        # coefficients will be automatically padded to number of dofs
        joint_target_ke = np.resize(np.atleast_1d(joint_target_ke), dof_count)
        joint_target_kd = np.resize(np.atleast_1d(joint_target_kd), dof_count)
        joint_limit_ke = np.resize(np.atleast_1d(joint_limit_ke), dof_count)
        joint_limit_kd = np.resize(np.atleast_1d(joint_limit_kd), dof_count)
        joint_limit_lower = np.resize(np.atleast_1d(joint_limit_lower), dof_count)
        joint_limit_upper = np.resize(np.atleast_1d(joint_limit_upper), dof_count)
        joint_twist_lower = np.resize(np.atleast_1d(joint_twist_lower), dof_count)
        joint_twist_upper = np.resize(np.atleast_1d(joint_twist_upper), dof_count)
        if joint_target is not None:
            joint_target = np.resize(np.atleast_1d(joint_target), dof_count)
       
        for i in range(coord_count):
            self.joint_q.append(0.0)

        for i in range(dof_count):
            self.joint_qd.append(0.0)
            self.joint_act.append(0.0)
            self.joint_limit_lower.append(joint_limit_lower[i])
            self.joint_limit_upper.append(joint_limit_upper[i])
            self.joint_limit_ke.append(joint_limit_ke[i])
            self.joint_limit_kd.append(joint_limit_kd[i])
            self.joint_twist_lower.append(joint_twist_lower[i])
            self.joint_twist_upper.append(joint_twist_upper[i])
            self.joint_target_ke.append(joint_target_ke[i])
            self.joint_target_kd.append(joint_target_kd[i])
            if joint_target is not None:
                self.joint_target.append(joint_target[i])
            else:
                if joint_limit_lower[i] > 0.0 or joint_limit_upper[i] < 0.0:
                    self.joint_target.append(0.5 * (joint_limit_lower[i] + joint_limit_upper[i]))
                else:
                    self.joint_target.append(0.0)
        if (joint_type == JOINT_FREE):
            # ensure that a valid quaternion is used for the free joint angular dofs
            self.joint_q[-1] = 1.0
            
        self.joint_linear_compliance.append(joint_linear_compliance)
        self.joint_angular_compliance.append(joint_angular_compliance)

        self.joint_q_start.append(self.joint_coord_count)
        self.joint_qd_start.append(self.joint_dof_count)

        self.joint_count += 1
        self.joint_dof_count += dof_count
        self.joint_coord_count += coord_count

        # return index of child body / joint
        return child


    # muscles
    def add_muscle(self, bodies: List[int], positions: List[Vec3], f0: float, lm: float, lt: float, lmax: float, pen: float) -> float:
        """Adds a muscle-tendon activation unit

        Args:
            bodies: A list of body indices for each waypoint
            positions: A list of positions of each waypoint in the body's local frame
            f0: Force scaling
            lm: Muscle length
            lt: Tendon length
            lmax: Maximally efficient muscle length

        Returns:
            The index of the muscle in the model

        """

        n = len(bodies)

        self.muscle_start.append(len(self.muscle_bodies))
        self.muscle_params.append((f0, lm, lt, lmax, pen))
        self.muscle_activation.append(0.0)

        for i in range(n):

            self.muscle_bodies.append(bodies[i])
            self.muscle_points.append(positions[i])

        # return the index of the muscle
        return len(self.muscle_start)-1

    # shapes
    def add_shape_plane(self,
                        plane: Vec4=(0.0, 1.0, 0.0, 0.0),
                        pos: Vec3=None,
                        rot: Quat=None,
                        width: float=10.0,
                        length: float=10.0,
                        body: int = -1,
                        ke: float=default_shape_ke,
                        kd: float=default_shape_kd,
                        kf: float=default_shape_kf,
                        mu: float=default_shape_mu,
                        restitution: float=default_shape_restitution):
        """
        Adds a plane collision shape.
        If pos and rot are defined, the plane is assumed to have its normal as (0, 1, 0).
        Otherwise, the plane equation is used.

        Args: 
            plane: The plane equation in form a*x + b*y + c*z + d = 0
            pos: The position of the plane in world coordinates
            rot: The rotation of the plane in world coordinates
            width: The extent along x of the plane (infinite if 0)
            length: The extent along z of the plane (infinite if 0)
            body: The body index to attach the shape to (-1 by default to keep the plane static)
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction
            restitution: The coefficient of restitution

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
                axis = c / np.linalg.norm(c)
                rot = wp.quat_from_axis_angle(axis, angle)
        scale = (width, length, 0.0)
        self._add_shape(body, pos, rot, GEO_PLANE, scale, None, 0.0, ke, kd, kf, mu, restitution)

    def add_shape_sphere(self,
                         body,
                         pos: Vec3=(0.0, 0.0, 0.0),
                         rot: Quat=(0.0, 0.0, 0.0, 1.0),
                         radius: float=1.0,
                         density: float=default_shape_density,
                         ke: float=default_shape_ke,
                         kd: float=default_shape_kd,
                         kf: float=default_shape_kf,
                         mu: float=default_shape_mu,
                         restitution: float=default_shape_restitution):
        """Adds a sphere collision shape to a body.

        Args:
            body: The index of the parent body this shape belongs to
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            radius: The radius of the sphere
            density: The density of the shape
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction
            restitution: The coefficient of restitution

        """

        self._add_shape(body, pos, rot, GEO_SPHERE, (radius, 0.0, 0.0, 0.0), None, density, ke, kd, kf, mu, restitution, thickness=radius)

    def add_shape_box(self,
                      body: int,
                      pos: Vec3=(0.0, 0.0, 0.0),
                      rot: Quat=(0.0, 0.0, 0.0, 1.0),
                      hx: float=0.5,
                      hy: float=0.5,
                      hz: float=0.5,
                      density: float=default_shape_density,
                      ke: float=default_shape_ke,
                      kd: float=default_shape_kd,
                      kf: float=default_shape_kf,
                      mu: float=default_shape_mu,
                      restitution: float=default_shape_restitution,
                      contact_thickness: float=0.0):
        """Adds a box collision shape to a body.

        Args:
            body: The index of the parent body this shape belongs to
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            hx: The half-extents along the x-axis
            hy: The half-extents along the y-axis
            hz: The half-extents along the z-axis
            density: The density of the shape
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction
            restitution: The coefficient of restitution
            contact_thickness: Radius around the box to be used for contact mechanics

        """

        self._add_shape(body, pos, rot, GEO_BOX, (hx, hy, hz, 0.0), None, density, ke, kd, kf, mu, restitution, thickness=contact_thickness)

    def add_shape_capsule(self,
                          body: int,
                          pos: Vec3=(0.0, 0.0, 0.0),
                          rot: Quat=(0.0, 0.0, 0.0, 1.0),
                          radius: float=1.0,
                          half_width: float=0.5,
                          density: float=default_shape_density,
                          ke: float=default_shape_ke,
                          kd: float=default_shape_kd,
                          kf: float=default_shape_kf,
                          mu: float=default_shape_mu,
                          restitution: float=default_shape_restitution):
        """Adds a capsule collision shape to a body.

        Args:
            body: The index of the parent body this shape belongs to
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            radius: The radius of the capsule
            half_width: The half length of the center cylinder along the x-axis
            density: The density of the shape
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction
            restitution: The coefficient of restitution

        """

        self._add_shape(body, pos, rot, GEO_CAPSULE, (radius, half_width, 0.0, 0.0), None, density, ke, kd, kf, mu, restitution, thickness=radius)

    def add_shape_mesh(self,
                       body: int,
                       pos: Vec3=(0.0, 0.0, 0.0),
                       rot: Quat=(0.0, 0.0, 0.0, 1.0),
                       mesh: Mesh=None,
                       scale: Vec3=(1.0, 1.0, 1.0),
                       density: float=default_shape_density,
                       ke: float=default_shape_ke,
                       kd: float=default_shape_kd,
                       kf: float=default_shape_kf,
                       mu: float=default_shape_mu,
                       restitution: float=default_shape_restitution,
                       contact_thickness: float=0.0):
        """Adds a triangle mesh collision shape to a body.

        Args:
            body: The index of the parent body this shape belongs to
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            mesh: The mesh object
            scale: Scale to use for the collider
            density: The density of the shape
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction
            restitution: The coefficient of restitution
            contact_thickness: The thickness of the contact surface around the shape

        """

        self._add_shape(body, pos, rot, GEO_MESH, (scale[0], scale[1], scale[2], 0.0), mesh, density, ke, kd, kf, mu, restitution, thickness=contact_thickness)

    def _shape_radius(self, type, scale, src):
        """
        Calculates the squared radius of a sphere that encloses the shape, used for broadphase collision detection.
        """
        if type == GEO_SPHERE:
            return scale[0]
        elif type == GEO_BOX:
            return np.linalg.norm(scale)
        elif type == GEO_CAPSULE:
            return scale[0] + scale[1]
        elif type == GEO_MESH:
            vmax = np.max(np.abs(src.vertices), axis=0) * scale[0]
            return np.linalg.norm(vmax)
        elif type == GEO_PLANE:
            if scale[0] > 0.0 and scale[1] > 0.0:
                # finite plane
                return np.linalg.norm(scale)
            else:
                return 1.0e6
        else:
            return 10.0
    
    def _add_shape(self, body, pos, rot, type, scale, src, density, ke, kd, kf, mu, restitution, thickness=0.0, collision_group=-1, collision_filter_parent=True, has_ground_collision=True):
        self.shape_body.append(body)
        shape = len(self.shape_geo_type)
        if body in self.body_shapes:
            # no contacts between shapes of the same body
            for same_body_shape in self.body_shapes[body]:
                self.shape_collision_filter_pairs.add((same_body_shape, shape))
            self.body_shapes[body].append(shape)
        else:
            self.body_shapes[body] = [shape]
        self.shape_transform.append(wp.transform(pos, rot))
        self.shape_geo_type.append(type.val)
        self.shape_geo_scale.append((scale[0], scale[1], scale[2]))
        self.shape_geo_src.append(src)
        self.shape_material_ke.append(ke)
        self.shape_material_kd.append(kd)
        self.shape_material_kf.append(kf)
        self.shape_material_mu.append(mu)
        self.shape_material_restitution.append(restitution)
        self.shape_contact_thickness.append(thickness)
        self.shape_collision_group.append(collision_group)
        if collision_group not in self.shape_collision_group_map:
            self.shape_collision_group_map[collision_group] = []
        self.last_collision_group = max(self.last_collision_group, collision_group)
        self.shape_collision_group_map[collision_group].append(shape)
        self.shape_collision_radius.append(self._shape_radius(type, scale, src))
        if collision_filter_parent and body > -1:
            # XXX we assume joint ID == body ID here
            parent_body = self.joint_parent[body]
            if parent_body > -1:
                for parent_shape in self.body_shapes[parent_body]:
                    self.shape_collision_filter_pairs.add((parent_shape, shape))
        if body == -1:
            has_ground_collision = False
        self.shape_ground_collision.append(has_ground_collision)

        (m, I) = self._compute_shape_mass(type, scale, src, density)

        self._update_body_mass(body, m, I, np.array(pos), np.array(rot))

    # particles
    def add_particle(self, pos : Vec3, vel : Vec3, mass : float) -> int:
        """Adds a single particle to the model

        Args:
            pos: The initial position of the particle
            vel: The initial velocity of the particle
            mass: The mass of the particle

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that does is not subject to dynamics.

        Returns:
            The index of the particle in the system
        """
        self.particle_q.append(pos)
        self.particle_qd.append(vel)
        self.particle_mass.append(mass)

        return len(self.particle_q) - 1

    def add_spring(self, i : int, j, ke : float, kd : float, control: float):
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

    def add_triangle(self, i : int, j : int, k : int, tri_ke : float=default_tri_ke, tri_ka : float=default_tri_ka, tri_kd :float=default_tri_kd, tri_drag : float=default_tri_drag, tri_lift : float = default_tri_lift) -> float:

        """Adds a trianglular FEM element between three particles in the system. 

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        Parameters specfied on the model. See model.tri_ke, model.tri_kd.

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle

        Return:
            The area of the triangle

        Note:
            The triangle is created with a rest-length based on the distance
            between the particles in their initial configuration.

        Todo:
            * Expose elastic paramters on a per-element basis

        """      
        # compute basis for 2D rest pose
        p = np.array(self.particle_q[i])
        q = np.array(self.particle_q[j])
        r = np.array(self.particle_q[k])

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

        if (area <= 0.0):

            print("inverted or degenerate triangle element")
            return 0.0
        else:
    
            inv_D = np.linalg.inv(D)

            self.tri_indices.append((i, j, k))
            self.tri_poses.append(inv_D.tolist())
            self.tri_activations.append(0.0)
            self.tri_materials.append((tri_ke, tri_ka, tri_kd, tri_drag, tri_lift))
            return area


    def add_triangles(self, i:List[int], j:List[int], k:List[int], tri_ke : Optional[List[float]] = None, tri_ka : Optional[List[float]] = None, tri_kd :Optional[List[float]] = None, tri_drag :Optional[List[float]] = None, tri_lift :Optional[List[float]] = None) -> List[float]:

        """Adds trianglular FEM elements between groups of three particles in the system. 

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        Parameters specfied on the model. See model.tri_ke, model.tri_kd.

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
            l = np.linalg.norm(a,axis=-1,keepdims=True)
            l[l==0] = 1.0
            return a / l

        n = normalized(np.cross(qp,rp))
        e1 = normalized(qp)
        e2 = normalized(np.cross(n,e1))

        R = np.concatenate((e1[...,None],e2[...,None]),axis=-1)
        M = np.concatenate((qp[...,None],rp[...,None]),axis=-1)

        D = np.matmul(R.transpose(0,2,1),M)
        
        areas = np.linalg.det(D) / 2.0
        areas[areas < 0.0] = 0.0
        valid_inds = (areas>0.0).nonzero()[0]
        if len(valid_inds) < len(areas):
            print("inverted or degenerate triangle elements")
        
        D[areas == 0.0] = np.eye(2)[None,...]
        inv_D = np.linalg.inv(D)

        inds = np.concatenate( (i[valid_inds,None],j[valid_inds,None],k[valid_inds,None]), axis=-1 )

        self.tri_indices.extend(inds.tolist())
        self.tri_poses.extend(inv_D[valid_inds].tolist())
        self.tri_activations.extend([0.0] * len(valid_inds))

        def init_if_none( arr, defaultValue ):
            if arr is None:
                return [defaultValue] * len(areas)
            return arr
        
        tri_ke = init_if_none( tri_ke, self.default_tri_ke )
        tri_ka = init_if_none( tri_ka, self.default_tri_ka )
        tri_kd = init_if_none( tri_kd, self.default_tri_kd )
        tri_drag = init_if_none( tri_drag, self.default_tri_drag )
        tri_lift = init_if_none( tri_lift, self.default_tri_lift )

        self.tri_materials.extend( zip(
            np.array(tri_ke)[valid_inds],
            np.array(tri_ka)[valid_inds],
            np.array(tri_kd)[valid_inds],
            np.array(tri_drag)[valid_inds],
            np.array(tri_lift)[valid_inds]
        ) )
        return areas.tolist()

    def add_tetrahedron(self, i: int, j: int, k: int, l: int, k_mu: float=1.e+3, k_lambda: float=1.e+3, k_damp: float=0.0) -> float:
        """Adds a tetrahedral FEM element between four particles in the system. 

        Tetrahdera are modeled as viscoelastic elements with a NeoHookean energy
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
            The tetrahedron is created with a rest-pose based on the particle's initial configruation

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

        if (volume <= 0.0):
            print("inverted tetrahedral element")
        else:

            inv_Dm = np.linalg.inv(Dm)

            self.tet_indices.append((i, j, k, l))
            self.tet_poses.append(inv_Dm.tolist())
            self.tet_activations.append(0.0)
            self.tet_materials.append((k_mu, k_lambda, k_damp))

        return volume

    def add_edge(self, i: int, j: int, k: int, l: int, rest: float=None, edge_ke: float=default_edge_ke, edge_kd: float=default_edge_kd):
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
        if (rest == None):

            x1 = np.array(self.particle_q[i])
            x2 = np.array(self.particle_q[j])
            x3 = np.array(self.particle_q[k])
            x4 = np.array(self.particle_q[l])

            n1 = wp.normalize(np.cross(x3 - x1, x4 - x1))
            n2 = wp.normalize(np.cross(x4 - x2, x3 - x2))
            e = wp.normalize(x4 - x3)

            d = np.clip(np.dot(n2, n1), -1.0, 1.0)

            angle = math.acos(d)
            sign = np.sign(np.dot(np.cross(n2, n1), e))

            rest = angle * sign

        self.edge_indices.append((i, j, k, l))
        self.edge_rest_angle.append(rest)
        self.edge_bending_properties.append((edge_ke, edge_kd))

    def add_edges(self, i, j, k, l, rest: Optional[List[float]] = None, edge_ke: Optional[List[float]] = None, edge_kd: Optional[List[float]] = None):
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
                l = np.linalg.norm(a,axis=-1,keepdims=True)
                l[l==0] = 1.0
                return a / l

            n1 = normalized(np.cross(x3 - x1, x4 - x1))
            n2 = normalized(np.cross(x4 - x2, x3 - x2))
            e = normalized(x4 - x3)

            def dot(a,b):
                return (a * b).sum(axis=-1)

            d = np.clip(dot(n2, n1), -1.0, 1.0)

            angle = np.arccos(d)
            sign = np.sign(dot(np.cross(n2, n1), e))

            rest = angle * sign

        inds = np.concatenate( (i[:,None],j[:,None],k[:,None],l[:,None]), axis=-1 )

        self.edge_indices.extend(inds.tolist())
        self.edge_rest_angle.extend(rest.tolist())
        
        def init_if_none( arr, defaultValue ):
            if arr is None:
                return [defaultValue] * len(i)
            return arr
        
        edge_ke = init_if_none( edge_ke, self.default_edge_ke )
        edge_kd = init_if_none( edge_kd, self.default_edge_kd )

        self.edge_bending_properties.extend(zip(edge_ke, edge_kd))

    def add_cloth_grid(self,
                       pos: Vec3,
                       rot: Quat,
                       vel: Vec3,
                       dim_x: int,
                       dim_y: int,
                       cell_x: float,
                       cell_y: float,
                       mass: float,
                       reverse_winding: bool=False,
                       fix_left: bool=False,
                       fix_right: bool=False,
                       fix_top: bool=False,
                       fix_bottom: bool=False,
                       tri_ke: float=default_tri_ke,
                       tri_ka: float=default_tri_ka,
                       tri_kd: float=default_tri_kd,
                       tri_drag: float=default_tri_drag,
                       tri_lift: float=default_tri_lift, 
                       edge_ke: float=default_edge_ke,
                       edge_kd: float=default_edge_kd):

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

                g = np.array((x * cell_x, y * cell_y, 0.0))
                p = np.array(wp.quat_rotate(rot, g)) + pos
                m = mass

                if (x == 0 and fix_left):
                    m = 0.0
                elif (x == dim_x and fix_right):
                    m = 0.0
                elif (y == 0 and fix_bottom):
                    m = 0.0
                elif (y == dim_y and fix_top):
                    m = 0.0

                self.add_particle(p, vel, m)

                if (x > 0 and y > 0):

                    if (reverse_winding):
                        tri1 = (start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                                start_vertex + grid_index(x, y - 1, dim_x + 1),
                                start_vertex + grid_index(x, y, dim_x + 1))

                        tri2 = (start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                                start_vertex + grid_index(x, y, dim_x + 1),
                                start_vertex + grid_index(x - 1, y, dim_x + 1))

                        self.add_triangle(*tri1, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
                        self.add_triangle(*tri2, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

                    else:

                        tri1 = (start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                                start_vertex + grid_index(x, y - 1, dim_x + 1),
                                start_vertex + grid_index(x - 1, y, dim_x + 1))

                        tri2 = (start_vertex + grid_index(x, y - 1, dim_x + 1),
                                start_vertex + grid_index(x, y, dim_x + 1),
                                start_vertex + grid_index(x - 1, y, dim_x + 1))

                        self.add_triangle(*tri1, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
                        self.add_triangle(*tri2, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

        end_vertex = len(self.particle_q)
        end_tri = len(self.tri_indices)

        # bending constraints, could create these explicitly for a grid but this
        # is a good test of the adjacency structure
        adj = wp.utils.MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        for k, e in adj.edges.items():

            # skip open edges
            if (e.f0 == -1 or e.f1 == -1):
                continue

            self.add_edge(e.o0, e.o1, e.v0, e.v1, edge_ke=edge_ke, edge_kd=edge_kd)          # opposite 0, opposite 1, vertex 0, vertex 1

    def add_cloth_mesh(self, pos: Vec3, rot: Quat, scale: float, vel: Vec3, vertices: List[Vec3], indices: List[int], density: float, edge_callback=None, face_callback=None,
                       tri_ke: float=default_tri_ke,
                       tri_ka: float=default_tri_ka,
                       tri_kd: float=default_tri_kd,
                       tri_drag: float=default_tri_drag,
                       tri_lift: float=default_tri_lift,
                       edge_ke: float=default_edge_ke,
                       edge_kd: float=default_edge_kd):
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

            p = np.array(wp.quat_rotate(rot, v * scale)) + pos

            self.add_particle(p, vel, 0.0)

        # triangles
        inds = start_vertex + np.array(indices)
        inds = inds.reshape(-1,3)
        areas = self.add_triangles(
            inds[:,0],inds[:,1],inds[:,2],
            [tri_ke] * num_tris,
            [tri_ka] * num_tris,
            [tri_kd] * num_tris,
            [tri_drag] * num_tris,
            [tri_lift] * num_tris
        )
        
        for t in range(num_tris):
            area = areas[t]

            self.particle_mass[inds[t,0]] += density * area / 3.0
            self.particle_mass[inds[t,1]] += density * area / 3.0
            self.particle_mass[inds[t,2]] += density * area / 3.0

        end_tri = len(self.tri_indices)

        adj = wp.utils.MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        edgeinds = np.array([[e.o0,e.o1,e.v0,e.v1] for k,e in adj.edges.items()])
        self.add_edges(
            edgeinds[:,0], edgeinds[:,1], edgeinds[:,2], edgeinds[:,0],
            edge_ke=[edge_ke] * len(edgeinds),
            edge_kd=[edge_kd] * len(edgeinds)
        )

    def add_particle_grid(self,
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
                      jitter: float):

        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):

                    v = np.array((x * cell_x, y * cell_y, z * cell_z))
                    m = mass

                    p = np.array(wp.quat_rotate(rot, v)) + pos + np.random.rand(3)*jitter

                    self.add_particle(p, vel, m)

    def add_soft_grid(self,
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
                      fix_left: bool=False,
                      fix_right: bool=False,
                      fix_top: bool=False,
                      fix_bottom: bool=False,
                      tri_ke: float=default_tri_ke,
                      tri_ka: float=default_tri_ka,
                      tri_kd: float=default_tri_kd,
                      tri_drag: float=default_tri_drag,
                      tri_lift: float=default_tri_lift):
        """Helper to create a rectangular tetrahedral FEM grid

        Creates a regular grid of FEM tetrhedra and surface triangles. Useful for example
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

                    v = np.array((x * cell_x, y * cell_y, z * cell_z))
                    m = mass

                    if (fix_left and x == 0):
                        m = 0.0

                    if (fix_right and x == dim_x):
                        m = 0.0

                    if (fix_top and y == dim_y):
                        m = 0.0

                    if (fix_bottom and y == 0):
                        m = 0.0

                    p = np.array(wp.quat_rotate(rot, v)) + pos

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

                    if (((x & 1) ^ (y & 1) ^ (z & 1))):

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
        for k, v in faces.items():
            self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

    def add_soft_mesh(self, pos: Vec3, rot: Quat, scale: float, vel: Vec3, vertices: List[Vec3], indices: List[int], density: float, k_mu: float, k_lambda: float, k_damp: float,
                       tri_ke: float=default_tri_ke,
                       tri_ka: float=default_tri_ka,
                       tri_kd: float=default_tri_kd,
                       tri_drag: float=default_tri_drag,
                       tri_lift: float=default_tri_lift):
        """Helper to create a tetrahedral model from an input tetrahedral mesh

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            vertices: A list of vertex positions
            indices: A list of tetrahedron indices, 4 entries per-element
            density: The density per-area of the mesh
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
        """
        num_tets = int(len(indices) / 4)

        start_vertex = len(self.particle_q)
        start_tri = len(self.tri_indices)

        # dict of open faces
        faces = {}

        def add_face(i, j, k):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        # add particles
        for v in vertices:

            p = wp.quat_rotate(rot, v * scale) + np.array(pos)

            self.add_particle(p, vel, 0.0)

        # add tetrahedra
        for t in range(num_tets):

            v0 = start_vertex + indices[t * 4 + 0]
            v1 = start_vertex + indices[t * 4 + 1]
            v2 = start_vertex + indices[t * 4 + 2]
            v3 = start_vertex + indices[t * 4 + 3]

            volume = self.add_tetrahedron(v0, v1, v2, v3, k_mu, k_lambda, k_damp)

            # distribute volume fraction to particles
            if (volume > 0.0):

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
        for k, v in faces.items():
            try:
                self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
            except np.linalg.LinAlgError:
                continue

    def compute_sphere_inertia(self, density: float, r: float) -> tuple:
        """Helper to compute mass and inertia of a sphere

        Args:
            density: The sphere density
            r: The sphere radius

        Returns:

            A tuple of (mass, inertia) with inertia specified around the origin
        """

        v = 4.0 / 3.0 * math.pi * r * r * r

        m = density * v
        Ia = 2.0 / 5.0 * m * r * r

        I = np.array([[Ia, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])

        return (m, I)

    def compute_capsule_inertia(self, density: float, r: float, l: float) -> tuple:
        """Helper to compute mass and inertia of a capsule

        Args:
            density: The capsule density
            r: The capsule radius
            l: The capsule length (full width of the interior cylinder)

        Returns:

            A tuple of (mass, inertia) with inertia specified around the origin
        """

        ms = density * (4.0 / 3.0) * math.pi * r * r * r
        mc = density * math.pi * r * r * l

        # total mass
        m = ms + mc

        # adapted from ODE
        Ia = mc * (0.25 * r * r + (1.0 / 12.0) * l * l) + ms * (0.4 * r * r + 0.375 * r * l + 0.25 * l * l)
        Ib = (mc * 0.5 + ms * 0.4) * r * r

        I = np.array([[Ib, 0.0, 0.0], [0.0, Ia, 0.0], [0.0, 0.0, Ia]])

        return (m, I)

    def compute_box_inertia(self, density: float, w: float, h: float, d: float) -> tuple:
        """Helper to compute mass and inertia of a box

        Args:
            density: The box density
            w: The box width along the x-axis
            h: The box height along the y-axis
            d: The box depth along the z-axis

        Returns:

            A tuple of (mass, inertia) with inertia specified around the origin
        """

        v = w * h * d
        m = density * v

        Ia = 1.0 / 12.0 * m * (h * h + d * d)
        Ib = 1.0 / 12.0 * m * (w * w + d * d)
        Ic = 1.0 / 12.0 * m * (w * w + h * h)

        I = np.array([[Ia, 0.0, 0.0], [0.0, Ib, 0.0], [0.0, 0.0, Ic]])

        return (m, I)

    def compute_mesh_inertia(self, density: float, vertices: list, indices: list) -> tuple:
        com = np.mean(vertices, 0)

        num_tris = len(indices) // 3

        # compute signed inertia for each tetrahedron
        # formed with the interior point, using an order-2
        # quadrature: https://www.sciencedirect.com/science/article/pii/S0377042712001604#br000040

        weight = 0.25
        alpha = math.sqrt(5.0) / 5.0

        I = np.zeros((3, 3))
        mass = 0.0

        for i in range(num_tris):

            p = np.array(vertices[indices[i * 3 + 0]])
            q = np.array(vertices[indices[i * 3 + 1]])
            r = np.array(vertices[indices[i * 3 + 2]])

            mid = (com + p + q + r) / 4.0

            pcom = p - com
            qcom = q - com
            rcom = r - com

            Dm = np.matrix((pcom, qcom, rcom)).T
            volume = np.linalg.det(Dm) / 6.0
            if volume == 0.0:
                continue

            # quadrature points lie on the line between the
            # centroid and each vertex of the tetrahedron
            quads = (mid + (p - mid) * alpha, mid + (q - mid) * alpha, mid + (r - mid) * alpha, mid + (com - mid) * alpha)

            for j in range(4):

                # displacement of quadrature point from COM
                d = quads[j] - com

                I += weight * volume * (np.dot(d, d) * np.eye(3, 3) - np.outer(d, d))
                mass += weight * volume

        return (mass * density, I * density)

    def _compute_shape_mass(self, type, scale, src, density):
      
        if density == 0:     # zero density means fixed
            return 0, np.zeros((3, 3))

        if (type == GEO_SPHERE):
            return self.compute_sphere_inertia(density, scale[0])
        elif (type == GEO_BOX):
            return self.compute_box_inertia(density, scale[0] * 2.0, scale[1] * 2.0, scale[2] * 2.0)
        elif (type == GEO_CAPSULE):
            return self.compute_capsule_inertia(density, scale[0], scale[1] * 2.0)
        elif (type == GEO_MESH):
            #todo: non-uniform scale of inertia tensor
            if src.mass > 0.0:
                s = scale[0]
                return (density * src.mass * s * s * s, density * src.I * s * s * s * s * s)
            else:
                # fall back to computing inertia from mesh geometry
                return self.compute_mesh_inertia(density, src.vertices, src.indices)


    def _transform_inertia(self, m, I, p, q):
        R = np.array(wp.quat_to_matrix(q)).reshape(3,3)

        # Steiner's theorem
        return R @ I @ R.T + m * (np.dot(p, p) * np.eye(3) - np.outer(p, p))

    
    # incrementally updates rigid body mass with additional mass and inertia expressed at a local to the body
    def _update_body_mass(self, i, m, I, p, q):
        
        if (i == -1):
            return
            
        # find new COM
        new_mass = self.body_mass[i] + m

        if new_mass == 0.0:    # no mass
            return

        new_com = (self.body_com[i] * self.body_mass[i] + p * m) / new_mass

        # shift inertia to new COM
        com_offset = new_com - self.body_com[i]
        shape_offset = new_com - p

        new_inertia = self._transform_inertia(self.body_mass[i], self.body_inertia[i], com_offset, wp.quat_identity()) + self._transform_inertia(
            m, I, shape_offset, q)

        self.body_mass[i] = new_mass
        self.body_inertia[i] = new_inertia
        self.body_com[i] = new_com

        if (new_mass > 0.0):
            self.body_inv_mass[i] = 1.0/new_mass
        else:
            self.body_inv_mass[i] = 0.0
            
        if new_inertia.any():
            self.body_inv_inertia[i] = np.linalg.inv(new_inertia)
        else:
            self.body_inv_inertia[i] = new_inertia

    def set_ground_plane(self,
                         normal=None,
                         ke: float=default_shape_ke,
                         kd: float=default_shape_kd,
                         kf: float=default_shape_kf,
                         mu: float=default_shape_mu,
                         restitution: float=default_shape_restitution):
        """
        Creates a ground plane for the world. If the normal is not specified,
        the upvector of the ModelBuilder is used.
        """
        if normal is None:
            normal = self.upvector
        self._ground_params = dict(
            plane=(*normal, 0.0),
            width=0.0,
            length=0.0,
            ke=ke,
            kd=kd,
            kf=kf,
            mu=mu,
            restitution=restitution)

    def _create_ground_plane(self):
        self.add_shape_plane(**self._ground_params)
        self._ground_created = True
        # disable ground collisions as they will be treated separately
        ground_id = len(self.shape_geo_type)-1
        for i in range(len(self.shape_geo_type)-1):
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
        particle_inv_mass = []
        
        for m in self.particle_mass:
            if (m > 0.0):
                particle_inv_mass.append(1.0 / m)
            else:
                particle_inv_mass.append(0.0)


        with wp.ScopedDevice(device):

            #-------------------------------------
            # construct Model (non-time varying) data

            m = Model(device)
            m.requires_grad = requires_grad

            #---------------------        
            # particles

            # state (initial)
            m.particle_q = wp.array(self.particle_q, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_qd = wp.array(self.particle_qd, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_mass = wp.array(self.particle_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_inv_mass = wp.array(particle_inv_mass, dtype=wp.float32, requires_grad=requires_grad)

            #---------------------
            # collision geometry

            m.shape_transform = wp.array(self.shape_transform, dtype=wp.transform, requires_grad=requires_grad)
            m.shape_body = wp.array(self.shape_body, dtype=wp.int32)
            m.shape_geo_type = wp.array(self.shape_geo_type, dtype=wp.int32)
            m.shape_geo_src = self.shape_geo_src

            # build list of ids for geometry sources (meshes, sdfs)
            shape_geo_id = []
            finalized_meshes = {}  # do not duplicate meshes
            for geo in self.shape_geo_src:
                geo_hash = hash(geo)  # avoid repeated hash computations
                if (geo):
                    if geo_hash not in finalized_meshes:
                        finalized_meshes[geo_hash] = geo.finalize(device=device)
                    shape_geo_id.append(finalized_meshes[geo_hash])
                else:
                    shape_geo_id.append(-1)

            m.shape_geo_id = wp.array(shape_geo_id, dtype=wp.uint64)
            m.shape_geo_scale = wp.array(self.shape_geo_scale, dtype=wp.vec3, requires_grad=requires_grad)
            m.shape_materials = ShapeContactMaterial()
            m.shape_materials.ke = wp.array(self.shape_material_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.kd = wp.array(self.shape_material_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.kf = wp.array(self.shape_material_kf, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.mu = wp.array(self.shape_material_mu, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.restitution = wp.array(self.shape_material_restitution, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_contact_thickness = wp.array(self.shape_contact_thickness, dtype=wp.float32, requires_grad=requires_grad)

            m.shape_collision_filter_pairs = self.shape_collision_filter_pairs
            m.shape_collision_group = self.shape_collision_group
            m.shape_collision_group_map = self.shape_collision_group_map
            m.shape_collision_radius = wp.array(self.shape_collision_radius, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_ground_collision = self.shape_ground_collision

            #---------------------
            # springs

            m.spring_indices = wp.array(self.spring_indices, dtype=wp.int32)
            m.spring_rest_length = wp.array(self.spring_rest_length, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_stiffness = wp.array(self.spring_stiffness, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_damping = wp.array(self.spring_damping, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_control = wp.array(self.spring_control, dtype=wp.float32, requires_grad=requires_grad)

            #---------------------
            # triangles

            m.tri_indices = wp.array(self.tri_indices, dtype=wp.int32)
            m.tri_poses = wp.array(self.tri_poses, dtype=wp.mat22, requires_grad=requires_grad)
            m.tri_activations = wp.array(self.tri_activations, dtype=wp.float32, requires_grad=requires_grad)
            m.tri_materials = wp.array(self.tri_materials, dtype=wp.float32, requires_grad=requires_grad)

            #---------------------
            # edges

            m.edge_indices = wp.array(self.edge_indices, dtype=wp.int32)
            m.edge_rest_angle = wp.array(self.edge_rest_angle, dtype=wp.float32, requires_grad=requires_grad)
            m.edge_bending_properties = wp.array(self.edge_bending_properties, dtype=wp.float32, requires_grad=requires_grad)

            #---------------------
            # tetrahedra

            m.tet_indices = wp.array(self.tet_indices, dtype=wp.int32)
            m.tet_poses = wp.array(self.tet_poses, dtype=wp.mat33, requires_grad=requires_grad)
            m.tet_activations = wp.array(self.tet_activations, dtype=wp.float32, requires_grad=requires_grad)
            m.tet_materials = wp.array(self.tet_materials, dtype=wp.float32, requires_grad=requires_grad)

            #-----------------------
            # muscles

            # close the muscle waypoint indices
            muscle_start = copy.copy(self.muscle_start)
            muscle_start.append(len(self.muscle_bodies))

            m.muscle_start = wp.array(muscle_start, dtype=wp.int32)
            m.muscle_params = wp.array(self.muscle_params, dtype=wp.float32, requires_grad=requires_grad)
            m.muscle_bodies = wp.array(self.muscle_bodies, dtype=wp.int32)
            m.muscle_points = wp.array(self.muscle_points, dtype=wp.vec3, requires_grad=requires_grad)
            m.muscle_activation = wp.array(self.muscle_activation, dtype=wp.float32, requires_grad=requires_grad)

            #--------------------------------------
            # rigid bodies
            
            m.body_q = wp.array(self.body_q, dtype=wp.transform, requires_grad=requires_grad)
            m.body_qd = wp.array(self.body_qd, dtype=wp.spatial_vector, requires_grad=requires_grad)
            m.body_inertia = wp.array(self.body_inertia, dtype=wp.mat33, requires_grad=requires_grad)
            m.body_inv_inertia = wp.array(self.body_inv_inertia, dtype=wp.mat33, requires_grad=requires_grad)
            m.body_mass = wp.array(self.body_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.body_inv_mass = wp.array(self.body_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.body_com = wp.array(self.body_com, dtype=wp.vec3, requires_grad=requires_grad)
            m.body_name = self.body_name

            # model
            m.joint_count = self.joint_count
            m.joint_type = wp.array(self.joint_type, dtype=wp.int32)
            m.joint_parent = wp.array(self.joint_parent, dtype=wp.int32)
            m.joint_child = wp.array(self.joint_child, dtype=wp.int32)
            m.joint_X_p = wp.array(self.joint_X_p, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_X_c = wp.array(self.joint_X_c, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_axis = wp.array(self.joint_axis, dtype=wp.vec3, requires_grad=requires_grad)
            m.joint_q = wp.array(self.joint_q, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_qd = wp.array(self.joint_qd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_name = self.joint_name

            # dynamics properties
            m.joint_armature = wp.array(self.joint_armature, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target = wp.array(self.joint_target, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_ke = wp.array(self.joint_target_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_kd = wp.array(self.joint_target_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_act = wp.array(self.joint_act, dtype=wp.float32, requires_grad=requires_grad)

            m.joint_limit_lower = wp.array(self.joint_limit_lower, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_upper = wp.array(self.joint_limit_upper, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_ke = wp.array(self.joint_limit_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_kd = wp.array(self.joint_limit_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_linear_compliance = wp.array(self.joint_linear_compliance, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_angular_compliance = wp.array(self.joint_angular_compliance, dtype=wp.float32, requires_grad=requires_grad)

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
            m.allocate_soft_contacts(1*1024, requires_grad=requires_grad)
            m.find_shape_contact_pairs()
            if self.num_rigid_contacts_per_env is None:
                contact_count = m.count_contact_points()
            else:
                contact_count = self.num_rigid_contacts_per_env*self.num_envs
            if wp.config.verbose:
                print(f"Allocating {contact_count} rigid contacts.")
            m.allocate_rigid_contacts(contact_count, requires_grad=requires_grad)
            m.rigid_contact_margin = self.rigid_contact_margin            
            m.rigid_contact_torsional_friction = self.rigid_contact_torsional_friction
            m.rigid_contact_rolling_friction = self.rigid_contact_rolling_friction
            
            m.joint_dof_count = self.joint_dof_count
            m.joint_coord_count = self.joint_coord_count
            
            # hash-grid for particle interactions
            m.particle_grid = wp.HashGrid(128, 128, 128)

            # store refs to geometry
            m.geo_meshes = self.geo_meshes
            m.geo_sdfs = self.geo_sdfs

            # enable ground plane
            m.ground_plane = wp.array(self._ground_params["plane"], dtype=wp.float32, requires_grad=requires_grad)
            m.gravity = np.array(self.upvector) * self.gravity

            m.enable_tri_collisions = False

            return m



