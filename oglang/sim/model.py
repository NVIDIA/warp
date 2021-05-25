"""A module for building simulation models and state.
"""

from operator import pos
import oglang as og

import math
import numpy as np

from typing import Tuple
from typing import List
Vec3 = List[float]
Vec4 = List[float]
Quat = List[float]
Mat33 = List[float]
Transform = Tuple[Vec3, Quat]


# shape geometry types
GEO_SPHERE = 0
GEO_BOX = 1
GEO_CAPSULE = 2
GEO_MESH = 3
GEO_SDF = 4
GEO_PLANE = 5
GEO_NONE = 6

# body joint types
JOINT_PRISMATIC = 0 
JOINT_REVOLUTE = 1
JOINT_BALL = 2
JOINT_FIXED = 3
JOINT_FREE = 4

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

            num_tris = int(len(indices) / 3)

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

                # quadrature points lie on the line between the
                # centroid and each vertex of the tetrahedron
                quads = (mid + (p - mid) * alpha, mid + (q - mid) * alpha, mid + (r - mid) * alpha, mid + (com - mid) * alpha)

                for j in range(4):

                    # displacement of quadrature point from COM
                    d = quads[j] - com

                    I += weight * volume * (og.length_sq(d) * np.eye(3, 3) - np.outer(d, d))
                    mass += weight * volume

            self.I = I
            self.mass = mass
            self.com = com

        else:
            
            self.I = np.eye(3, dtype=np.float32)
            self.mass = 1.0
            self.com = np.array((0.0, 0.0, 0.0))



    # construct simulation ready buffers from points
    def finalize(self, device):

        pos = og.array(self.vertices, dtype=og.vec3, device=device)
        vel = og.zeros_like(pos)
        indices = og.array(self.indices, dtype=og.int32, device=device)

        self.mesh = og.Mesh(pos, vel, indices)
        return self.mesh.id


class State:
    """The State object holds all *time-varying* data for a model.
    
    Time-varying data includes particle positions, velocities, rigid body states, and
    anything that is output from the integrator as derived data, e.g.: forces. 
    
    The exact attributes depend on the contents of the model. State objects should
    generally be created using the :func:`Model.state()` function.

    Attributes:

        particle_q (og.array): Tensor of particle positions
        particle_qd (og.array): Tensor of particle velocities

        joint_q (og.array): Tensor of joint coordinates
        joint_qd (og.array): Tensor of joint velocities
        joint_act (og.array): Tensor of joint actuation values

    """

    def __init__(self):
        
        self.particle_count = 0
        self.link_count = 0

    # def flatten(self):
    #     """Returns a list of Tensors stored by the state

    #     This function is intended to be used internal-only but can be used to obtain
    #     a set of all tensors owned by the state.
    #     """

    #     tensors = []

    #     # particles
    #     if (self.particle_count):
    #         tensors.append(self.particle_q)
    #         tensors.append(self.particle_qd)

    #     # articulations
    #     if (self.link_count):
    #         tensors.append(self.joint_q)
    #         tensors.append(self.joint_qd)
    #         tensors.append(self.joint_act)

    #     return tensors


    def flatten(self):
        """Returns a list of Tensors stored by the state

        This function is intended to be used internal-only but can be used to obtain
        a set of all tensors owned by the state.
        """

        tensors = []

        # build a list of all tensor attributes
        for attr, value in self.__dict__.items():
            if (og.is_tensor(value)):
                tensors.append(value)

        return tensors


class Model:
    """Holds the definition of the simulation model

    This class holds the non-time varying description of the system, i.e.:
    all geometry, constraints, and parameters used to describe the simulation.

    Attributes:
        particle_q (og.array): Particle positions, shape [particle_count, 3], float
        particle_qd (og.array): Particle velocities, shape [particle_count, 3], float
        particle_mass (og.array): Particle mass, shape [particle_count], float
        particle_inv_mass (og.array): Particle inverse mass, shape [particle_count], float

        shape_transform (og.array): Rigid shape transforms, shape [shape_count, 7], float
        shape_body (og.array): Rigid shape body index, shape [shape_count], int
        shape_geo_type (og.array): Rigid shape geometry type, [shape_count], int
        shape_geo_src (og.array): Rigid shape geometry source, shape [shape_count], int
        shape_geo_scale (og.array): Rigid shape geometry scale, shape [shape_count, 3], float
        shape_materials (og.array): Rigid shape contact materials, shape [shape_count, 4], float

        spring_indices (og.array): Particle spring indices, shape [spring_count*2], int
        spring_rest_length (og.array): Particle spring rest length, shape [spring_count], float
        spring_stiffness (og.array): Particle spring stiffness, shape [spring_count], float
        spring_damping (og.array): Particle spring damping, shape [spring_count], float
        spring_control (og.array): Particle spring activation, shape [spring_count], float

        tri_indices (og.array): Triangle element indices, shape [tri_count*3], int
        tri_poses (og.array): Triangle element rest pose, shape [tri_count, 2, 2], float
        tri_activations (og.array): Triangle element activations, shape [tri_count], float

        edge_indices (og.array): Bending edge indices, shape [edge_count*2], int
        edge_rest_angle (og.array): Bending edge rest angle, shape [edge_count], float

        tet_indices (og.array): Tetrahedral element indices, shape [tet_count*4], int
        tet_poses (og.array): Tetrahedral rest poses, shape [tet_count, 3, 3], float
        tet_activations (og.array): Tetrahedral volumetric activations, shape [tet_count], float
        tet_materials (og.array): Tetrahedral elastic parameters in form :math:`k_{mu}, k_{lambda}, k_{damp}`, shape [tet_count, 3]
        
        body_X_cm (og.array): Rigid body center of mass (in local frame), shape [link_count, 7], float
        body_I_m (og.array): Rigid body inertia tensor (relative to COM), shape [link_count, 3, 3], float

        articulation_start (og.array): Articulation start offset, shape [num_articulations], int

        joint_q (og.array): Joint coordinate, shape [joint_coord_count], float
        joint_qd (og.array): Joint velocity, shape [joint_dof_count], float
        joint_type (og.array): Joint type, shape [joint_count], int
        joint_parent (og.array): Joint parent, shape [joint_count], int
        joint_X_pj (og.array): Joint transform in parent frame, shape [joint_count, 7], float
        joint_X_cm (og.array): Joint mass frame in child frame, shape [joint_count, 7], float
        joint_axis (og.array): Joint axis in child frame, shape [joint_count, 3], float
        joint_q_start (og.array): Joint coordinate offset, shape [joint_count], int
        joint_qd_start (og.array): Joint velocity offset, shape [joint_count], int

        joint_armature (og.array): Armature for each joint, shape [joint_count], float
        joint_target_ke (og.array): Joint stiffness, shape [joint_count], float
        joint_target_kd (og.array): Joint damping, shape [joint_count], float
        joint_target (og.array): Joint target, shape [joint_count], float

        particle_count (int): Total number of particles in the system
        joint_coord_count (int): Total number of joint coordinates in the system
        joint_dof_count (int): Total number of joint dofs in the system
        link_count (int): Total number of links in the system
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

    def __init__(self, device):

        self.particle_q = None
        self.particle_qd = None
        self.particle_mass = None
        self.particle_inv_mass = None

        self.shape_transform = None
        self.shape_body = None
        self.shape_geo_type = None
        self.shape_geo_src = None
        self.shape_geo_scale = None
        self.shape_materials = None

        self.spring_indices = None
        self.spring_rest_length = None
        self.spring_stiffness = None
        self.spring_damping = None
        self.spring_control = None

        self.tri_indices = None
        self.tri_poses = None
        self.tri_activations = None

        self.edge_indices = None
        self.edge_rest_angle = None

        self.tet_indices = None
        self.tet_poses = None
        self.tet_activations = None
        self.tet_materials = None
        
        self.body_X_cm = None
        self.body_I_m = None

        self.articulation_start = None

        self.joint_q = None
        self.joint_qd = None
        self.joint_type = None
        self.joint_parent = None
        self.joint_X_pj = None
        self.joint_X_cm = None
        self.joint_axis = None
        self.joint_q_start = None
        self.joint_qd_start = None

        self.joint_armature = None
        self.joint_target_ke = None
        self.joint_target_kd = None
        self.joint_target = None

        self.particle_count = 0
        self.joint_coord_count = 0
        self.joint_dof_count = 0
        self.link_count = 0
        self.shape_count = 0
        self.tri_count = 0
        self.tet_count = 0
        self.edge_count = 0
        self.spring_count = 0
        self.contact_count = 0

        self.gravity = np.array((0.0, -9.8, 0.0))

        self.soft_contact_distance = 0.1
        self.soft_contact_margin = 0.2
        self.soft_contact_ke = 1.e+3
        self.soft_contact_kd = 0.0
        self.soft_contact_kf = 1.e+3
        self.soft_contact_mu = 0.5

        self.tri_ke = 100.0
        self.tri_ka = 100.0
        self.tri_kd = 10.0
        self.tri_kb = 100.0
        self.tri_drag = 0.0
        self.tri_lift = 0.0

        self.edge_ke = 100.0
        self.edge_kd = 0.0

        self.particle_radius = 0.1
        self.device = device

    def state(self) -> State:
        """Returns a state object for the model

        The returned state will be initialized with the initial configuration given in
        the model description.
        """

        s = State()

        s.particle_count = self.particle_count
        s.link_count = self.link_count

        #--------------------------------
        # dynamic state (input, output)
          
        # particles
        if (self.particle_count):
            s.particle_q = og.clone(self.particle_q)
            s.particle_qd = og.clone(self.particle_qd)

        # articulations
        if (self.link_count):
            s.joint_q = og.clone(self.joint_q)
            s.joint_qd = og.clone(self.joint_qd)
            s.joint_act = og.zeros_like(self.joint_qd)

        #--------------------------------
        # derived state (output only)
        
        if (self.particle_count):
            s.particle_f = og.empty_like(self.particle_qd, requires_grad=True)

        if (self.link_count):

            # joints
            s.joint_qdd = og.empty_like(self.joint_qd, requires_grad=True)
            s.joint_tau = og.empty_like(self.joint_qd, requires_grad=True)
            s.joint_S_s = og.empty((self.joint_dof_count, 6), dtype=og.float32, device=self.device, requires_grad=True)            

            # rigids
            s.body_X_sc = og.empty((self.link_count, 7), dtype=og.float32, device=self.device, requires_grad=True)
            s.body_X_sm = og.empty((self.link_count, 7), dtype=og.float32, device=self.device, requires_grad=True)
            s.body_I_s = og.empty((self.link_count, 6, 6), dtype=og.float32, device=self.device, requires_grad=True)
            s.body_v_s = og.empty((self.link_count, 6), dtype=og.float32, device=self.device, requires_grad=True)
            s.body_a_s = og.empty((self.link_count, 6), dtype=og.float32, device=self.device, requires_grad=True)
            s.body_f_s = og.zeros((self.link_count, 6), dtype=og.float32, device=self.device, requires_grad=True)
            #s.body_ft_s = og.zeros((self.link_count, 6), dtype=og.float32, device=self.device, requires_grad=True)
            #s.body_f_ext_s = og.zeros((self.link_count, 6), dtype=og.float32, device=self.device, requires_grad=True)

            # system matrices
            s.M = og.zeros(self.M_size, dtype=og.float32, device=self.device, requires_grad=True)
            s.J = og.zeros(self.J_size, dtype=og.float32, device=self.device, requires_grad=True)
            s.P = og.empty(self.J_size, dtype=og.float32, device=self.device, requires_grad=True)
            s.H = og.empty(self.H_size, dtype=og.float32, device=self.device, requires_grad=True)
            
            # zero since only upper triangle is set which can trigger NaN detection
            s.L = og.zeros(self.H_size, dtype=og.float32, device=self.device, requires_grad=True)

        else:

            s.body_X_sc = None
            s.body_v_s = None

        return s

    def flatten(self):
        """Returns a list of Tensors stored by the model

        This function is intended to be used internal-only but can be used to obtain
        a set of all tensors owned by the model.
        """

        tensors = []

        # build a list of all tensor attributes
        for attr, value in self.__dict__.items():
            if (og.is_tensor(value)):
                tensors.append(value)

        return tensors

    # builds contacts
    def collide(self, state: State):
        """Constructs a set of contacts between rigid bodies and ground

        This method performs collision detection between rigid body vertices in the scene and updates
        the model's set of contacts stored as the following attributes:

            * **contact_body0**: Tensor of ints with first rigid body index 
            * **contact_body1**: Tensor of ints with second rigid body index (currently always -1 to indicate ground)
            * **contact_point0**: Tensor of Vec3 representing contact point in local frame of body0
            * **contact_dist**: Tensor of float values representing the distance to maintain
            * **contact_material**: Tensor contact material indices

        Args:
            state: The state of the simulation at which to perform collision detection

        Note:
            Currently this method uses an 'all pairs' approach to contact generation that is
            state indepdendent. In the future this will change and will create a node in
            the computational graph to propagate gradients as a function of state.

        Todo:

            Only ground-plane collision is currently implemented. Since the ground is static
            it is acceptable to call this method once at initialization time.
        """

        body0 = []
        body1 = []
        point = []
        dist = []
        mat = []

        def add_contact(b0, b1, t, p0, d, m):
            body0.append(b0)
            body1.append(b1)
            point.append(transform_point(t, np.array(p0)))
            dist.append(d)
            mat.append(m)

        for i in range(self.shape_count):

            # transform from shape to body
            X_bs = transform_expand(self.shape_transform[i].tolist())

            geo_type = self.shape_geo_type[i].item()

            if (geo_type == GEO_SPHERE):

                radius = self.shape_geo_scale[i][0].item()

                add_contact(self.shape_body[i], -1, X_bs, (0.0, 0.0, 0.0), radius, i)

            elif (geo_type == GEO_CAPSULE):

                radius = self.shape_geo_scale[i][0].item()
                half_width = self.shape_geo_scale[i][1].item()

                add_contact(self.shape_body[i], -1, X_bs, (-half_width, 0.0, 0.0), radius, i)
                add_contact(self.shape_body[i], -1, X_bs, (half_width, 0.0, 0.0), radius, i)

            elif (geo_type == GEO_BOX):

                edges = self.shape_geo_scale[i].tolist()

                add_contact(self.shape_body[i], -1, X_bs, (-edges[0], -edges[1], -edges[2]), 0.0, i)        
                add_contact(self.shape_body[i], -1, X_bs, ( edges[0], -edges[1], -edges[2]), 0.0, i)
                add_contact(self.shape_body[i], -1, X_bs, (-edges[0],  edges[1], -edges[2]), 0.0, i)
                add_contact(self.shape_body[i], -1, X_bs, (edges[0], edges[1], -edges[2]), 0.0, i)
                add_contact(self.shape_body[i], -1, X_bs, (-edges[0], -edges[1], edges[2]), 0.0, i)
                add_contact(self.shape_body[i], -1, X_bs, (edges[0], -edges[1], edges[2]), 0.0, i)
                add_contact(self.shape_body[i], -1, X_bs, (-edges[0], edges[1], edges[2]), 0.0, i)
                add_contact(self.shape_body[i], -1, X_bs, (edges[0], edges[1], edges[2]), 0.0, i)

            elif (geo_type == GEO_MESH):

                mesh = self.shape_geo_src[i]
                scale = self.shape_geo_scale[i]

                for v in mesh.vertices:

                    p = (v[0] * scale[0], v[1] * scale[1], v[2] * scale[2])

                    add_contact(self.shape_body[i], -1, X_bs, p, 0.0, i)

        # send to og
        self.contact_body0 = og.array(body0, dtype=og.int32, device=self.device)
        self.contact_body1 = og.array(body1, dtype=og.int32, device=self.device)
        self.contact_point0 = og.array(point, dtype=og.float32, device=self.device)
        self.contact_dist = og.array(dist, dtype=og.float32, device=self.device)
        self.contact_material = og.array(mat, dtype=og.int32, device=self.device)

        self.contact_count = len(body0)





class ModelBuilder:
    """A helper class for building simulation models at runtime.

    Use the ModelBuilder to construct a simulation scene. The ModelBuilder
    is independent of Pyog and builds the scene representation using
    standard Python data structures, this means it is not differentiable. Once :func:`finalize()` 
    has been called the ModelBuilder transfers all data to og tensors and returns 
    an object that may be used for simulation.

    Example:

        >>> import og as og
        >>>
        >>> builder = og.ModelBuilder()
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
        >>> model = builder.finalize()

    Note:
        It is strongly recommended to use the ModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if 
        desired.
    """
    
    def __init__(self):

        # particles
        self.particle_q = []
        self.particle_qd = []
        self.particle_mass = []

        # shapes
        self.shape_transform = []
        self.shape_body = []
        self.shape_geo_type = []
        self.shape_geo_scale = []
        self.shape_geo_src = []
        self.shape_materials = []

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

        # edges (bending)
        self.edge_indices = []
        self.edge_rest_angle = []

        # tetrahedra
        self.tet_indices = []
        self.tet_poses = []
        self.tet_activations = []
        self.tet_materials = []

        # muscles
        self.muscle_start = []
        self.muscle_params = []
        self.muscle_activation = []
        self.muscle_links = []
        self.muscle_points = []

        # rigid bodies
        self.joint_parent = []         # index of the parent body                      (constant)
        self.joint_child = []          # index of the child body                       (constant)
        self.joint_axis = []           # joint axis in child joint frame               (constant)
        self.joint_X_pj = []           # frame of joint in parent                      (constant)
        self.joint_X_cm = []           # frame of child com (in child coordinates)     (constant)

        self.joint_q_start = []        # joint offset in the q array
        self.joint_qd_start = []       # joint offset in the qd array
        self.joint_type = []
        self.joint_armature = []
        self.joint_target_ke = []
        self.joint_target_kd = []
        self.joint_target = []
        self.joint_limit_lower = []
        self.joint_limit_upper = []
        self.joint_limit_ke = []
        self.joint_limit_kd = []

        self.joint_q = []              # generalized coordinates       (input)
        self.joint_qd = []             # generalized velocities        (input)
        self.joint_qdd = []            # generalized accelerations     (id,fd)
        self.joint_tau = []            # generalized actuation         (input)
        self.joint_u = []              # generalized total torque      (fd)

        self.body_mass = []
        self.body_inertia = []
        self.body_com = []

        self.articulation_start = []

    def add_articulation(self) -> int:
        """Add an articulation object, all subsequently added links (see: :func:`add_link`) will belong to this articulation object. 
        Calling this method multiple times 'closes' any previous articulations and begins a new one.

        Returns:
            The index of the articulation
        """
        self.articulation_start.append(len(self.joint_type))
        return len(self.articulation_start)-1


    # rigids, register a rigid body and return its index.
    def add_link(
        self, 
        parent : int, 
        X_pj : Transform, 
        axis : Vec3, 
        type : int, 
        armature: float=0.0, 
        stiffness: float=0.0, 
        damping: float=0.0,
        limit_lower: float=-1.e+3,
        limit_upper: float=1.e+3,
        limit_ke: float=100.0,
        limit_kd: float=10.0,
        com: Vec3=np.zeros(3), 
        I_m: Mat33=np.zeros((3, 3)), 
        m: float=0.0) -> int:
        """Adds a rigid body to the model.

        Args:
            parent: The index of the parent body
            X_pj: The location of the joint in the parent's local frame connecting this body
            axis: The joint axis
            type: The type of joint, should be one of: JOINT_PRISMATIC, JOINT_REVOLUTE, JOINT_BALL, JOINT_FIXED, or JOINT_FREE
            armature: Additional inertia around the joint axis
            stiffness: Spring stiffness that attempts to return joint to zero position
            damping: Spring damping that attempts to remove joint velocity
            com: The center of mass of the body w.r.t its origin
            I_m: The 3x3 inertia tensor of the body (specified relative to the center of mass)
            m: The mass of the body

        Returns:
            The index of the body in the model

        Note:
            If the mass (m) is zero then the body is treated as kinematic with no dynamics

        """

        # joint data
        self.joint_type.append(type)
        self.joint_axis.append(np.array(axis))
        self.joint_parent.append(parent)
        self.joint_X_pj.append(X_pj)

        self.joint_target_ke.append(stiffness)
        self.joint_target_kd.append(damping)
        self.joint_limit_ke.append(limit_ke)
        self.joint_limit_kd.append(limit_kd)
        self.joint_armature.append(armature)

        self.joint_q_start.append(len(self.joint_q))
        self.joint_qd_start.append(len(self.joint_qd))

        if (type == JOINT_PRISMATIC):
            self.joint_q.append(0.0)
            self.joint_qd.append(0.0)
            self.joint_target.append(0.0)
            self.joint_limit_lower.append(limit_lower)
            self.joint_limit_upper.append(limit_upper)

        elif (type == JOINT_REVOLUTE):
            self.joint_q.append(0.0)
            self.joint_qd.append(0.0)
            self.joint_target.append(0.0)
            self.joint_limit_lower.append(limit_lower)
            self.joint_limit_upper.append(limit_upper)

        elif (type == JOINT_BALL):
            
            # quaternion
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(1.0)

            # angular velocity
            self.joint_qd.append(0.0)
            self.joint_qd.append(0.0)
            self.joint_qd.append(0.0)

            # pd targets
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)

            self.joint_limit_lower.append(limit_lower)
            self.joint_limit_lower.append(limit_lower)
            self.joint_limit_lower.append(limit_lower)
            self.joint_limit_lower.append(0.0)
                       
            self.joint_limit_upper.append(limit_upper)
            self.joint_limit_upper.append(limit_upper)
            self.joint_limit_upper.append(limit_upper)
            self.joint_limit_upper.append(0.0)


        elif (type == JOINT_FIXED):
            pass
        elif (type == JOINT_FREE):

            # translation
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)

            # quaternion
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(0.0)
            self.joint_q.append(1.0)

            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)
            self.joint_target.append(0.0)

            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)
            self.joint_limit_lower.append(0.0)
                       
            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)
            self.joint_limit_upper.append(0.0)

            # joint velocities
            for i in range(6):
                self.joint_qd.append(0.0)

        self.body_inertia.append(np.zeros((3, 3)))
        self.body_mass.append(0.0)
        self.body_com.append(np.zeros(3))

        # return index of body
        return len(self.joint_type) - 1


    # muscles
    def add_muscle(self, links: List[int], positions: List[Vec3], f0: float, lm: float, lt: float, lmax: float, pen: float) -> float:
        """Adds a muscle-tendon activation unit

        Args:
            links: A list of link indices for each waypoint
            positions: A list of positions of each waypoint in the link's local frame
            f0: Force scaling
            lm: Muscle length
            lt: Tendon length
            lmax: Maximally efficient muscle length

        Returns:
            The index of the muscle in the model

        """

        n = len(links)

        self.muscle_start.append(len(self.muscle_links))
        self.muscle_params.append((f0, lm, lt, lmax, pen))
        self.muscle_activation.append(0.0)

        for i in range(n):

            self.muscle_links.append(links[i])
            self.muscle_points.append(positions[i])

        # return the index of the muscle
        return len(self.muscle_start)-1

    # shapes
    def add_shape_plane(self, plane: Vec4=(0.0, 1.0, 0.0, 0.0), ke: float=1.e+5, kd: float=1000.0, kf: float=1000.0, mu: float=0.5):
        """Adds a plane collision shape

        Args:
            plane: The plane equation in form a*x + b*y + c*z + d = 0
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction

        """
        self._add_shape(-1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), GEO_PLANE, plane, None, 0.0, ke, kd, kf, mu)

    def add_shape_sphere(self, body, pos: Vec3=(0.0, 0.0, 0.0), rot: Quat=(0.0, 0.0, 0.0, 1.0), radius: float=1.0, density: float=1000.0, ke: float=1.e+5, kd: float=1000.0, kf: float=1000.0, mu: float=0.5):
        """Adds a sphere collision shape to a link.

        Args:
            body: The index of the parent link this shape belongs to
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            radius: The radius of the sphere
            density: The density of the shape
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction

        """

        self._add_shape(body, pos, rot, GEO_SPHERE, (radius, 0.0, 0.0, 0.0), None, density, ke, kd, kf, mu)

    def add_shape_box(self,
                      body : int,
                      pos: Vec3=(0.0, 0.0, 0.0),
                      rot: Quat=(0.0, 0.0, 0.0, 1.0),
                      hx: float=0.5,
                      hy: float=0.5,
                      hz: float=0.5,
                      density: float=1000.0,
                      ke: float=1.e+5,
                      kd: float=1000.0,
                      kf: float=1000.0,
                      mu: float=0.5):
        """Adds a box collision shape to a link.

        Args:
            body: The index of the parent link this shape belongs to
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

        """

        self._add_shape(body, pos, rot, GEO_BOX, (hx, hy, hz, 0.0), None, density, ke, kd, kf, mu)

    def add_shape_capsule(self,
                          body: int,
                          pos: Vec3=(0.0, 0.0, 0.0),
                          rot: Quat=(0.0, 0.0, 0.0, 1.0),
                          radius: float=1.0,
                          half_width: float=0.5,
                          density: float=1000.0,
                          ke: float=1.e+5,
                          kd: float=1000.0,
                          kf: float=1000.0,
                          mu: float=0.5):
        """Adds a capsule collision shape to a link.

        Args:
            body: The index of the parent link this shape belongs to
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            radius: The radius of the capsule
            half_width: The half length of the center cylinder along the x-axis
            density: The density of the shape
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction

        """

        self._add_shape(body, pos, rot, GEO_CAPSULE, (radius, half_width, 0.0, 0.0), None, density, ke, kd, kf, mu)

    def add_shape_mesh(self,
                       body: int,
                       pos: Vec3=(0.0, 0.0, 0.0),
                       rot: Quat=(0.0, 0.0, 0.0, 1.0),
                       mesh: Mesh=None,
                       scale: Vec3=(1.0, 1.0, 1.0),
                       density: float=1000.0,
                       ke: float=1.e+5,
                       kd: float=1000.0,
                       kf: float=1000.0,
                       mu: float=0.5):
        """Adds a triangle mesh collision shape to a link.

        Args:
            body: The index of the parent link this shape belongs to
            pos: The location of the shape with respect to the parent frame
            rot: The rotation of the shape with respect to the parent frame
            mesh: The mesh object
            scale: Scale to use for the collider
            density: The density of the shape
            ke: The contact elastic stiffness
            kd: The contact damping stiffness
            kf: The contact friction stiffness
            mu: The coefficient of friction

        """


        self._add_shape(body, pos, rot, GEO_MESH, (scale[0], scale[1], scale[2], 0.0), mesh, density, ke, kd, kf, mu)

    def _add_shape(self, body , pos, rot, type, scale, src, density, ke, kd, kf, mu):
        self.shape_body.append(body)
        self.shape_transform.append(og.transform(pos, rot))
        self.shape_geo_type.append(type)
        self.shape_geo_scale.append((scale[0], scale[1], scale[2]))
        self.shape_geo_src.append(src)
        self.shape_materials.append((ke, kd, kf, mu))

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

    def add_triangle(self, i : int, j : int, k : int) -> float:
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
        n = og.normalize(og.cross(qp, rp))
        e1 = og.normalize(qp)
        e2 = og.normalize(og.cross(n, e1))

        R = np.matrix((e1, e2))
        M = np.matrix((qp, rp))

        D = R * M.T
        inv_D = np.linalg.inv(D)

        area = np.linalg.det(D) / 2.0

        if (area < 0.0):
            print("inverted triangle element")

        self.tri_indices.append((i, j, k))
        self.tri_poses.append(inv_D.tolist())
        self.tri_activations.append(0.0)

        return area

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

        Dm = np.matrix((qp, rp, sp)).T
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

    def add_edge(self, i: int, j: int, k: int, l: int, rest: float=None):
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

            n1 = og.normalize(np.cross(x3 - x1, x4 - x1))
            n2 = og.normalize(np.cross(x4 - x2, x3 - x2))
            e = og.normalize(x4 - x3)

            d = np.clip(np.dot(n2, n1), -1.0, 1.0)

            angle = math.acos(d)
            sign = np.sign(np.dot(np.cross(n2, n1), e))

            rest = angle * sign

        self.edge_indices.append((i, j, k, l))
        self.edge_rest_angle.append(rest)

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
                       fix_bottom: bool=False):

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
                p = og.quat_rotate(rot, g) + pos
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

                        self.add_triangle(*tri1)
                        self.add_triangle(*tri2)

                    else:

                        tri1 = (start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                                start_vertex + grid_index(x, y - 1, dim_x + 1),
                                start_vertex + grid_index(x - 1, y, dim_x + 1))

                        tri2 = (start_vertex + grid_index(x, y - 1, dim_x + 1),
                                start_vertex + grid_index(x, y, dim_x + 1),
                                start_vertex + grid_index(x - 1, y, dim_x + 1))

                        self.add_triangle(*tri1)
                        self.add_triangle(*tri2)

        end_vertex = len(self.particle_q)
        end_tri = len(self.tri_indices)

        # bending constraints, could create these explicitly for a grid but this
        # is a good test of the adjacency structure
        adj = og.MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        for k, e in adj.edges.items():

            # skip open edges
            if (e.f0 == -1 or e.f1 == -1):
                continue

            self.add_edge(e.o0, e.o1, e.v0, e.v1)          # opposite 0, opposite 1, vertex 0, vertex 1

    def add_cloth_mesh(self, pos: Vec3, rot: Quat, scale: float, vel: Vec3, vertices: List[Vec3], indices: List[int], density: float, edge_callback=None, face_callback=None):
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
        for i, v in enumerate(vertices):

            p = og.quat_rotate(rot, v * scale) + pos

            self.add_particle(p, vel, 0.0)

        # triangles
        for t in range(num_tris):

            i = start_vertex + indices[t * 3 + 0]
            j = start_vertex + indices[t * 3 + 1]
            k = start_vertex + indices[t * 3 + 2]

            if (face_callback):
                face_callback(i, j, k)

            area = self.add_triangle(i, j, k)

            # add area fraction to particles
            if (area > 0.0):

                self.particle_mass[i] += density * area / 3.0
                self.particle_mass[j] += density * area / 3.0
                self.particle_mass[k] += density * area / 3.0

        end_vertex = len(self.particle_q)
        end_tri = len(self.tri_indices)

        adj = og.MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        # bend constraints
        for k, e in adj.edges.items():

            # skip open edges
            if (e.f0 == -1 or e.f1 == -1):
                continue

            if (edge_callback):
                edge_callback(e.f0, e.f1)

            self.add_edge(e.o0, e.o1, e.v0, e.v1)

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
                      fix_bottom: bool=False):
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

                    p = og.quat_rotate(rot, v) + pos

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
            self.add_triangle(v[0], v[1], v[2])

    def add_soft_mesh(self, pos: Vec3, rot: Quat, scale: float, vel: Vec3, vertices: List[Vec3], indices: List[int], density: float, k_mu: float, k_lambda: float, k_damp: float):
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

            p = og.quat_rotate(rot, v * scale) + pos

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
                self.add_triangle(v[0], v[1], v[2])
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
            s = scale[0]     # eventually want to compute moment of inertia for mesh.
            return (density * src.mass * s * s * s, density * src.I * s * s * s * s * s)

    
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

        new_inertia = transform_inertia(self.body_mass[i], self.body_inertia[i], com_offset, quat_identity()) + transform_inertia(
            m, I, shape_offset, q)

        self.body_mass[i] = new_mass
        self.body_inertia[i] = new_inertia
        self.body_com[i] = new_com

    # returns a (model, state) pair given the description
    def finalize(self, device: str) -> Model:
        """Convert this builder object to a concrete model for simulation.

        After building simulation elements this method should be called to transfer
        all data to Pyog tensors ready for simulation.

        Args:
            device: The simulation device to use, e.g.: 'cpu', 'cuda'

        Returns:

            A model object.
        """

        # construct particle inv masses
        particle_inv_mass = []
        for m in self.particle_mass:
            if (m > 0.0):
                particle_inv_mass.append(1.0 / m)
            else:
                particle_inv_mass.append(0.0)

        #-------------------------------------
        # construct Model (non-time varying) data

        m = Model(device)

        #---------------------        
        # particles

        # state (initial)
        m.particle_q = og.array(self.particle_q, dtype=og.vec3, device=device)
        m.particle_qd = og.array(self.particle_qd, dtype=og.vec3, device=device)

        # model 
        m.particle_mass = og.array(self.particle_mass, dtype=og.float32, device=device)
        m.particle_inv_mass = og.array(particle_inv_mass, dtype=og.float32, device=device)

        #---------------------
        # collision geometry
        m.shape_transform = og.array(og.transform_flatten_list(self.shape_transform), dtype=og.spatial_transform, device=device)
        m.shape_body = og.array(self.shape_body, dtype=og.int32, device=device)
        m.shape_geo_type = og.array(self.shape_geo_type, dtype=og.int32, device=device)
        m.shape_geo_src = self.shape_geo_src

        # build list of ids for geometry sources (meshes, sdfs)
        shape_geo_id = []
        for geo in self.shape_geo_src:
            if (geo):
                shape_geo_id.append(geo.finalize(device=device))
            else:
                shape_geo_id.append(-1)

        m.shape_geo_id = og.array(shape_geo_id, dtype=og.uint64, device=device)
        m.shape_geo_scale = og.array(self.shape_geo_scale, dtype=og.vec3, device=device)
        m.shape_materials = og.array(self.shape_materials, dtype=og.float32, device=device)

        #---------------------
        # springs

        m.spring_indices = og.array(self.spring_indices, dtype=og.int32, device=device)
        m.spring_rest_length = og.array(self.spring_rest_length, dtype=og.float32, device=device)
        m.spring_stiffness = og.array(self.spring_stiffness, dtype=og.float32, device=device)
        m.spring_damping = og.array(self.spring_damping, dtype=og.float32, device=device)
        m.spring_control = og.array(self.spring_control, dtype=og.float32, device=device)

        #---------------------
        # triangles

        m.tri_indices = og.array(self.tri_indices, dtype=og.int32, device=device)
        m.tri_poses = og.array(self.tri_poses, dtype=og.mat22, device=device)
        m.tri_activations = og.array(self.tri_activations, dtype=og.float32, device=device)

        #---------------------
        # edges

        m.edge_indices = og.array(self.edge_indices, dtype=og.int32, device=device)
        m.edge_rest_angle = og.array(self.edge_rest_angle, dtype=og.float32, device=device)

        #---------------------
        # tetrahedra

        m.tet_indices = og.array(self.tet_indices, dtype=og.int32, device=device)
        m.tet_poses = og.array(self.tet_poses, dtype=og.mat33, device=device)
        m.tet_activations = og.array(self.tet_activations, dtype=og.float32, device=device)
        m.tet_materials = og.array(self.tet_materials, dtype=og.float32, device=device)

        #-----------------------
        # muscles

        muscle_count = len(self.muscle_start)

        # close the muscle waypoint indices
        self.muscle_start.append(len(self.muscle_links))

        m.muscle_start = og.array(self.muscle_start, dtype=og.int32, device=device)
        m.muscle_params = og.array(self.muscle_params, dtype=og.float32, device=device)
        m.muscle_links = og.array(self.muscle_links, dtype=og.int32, device=device)
        m.muscle_points = og.array(self.muscle_points, dtype=og.float32, device=device)
        m.muscle_activation = og.array(self.muscle_activation, dtype=og.float32, device=device)

        #--------------------------------------
        # articulations

        # build 6x6 spatial inertia and COM transform
        body_X_cm = []
        body_I_m = [] 

        for i in range(len(self.body_inertia)):
            body_I_m.append(og.spatial_matrix_from_inertia(self.body_inertia[i], self.body_mass[i]))
            body_X_cm.append(og.transform(self.body_com[i], og.quat_identity()))
        
        m.body_I_m = og.array(body_I_m, dtype=og.float32, device=device)


        articulation_count = len(self.articulation_start)
        joint_coord_count = len(self.joint_q)
        joint_dof_count = len(self.joint_qd)

        # 'close' the start index arrays with a sentinel value
        self.joint_q_start.append(len(self.joint_q))
        self.joint_qd_start.append(len(self.joint_qd))
        self.articulation_start.append(len(self.joint_type))        

        # calculate total size and offsets of Jacobian and mass matrices for entire system
        m.J_size = 0
        m.M_size = 0
        m.H_size = 0

        articulation_J_start = []
        articulation_M_start = []
        articulation_H_start = []

        articulation_M_rows = []
        articulation_H_rows = []
        articulation_J_rows = []
        articulation_J_cols = []

        articulation_dof_start = []
        articulation_coord_start = []

        for i in range(articulation_count):

            first_joint = self.articulation_start[i]
            last_joint = self.articulation_start[i+1]

            first_coord = self.joint_q_start[first_joint]
            last_coord = self.joint_q_start[last_joint]

            first_dof = self.joint_qd_start[first_joint]
            last_dof = self.joint_qd_start[last_joint]

            joint_count = last_joint-first_joint
            dof_count = last_dof-first_dof
            coord_count = last_coord-first_coord

            articulation_J_start.append(m.J_size)
            articulation_M_start.append(m.M_size)
            articulation_H_start.append(m.H_size)
            articulation_dof_start.append(first_dof)
            articulation_coord_start.append(first_coord)

            # bit of data duplication here, but will leave it as such for clarity
            articulation_M_rows.append(joint_count*6)
            articulation_H_rows.append(dof_count)
            articulation_J_rows.append(joint_count*6)
            articulation_J_cols.append(dof_count)

            m.J_size += 6*joint_count*dof_count
            m.M_size += 6*joint_count*6*joint_count
            m.H_size += dof_count*dof_count
            

        m.articulation_joint_start = og.array(self.articulation_start, dtype=og.int32, device=device)

        # matrix offsets for batched gemm
        m.articulation_J_start = og.array(articulation_J_start, dtype=og.int32, device=device)
        m.articulation_M_start = og.array(articulation_M_start, dtype=og.int32, device=device)
        m.articulation_H_start = og.array(articulation_H_start, dtype=og.int32, device=device)
        
        m.articulation_M_rows = og.array(articulation_M_rows, dtype=og.int32, device=device)
        m.articulation_H_rows = og.array(articulation_H_rows, dtype=og.int32, device=device)
        m.articulation_J_rows = og.array(articulation_J_rows, dtype=og.int32, device=device)
        m.articulation_J_cols = og.array(articulation_J_cols, dtype=og.int32, device=device)

        m.articulation_dof_start = og.array(articulation_dof_start, dtype=og.int32, device=device)
        m.articulation_coord_start = og.array(articulation_coord_start, dtype=og.int32, device=device)

        # state (initial)
        m.joint_q = og.array(self.joint_q, dtype=og.float32, device=device)
        m.joint_qd = og.array(self.joint_qd, dtype=og.float32, device=device)

        # model
        m.joint_type = og.array(self.joint_type, dtype=og.int32, device=device)
        m.joint_parent = og.array(self.joint_parent, dtype=og.int32, device=device)
        m.joint_X_pj = og.array(og.transform_flatten_list(self.joint_X_pj), dtype=og.float32, device=device)
        m.joint_X_cm = og.array(og.transform_flatten_list(body_X_cm), dtype=og.float32, device=device)
        m.joint_axis = og.array(self.joint_axis, dtype=og.float32, device=device)
        m.joint_q_start = og.array(self.joint_q_start, dtype=og.int32, device=device) 
        m.joint_qd_start = og.array(self.joint_qd_start, dtype=og.int32, device=device)

        # dynamics properties
        m.joint_armature = og.array(self.joint_armature, dtype=og.float32, device=device)
        
        m.joint_target = og.array(self.joint_target, dtype=og.float32, device=device)
        m.joint_target_ke = og.array(self.joint_target_ke, dtype=og.float32, device=device)
        m.joint_target_kd = og.array(self.joint_target_kd, dtype=og.float32, device=device)

        m.joint_limit_lower = og.array(self.joint_limit_lower, dtype=og.float32, device=device)
        m.joint_limit_upper = og.array(self.joint_limit_upper, dtype=og.float32, device=device)
        m.joint_limit_ke = og.array(self.joint_limit_ke, dtype=og.float32, device=device)
        m.joint_limit_kd = og.array(self.joint_limit_kd, dtype=og.float32, device=device)

        # contacts
        m.soft_contact_max = 64*1024

        m.soft_contact_count = og.zeros(1, dtype=og.int32, device=device)
        m.soft_contact_particle = og.zeros(m.soft_contact_max, dtype=int, device=device)
        m.soft_contact_body = og.zeros(m.soft_contact_max, dtype=int, device=device)
        m.soft_contact_body_pos = og.zeros(m.soft_contact_max, dtype=og.vec3, device=device)
        m.soft_contact_body_vel = og.zeros(m.soft_contact_max, dtype=og.vec3, device=device)
        m.soft_contact_normal = og.zeros(m.soft_contact_max, dtype=og.vec3, device=device)

        # counts
        m.particle_count = len(self.particle_q)

        m.articulation_count = articulation_count
        m.joint_coord_count = joint_coord_count
        m.joint_dof_count = joint_dof_count
        m.muscle_count = muscle_count

        m.link_count = len(self.joint_type)        
        m.shape_count = len(self.shape_geo_type)
        m.tri_count = len(self.tri_poses)
        m.tet_count = len(self.tet_poses)
        m.edge_count = len(self.edge_rest_angle)
        m.spring_count = len(self.spring_rest_length)
        m.contact_count = 0
        
        # store refs to geometry
        m.geo_meshes = self.geo_meshes
        m.geo_sdfs = self.geo_sdfs

        # enable ground plane
        m.ground = True
        m.ground_plane = np.array((0.0, 1.0, 0.0, 0.0))

        m.enable_tri_collisions = False

        #-------------------------------------
        # construct generalized State (time-varying) vector

        return m
