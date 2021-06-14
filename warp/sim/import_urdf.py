
import urdfpy
import math
import numpy as np
import os

import xml.etree.ElementTree as ET

import warp as wp

def urwp_add_collision(builder, link, collisions, shape_ke, shape_kd, shape_kf, shape_mu):
        
    # add geometry
    for collision in collisions:
        
        origin = urdfpy.matrix_to_xyz_rpy(collision.origin)

        pos = origin[0:3]
        rot = wp.rpy2quat(*origin[3:6])

        geo = collision.geometry

        if (geo.box):
            builder.add_shape_box(
                link,
                pos, 
                rot, 
                geo.box.size[0]*0.5, 
                geo.box.size[1]*0.5, 
                geo.box.size[2]*0.5,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)
        
        if (geo.sphere):
            builder.add_shape_sphere(
                link, 
                pos, 
                rot, 
                geo.sphere.radius,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)
         
        if (geo.cylinder):
            
            # cylinders in URwp are aligned with z-axis, while wplex uses x-axis
            r = wp.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.5)

            builder.add_shape_capsule(
                link, 
                pos, 
                wp.quat_multiply(rot, r), 
                geo.cylinder.radius, 
                geo.cylinder.length*0.5,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)
 
        if (geo.mesh):

            for m in geo.mesh.meshes:
                faces = []
                vertices = []

                for v in m.vertices:
                    vertices.append(np.array(v))
                    
                for f in m.faces:
                    faces.append(int(f[0]))
                    faces.append(int(f[1]))
                    faces.append(int(f[2]))
                                    
                mesh = wp.Mesh(vertices, faces)
                
                builder.add_shape_mesh(
                    link,
                    pos,
                    rot,
                    mesh,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                    mu=shape_mu)

def urwp_load(
    builder, 
    filename, 
    xform, 
    floating=False, 
    armature=0.0, 
    shape_ke=1.e+4, 
    shape_kd=1.e+3, 
    shape_kf=1.e+2, 
    shape_mu=0.25,
    limit_ke=100.0,
    limit_kd=10.0):

    robot = urdfpy.URwp.load(filename)

    # maps from link name -> link index
    link_index = {}

    builder.add_articulation()

    # add base
    if (floating):
        root = builder.add_link(-1, wp.transform_identity(), (0,0,0), wp.JOINT_FREE)

        # set dofs to transform
        start = builder.joint_q_start[root]

        builder.joint_q[start + 0] = xform[0][0]
        builder.joint_q[start + 1] = xform[0][1]
        builder.joint_q[start + 2] = xform[0][2]

        builder.joint_q[start + 3] = xform[1][0]
        builder.joint_q[start + 4] = xform[1][1]
        builder.joint_q[start + 5] = xform[1][2]
        builder.joint_q[start + 6] = xform[1][3]
    else:    
        root = builder.add_link(-1, xform, (0,0,0), wp.JOINT_FIXED)

    urwp_add_collision(builder, root, robot.links[0].collisions, shape_ke, shape_kd, shape_kf, shape_mu)
    link_index[robot.links[0].name] = root

    # add children
    for joint in robot.joints:

        type = None
        axis = (0.0, 0.0, 0.0)

        if (joint.joint_type == "revolute" or joint.joint_type == "continuous"):
            type = wp.JOINT_REVOLUTE
            axis = joint.axis
        if (joint.joint_type == "prismatic"):
            type = wp.JOINT_PRISMATIC
            axis = joint.axis
        if (joint.joint_type == "fixed"):
            type = wp.JOINT_FIXED
        if (joint.joint_type == "floating"):
            type = wp.JOINT_FREE
        
        parent = -1

        if joint.parent in link_index:
            parent = link_index[joint.parent]

        origin = urdfpy.matrix_to_xyz_rpy(joint.origin)

        pos = origin[0:3]
        rot = wp.rpy2quat(*origin[3:6])

        lower = -1.e+3
        upper = 1.e+3
        damping = 0.0

        # limits
        if (joint.limit):
            
            if (joint.limit.lower != None):
                lower = joint.limit.lower
            if (joint.limit.upper != None):
                upper = joint.limit.upper

        # damping
        if (joint.dynamics):
            if (joint.dynamics.damping):
                damping = joint.dynamics.damping

        # add link
        link = builder.add_link(
            parent=parent, 
            X_pj=wp.transform(pos, rot), 
            axis=axis, 
            type=type,
            limit_lower=lower,
            limit_upper=upper,
            limit_ke=limit_ke,
            limit_kd=limit_kd,
            damping=damping)
       
        # add collisions
        urwp_add_collision(builder, link, robot.link_map[joint.child].collisions, shape_ke, shape_kd, shape_kf, shape_mu)

        # add ourselves to the index
        link_index[joint.child] = link





# build an articulated tree
def build_tree(
    builder, 
    angle,
    max_depth,    
    width=0.05,
    length=0.25,
    density=1000.0,
    joint_stiffness=0.0,
    joint_damping=0.0,
    shape_ke = 1.e+4,
    shape_kd = 1.e+3,
    shape_kf = 1.e+2,
    shape_mu = 0.5,
    floating=False):


    def build_recursive(parent, depth):

        if (depth >= max_depth):
            return

        X_pj = wp.transform((length * 2.0, 0.0, 0.0), wp.quat_from_axis_angle((0.0, 0.0, 1.0), angle))

        type = wp.JOINT_REVOLUTE
        axis = (0.0, 0.0, 1.0)

        if (depth == 0 and floating == True):
            X_pj = wp.transform((0.0, 0.0, 0.0), wp.quat_identity())
            type = wp.JOINT_FREE

        link = builder.add_link(
            parent, 
            X_pj, 
            axis, 
            type,
            stiffness=joint_stiffness,
            damping=joint_damping)
        
        # box
        # shape = builder.add_shape_box(
        #     link, 
        #     pos=(length, 0.0, 0.0),
        #     hx=length, 
        #     hy=width, 
        #     hz=width,
        #     ke=shape_ke,
        #     kd=shape_kd,
        #     kf=shape_kf,
        #     mu=shape_mu)
        
        # capsule
        shape = builder.add_shape_capsule(
            link, 
            pos=(length, 0.0, 0.0), 
            radius=width, 
            half_width=length, 
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
            mu=shape_mu)

        # recurse
        #build_tree_recursive(builder, link, angle, width, depth + 1, max_depth, shape_ke, shape_kd, shape_kf, shape_mu, floating)
        build_recursive(link, depth + 1)

    # 
    build_recursive(-1, 0)



# SNU file format parser

class MuscleUnit:

    def __init__(self):
        
        self.name = ""
        self.bones = []
        self.points = []

class Skeleton:

    def __init__(self, skeleton_file, muscle_file, builder, filter):

        self.parse_skeleton(skeleton_file, builder, filter)
        self.parse_muscles(muscle_file, builder)

    def parse_skeleton(self, filename, builder, filter):
        file = ET.parse(filename)
        root = file.getroot()
        
        self.node_map = {}       # map node names to link indices
        self.xform_map = {}      # map node names to parent transforms
        self.mesh_map = {}       # map mesh names to link indices objects

        self.coord_start = len(builder.joint_q)
        self.dof_start = len(builder.joint_qd)

    
        type_map = { 
            "Ball": wp.JOINT_BALL, 
            "Revolute": wp.JOINT_REVOLUTE, 
            "Prismatic": wp.JOINT_PRISMATIC, 
            "Free": wp.JOINT_FREE, 
            "Fixed": wp.JOINT_FIXED
        }

        builder.add_articulation()

        for child in root:

            if (child.tag == "Node"):

                body = child.find("Body")
                joint = child.find("Joint")

                name = child.attrib["name"]
                parent = child.attrib["parent"]
                parent_X_s = wp.transform_identity()

                if parent in self.node_map:
                    parent_link = self.node_map[parent]
                    parent_X_s = self.xform_map[parent]
                else:
                    parent_link = -1

                body_xform = body.find("Transformation")
                joint_xform = joint.find("Transformation")

                body_mesh = body.attrib["obj"]
                body_size = np.fromstring(body.attrib["size"], sep=" ")
                body_type = body.attrib["type"]
                body_mass = body.attrib["mass"]

                body_R_s = np.fromstring(body_xform.attrib["linear"], sep=" ").reshape((3,3))
                body_t_s = np.fromstring(body_xform.attrib["translation"], sep=" ")

                joint_R_s = np.fromstring(joint_xform.attrib["linear"], sep=" ").reshape((3,3))
                joint_t_s = np.fromstring(joint_xform.attrib["translation"], sep=" ")
            
                joint_type = type_map[joint.attrib["type"]]
                
                #joint_lower = np.fromstring(joint.attrib["lower"], sep=" ")
                #joint_uppper = np.fromstring(joint.attrib["upper"], sep=" ")

                if ("axis" in joint.attrib):
                    joint_axis = np.fromstring(joint.attrib["axis"], sep=" ")
                else:
                    joint_axis = np.array((0.0, 0.0, 0.0))

                body_X_s = wp.transform(body_t_s, wp.quat_from_matrix(body_R_s))
                joint_X_s = wp.transform(joint_t_s, wp.quat_from_matrix(joint_R_s))

                mesh_base = os.path.splitext(body_mesh)[0]
                mesh_file = mesh_base + ".usd"

                #-----------------------------------
                # one time conversion, put meshes into local body space (and meter units)

                # stage = Usd.Stage.Open("./assets/snu/OBJ/" + mesh_file)
                # geom = UsdGeom.Mesh.Get(stage, "/" + mesh_base + "_obj/defaultobject/defaultobject")

                # body_X_bs = wp.transform_inverse(body_X_s)
                # joint_X_bs = wp.transform_inverse(joint_X_s)

                # points = geom.GetPointsAttr().Get()
                # for i in range(len(points)):

                #     p = wp.transform_point(joint_X_bs, points[i]*0.01)
                #     points[i] = Gf.Vec3f(p.tolist())  # cm -> meters
                

                # geom.GetPointsAttr().Set(points)

                # extent = UsdGeom.Boundable.ComputeExtentFromPlugins(geom, 0.0)
                # geom.GetExtentAttr().Set(extent)
                # stage.Save()
                
                #--------------------------------------
                link = -1

                if len(filter) == 0 or name in filter:

                    joint_X_p = wp.transform_multiply(wp.transform_inverse(parent_X_s), joint_X_s)
                    body_X_c = wp.transform_multiply(wp.transform_inverse(joint_X_s), body_X_s)

                    if (parent_link == -1):
                        joint_X_p = wp.transform_identity()

                    # add link
                    link = builder.add_link(
                        parent=parent_link, 
                        X_pj=joint_X_p,
                        axis=joint_axis,
                        type=joint_type,
                        damping=2.0,
                        stiffness=10.0)

                    # add shape
                    shape = builder.add_shape_box(
                        body=link, 
                        pos=body_X_c[0],
                        rot=body_X_c[1],
                        hx=body_size[0]*0.5,
                        hy=body_size[1]*0.5,
                        hz=body_size[2]*0.5,
                        ke=1.e+3*5.0,
                        kd=1.e+2*2.0,
                        kf=1.e+2,
                        mu=0.5)

                # add lookup in name->link map
                # save parent transform
                self.xform_map[name] = joint_X_s
                self.node_map[name] = link
                self.mesh_map[mesh_base] = link

    def parse_muscles(self, filename, builder):

        # list of MuscleUnits
        muscles = []

        file = ET.parse(filename)
        root = file.getroot()

        self.muscle_start = len(builder.muscle_activation)

        for child in root:

                if (child.tag == "Unit"):

                    unit_name = child.attrib["name"]
                    unit_f0 = float(child.attrib["f0"])
                    unit_lm = float(child.attrib["lm"])
                    unit_lt = float(child.attrib["lt"])
                    unit_lmax = float(child.attrib["lmax"])
                    unit_pen = float(child.attrib["pen_angle"])

                    m = MuscleUnit()
                    m.name = unit_name

                    incomplete = False

                    for waypoint in child.iter("Waypoint"):
                    
                        way_bone = waypoint.attrib["body"]
                        way_link = self.node_map[way_bone]
                        way_loc = np.fromstring(waypoint.attrib["p"], sep=" ", dtype=np.float32)

                        if (way_link == -1):
                            incomplete = True
                            break

                        # transform loc to joint local space
                        joint_X_s = self.xform_map[way_bone]

                        way_loc = wp.transform_point(wp.transform_inverse(joint_X_s), way_loc)

                        m.bones.append(way_link)
                        m.points.append(way_loc)

                    if not incomplete:

                        muscles.append(m)
                        builder.add_muscle(m.bones, m.points, f0=unit_f0, lm=unit_lm, lt=unit_lt, lmax=unit_lmax, pen=unit_pen)

        self.muscles = muscles



