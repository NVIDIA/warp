import xml.etree.ElementTree as ET

import numpy as np

import warp as wp
import warpsim

def parse_mjcf(
    filename, 
    builder, 
    density=1000.0, 
    stiffness=0.0, 
    damping=0.0, 
    contact_ke=1000.0,
    contact_kd=100.0,
    contact_kf=100.0,
    contact_mu=0.5,
    limit_ke=100.0,
    limit_kd=10.0):

    file = ET.parse(filename)
    root = file.getroot()

    # map node names to link indices
    self.node_map = {}
    self.xform_map = {}
    self.mesh_map = {}

    type_map = { 
        "ball": wp.JOINT_BALL, 
        "hinge": wp.JOINT_REVOLUTE, 
        "slide": wp.JOINT_PRISMATIC, 
        "free": wp.JOINT_FREE, 
        "fixed": wp.JOINT_FIXED
    }


    def parse_vec(node, key, default):
        if key in node.attrib:
            return np.fromstring(node.attrib[key], sep=" ")
        else:
            return np.array(default)

    def parse_body(body, parent):

        body_name = body.attrib["name"]
        body_pos = np.fromstring(body.attrib["pos"], sep=" ")

        # note: only supports single joint per-body
        joint = body.find("joint")
        
        
        joint_name = joint.attrib["name"],
        joint_type = type_map[joint.attrib["type"]]
        joint_axis = parse_vec(joint, "axis", (0.0, 0.0, 0.0))
        joint_pos = parse_vec(joint, "pos", (0.0, 0.0, 0.0))
        joint_range = parse_vec(joint, "range", (-3.0, 3.0))

        joint_axis = wp.normalize(joint_axis)

        if (parent == -1):
            body_pos = np.array((0.0, 0.0, 0.0))

        #-----------------
        # add body
        
        link = builder.add_link(
            parent, 
            X_pj=wp.transform(body_pos, wp.quat_identity()), 
            axis=joint_axis, 
            type=joint_type,
            limit_lower=np.deg2rad(joint_range[0]),
            limit_upper=np.deg2rad(joint_range[1]),
            limit_ke=limit_ke,
            limit_kd=limit_kd,
            stiffness=stiffness,
            damping=damping,
            armature=0.0)

        #-----------------
        # add shapes

        for geom in body.findall("geom"):
            
            geom_name = geom.attrib["name"]
            geom_type = geom.attrib["type"]

            geom_size = parse_vec(geom, "size", [1.0])                
            geom_pos = parse_vec(geom, "pos", (0.0, 0.0, 0.0)) 
            geom_rot = parse_vec(geom, "quat", (0.0, 0.0, 0.0, 1.0))

            if (geom_type == "sphere"):

                builder.add_shape_sphere(
                    link, 
                    pos=geom_pos, 
                    rot=geom_rot,
                    radius=geom_size[0],
                    density=density,
                    ke=contact_ke,
                    kd=contact_kd,
                    kf=contact_kf,
                    mu=contact_mu)

            elif (geom_type == "capsule"):

                if ("fromto" in geom.attrib):
                    geom_fromto = parse_vec(geom, "fromto", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

                    start = geom_fromto[0:3]
                    end = geom_fromto[3:6]

                    # compute rotation to align wplex capsule (along x-axis), with mjcf fromto direction                        
                    axis = wp.normalize(end-start)
                    angle = math.acos(np.dot(axis, (1.0, 0.0, 0.0)))
                    axis = wp.normalize(np.cross(axis, (1.0, 0.0, 0.0)))

                    geom_pos = (start + end)*0.5
                    geom_rot = wp.quat_from_axis_angle(axis, -angle)

                    geom_radius = geom_size[0]
                    geom_width = np.linalg.norm(end-start)*0.5

                else:

                    geom_radius = geom_size[0]
                    geom_width = geom_size[1]
                    geom_pos = parse_vec(geom, "pos", (0.0, 0.0, 0.0))


                builder.add_shape_capsule(
                    link,
                    pos=geom_pos,
                    rot=geom_rot,
                    radius=geom_radius,
                    half_width=geom_width,
                    density=density,
                    ke=contact_ke,
                    kd=contact_kd,
                    kf=contact_kf,
                    mu=contact_mu)
                
            else:
                print("Type: " + geom_type + " unsupported")
            

        #-----------------
        # recurse

        for child in body.findall("body"):
            parse_body(child, link)


    #-----------------
    # start articulation

    builder.add_articulation()

    world = root.find("worldbody")
    for body in world.findall("body"):
        parse_body(body, -1)

