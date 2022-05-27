# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from warp.sim.model import JOINT_COMPOUND, JOINT_REVOLUTE, JOINT_UNIVERSAL
from warp.utils import transform_identity

import math
import numpy as np
import os

import xml.etree.ElementTree as ET

import warp as wp
import warp.sim 

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
    limit_ke=10000.0,
    limit_kd=1000.0,
    armature=0.0,
    armature_scale=1.0):

    file = ET.parse(filename)
    root = file.getroot()

    # map node names to link indices
    node_map = {}
    xform_map = {}
    mesh_map = {}
    
    type_map = { 
        "ball": wp.sim.JOINT_BALL, 
        "hinge": wp.sim.JOINT_REVOLUTE, 
        "slide": wp.sim.JOINT_PRISMATIC, 
        "free": wp.sim.JOINT_FREE, 
        "fixed": wp.sim.JOINT_FIXED
    }


    def parse_float(node, key, default):
        if key in node.attrib:
            return float(node.attrib[key])
        else:
            return default

    def parse_vec(node, key, default):
        if key in node.attrib:
            return np.fromstring(node.attrib[key], sep=" ")
        else:
            return np.array(default)

    def parse_body(body, parent):

        body_name = body.attrib["name"]
        body_pos = np.fromstring(body.attrib["pos"], sep=" ")

        #-----------------
        # add body for each joint
        joints = body.findall("joint")

        if (parent == -1):
            body_pos = np.array((0.0, 0.0, 0.0))

        start_dof = builder.joint_dof_count
        start_coord = builder.joint_coord_count

        if (len(joints) == 1):

            joint = joints[0]

            # default to hinge if not specified
            if ("type" not in joint.attrib):
                joint.attrib["type"] = "hinge"

            joint_name = joint.attrib["name"],
            joint_type = type_map[joint.attrib["type"]]
            joint_axis = wp.normalize(parse_vec(joint, "axis", (0.0, 0.0, 0.0)))
            joint_pos = parse_vec(joint, "pos", (0.0, 0.0, 0.0))
            joint_range = parse_vec(joint, "range", (-3.0, 3.0))
            joint_armature = parse_float(joint, "armature", armature)*armature_scale
            joint_stiffness = parse_float(joint, "stiffness", stiffness)
            joint_damping = parse_float(joint, "damping", damping)

            link = builder.add_body(
                parent=parent,
                origin=wp.transform_identity(),  # will be evaluated in fk()
                joint_xform=wp.transform(body_pos, wp.quat_identity()), 
                joint_axis=joint_axis, 
                joint_type=joint_type,
                joint_limit_lower=np.deg2rad(joint_range[0]),
                joint_limit_upper=np.deg2rad(joint_range[1]),
                joint_limit_ke=limit_ke,
                joint_limit_kd=limit_kd,
                joint_target_ke=joint_stiffness,
                joint_target_kd=joint_damping,
                joint_armature=joint_armature)

            #print(f"{joint_name} coord: {start_coord} dof: {start_dof} body index: {link}")

        else:
            
            if (len(joints) == 2):
                type = JOINT_UNIVERSAL
            elif (len(joints) == 3):
                type = JOINT_COMPOUND
            else:
                raise RuntimeError("Bodies must have 1-3 joints")

            # universal / compound joint
            joint_stiffness = []
            joint_damping = []
            joint_lower = []
            joint_upper = []
            joint_armature = []
            joint_axis = []

            for i, joint in enumerate(joints):

                # default to hinge if not specified
                if ("type" not in joint.attrib):
                    joint.attrib["type"] = "hinge"
                
                if (joint.attrib["type"] != "hinge"):
                    print("Compound joints must all be hinges")

                joint_name = joint.attrib["name"],
                joint_pos = parse_vec(joint, "pos", (0.0, 0.0, 0.0))
                joint_range = parse_vec(joint, "range", (-3.0, 3.0))
                joint_lower.append(np.deg2rad(joint_range[0]))
                joint_upper.append(np.deg2rad(joint_range[1]))
                joint_armature.append(parse_float(joint, "armature", armature)*armature_scale)
                joint_stiffness.append(parse_float(joint, "stiffness", stiffness))
                joint_damping.append(parse_float(joint, "damping", damping))
                joint_axis.append(wp.normalize(parse_vec(joint, "axis", (0.0, 0.0, 0.0))))

            # align MuJoCo axes with joint coordinates

            if len(joints) == 2:
                M = np.array([joint_axis[0], joint_axis[1], wp.cross(joint_axis[0], joint_axis[1])]).T

            elif len(joints) == 3:
                M = np.array([joint_axis[0], joint_axis[1], joint_axis[2]]).T
            
            q = wp.quat_from_matrix(M)            

            link = builder.add_body(
                parent=parent,
                origin=wp.transform_identity(),  # will be evaluated in fk()
                joint_xform=wp.transform(body_pos, wp.quat_identity()),
                joint_xform_child=wp.transform([0.0, 0.0, 0.0], q),
                joint_type=type,
                joint_limit_lower=joint_lower,
                joint_limit_upper=joint_upper,
                joint_limit_ke=limit_ke,
                joint_limit_kd=limit_kd,
                joint_target_ke=joint_stiffness,
                joint_target_kd=joint_damping,
                joint_armature=joint_armature[0])
            
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

                    # compute rotation to align the Warp capsule (along x-axis), with mjcf fromto direction                        
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