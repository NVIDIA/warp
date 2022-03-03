# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import math
import numpy as np
import os

import xml.etree.ElementTree as ET

import warp as wp

# SNU file format parser

class MuscleUnit:

    def __init__(self):
        
        self.name = ""
        self.bones = []
        self.points = []

class Skeleton:

    def __init__(self, root_xform, skeleton_file, muscle_file, builder, filter, armature=0.0):

        self.parse_skeleton(skeleton_file, builder, filter, root_xform, armature)
        self.parse_muscles(muscle_file, builder)

    def parse_skeleton(self, filename, builder, filter, root_xform, armature):
        file = ET.parse(filename)
        root = file.getroot()
        
        self.node_map = {}       # map node names to link indices
        self.xform_map = {}      # map node names to parent transforms
        self.mesh_map = {}       # map mesh names to link indices objects

        self.coord_start = builder.joint_coord_count
        self.dof_start = builder.joint_dof_count
    
        type_map = { 
            "Ball": wp.sim.JOINT_BALL, 
            "Revolute": wp.sim.JOINT_REVOLUTE, 
            "Prismatic": wp.sim.JOINT_PRISMATIC, 
            "Free": wp.sim.JOINT_FREE, 
            "Fixed": wp.sim.JOINT_FIXED
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
                
                joint_lower = np.array([-1.e+3])
                joint_upper = np.array([1.e+3])

                try:
                    joint_lower = np.fromstring(joint.attrib["lower"], sep=" ")
                    joint_upper = np.fromstring(joint.attrib["upper"], sep=" ")
                except:
                    pass

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
                    link = builder.add_body(
                        parent=parent_link, 
                        origin=wp.transform_multiply(root_xform, joint_X_s),
                        joint_xform=joint_X_p,
                        joint_axis=joint_axis,
                        joint_type=joint_type,
                        joint_target_ke=5.0,
                        joint_target_kd=2.0,
                        joint_limit_lower=joint_lower[0],
                        joint_limit_upper=joint_upper[0],
                        joint_limit_ke=1.e+3,
                        joint_limit_kd=1.e+2,
                        joint_armature=armature)

                    # add shape
                    shape = builder.add_shape_box(
                        body=link, 
                        pos=body_X_c.p,
                        rot=body_X_c.q,
                        hx=body_size[0]*0.5,
                        hy=body_size[1]*0.5,
                        hz=body_size[2]*0.5,
                        ke=1.e+3*5.0,
                        kd=1.e+2*2.0,
                        kf=1.e+3,
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




def parse_snu(root_xform, skeleton_file, muscle_file, builder, filter, armature=0.0):
    return Skeleton(root_xform, skeleton_file, muscle_file, builder, filter, armature=0.0)

