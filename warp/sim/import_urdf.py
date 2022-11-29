# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


try:
    import urdfpy
except:
    pass

import math
import numpy as np
import os
import xml.etree.ElementTree as ET

import warp as wp
from warp.sim.model import Mesh


def urdf_add_collision(builder, link, collisions, density, shape_ke, shape_kd, shape_kf, shape_mu, shape_restitution):

    # add geometry
    for collision in collisions:

        origin = urdfpy.matrix_to_xyz_rpy(collision.origin)

        pos = origin[0:3]
        rot = wp.quat_rpy(*origin[3:6])

        geo = collision.geometry

        if geo.box:
            builder.add_shape_box(
                body=link,
                pos=pos,
                rot=rot,
                hx=geo.box.size[0]*0.5,
                hy=geo.box.size[1]*0.5,
                hz=geo.box.size[2]*0.5,
                density=density,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu,
                restitution=shape_restitution)

        if geo.sphere:
            builder.add_shape_sphere(
                body=link,
                pos=pos,
                rot=rot,
                radius=geo.sphere.radius,
                density=density,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu,
                restitution=shape_restitution)

        if geo.cylinder:

            # cylinders in URDF are aligned with z-axis, while Warp uses x-axis
            r = wp.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.5)

            builder.add_shape_capsule(
                body=link,
                pos=pos,
                rot=wp.mul(rot, r),
                radius=geo.cylinder.radius,
                half_width=geo.cylinder.length*0.5,
                density=density,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu,
                restitution=shape_restitution)

        if geo.mesh:

            for m in geo.mesh.meshes:
                faces = list(np.array(m.faces).astype('int').flatten())
                vertices = np.array(m.vertices, dtype=np.float32).reshape((-1, 3))
                if geo.mesh.scale is not None:
                    vertices *= geo.mesh.scale
                mesh = Mesh(vertices, faces)
                builder.add_shape_mesh(
                    body=link,
                    pos=pos,
                    rot=rot,
                    mesh=mesh,
                    density=density,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                    mu=shape_mu,
                    restitution=shape_restitution)

def parse_urdf(
        filename,
        builder,
        xform,
        floating=False,
        density=0.0,
        stiffness=100.0,
        damping=10.0,
        armature=0.0,
        shape_ke=1.e+4,
        shape_kd=1.e+3,
        shape_kf=1.e+2,
        shape_mu=0.25,
        shape_restitution=0.5,
        limit_ke=100.0,
        limit_kd=10.0,
        parse_visuals_as_colliders=False,
        enable_self_collisions=True):

    robot = urdfpy.URDF.load(filename)

    # maps from link name -> link index
    link_index = {}

    builder.add_articulation()
    
    start_shape_count = len(builder.shape_geo_type)

    # import inertial properties from URDF if density is zero
    if density == 0.0:
        com = urdfpy.matrix_to_xyz_rpy(robot.base_link.inertial.origin)[0:3]
        I_m = robot.base_link.inertial.inertia
        m = robot.base_link.inertial.mass
    else:
        com = np.zeros(3)
        I_m = np.zeros((3, 3))
        m = 0.0

    if parse_visuals_as_colliders:
        colliders = robot.links[0].visuals
    else:
        colliders = robot.links[0].collisions

    # add base
    if floating:
        root = builder.add_body(origin=wp.transform_identity(),
                                parent=-1,
                                joint_type=wp.sim.JOINT_FREE,
                                joint_armature=armature,
                                com=com,
                                I_m=I_m,
                                m=m,
                                body_name=robot.base_link.name,
                                joint_name="floating_base")

        # set dofs to transform
        start = builder.joint_q_start[root]

        builder.joint_q[start + 0] = xform.p[0]
        builder.joint_q[start + 1] = xform.p[1]
        builder.joint_q[start + 2] = xform.p[2]

        builder.joint_q[start + 3] = xform.q[0]
        builder.joint_q[start + 4] = xform.q[1]
        builder.joint_q[start + 5] = xform.q[2]
        builder.joint_q[start + 6] = xform.q[3]
        # make sure we do not reset inertia to zero if density is zero where we use the inertia from the URDF
        actual_density = 1.0 if m > 0.0 and density == 0.0 else density
        urdf_add_collision(
            builder, root, colliders, actual_density, shape_ke, shape_kd, shape_kf, shape_mu, shape_restitution)
        
    else:
        root = builder.add_body(origin=wp.transform_identity(),
                                parent=-1,
                                joint_xform=xform,
                                joint_type=wp.sim.JOINT_FIXED,
                                body_name=robot.base_link.name,
                                joint_name="fixed_base")
        urdf_add_collision(
            builder, root, colliders, 0.0, shape_ke, shape_kd, shape_kf, shape_mu, shape_restitution)

    link_index[robot.links[0].name] = root

    # add children
    for joint in robot.joints:

        type = None
        axis = (0.0, 0.0, 0.0)

        if joint.joint_type == "revolute" or joint.joint_type == "continuous":
            type = wp.sim.JOINT_REVOLUTE
            axis = joint.axis
        if joint.joint_type == "prismatic":
            type = wp.sim.JOINT_PRISMATIC
            axis = joint.axis
        if joint.joint_type == "fixed":
            type = wp.sim.JOINT_FIXED
        if joint.joint_type == "floating":
            type = wp.sim.JOINT_FREE

        parent = root
        if joint.parent in link_index:
            parent = link_index[joint.parent]

        origin = urdfpy.matrix_to_xyz_rpy(joint.origin)
        pos = origin[0:3]
        rot = wp.quat_rpy(*origin[3:6])

        lower = -1.e+3
        upper = 1.e+3
        joint_damping = damping

        # limits
        if joint.limit:
            if joint.limit.lower != None:
                lower = joint.limit.lower
            if joint.limit.upper != None:
                upper = joint.limit.upper

        # overwrite damping if defined in URDF
        if joint.dynamics:
            if joint.dynamics.damping:
                joint_damping = joint.dynamics.damping

        if density == 0.0:
            com = urdfpy.matrix_to_xyz_rpy(robot.link_map[joint.child].inertial.origin)[0:3]
            I_m = robot.link_map[joint.child].inertial.inertia
            m = robot.link_map[joint.child].inertial.mass
        else:
            com = np.zeros(3)
            I_m = np.zeros((3, 3))
            m = 0.0

        # add link
        link = builder.add_body(
            origin=wp.transform_identity(),
            parent=parent,
            joint_xform=wp.transform(pos, rot),
            joint_axis=axis,
            joint_type=type,
            joint_limit_lower=lower,
            joint_limit_upper=upper,
            joint_limit_ke=limit_ke,
            joint_limit_kd=limit_kd,
            joint_target_ke=stiffness,
            joint_target_kd=joint_damping,
            joint_armature=armature,
            com=com,
            I_m=I_m,
            m=m,
            joint_name=joint.name,
            body_name=joint.parent)

        if parse_visuals_as_colliders:
            child_colliders = robot.link_map[joint.child].visuals
        else:
            child_colliders = robot.link_map[joint.child].collisions

        # add collisions
        urdf_add_collision(
            builder, link, child_colliders, density, shape_ke, shape_kd, shape_kf, shape_mu, shape_restitution)

        # add ourselves to the index
        link_index[joint.child] = link
    
    end_shape_count = len(builder.shape_geo_type)

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            for j in range(i+1, end_shape_count):
                builder.shape_collision_filter_pairs.add((i, j))
