
try:
    import urdfpy
except:
    pass

import math
import numpy as np
import os

import xml.etree.ElementTree as ET

import warp as wp

def urdf_add_collision(builder, link, collisions, shape_ke, shape_kd, shape_kf, shape_mu):
        
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
            
            # cylinders in URDF are aligned with z-axis, while Warp uses x-axis
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

def parse_urdf(
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

    robot = urdfpy.urdf.load(filename)

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

    urdf_add_collision(builder, root, robot.links[0].collisions, shape_ke, shape_kd, shape_kf, shape_mu)
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
        urdf_add_collision(builder, link, robot.link_map[joint.child].collisions, shape_ke, shape_kd, shape_kf, shape_mu)

        # add ourselves to the index
        link_index[joint.child] = link


