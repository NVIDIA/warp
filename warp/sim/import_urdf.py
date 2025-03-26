# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import xml.etree.ElementTree as ET
from typing import Union

import numpy as np

import warp as wp
from warp.sim.model import Mesh


def parse_urdf(
    urdf_filename,
    builder,
    xform=None,
    floating=False,
    base_joint: Union[dict, str, None] = None,
    density=1000.0,
    stiffness=100.0,
    damping=10.0,
    armature=0.0,
    contact_ke=1.0e4,
    contact_kd=1.0e3,
    contact_kf=1.0e2,
    contact_ka=0.0,
    contact_mu=0.25,
    contact_restitution=0.5,
    contact_thickness=0.0,
    limit_ke=100.0,
    limit_kd=10.0,
    joint_limit_lower=-1e6,
    joint_limit_upper=1e6,
    scale=1.0,
    hide_visuals=False,
    parse_visuals_as_colliders=False,
    force_show_colliders=False,
    enable_self_collisions=True,
    ignore_inertial_definitions=True,
    ensure_nonstatic_links=True,
    static_link_mass=1e-2,
    collapse_fixed_joints=False,
):
    """
    Parses a URDF file and adds the bodies and joints to the given ModelBuilder.

    Args:
        urdf_filename (str): The filename of the URDF file to parse.
        builder (ModelBuilder): The :class:`ModelBuilder` to add the bodies and joints to.
        xform (:ref:`transform <transform>`): The transform to apply to the root body.
        floating (bool): If True, the root body is a free joint. If False, the root body is connected via a fixed joint to the world, unless a `base_joint` is defined.
        base_joint (Union[str, dict]): The joint by which the root body is connected to the world. This can be either a string defining the joint axes of a D6 joint with comma-separated positional and angular axis names (e.g. "px,py,rz" for a D6 joint with linear axes in x, y and an angular axis in z) or a dict with joint parameters (see :meth:`ModelBuilder.add_joint`).
        density (float): The density of the shapes in kg/m^3 which will be used to calculate the body mass and inertia.
        stiffness (float): The stiffness of the joints.
        damping (float): The damping of the joints.
        armature (float): The armature of the joints (bias to add to the inertia diagonals that may stabilize the simulation).
        contact_ke (float): The stiffness of the shape contacts (used by the Euler integrators).
        contact_kd (float): The damping of the shape contacts (used by the Euler integrators).
        contact_kf (float): The friction stiffness of the shape contacts (used by the Euler integrators).
        contact_ka (float): The adhesion distance of the shape contacts (used by the Euler integrators).
        contact_mu (float): The friction coefficient of the shape contacts.
        contact_restitution (float): The restitution coefficient of the shape contacts.
        contact_thickness (float): The thickness to add to the shape geometry.
        limit_ke (float): The stiffness of the joint limits (used by the Euler integrators).
        limit_kd (float): The damping of the joint limits (used by the Euler integrators).
        joint_limit_lower (float): The default lower joint limit if not specified in the URDF.
        joint_limit_upper (float): The default upper joint limit if not specified in the URDF.
        scale (float): The scaling factor to apply to the imported mechanism.
        hide_visuals (bool): If True, hide visual shapes.
        parse_visuals_as_colliders (bool): If True, the geometry defined under the `<visual>` tags is used for collision handling instead of the `<collision>` geometries.
        force_show_colliders (bool): If True, the collision shapes are always shown, even if there are visual shapes.
        enable_self_collisions (bool): If True, self-collisions are enabled.
        ignore_inertial_definitions (bool): If True, the inertial parameters defined in the URDF are ignored and the inertia is calculated from the shape geometry.
        ensure_nonstatic_links (bool): If True, links with zero mass are given a small mass (see `static_link_mass`) to ensure they are dynamic.
        static_link_mass (float): The mass to assign to links with zero mass (if `ensure_nonstatic_links` is set to True).
        collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged.
    """
    if xform is None:
        xform = wp.transform()

    file = ET.parse(urdf_filename)
    root = file.getroot()

    contact_vars = {
        "ke": contact_ke,
        "kd": contact_kd,
        "kf": contact_kf,
        "ka": contact_ka,
        "mu": contact_mu,
        "restitution": contact_restitution,
        "thickness": contact_thickness,
    }

    def parse_transform(element):
        if element is None or element.find("origin") is None:
            return wp.transform()
        origin = element.find("origin")
        xyz = origin.get("xyz") or "0 0 0"
        rpy = origin.get("rpy") or "0 0 0"
        xyz = [float(x) * scale for x in xyz.split()]
        rpy = [float(x) for x in rpy.split()]
        return wp.transform(xyz, wp.quat_rpy(*rpy))

    def parse_shapes(link, geoms, density, incoming_xform=None, visible=True, just_visual=False):
        shapes = []
        # add geometry
        for geom_group in geoms:
            geo = geom_group.find("geometry")
            if geo is None:
                continue

            tf = parse_transform(geom_group)
            if incoming_xform is not None:
                tf = incoming_xform * tf

            for box in geo.findall("box"):
                size = box.get("size") or "1 1 1"
                size = [float(x) for x in size.split()]
                s = builder.add_shape_box(
                    body=link,
                    pos=wp.vec3(tf.p),
                    rot=wp.quat(tf.q),
                    hx=size[0] * 0.5 * scale,
                    hy=size[1] * 0.5 * scale,
                    hz=size[2] * 0.5 * scale,
                    density=density,
                    is_visible=visible,
                    has_ground_collision=not just_visual,
                    has_shape_collision=not just_visual,
                    **contact_vars,
                )
                shapes.append(s)

            for sphere in geo.findall("sphere"):
                s = builder.add_shape_sphere(
                    body=link,
                    pos=wp.vec3(tf.p),
                    rot=wp.quat(tf.q),
                    radius=float(sphere.get("radius") or "1") * scale,
                    density=density,
                    is_visible=visible,
                    has_ground_collision=not just_visual,
                    has_shape_collision=not just_visual,
                    **contact_vars,
                )
                shapes.append(s)

            for cylinder in geo.findall("cylinder"):
                s = builder.add_shape_capsule(
                    body=link,
                    pos=wp.vec3(tf.p),
                    rot=wp.quat(tf.q),
                    radius=float(cylinder.get("radius") or "1") * scale,
                    half_height=float(cylinder.get("length") or "1") * 0.5 * scale,
                    density=density,
                    up_axis=2,  # cylinders in URDF are aligned with z-axis
                    is_visible=visible,
                    has_ground_collision=not just_visual,
                    has_shape_collision=not just_visual,
                    **contact_vars,
                )
                shapes.append(s)

            for capsule in geo.findall("capsule"):
                s = builder.add_shape_capsule(
                    body=link,
                    pos=wp.vec3(tf.p),
                    rot=wp.quat(tf.q),
                    radius=float(capsule.get("radius") or "1") * scale,
                    half_height=float(capsule.get("height") or "1") * 0.5 * scale,
                    density=density,
                    up_axis=2,  # capsules in URDF are aligned with z-axis
                    is_visible=visible,
                    has_ground_collision=not just_visual,
                    has_shape_collision=not just_visual,
                    **contact_vars,
                )
                shapes.append(s)

            for mesh in geo.findall("mesh"):
                filename = mesh.get("filename")
                if filename is None:
                    continue
                if filename.startswith("package://"):
                    fn = filename.replace("package://", "")
                    package_name = fn.split("/")[0]
                    urdf_folder = os.path.dirname(urdf_filename)
                    # resolve file path from package name, i.e. find
                    # the package folder from the URDF folder
                    if package_name in urdf_folder:
                        filename = os.path.join(urdf_folder[: urdf_folder.index(package_name)], fn)
                    else:
                        wp.utils.warn(
                            f'Warning: package "{package_name}" not found in URDF folder while loading mesh at "{filename}"'
                        )
                elif filename.startswith("http://") or filename.startswith("https://"):
                    # download mesh
                    import shutil
                    import tempfile

                    import requests

                    with tempfile.TemporaryDirectory() as tmpdir:
                        # get filename extension
                        extension = os.path.splitext(filename)[1]
                        tmpfile = os.path.join(tmpdir, "mesh" + extension)
                        with requests.get(filename, stream=True) as r:
                            with open(tmpfile, "wb") as f:
                                shutil.copyfileobj(r.raw, f)
                        filename = tmpfile
                else:
                    filename = os.path.join(os.path.dirname(urdf_filename), filename)
                if not os.path.exists(filename):
                    wp.utils.warn(f"Warning: mesh file {filename} does not exist")
                    continue

                import trimesh

                # use force='mesh' to load the mesh as a trimesh object
                # with baked in transforms, e.g. from COLLADA files
                m = trimesh.load(filename, force="mesh")
                scaling = mesh.get("scale") or "1 1 1"
                scaling = np.array([float(x) * scale for x in scaling.split()])
                if hasattr(m, "geometry"):
                    # multiple meshes are contained in a scene
                    for m_geom in m.geometry.values():
                        m_vertices = np.array(m_geom.vertices, dtype=np.float32) * scaling
                        m_faces = np.array(m_geom.faces.flatten(), dtype=np.int32)
                        m_mesh = Mesh(m_vertices, m_faces)
                        s = builder.add_shape_mesh(
                            body=link,
                            pos=wp.vec3(tf.p),
                            rot=wp.quat(tf.q),
                            mesh=m_mesh,
                            density=density,
                            is_visible=visible,
                            has_ground_collision=not just_visual,
                            has_shape_collision=not just_visual,
                            **contact_vars,
                        )
                        shapes.append(s)
                else:
                    # a single mesh
                    m_vertices = np.array(m.vertices, dtype=np.float32) * scaling
                    m_faces = np.array(m.faces.flatten(), dtype=np.int32)
                    m_mesh = Mesh(m_vertices, m_faces)
                    s = builder.add_shape_mesh(
                        body=link,
                        pos=wp.vec3(tf.p),
                        rot=wp.quat(tf.q),
                        mesh=m_mesh,
                        density=density,
                        is_visible=visible,
                        has_ground_collision=not just_visual,
                        has_shape_collision=not just_visual,
                        **contact_vars,
                    )
                    shapes.append(s)

        return shapes

    # maps from link name -> link index
    link_index = {}

    visual_shapes = []

    builder.add_articulation()

    start_shape_count = len(builder.shape_geo_type)

    # add links
    for _i, urdf_link in enumerate(root.findall("link")):
        name = urdf_link.get("name")
        link = builder.add_body(origin=wp.transform_identity(), armature=armature, name=name)

        # add ourselves to the index
        link_index[name] = link

        visuals = urdf_link.findall("visual")
        colliders = urdf_link.findall("collision")

        if parse_visuals_as_colliders:
            colliders = visuals
        else:
            s = parse_shapes(link, visuals, density=0.0, just_visual=True, visible=not hide_visuals)
            visual_shapes.extend(s)

        show_colliders = force_show_colliders
        if parse_visuals_as_colliders:
            show_colliders = True
        elif len(visuals) == 0:
            # we need to show the collision shapes since there are no visual shapes
            show_colliders = True

        parse_shapes(link, colliders, density=density, visible=show_colliders)
        m = builder.body_mass[link]
        if not ignore_inertial_definitions and urdf_link.find("inertial") is not None:
            # overwrite inertial parameters if defined
            inertial = urdf_link.find("inertial")
            inertial_frame = parse_transform(inertial)
            com = inertial_frame.p
            I_m = np.zeros((3, 3))
            I_m[0, 0] = float(inertial.find("inertia").get("ixx") or "0") * scale**2
            I_m[1, 1] = float(inertial.find("inertia").get("iyy") or "0") * scale**2
            I_m[2, 2] = float(inertial.find("inertia").get("izz") or "0") * scale**2
            I_m[0, 1] = float(inertial.find("inertia").get("ixy") or "0") * scale**2
            I_m[0, 2] = float(inertial.find("inertia").get("ixz") or "0") * scale**2
            I_m[1, 2] = float(inertial.find("inertia").get("iyz") or "0") * scale**2
            I_m[1, 0] = I_m[0, 1]
            I_m[2, 0] = I_m[0, 2]
            I_m[2, 1] = I_m[1, 2]
            rot = wp.quat_to_matrix(inertial_frame.q)
            I_m = rot @ wp.mat33(I_m)
            m = float(inertial.find("mass").get("value") or "0")
            builder.body_mass[link] = m
            builder.body_inv_mass[link] = 1.0 / m if m > 0.0 else 0.0
            builder.body_com[link] = com
            builder.body_inertia[link] = I_m
            if any(x for x in I_m):
                builder.body_inv_inertia[link] = wp.inverse(I_m)
            else:
                builder.body_inv_inertia[link] = I_m
        if m == 0.0 and ensure_nonstatic_links:
            # set the mass to something nonzero to ensure the body is dynamic
            m = static_link_mass
            # cube with side length 0.5
            I_m = wp.mat33(np.eye(3)) * m / 12.0 * (0.5 * scale) ** 2 * 2.0
            I_m += wp.mat33(armature * np.eye(3))
            builder.body_mass[link] = m
            builder.body_inv_mass[link] = 1.0 / m
            builder.body_inertia[link] = I_m
            builder.body_inv_inertia[link] = wp.inverse(I_m)

    end_shape_count = len(builder.shape_geo_type)

    # find joints per body
    body_children = {name: [] for name in link_index.keys()}
    # mapping from parent, child link names to joint
    parent_child_joint = {}

    joints = []
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        body_children[parent].append(child)
        joint_data = {
            "name": joint.get("name"),
            "parent": parent,
            "child": child,
            "type": joint.get("type"),
            "origin": parse_transform(joint),
            "damping": damping,
            "friction": 0.0,
            "limit_lower": joint_limit_lower,
            "limit_upper": joint_limit_upper,
        }
        if joint.find("axis") is not None:
            joint_data["axis"] = joint.find("axis").get("xyz")
            joint_data["axis"] = np.array([float(x) for x in joint_data["axis"].split()])
        if joint.find("dynamics") is not None:
            dynamics = joint.find("dynamics")
            joint_data["damping"] = float(dynamics.get("damping") or str(damping))
            joint_data["friction"] = float(dynamics.get("friction") or "0")
        if joint.find("limit") is not None:
            limit = joint.find("limit")
            joint_data["limit_lower"] = float(limit.get("lower") or "-1e6")
            joint_data["limit_upper"] = float(limit.get("upper") or "1e6")
        if joint.find("mimic") is not None:
            mimic = joint.find("mimic")
            joint_data["mimic_joint"] = mimic.get("joint")
            joint_data["mimic_multiplier"] = float(mimic.get("multiplier") or "1")
            joint_data["mimic_offset"] = float(mimic.get("offset") or "0")

        parent_child_joint[(parent, child)] = joint_data
        joints.append(joint_data)

    # topological sorting of joints because the FK solver will resolve body transforms
    # in joint order and needs the parent link transform to be resolved before the child
    visited = dict.fromkeys(link_index.keys(), False)
    sorted_joints = []

    # depth-first search
    def dfs(joint):
        link = joint["child"]
        if visited[link]:
            return
        visited[link] = True

        for child in body_children[link]:
            if not visited[child]:
                dfs(parent_child_joint[(link, child)])

        sorted_joints.insert(0, joint)

    # start DFS from each unvisited joint
    for joint in joints:
        if not visited[joint["parent"]]:
            dfs(joint)

    # add base joint
    if len(sorted_joints) > 0:
        base_link_name = sorted_joints[0]["parent"]
    else:
        base_link_name = next(iter(link_index.keys()))
    root = link_index[base_link_name]
    if base_joint is not None:
        # in case of a given base joint, the position is applied first, the rotation only
        # after the base joint itself to not rotate its axis
        base_parent_xform = wp.transform(xform.p, wp.quat_identity())
        base_child_xform = wp.transform((0.0, 0.0, 0.0), wp.quat_inverse(xform.q))
        if isinstance(base_joint, str):
            axes = base_joint.lower().split(",")
            axes = [ax.strip() for ax in axes]
            linear_axes = [ax[-1] for ax in axes if ax[0] in {"l", "p"}]
            angular_axes = [ax[-1] for ax in axes if ax[0] in {"a", "r"}]
            axes = {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
            }
            builder.add_joint_d6(
                linear_axes=[wp.sim.JointAxis(axes[a]) for a in linear_axes],
                angular_axes=[wp.sim.JointAxis(axes[a]) for a in angular_axes],
                parent_xform=base_parent_xform,
                child_xform=base_child_xform,
                parent=-1,
                child=root,
                name="base_joint",
            )
        elif isinstance(base_joint, dict):
            base_joint["parent"] = -1
            base_joint["child"] = root
            base_joint["parent_xform"] = base_parent_xform
            base_joint["child_xform"] = base_child_xform
            base_joint["name"] = "base_joint"
            builder.add_joint(**base_joint)
        else:
            raise ValueError(
                "base_joint must be a comma-separated string of joint axes or a dict with joint parameters"
            )
    elif floating:
        builder.add_joint_free(root, name="floating_base")

        # set dofs to transform
        start = builder.joint_q_start[root]

        builder.joint_q[start + 0] = xform.p[0]
        builder.joint_q[start + 1] = xform.p[1]
        builder.joint_q[start + 2] = xform.p[2]

        builder.joint_q[start + 3] = xform.q[0]
        builder.joint_q[start + 4] = xform.q[1]
        builder.joint_q[start + 5] = xform.q[2]
        builder.joint_q[start + 6] = xform.q[3]
    else:
        builder.add_joint_fixed(-1, root, parent_xform=xform, name="fixed_base")

    # add joints, in topological order starting from root body
    for joint in sorted_joints:
        parent = link_index[joint["parent"]]
        child = link_index[joint["child"]]
        if child == -1:
            # we skipped the insertion of the child body
            continue

        lower = joint["limit_lower"]
        upper = joint["limit_upper"]
        joint_damping = joint["damping"]

        parent_xform = joint["origin"]
        child_xform = wp.transform_identity()

        joint_mode = wp.sim.JOINT_MODE_FORCE
        if stiffness > 0.0:
            joint_mode = wp.sim.JOINT_MODE_TARGET_POSITION

        joint_params = {
            "parent": parent,
            "child": child,
            "parent_xform": parent_xform,
            "child_xform": child_xform,
            "name": joint["name"],
            "armature": armature,
        }

        if joint["type"] == "revolute" or joint["type"] == "continuous":
            builder.add_joint_revolute(
                axis=joint["axis"],
                target_ke=stiffness,
                target_kd=joint_damping,
                limit_lower=lower,
                limit_upper=upper,
                limit_ke=limit_ke,
                limit_kd=limit_kd,
                mode=joint_mode,
                **joint_params,
            )
        elif joint["type"] == "prismatic":
            builder.add_joint_prismatic(
                axis=joint["axis"],
                target_ke=stiffness,
                target_kd=joint_damping,
                limit_lower=lower * scale,
                limit_upper=upper * scale,
                limit_ke=limit_ke,
                limit_kd=limit_kd,
                mode=joint_mode,
                **joint_params,
            )
        elif joint["type"] == "fixed":
            builder.add_joint_fixed(**joint_params)
        elif joint["type"] == "floating":
            builder.add_joint_free(**joint_params)
        elif joint["type"] == "planar":
            # find plane vectors perpendicular to axis
            axis = np.array(joint["axis"])
            axis /= np.linalg.norm(axis)

            # create helper vector that is not parallel to the axis
            helper = np.array([1, 0, 0]) if np.allclose(axis, [0, 1, 0]) else np.array([0, 1, 0])

            u = np.cross(helper, axis)
            u /= np.linalg.norm(u)

            v = np.cross(axis, u)
            v /= np.linalg.norm(v)

            builder.add_joint_d6(
                linear_axes=[
                    wp.sim.JointAxis(
                        u,
                        limit_lower=lower * scale,
                        limit_upper=upper * scale,
                        limit_ke=limit_ke,
                        limit_kd=limit_kd,
                        target_ke=stiffness,
                        target_kd=joint_damping,
                        mode=joint_mode,
                    ),
                    wp.sim.JointAxis(
                        v,
                        limit_lower=lower * scale,
                        limit_upper=upper * scale,
                        limit_ke=limit_ke,
                        limit_kd=limit_kd,
                        target_ke=stiffness,
                        target_kd=joint_damping,
                        mode=joint_mode,
                    ),
                ],
                **joint_params,
            )
        else:
            raise Exception("Unsupported joint type: " + joint["type"])

    for i in range(start_shape_count, end_shape_count):
        for j in visual_shapes:
            builder.shape_collision_filter_pairs.add((i, j))

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            for j in range(i + 1, end_shape_count):
                builder.shape_collision_filter_pairs.add((i, j))

    if collapse_fixed_joints:
        builder.collapse_fixed_joints()
