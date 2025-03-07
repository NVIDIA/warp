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

import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Union

import numpy as np

import warp as wp
from warp.sim.model import Mesh


def parse_mjcf(
    mjcf_filename,
    builder,
    xform=None,
    floating=False,
    base_joint: Union[dict, str, None] = None,
    density=1000.0,
    stiffness=100.0,
    damping=10.0,
    armature=0.0,
    armature_scale=1.0,
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
    parse_meshes=True,
    up_axis="Z",
    ignore_names=(),
    ignore_classes=None,
    visual_classes=("visual",),
    collider_classes=("collision",),
    no_class_as_colliders=True,
    force_show_colliders=False,
    enable_self_collisions=False,
    ignore_inertial_definitions=True,
    ensure_nonstatic_links=True,
    static_link_mass=1e-2,
    collapse_fixed_joints=False,
    verbose=False,
):
    """
    Parses MuJoCo XML (MJCF) file and adds the bodies and joints to the given ModelBuilder.

    Args:
        mjcf_filename (str): The filename of the MuJoCo file to parse.
        builder (ModelBuilder): The :class:`ModelBuilder` to add the bodies and joints to.
        xform (:ref:`transform <transform>`): The transform to apply to the imported mechanism.
        floating (bool): If True, the root body is a free joint. If False, the root body is connected via a fixed joint to the world, unless a `base_joint` is defined.
        base_joint (Union[str, dict]): The joint by which the root body is connected to the world. This can be either a string defining the joint axes of a D6 joint with comma-separated positional and angular axis names (e.g. "px,py,rz" for a D6 joint with linear axes in x, y and an angular axis in z) or a dict with joint parameters (see :meth:`ModelBuilder.add_joint`).
        density (float): The density of the shapes in kg/m^3 which will be used to calculate the body mass and inertia.
        stiffness (float): The stiffness of the joints.
        damping (float): The damping of the joints.
        armature (float): Default joint armature to use if `armature` has not been defined for a joint in the MJCF.
        armature_scale (float): Scaling factor to apply to the MJCF-defined joint armature values.
        contact_ke (float): The stiffness of the shape contacts.
        contact_kd (float): The damping of the shape contacts.
        contact_kf (float): The friction stiffness of the shape contacts.
        contact_ka (float): The adhesion distance of the shape contacts.
        contact_mu (float): The friction coefficient of the shape contacts.
        contact_restitution (float): The restitution coefficient of the shape contacts.
        contact_thickness (float): The thickness to add to the shape geometry.
        limit_ke (float): The stiffness of the joint limits.
        limit_kd (float): The damping of the joint limits.
        joint_limit_lower (float): The default lower joint limit if not specified in the MJCF.
        joint_limit_upper (float): The default upper joint limit if not specified in the MJCF.
        scale (float): The scaling factor to apply to the imported mechanism.
        hide_visuals (bool): If True, hide visual shapes.
        parse_visuals_as_colliders (bool): If True, the geometry defined under the `visual_classes` tags is used for collision handling instead of the `collider_classes` geometries.
        parse_meshes (bool): Whether geometries of type `"mesh"` should be parsed. If False, geometries of type `"mesh"` are ignored.
        up_axis (str): The up axis of the mechanism. Can be either `"X"`, `"Y"` or `"Z"`. The default is `"Z"`.
        ignore_names (List[str]): A list of regular expressions. Bodies and joints with a name matching one of the regular expressions will be ignored.
        ignore_classes (List[str]): A list of regular expressions. Bodies and joints with a class matching one of the regular expressions will be ignored.
        visual_classes (List[str]): A list of regular expressions. Visual geometries with a class matching one of the regular expressions will be parsed.
        collider_classes (List[str]): A list of regular expressions. Collision geometries with a class matching one of the regular expressions will be parsed.
        no_class_as_colliders: If True, geometries without a class are parsed as collision geometries. If False, geometries without a class are parsed as visual geometries.
        force_show_colliders (bool): If True, the collision shapes are always shown, even if there are visual shapes.
        enable_self_collisions (bool): If True, self-collisions are enabled.
        ignore_inertial_definitions (bool): If True, the inertial parameters defined in the MJCF are ignored and the inertia is calculated from the shape geometry.
        ensure_nonstatic_links (bool): If True, links with zero mass are given a small mass (see `static_link_mass`) to ensure they are dynamic.
        static_link_mass (float): The mass to assign to links with zero mass (if `ensure_nonstatic_links` is set to True).
        collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged.
        verbose (bool): If True, print additional information about parsing the MJCF.
    """
    if xform is None:
        xform = wp.transform()

    if ignore_classes is None:
        ignore_classes = []

    mjcf_dirname = os.path.dirname(mjcf_filename)
    file = ET.parse(mjcf_filename)
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

    use_degrees = True  # angles are in degrees by default
    euler_seq = [0, 1, 2]  # XYZ by default

    compiler = root.find("compiler")
    if compiler is not None:
        use_degrees = compiler.attrib.get("angle", "degree").lower() == "degree"
        euler_seq = ["xyz".index(c) for c in compiler.attrib.get("eulerseq", "xyz").lower()]
        mesh_dir = compiler.attrib.get("meshdir", ".")
    else:
        mesh_dir = "."

    mesh_assets = {}
    for asset in root.findall("asset"):
        for mesh in asset.findall("mesh"):
            if "file" in mesh.attrib:
                fname = os.path.join(mesh_dir, mesh.attrib["file"])
                # handle stl relative paths
                if not os.path.isabs(fname):
                    fname = os.path.abspath(os.path.join(mjcf_dirname, fname))
                name = mesh.attrib.get("name", ".".join(os.path.basename(fname).split(".")[:-1]))
                s = mesh.attrib.get("scale", "1.0 1.0 1.0")
                s = np.fromstring(s, sep=" ", dtype=np.float32)
                mesh_assets[name] = {"file": fname, "scale": s}

    class_parent = {}
    class_children = {}
    class_defaults = {"__all__": {}}

    def get_class(element):
        return element.get("class", "__all__")

    def parse_default(node, parent):
        nonlocal class_parent
        nonlocal class_children
        nonlocal class_defaults
        class_name = "__all__"
        if "class" in node.attrib:
            class_name = node.attrib["class"]
            class_parent[class_name] = parent
            parent = parent or "__all__"
            if parent not in class_children:
                class_children[parent] = []
            class_children[parent].append(class_name)

        if class_name not in class_defaults:
            class_defaults[class_name] = {}
        for child in node:
            if child.tag == "default":
                parse_default(child, node.get("class"))
            else:
                class_defaults[class_name][child.tag] = child.attrib

    for default in root.findall("default"):
        parse_default(default, None)

    def merge_attrib(default_attrib: dict, incoming_attrib: dict):
        attrib = default_attrib.copy()
        attrib.update(incoming_attrib)
        return attrib

    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    sqh = np.sqrt(0.5)
    if up_axis == 0:
        xform = wp.transform(xform.p, wp.quat(0.0, 0.0, -sqh, sqh) * xform.q)
    elif up_axis == 2:
        xform = wp.transform(xform.p, wp.quat(sqh, 0.0, 0.0, -sqh) * xform.q)
    # do not apply scaling to the root transform
    xform = wp.transform(np.array(xform.p) / scale, xform.q)

    def parse_float(attrib, key, default):
        if key in attrib:
            return float(attrib[key])
        else:
            return default

    def parse_vec(attrib, key, default):
        if key in attrib:
            out = np.fromstring(attrib[key], sep=" ", dtype=np.float32)
        else:
            out = np.array(default, dtype=np.float32)

        length = len(out)
        if length == 1:
            return wp.vec(len(default), wp.float32)(out[0], out[0], out[0])

        return wp.vec(length, wp.float32)(out)

    def parse_orientation(attrib):
        if "quat" in attrib:
            wxyz = np.fromstring(attrib["quat"], sep=" ")
            return wp.normalize(wp.quat(*wxyz[1:], wxyz[0]))
        if "euler" in attrib:
            euler = np.fromstring(attrib["euler"], sep=" ")
            if use_degrees:
                euler *= np.pi / 180
            return wp.sim.quat_from_euler(wp.vec3(euler), *euler_seq)
        if "axisangle" in attrib:
            axisangle = np.fromstring(attrib["axisangle"], sep=" ")
            angle = axisangle[3]
            if use_degrees:
                angle *= np.pi / 180
            axis = wp.normalize(wp.vec3(*axisangle[:3]))
            return wp.quat_from_axis_angle(axis, float(angle))
        if "xyaxes" in attrib:
            xyaxes = np.fromstring(attrib["xyaxes"], sep=" ")
            xaxis = wp.normalize(wp.vec3(*xyaxes[:3]))
            zaxis = wp.normalize(wp.vec3(*xyaxes[3:]))
            yaxis = wp.normalize(wp.cross(zaxis, xaxis))
            rot_matrix = np.array([xaxis, yaxis, zaxis]).T
            return wp.quat_from_matrix(rot_matrix)
        if "zaxis" in attrib:
            zaxis = np.fromstring(attrib["zaxis"], sep=" ")
            zaxis = wp.normalize(wp.vec3(*zaxis))
            xaxis = wp.normalize(wp.cross(wp.vec3(0, 0, 1), zaxis))
            yaxis = wp.normalize(wp.cross(zaxis, xaxis))
            rot_matrix = np.array([xaxis, yaxis, zaxis]).T
            return wp.quat_from_matrix(rot_matrix)
        return wp.quat_identity()

    def parse_shapes(defaults, body_name, link, geoms, density, visible=True, just_visual=False, incoming_xform=None):
        shapes = []
        for geo_count, geom in enumerate(geoms):
            geom_defaults = defaults
            if "class" in geom.attrib:
                geom_class = geom.attrib["class"]
                ignore_geom = False
                for pattern in ignore_classes:
                    if re.match(pattern, geom_class):
                        ignore_geom = True
                        break
                if ignore_geom:
                    continue
                if geom_class in class_defaults:
                    geom_defaults = merge_attrib(defaults, class_defaults[geom_class])
            if "geom" in geom_defaults:
                geom_attrib = merge_attrib(geom_defaults["geom"], geom.attrib)
            else:
                geom_attrib = geom.attrib

            geom_name = geom_attrib.get("name", f"{body_name}_geom_{geo_count}{'_visual' if just_visual else ''}")
            geom_type = geom_attrib.get("type", "sphere")
            if "mesh" in geom_attrib:
                geom_type = "mesh"

            ignore_geom = False
            for pattern in ignore_names:
                if re.match(pattern, geom_name):
                    ignore_geom = True
                    break
            if ignore_geom:
                continue

            geom_size = parse_vec(geom_attrib, "size", [1.0, 1.0, 1.0]) * scale
            geom_pos = parse_vec(geom_attrib, "pos", (0.0, 0.0, 0.0)) * scale
            geom_rot = parse_orientation(geom_attrib)
            geom_density = parse_float(geom_attrib, "density", density)

            if incoming_xform is not None:
                geom_pos = wp.transform_point(incoming_xform, geom_pos)
                geom_rot = incoming_xform.q * geom_rot

            if geom_type == "sphere":
                s = builder.add_shape_sphere(
                    link,
                    pos=geom_pos,
                    rot=geom_rot,
                    radius=geom_size[0],
                    density=geom_density,
                    is_visible=visible,
                    has_ground_collision=not just_visual,
                    has_shape_collision=not just_visual,
                    **contact_vars,
                )
                shapes.append(s)

            elif geom_type == "box":
                s = builder.add_shape_box(
                    link,
                    pos=geom_pos,
                    rot=geom_rot,
                    hx=geom_size[0],
                    hy=geom_size[1],
                    hz=geom_size[2],
                    density=geom_density,
                    is_visible=visible,
                    has_ground_collision=not just_visual,
                    has_shape_collision=not just_visual,
                    **contact_vars,
                )
                shapes.append(s)

            elif geom_type == "mesh" and parse_meshes:
                import trimesh

                # use force='mesh' to load the mesh as a trimesh object
                # with baked in transforms, e.g. from COLLADA files
                stl_file = mesh_assets[geom_attrib["mesh"]]["file"]
                m = trimesh.load(stl_file, force="mesh")
                if "mesh" in geom_defaults:
                    mesh_scale = parse_vec(geom_defaults["mesh"], "scale", mesh_assets[geom_attrib["mesh"]]["scale"])
                else:
                    mesh_scale = mesh_assets[geom_attrib["mesh"]]["scale"]
                scaling = np.array(mesh_scale) * scale
                # as per the Mujoco XML reference, ignore geom size attribute
                assert len(geom_size) == 3, "need to specify size for mesh geom"

                if hasattr(m, "geometry"):
                    # multiple meshes are contained in a scene
                    for m_geom in m.geometry.values():
                        m_vertices = np.array(m_geom.vertices, dtype=np.float32) * scaling
                        m_faces = np.array(m_geom.faces.flatten(), dtype=np.int32)
                        m_mesh = Mesh(m_vertices, m_faces)
                        s = builder.add_shape_mesh(
                            body=link,
                            pos=geom_pos,
                            rot=geom_rot,
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
                        pos=geom_pos,
                        rot=geom_rot,
                        mesh=m_mesh,
                        density=density,
                        is_visible=visible,
                        has_ground_collision=not just_visual,
                        has_shape_collision=not just_visual,
                        **contact_vars,
                    )
                    shapes.append(s)

            elif geom_type in {"capsule", "cylinder"}:
                if "fromto" in geom_attrib:
                    geom_fromto = parse_vec(geom_attrib, "fromto", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

                    start = wp.vec3(geom_fromto[0:3]) * scale
                    end = wp.vec3(geom_fromto[3:6]) * scale

                    # compute rotation to align the Warp capsule (along x-axis), with mjcf fromto direction
                    axis = wp.normalize(end - start)
                    angle = math.acos(wp.dot(axis, wp.vec3(0.0, 1.0, 0.0)))
                    axis = wp.normalize(wp.cross(axis, wp.vec3(0.0, 1.0, 0.0)))

                    geom_pos = (start + end) * 0.5
                    geom_rot = wp.quat_from_axis_angle(axis, -angle)

                    geom_radius = geom_size[0]
                    geom_height = wp.length(end - start) * 0.5
                    geom_up_axis = 1

                else:
                    geom_radius = geom_size[0]
                    geom_height = geom_size[1]
                    geom_up_axis = up_axis

                if geom_type == "cylinder":
                    s = builder.add_shape_cylinder(
                        link,
                        pos=geom_pos,
                        rot=geom_rot,
                        radius=geom_radius,
                        half_height=geom_height,
                        density=density,
                        up_axis=geom_up_axis,
                        is_visible=visible,
                        has_ground_collision=not just_visual,
                        has_shape_collision=not just_visual,
                        **contact_vars,
                    )
                    shapes.append(s)
                else:
                    s = builder.add_shape_capsule(
                        link,
                        pos=geom_pos,
                        rot=geom_rot,
                        radius=geom_radius,
                        half_height=geom_height,
                        density=density,
                        up_axis=geom_up_axis,
                        is_visible=visible,
                        has_ground_collision=not just_visual,
                        has_shape_collision=not just_visual,
                        **contact_vars,
                    )
                    shapes.append(s)

            elif geom_type == "plane":
                normal = wp.quat_rotate(geom_rot, wp.vec3(0.0, 0.0, 1.0))
                p = wp.dot(geom_pos, normal)
                s = builder.add_shape_plane(
                    body=link,
                    plane=(*normal, p),
                    width=geom_size[0],
                    length=geom_size[1],
                    is_visible=visible,
                    has_ground_collision=False,
                    has_shape_collision=not just_visual,
                    **contact_vars,
                )
                shapes.append(s)

            else:
                if verbose:
                    print(f"MJCF parsing shape {geom_name} issue: geom type {geom_type} is unsupported")

        return shapes

    def parse_body(body, parent, incoming_defaults: dict, childclass: str = None):
        body_class = body.get("class")
        if body_class is None:
            body_class = childclass
            defaults = incoming_defaults
        else:
            for pattern in ignore_classes:
                if re.match(pattern, body_class):
                    return
            defaults = merge_attrib(incoming_defaults, class_defaults[body_class])
        if "body" in defaults:
            body_attrib = merge_attrib(defaults["body"], body.attrib)
        else:
            body_attrib = body.attrib
        body_name = body_attrib["name"]
        body_name = body_name.replace("-", "_")  # ensure valid USD path
        body_pos = parse_vec(body_attrib, "pos", (0.0, 0.0, 0.0))
        body_ori = parse_orientation(body_attrib)
        if parent == -1:
            body_pos = wp.transform_point(xform, body_pos)
            body_ori = xform.q * body_ori
        body_pos *= scale

        joint_armature = []
        joint_name = []
        joint_pos = []

        linear_axes = []
        angular_axes = []
        joint_type = None

        freejoint_tags = body.findall("freejoint")
        if len(freejoint_tags) > 0:
            joint_type = wp.sim.JOINT_FREE
            joint_name.append(freejoint_tags[0].attrib.get("name", f"{body_name}_freejoint"))
            joint_armature.append(0.0)
        else:
            joints = body.findall("joint")
            for _i, joint in enumerate(joints):
                joint_defaults = defaults
                if "class" in joint.attrib:
                    joint_class = joint.attrib["class"]
                    if joint_class in class_defaults:
                        joint_defaults = merge_attrib(joint_defaults, class_defaults[joint_class])
                if "joint" in joint_defaults:
                    joint_attrib = merge_attrib(joint_defaults["joint"], joint.attrib)
                else:
                    joint_attrib = joint.attrib

                # default to hinge if not specified
                joint_type_str = joint_attrib.get("type", "hinge")

                joint_name.append(joint_attrib["name"])
                joint_pos.append(parse_vec(joint_attrib, "pos", (0.0, 0.0, 0.0)) * scale)
                joint_range = parse_vec(joint_attrib, "range", (joint_limit_lower, joint_limit_upper))
                joint_armature.append(parse_float(joint_attrib, "armature", armature) * armature_scale)

                if joint_type_str == "free":
                    joint_type = wp.sim.JOINT_FREE
                    break
                if joint_type_str == "fixed":
                    joint_type = wp.sim.JOINT_FIXED
                    break
                is_angular = joint_type_str == "hinge"
                mode = wp.sim.JOINT_MODE_FORCE
                if stiffness > 0.0 or "stiffness" in joint_attrib:
                    mode = wp.sim.JOINT_MODE_TARGET_POSITION
                axis_vec = parse_vec(joint_attrib, "axis", (0.0, 0.0, 0.0))
                limit_lower = np.deg2rad(joint_range[0]) if is_angular and use_degrees else joint_range[0]
                limit_upper = np.deg2rad(joint_range[1]) if is_angular and use_degrees else joint_range[1]
                ax = wp.sim.JointAxis(
                    axis=axis_vec,
                    limit_lower=limit_lower,
                    limit_upper=limit_upper,
                    target_ke=parse_float(joint_attrib, "stiffness", stiffness),
                    target_kd=parse_float(joint_attrib, "damping", damping),
                    limit_ke=limit_ke,
                    limit_kd=limit_kd,
                    mode=mode,
                )
                if is_angular:
                    angular_axes.append(ax)
                else:
                    linear_axes.append(ax)

        link = builder.add_body(
            origin=wp.transform(body_pos, body_ori),  # will be evaluated in fk()
            armature=joint_armature[0] if len(joint_armature) > 0 else armature,
            name=body_name,
        )

        if joint_type is None:
            if len(linear_axes) == 0:
                if len(angular_axes) == 0:
                    joint_type = wp.sim.JOINT_FIXED
                elif len(angular_axes) == 1:
                    joint_type = wp.sim.JOINT_REVOLUTE
                elif len(angular_axes) == 2:
                    joint_type = wp.sim.JOINT_UNIVERSAL
                elif len(angular_axes) == 3:
                    joint_type = wp.sim.JOINT_COMPOUND
            elif len(linear_axes) == 1 and len(angular_axes) == 0:
                joint_type = wp.sim.JOINT_PRISMATIC
            else:
                joint_type = wp.sim.JOINT_D6

        if len(freejoint_tags) > 0 and parent == -1 and (base_joint is not None or floating is not None):
            joint_pos = joint_pos[0] if len(joint_pos) > 0 else (0.0, 0.0, 0.0)
            _xform = wp.transform(body_pos + joint_pos, body_ori)

            if base_joint is not None:
                # in case of a given base joint, the position is applied first, the rotation only
                # after the base joint itself to not rotate its axis
                base_parent_xform = wp.transform(_xform.p, wp.quat_identity())
                base_child_xform = wp.transform((0.0, 0.0, 0.0), wp.quat_inverse(_xform.q))
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
                        child=link,
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
                builder.add_joint_free(link, name="floating_base")

                # set dofs to transform
                start = builder.joint_q_start[link]

                builder.joint_q[start + 0] = _xform.p[0]
                builder.joint_q[start + 1] = _xform.p[1]
                builder.joint_q[start + 2] = _xform.p[2]

                builder.joint_q[start + 3] = _xform.q[0]
                builder.joint_q[start + 4] = _xform.q[1]
                builder.joint_q[start + 5] = _xform.q[2]
                builder.joint_q[start + 6] = _xform.q[3]
            else:
                builder.add_joint_fixed(-1, link, parent_xform=_xform, name="fixed_base")

        else:
            joint_pos = joint_pos[0] if len(joint_pos) > 0 else (0.0, 0.0, 0.0)
            if len(joint_name) == 0:
                joint_name = [f"{body_name}_joint"]
            builder.add_joint(
                joint_type,
                parent,
                link,
                linear_axes,
                angular_axes,
                name="_".join(joint_name),
                parent_xform=wp.transform(body_pos + joint_pos, body_ori),
                child_xform=wp.transform(joint_pos, wp.quat_identity()),
                armature=joint_armature[0] if len(joint_armature) > 0 else armature,
            )

        # -----------------
        # add shapes

        geoms = body.findall("geom")
        visuals = []
        colliders = []
        for geo_count, geom in enumerate(geoms):
            geom_defaults = defaults
            if "class" in geom.attrib:
                geom_class = geom.attrib["class"]
                ignore_geom = False
                for pattern in ignore_classes:
                    if re.match(pattern, geom_class):
                        ignore_geom = True
                        break
                if ignore_geom:
                    continue
                if geom_class in class_defaults:
                    geom_defaults = merge_attrib(defaults, class_defaults[geom_class])
            if "geom" in geom_defaults:
                geom_attrib = merge_attrib(geom_defaults["geom"], geom.attrib)
            else:
                geom_attrib = geom.attrib

            geom_name = geom_attrib.get("name", f"{body_name}_geom_{geo_count}")

            if "class" in geom.attrib:
                for pattern in visual_classes:
                    if re.match(pattern, geom_class):
                        visuals.append(geom)
                        break
                for pattern in collider_classes:
                    if re.match(pattern, geom_class):
                        colliders.append(geom)
                        break
            else:
                no_class_class = "collision" if no_class_as_colliders else "visual"
                if verbose:
                    print(f"MJCF parsing shape {geom_name} issue: no class defined for geom, assuming {no_class_class}")
                if no_class_as_colliders:
                    colliders.append(geom)
                else:
                    visuals.append(geom)

        if parse_visuals_as_colliders:
            colliders = visuals
        else:
            s = parse_shapes(
                defaults, body_name, link, visuals, density=0.0, just_visual=True, visible=not hide_visuals
            )
            visual_shapes.extend(s)

        show_colliders = force_show_colliders
        if parse_visuals_as_colliders:
            show_colliders = True
        elif len(visuals) == 0:
            # we need to show the collision shapes since there are no visual shapes
            show_colliders = True

        parse_shapes(defaults, body_name, link, colliders, density, visible=show_colliders)

        m = builder.body_mass[link]
        if not ignore_inertial_definitions and body.find("inertial") is not None:
            inertial = body.find("inertial")
            if "inertial" in defaults:
                inertial_attrib = merge_attrib(defaults["inertial"], inertial.attrib)
            else:
                inertial_attrib = inertial.attrib
            # overwrite inertial parameters if defined
            inertial_pos = parse_vec(inertial_attrib, "pos", (0.0, 0.0, 0.0)) * scale
            inertial_rot = parse_orientation(inertial_attrib)

            inertial_frame = wp.transform(inertial_pos, inertial_rot)
            com = inertial_frame.p
            if inertial_attrib.get("diaginertia") is not None:
                diaginertia = parse_vec(inertial_attrib, "diaginertia", None)
                I_m = np.zeros((3, 3))
                I_m[0, 0] = diaginertia[0] * scale**2
                I_m[1, 1] = diaginertia[1] * scale**2
                I_m[2, 2] = diaginertia[2] * scale**2
            else:
                fullinertia = inertial_attrib.get("fullinertia")
                assert fullinertia is not None
                fullinertia = np.fromstring(fullinertia, sep=" ", dtype=np.float32)
                I_m = np.zeros((3, 3))
                I_m[0, 0] = fullinertia[0] * scale**2
                I_m[1, 1] = fullinertia[1] * scale**2
                I_m[2, 2] = fullinertia[2] * scale**2
                I_m[0, 1] = fullinertia[3] * scale**2
                I_m[0, 2] = fullinertia[4] * scale**2
                I_m[1, 2] = fullinertia[5] * scale**2
                I_m[1, 0] = I_m[0, 1]
                I_m[2, 0] = I_m[0, 2]
                I_m[2, 1] = I_m[1, 2]
            rot = wp.quat_to_matrix(inertial_frame.q)
            I_m = rot @ wp.mat33(I_m)
            m = float(inertial_attrib.get("mass", "0"))
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

        # -----------------
        # recurse

        for child in body.findall("body"):
            _childclass = body.get("childclass")
            if _childclass is None:
                _childclass = childclass
                _incoming_defaults = defaults
            else:
                _incoming_defaults = merge_attrib(defaults, class_defaults[_childclass])
            parse_body(child, link, _incoming_defaults, childclass=_childclass)

    # -----------------
    # start articulation

    visual_shapes = []
    start_shape_count = len(builder.shape_geo_type)
    builder.add_articulation()

    world = root.find("worldbody")
    world_class = get_class(world)
    world_defaults = merge_attrib(class_defaults["__all__"], class_defaults.get(world_class, {}))

    # -----------------
    # add bodies

    for body in world.findall("body"):
        parse_body(body, -1, world_defaults)

    # -----------------
    # add static geoms

    parse_shapes(world_defaults, "world", -1, world.findall("geom"), density, incoming_xform=xform)

    end_shape_count = len(builder.shape_geo_type)

    for i in range(start_shape_count, end_shape_count):
        for j in visual_shapes:
            builder.shape_collision_filter_pairs.add((i, j))

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            for j in range(i + 1, end_shape_count):
                builder.shape_collision_filter_pairs.add((i, j))

    if collapse_fixed_joints:
        builder.collapse_fixed_joints()
