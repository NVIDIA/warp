# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

import warp as wp


def parse_mjcf(
    mjcf_filename,
    builder,
    xform=None,
    density=1000.0,
    stiffness=0.0,
    damping=0.0,
    contact_ke=1000.0,
    contact_kd=100.0,
    contact_kf=100.0,
    contact_ka=0.0,
    contact_mu=0.5,
    contact_restitution=0.5,
    contact_thickness=0.0,
    limit_ke=100.0,
    limit_kd=10.0,
    scale=1.0,
    armature=0.0,
    armature_scale=1.0,
    parse_meshes=True,
    enable_self_collisions=False,
    up_axis="Z",
    ignore_classes=None,
    collapse_fixed_joints=False,
):
    """
    Parses MuJoCo XML (MJCF) file and adds the bodies and joints to the given ModelBuilder.

    Args:
        mjcf_filename (str): The filename of the MuJoCo file to parse.
        builder (ModelBuilder): The :class:`ModelBuilder` to add the bodies and joints to.
        xform (:ref:`transform <transform>`): The transform to apply to the imported mechanism.
        density (float): The density of the shapes in kg/m^3 which will be used to calculate the body mass and inertia.
        stiffness (float): The stiffness of the joints.
        damping (float): The damping of the joints.
        contact_ke (float): The stiffness of the shape contacts.
        contact_kd (float): The damping of the shape contacts.
        contact_kf (float): The friction stiffness of the shape contacts.
        contact_ka (float): The adhesion distance of the shape contacts.
        contact_mu (float): The friction coefficient of the shape contacts.
        contact_restitution (float): The restitution coefficient of the shape contacts.
        contact_thickness (float): The thickness to add to the shape geometry.
        limit_ke (float): The stiffness of the joint limits.
        limit_kd (float): The damping of the joint limits.
        scale (float): The scaling factor to apply to the imported mechanism.
        armature (float): Default joint armature to use if `armature` has not been defined for a joint in the MJCF.
        armature_scale (float): Scaling factor to apply to the MJCF-defined joint armature values.
        parse_meshes (bool): Whether geometries of type `"mesh"` should be parsed. If False, geometries of type `"mesh"` are ignored.
        enable_self_collisions (bool): If True, self-collisions are enabled.
        up_axis (str): The up axis of the mechanism. Can be either `"X"`, `"Y"` or `"Z"`. The default is `"Z"`.
        ignore_classes (List[str]): A list of regular expressions. Bodies and joints with a class matching one of the regular expressions will be ignored.
        collapse_fixed_joints (bool): If True, fixed joints are removed and the respective bodies are merged.

    Note:
        The inertia and masses of the bodies are calculated from the shape geometry and the given density. The values defined in the MJCF are not respected at the moment.

        The handling of advanced features, such as MJCF classes, is still experimental.
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
    euler_seq = [1, 2, 3]  # XYZ by default

    compiler = root.find("compiler")
    if compiler is not None:
        use_degrees = compiler.attrib.get("angle", "degree").lower() == "degree"
        euler_seq = ["xyz".index(c) + 1 for c in compiler.attrib.get("eulerseq", "xyz").lower()]
        mesh_dir = compiler.attrib.get("meshdir", ".")

    mesh_assets = {}
    for asset in root.findall("asset"):
        for mesh in asset.findall("mesh"):
            if "file" in mesh.attrib:
                fname = os.path.join(mesh_dir, mesh.attrib["file"])
                # handle stl relative paths
                if not os.path.isabs(fname):
                    fname = os.path.abspath(os.path.join(mjcf_dirname, fname))
                if "name" in mesh.attrib:
                    mesh_assets[mesh.attrib["name"]] = fname
                else:
                    name = ".".join(os.path.basename(fname).split(".")[:-1])
                    mesh_assets[name] = fname

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
            return wp.quat_from_euler(euler, *euler_seq)
        if "axisangle" in attrib:
            axisangle = np.fromstring(attrib["axisangle"], sep=" ")
            angle = axisangle[3]
            if use_degrees:
                angle *= np.pi / 180
            axis = wp.normalize(wp.vec3(*axisangle[:3]))
            return wp.quat_from_axis_angle(axis, angle)
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

    def parse_mesh(geom):
        import trimesh

        faces = []
        vertices = []
        stl_file = mesh_assets[geom["mesh"]]
        m = trimesh.load(stl_file)

        for v in m.vertices:
            vertices.append(np.array(v) * scale)

        for f in m.faces:
            faces.append(int(f[0]))
            faces.append(int(f[1]))
            faces.append(int(f[2]))
        return wp.sim.Mesh(vertices, faces), m.scale

    def parse_body(body, parent, incoming_defaults: dict):
        body_class = body.get("childclass")
        if body_class is None:
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
        else:
            joints = body.findall("joint")
            for _i, joint in enumerate(joints):
                if "joint" in defaults:
                    joint_attrib = merge_attrib(defaults["joint"], joint.attrib)
                else:
                    joint_attrib = joint.attrib

                # default to hinge if not specified
                joint_type_str = joint_attrib.get("type", "hinge")

                joint_name.append(joint_attrib["name"])
                joint_pos.append(parse_vec(joint_attrib, "pos", (0.0, 0.0, 0.0)) * scale)
                joint_range = parse_vec(joint_attrib, "range", (-3.0, 3.0))
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
                ax = wp.sim.model.JointAxis(
                    axis=axis_vec,
                    limit_lower=(np.deg2rad(joint_range[0]) if is_angular and use_degrees else joint_range[0]),
                    limit_upper=(np.deg2rad(joint_range[1]) if is_angular and use_degrees else joint_range[1]),
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

        joint_pos = joint_pos[0] if len(joint_pos) > 0 else (0.0, 0.0, 0.0)
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

        for geo_count, geom in enumerate(body.findall("geom")):
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
            geom_type = geom_attrib.get("type", "sphere")
            if "mesh" in geom_attrib:
                geom_type = "mesh"

            geom_size = parse_vec(geom_attrib, "size", [1.0, 1.0, 1.0]) * scale
            geom_pos = parse_vec(geom_attrib, "pos", (0.0, 0.0, 0.0)) * scale
            geom_rot = parse_orientation(geom_attrib)
            geom_density = parse_float(geom_attrib, "density", density)

            if geom_type == "sphere":
                builder.add_shape_sphere(
                    link,
                    pos=geom_pos,
                    rot=geom_rot,
                    radius=geom_size[0],
                    density=geom_density,
                    **contact_vars,
                )

            elif geom_type == "box":
                builder.add_shape_box(
                    link,
                    pos=geom_pos,
                    rot=geom_rot,
                    hx=geom_size[0],
                    hy=geom_size[1],
                    hz=geom_size[2],
                    density=geom_density,
                    **contact_vars,
                )

            elif geom_type == "mesh" and parse_meshes:
                mesh, _ = parse_mesh(geom_attrib)
                if "mesh" in defaults:
                    mesh_scale = parse_vec(defaults["mesh"], "scale", [1.0, 1.0, 1.0])
                else:
                    mesh_scale = [1.0, 1.0, 1.0]
                # as per the Mujoco XML reference, ignore geom size attribute
                assert len(geom_size) == 3, "need to specify size for mesh geom"
                builder.add_shape_mesh(
                    body=link,
                    pos=geom_pos,
                    rot=geom_rot,
                    mesh=mesh,
                    scale=mesh_scale,
                    density=density,
                    **contact_vars,
                )

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
                    builder.add_shape_cylinder(
                        link,
                        pos=geom_pos,
                        rot=geom_rot,
                        radius=geom_radius,
                        half_height=geom_height,
                        density=density,
                        up_axis=geom_up_axis,
                        **contact_vars,
                    )
                else:
                    builder.add_shape_capsule(
                        link,
                        pos=geom_pos,
                        rot=geom_rot,
                        radius=geom_radius,
                        half_height=geom_height,
                        density=density,
                        up_axis=geom_up_axis,
                        **contact_vars,
                    )

            else:
                print(f"MJCF parsing shape {geom_name} issue: geom type {geom_type} is unsupported")

        # -----------------
        # recurse

        for child in body.findall("body"):
            parse_body(child, link, defaults)

    # -----------------
    # start articulation

    start_shape_count = len(builder.shape_geo_type)
    builder.add_articulation()

    world = root.find("worldbody")
    world_class = get_class(world)
    world_defaults = merge_attrib(class_defaults["__all__"], class_defaults.get(world_class, {}))
    for body in world.findall("body"):
        parse_body(body, -1, world_defaults)

    end_shape_count = len(builder.shape_geo_type)

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            for j in range(i + 1, end_shape_count):
                builder.shape_collision_filter_pairs.add((i, j))

    if collapse_fixed_joints:
        builder.collapse_fixed_joints()
