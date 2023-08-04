# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
import re

import warp as wp
from . import ModelBuilder


def parse_usd(
    filename,
    builder: ModelBuilder,
    default_density=1.0e3,
    only_load_enabled_rigid_bodies=False,
    only_load_enabled_joints=True,
    default_ke=1e5,
    default_kd=250.0,
    default_kf=500.0,
    default_mu=0.0,
    default_restitution=0.0,
    default_thickness=0.0,
    joint_limit_ke=100.0,
    joint_limit_kd=10.0,
    verbose=False,
    ignore_paths=[],
    export_usda=False,
):
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        raise ImportError("Failed to import pxr. Please install USD.")

    if filename.startswith("http://") or filename.startswith("https://"):
        # download file
        import requests, os, datetime

        response = requests.get(filename, allow_redirects=True)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download USD file. Status code: {response.status_code}")
        file = response.content
        dot = os.path.extsep
        base = os.path.basename(filename)
        url_folder = os.path.dirname(filename)
        base_name = dot.join(base.split(dot)[:-1])
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        folder_name = os.path.join(".usd_cache", f"{base_name}_{timestamp}")
        os.makedirs(folder_name, exist_ok=True)
        target_filename = os.path.join(folder_name, base)
        with open(target_filename, "wb") as f:
            f.write(file)

        stage = Usd.Stage.Open(target_filename, Usd.Stage.LoadNone)
        stage_str = stage.GetRootLayer().ExportToString()
        print(f"Downloaded USD file to {target_filename}.")
        if export_usda:
            usda_filename = os.path.join(folder_name, base_name + ".usda")
            with open(usda_filename, "w") as f:
                f.write(stage_str)
                print(f"Exported USDA file to {usda_filename}.")

        # parse referenced USD files like `references = @./franka_collisions.usd@`
        downloaded = set()
        for match in re.finditer(r"references.=.@(.*?)@", stage_str):
            refname = match.group(1)
            if refname.startswith("./"):
                refname = refname[2:]
            if refname in downloaded:
                continue
            try:
                response = requests.get(f"{url_folder}/{refname}", allow_redirects=True)
                if response.status_code != 200:
                    print(f"Failed to download reference {refname}. Status code: {response.status_code}")
                    continue
                file = response.content
                refdir = os.path.dirname(refname)
                if refdir:
                    os.makedirs(os.path.join(folder_name, refdir), exist_ok=True)
                ref_filename = os.path.join(folder_name, refname)
                with open(ref_filename, "wb") as f:
                    f.write(file)
                downloaded.add(refname)
                print(f"Downloaded USD reference {refname} to {ref_filename}.")
                if export_usda:
                    ref_stage = Usd.Stage.Open(ref_filename, Usd.Stage.LoadNone)
                    ref_stage_str = ref_stage.GetRootLayer().ExportToString()
                    base = os.path.basename(ref_filename)
                    base_name = dot.join(base.split(dot)[:-1])
                    usda_filename = os.path.join(folder_name, base_name + ".usda")
                    with open(usda_filename, "w") as f:
                        f.write(ref_stage_str)
                        print(f"Exported USDA file to {usda_filename}.")
            except:
                print(f"Failed to download {refname}.")

        filename = target_filename

    def get_attribute(prim, name):
        if "*" in name:
            regex = name.replace("*", ".*")
            for attr in prim.GetAttributes():
                if re.match(regex, attr.GetName()):
                    return attr
        else:
            return prim.GetAttribute(name)

    def has_attribute(prim, name):
        attr = get_attribute(prim, name)
        return attr.IsValid() and attr.HasAuthoredValue()

    def parse_float(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        val = attr.Get()
        if np.isfinite(val):
            return val
        return default

    def parse_quat(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        val = attr.Get()
        quat = wp.quat(*val.imaginary, val.real)
        l = wp.length(quat)
        if np.isfinite(l) and l > 0.0:
            return quat
        return default

    def parse_vec(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        val = attr.Get()
        if np.isfinite(val).all():
            return np.array(val, dtype=np.float32)
        return default

    def parse_generic(prim, name, default=None):
        attr = get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        return attr.Get()

    def str2axis(s: str) -> np.ndarray:
        axis = np.zeros(3, dtype=np.float32)
        axis["XYZ".index(s.upper())] = 1.0
        return axis

    stage = Usd.Stage.Open(filename, Usd.Stage.LoadAll)
    if UsdPhysics.StageHasAuthoredKilogramsPerUnit(stage):
        mass_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    else:
        mass_unit = 1.0
    if UsdGeom.StageHasAuthoredMetersPerUnit(stage):
        linear_unit = UsdGeom.GetStageMetersPerUnit(stage)
    else:
        linear_unit = 1.0

    def parse_xform(prim):
        xform = UsdGeom.Xform(prim)
        mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
        rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].flatten()))
        pos = mat[3, :3] * linear_unit
        scale = np.ones(3, dtype=np.float32)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale = np.array(op.Get(), dtype=np.float32)
        return wp.transform(pos, rot), scale

    def parse_axis(prim, type, joint_data, is_angular, axis=None):
        # parse joint axis data
        schemas = prim.GetAppliedSchemas()
        schemas_str = "".join(schemas)
        if f"DriveAPI:{type}" not in schemas_str and f"PhysicsLimitAPI:{type}" not in schemas_str:
            return
        drive_type = parse_generic(prim, f"drive:{type}:physics:type", "force")
        if drive_type != "force":
            print(f"Warning: only force drive type is supported, ignoring drive:{type} for joint {path}")
            return
        stiffness = parse_float(prim, f"drive:{type}:physics:stiffness", 0.0)
        damping = parse_float(prim, f"drive:{type}:physics:damping", 0.0)
        low = parse_float(prim, f"limit:{type}:physics:low")
        high = parse_float(prim, f"limit:{type}:physics:high")
        target_pos = parse_float(prim, f"drive:{type}:physics:targetPosition")
        target_vel = parse_float(prim, f"drive:{type}:physics:targetVelocity")
        if is_angular:
            stiffness *= mass_unit * linear_unit**2
            stiffness = np.deg2rad(stiffness)
            damping *= mass_unit * linear_unit**2
            damping = np.deg2rad(damping)
            if target_pos is not None:
                target_pos = np.deg2rad(target_pos)
            if target_vel is not None:
                target_vel = np.deg2rad(target_vel)
            if low is None:
                low = joint_data["lowerLimit"]
            else:
                low = np.deg2rad(low)
            if high is None:
                high = joint_data["upperLimit"]
            else:
                high = np.deg2rad(high)
        else:
            stiffness *= mass_unit
            damping *= mass_unit
            if target_pos is not None:
                target_pos *= linear_unit
            if target_vel is not None:
                target_vel *= linear_unit
            if low is None:
                low = joint_data["lowerLimit"]
            else:
                low *= linear_unit
            if high is None:
                high = joint_data["upperLimit"]
            else:
                high *= linear_unit
        mode = wp.sim.JOINT_MODE_LIMIT
        if f"DriveAPI:{type}" in schemas_str:
            if target_vel is not None and target_vel != 0.0:
                mode = wp.sim.JOINT_MODE_TARGET_VELOCITY
            else:
                mode = wp.sim.JOINT_MODE_TARGET_POSITION
        if low > high:
            low = (low + high) / 2
            high = low
        axis = wp.sim.JointAxis(
            axis=(axis or joint_data["axis"]),
            limit_lower=low,
            limit_upper=high,
            target=(target_pos or target_vel or (low + high) / 2),
            target_ke=stiffness,
            target_kd=damping,
            mode=mode,
            limit_ke=joint_limit_ke,
            limit_kd=joint_limit_kd,
        )
        if is_angular:
            joint_data["angular_axes"].append(axis)
        else:
            joint_data["linear_axes"].append(axis)

    upaxis = str2axis(UsdGeom.GetStageUpAxis(stage))

    shape_types = {"Cube", "Sphere", "Mesh", "Capsule", "Plane", "Cylinder", "Cone"}

    path_body_map = {}
    path_shape_map = {}
    # maps prim path name to its world transform
    path_world_poses = {}
    # transform from body frame to where the actual joint child frame is
    # so that the link's children will use the right parent tf for the joint
    prim_joint_xforms = {}
    path_collision_filters = set()
    no_collision_shapes = set()

    body_density = {}  # mapping from body ID to defined density

    # first find all joints and materials
    joint_data = {}  # mapping from path of child link to joint USD settings
    materials = {}  # mapping from material path to material USD settings
    joint_parents = set()  # paths of joint parents
    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName())
        path = str(prim.GetPath())
        # if verbose:
        #     print(path, type_name)
        if type_name.endswith("Joint"):
            # the type name can sometimes be "DistancePhysicsJoint" or "PhysicsDistanceJoint" ...
            type_name = type_name.replace("Physics", "").replace("Joint", "")
            child = str(prim.GetRelationship("physics:body1").GetTargets()[0])
            pos0 = parse_vec(prim, "physics:localPos0", np.zeros(3, dtype=np.float32)) * linear_unit
            pos1 = parse_vec(prim, "physics:localPos1", np.zeros(3, dtype=np.float32)) * linear_unit
            rot0 = parse_quat(prim, "physics:localRot0", wp.quat_identity())
            rot1 = parse_quat(prim, "physics:localRot1", wp.quat_identity())
            joint_data[child] = {
                "type": type_name,
                "name": str(prim.GetName()),
                "parent_tf": wp.transform(pos0, rot0),
                "child_tf": wp.transform(pos1, rot1),
                "enabled": parse_generic(prim, "physics:jointEnabled", True),
                "collisionEnabled": parse_generic(prim, "physics:collisionEnabled", False),
                "excludeFromArticulation": parse_generic(prim, "physics:excludeFromArticulation", False),
                "axis": str2axis(parse_generic(prim, "physics:axis", "X")),
                "breakForce": parse_float(prim, "physics:breakForce", np.inf),
                "breakTorque": parse_float(prim, "physics:breakTorque", np.inf),
                "linear_axes": [],
                "angular_axes": [],
            }
            if only_load_enabled_joints and not joint_data[child]["enabled"]:
                print("Skipping disabled joint", path)
                continue
            # parse joint limits
            lower = parse_float(prim, "physics:lowerLimit", -np.inf)
            upper = parse_float(prim, "physics:upperLimit", np.inf)
            if type_name == "Distance":
                # if distance is negative the joint is not limited
                joint_data[child]["lowerLimit"] = parse_float(prim, "physics:minDistance", -1.0) * linear_unit
                joint_data[child]["upperLimit"] = parse_float(prim, "physics:maxDistance", -1.0) * linear_unit
            elif type_name == "Prismatic":
                joint_data[child]["lowerLimit"] = lower * linear_unit
                joint_data[child]["upperLimit"] = upper * linear_unit
            else:
                joint_data[child]["lowerLimit"] = np.deg2rad(lower) if np.isfinite(lower) else lower
                joint_data[child]["upperLimit"] = np.deg2rad(upper) if np.isfinite(upper) else upper

            if joint_data[child]["lowerLimit"] > joint_data[child]["upperLimit"]:
                joint_data[child]["lowerLimit"] = (
                    joint_data[child]["lowerLimit"] + joint_data[child]["upperLimit"]
                ) / 2
                joint_data[child]["upperLimit"] = joint_data[child]["lowerLimit"]
            parents = prim.GetRelationship("physics:body0").GetTargets()
            if len(parents) > 0:
                parent_path = str(parents[0])
                joint_data[child]["parent"] = parent_path
                joint_parents.add(parent_path)
            else:
                joint_data[child]["parent"] = None

            # parse joint drive
            parse_axis(prim, "angular", joint_data[child], is_angular=True)
            parse_axis(prim, "rotX", joint_data[child], is_angular=True, axis=(1.0, 0.0, 0.0))
            parse_axis(prim, "rotY", joint_data[child], is_angular=True, axis=(0.0, 1.0, 0.0))
            parse_axis(prim, "rotZ", joint_data[child], is_angular=True, axis=(0.0, 0.0, 1.0))
            parse_axis(prim, "linear", joint_data[child], is_angular=False)
            parse_axis(prim, "transX", joint_data[child], is_angular=False, axis=(1.0, 0.0, 0.0))
            parse_axis(prim, "transY", joint_data[child], is_angular=False, axis=(0.0, 1.0, 0.0))
            parse_axis(prim, "transZ", joint_data[child], is_angular=False, axis=(0.0, 0.0, 1.0))

        elif type_name == "Material":
            material = {}
            if has_attribute(prim, "physics:density"):
                material["density"] = parse_float(prim, "physics:density") * mass_unit  # / (linear_unit**3)
            if has_attribute(prim, "physics:restitution"):
                material["restitution"] = parse_float(prim, "physics:restitution", default_restitution)
            if has_attribute(prim, "physics:staticFriction"):
                material["staticFriction"] = parse_float(prim, "physics:staticFriction", default_mu)
            if has_attribute(prim, "physics:dynamicFriction"):
                material["dynamicFriction"] = parse_float(prim, "physics:dynamicFriction", default_mu)
            materials[path] = material

        elif type_name == "PhysicsScene":
            scene = UsdPhysics.Scene(prim)
            g_vec = scene.GetGravityDirectionAttr()
            g_mag = scene.GetGravityMagnitudeAttr()
            if g_mag.HasAuthoredValue() and np.isfinite(g_mag.Get()):
                builder.gravity = g_mag.Get() * linear_unit
            if g_vec.HasAuthoredValue() and np.linalg.norm(g_vec.Get()) > 0.0:
                builder.up_vector = np.array(g_vec.Get(), dtype=np.float32)  # TODO flip sign?
            else:
                builder.up_vector = upaxis

    def parse_prim(prim, incoming_xform, incoming_scale, parent_body: int = -1):
        nonlocal builder
        nonlocal joint_data
        nonlocal path_body_map
        nonlocal path_shape_map
        nonlocal path_world_poses
        nonlocal prim_joint_xforms
        nonlocal path_collision_filters
        nonlocal no_collision_shapes
        nonlocal body_density

        path = str(prim.GetPath())
        for pattern in ignore_paths:
            if re.match(pattern, path):
                return

        type_name = str(prim.GetTypeName())
        if (
            type_name.endswith("Joint")
            or type_name.endswith("Light")
            or type_name.endswith("Scene")
            or type_name.endswith("Material")
        ):
            return

        schemas = set(prim.GetAppliedSchemas())
        children_refs = prim.GetChildren()

        prim_joint_xforms[path] = wp.transform()

        local_xform, scale = parse_xform(prim)
        scale = incoming_scale * scale
        xform = wp.mul(incoming_xform, local_xform)
        path_world_poses[path] = xform

        geo_tf = local_xform
        body_id = parent_body
        is_rigid_body = "PhysicsRigidBodyAPI" in schemas and parent_body == -1
        create_rigid_body = is_rigid_body or path in joint_parents
        if create_rigid_body:
            body_id = builder.add_body(
                origin=xform,
                name=prim.GetName(),
            )
            path_body_map[path] = body_id
            body_density[body_id] = 0.0

            parent_body = body_id

            geo_tf = wp.transform()

            # set up joints between rigid bodies after the children have been added
            if path in joint_data:
                joint = joint_data[path]

                joint_params = dict(
                    child=body_id,
                    linear_axes=joint["linear_axes"],
                    angular_axes=joint["angular_axes"],
                    name=joint["name"],
                    enabled=joint["enabled"],
                    parent_xform=joint["parent_tf"],
                    child_xform=joint["child_tf"],
                )

                parent_path = joint["parent"]
                if parent_path is None:
                    joint_params["parent"] = -1
                    parent_tf = wp.transform()
                else:
                    joint_params["parent"] = path_body_map[parent_path]
                    parent_tf = path_world_poses[parent_path]

                # the joint to which we are connected will transform this body already
                geo_tf = wp.transform()

                if verbose:
                    print(f"Adding joint {joint['name']} between {joint['parent']} and {path}")
                    print("  parent_xform", joint["parent_tf"])
                    print("  child_xform ", joint["child_tf"])
                    print("  parent_tf   ", parent_tf)
                    print(f"  geo_tf at {path} = {geo_tf}  (xform was {xform})")

                if joint["type"] == "Revolute":
                    joint_params["joint_type"] = wp.sim.JOINT_REVOLUTE
                    if len(joint_params["angular_axes"]) == 0:
                        joint_params["angular_axes"].append(
                            wp.sim.JointAxis(
                                joint["axis"],
                                limit_lower=joint["lowerLimit"],
                                limit_upper=joint["upperLimit"],
                                limit_ke=joint_limit_ke,
                                limit_kd=joint_limit_kd,
                            )
                        )
                elif joint["type"] == "Prismatic":
                    joint_params["joint_type"] = wp.sim.JOINT_PRISMATIC
                    if len(joint_params["linear_axes"]) == 0:
                        joint_params["linear_axes"].append(
                            wp.sim.JointAxis(
                                joint["axis"],
                                limit_lower=joint["lowerLimit"],
                                limit_upper=joint["upperLimit"],
                                limit_ke=joint_limit_ke,
                                limit_kd=joint_limit_kd,
                            )
                        )
                elif joint["type"] == "Spherical":
                    joint_params["joint_type"] = wp.sim.JOINT_BALL
                elif joint["type"] == "Fixed":
                    joint_params["joint_type"] = wp.sim.JOINT_FIXED
                elif joint["type"] == "Distance":
                    joint_params["joint_type"] = wp.sim.JOINT_DISTANCE
                    # we have to add a dummy linear X axis to define the joint limits
                    joint_params["linear_axes"].append(
                        wp.sim.JointAxis(
                            (1.0, 0.0, 0.0),
                            limit_lower=joint["lowerLimit"],
                            limit_upper=joint["upperLimit"],
                            limit_ke=joint_limit_ke,
                            limit_kd=joint_limit_kd,
                        )
                    )
                elif joint["type"] == "":
                    joint_params["joint_type"] = wp.sim.JOINT_D6
                else:
                    print(f"Warning: unsupported joint type {joint['type']} for {path}")

                builder.add_joint(**joint_params)

            elif is_rigid_body:
                builder.add_joint_free(child=body_id)
                # free joint; we set joint_q/qd, not body_q/qd since eval_fk is used after model creation
                builder.joint_q[-4:] = xform.q
                builder.joint_q[-7:-4] = xform.p
                linear_vel = parse_vec(prim, "physics:velocity", np.zeros(3, dtype=np.float32)) * linear_unit
                angular_vel = parse_vec(prim, "physics:angularVelocity", np.zeros(3, dtype=np.float32)) * linear_unit
                builder.joint_qd[-6:-3] = angular_vel
                builder.joint_qd[-3:] = linear_vel

        if verbose:
            print(f"added {type_name} body {body_id} ({path}) at {xform}")

        density = None

        material = None
        if prim.HasRelationship("material:binding:physics"):
            other_paths = prim.GetRelationship("material:binding:physics").GetTargets()
            if len(other_paths) > 0:
                material = materials[str(other_paths[0])]
        if material is not None:
            if "density" in material:
                density = material["density"]
        if has_attribute(prim, "physics:density"):
            d = parse_float(prim, "physics:density")
            density = d * mass_unit  # / (linear_unit**3)

        # assert prim.GetAttribute('orientation').Get() == "rightHanded", "Only right-handed orientations are supported."
        enabled = parse_generic(prim, "physics:rigidBodyEnabled", True)
        if only_load_enabled_rigid_bodies and not enabled:
            if verbose:
                print("Skipping disabled rigid body", path)
            return
        mass = parse_float(prim, "physics:mass")
        if is_rigid_body:
            if density is None:
                density = default_density
            body_density[body_id] = density
        elif density is None:
            if body_id >= 0:
                density = body_density[body_id]
            else:
                density = 0.0

        com = parse_vec(prim, "physics:centerOfMass", np.zeros(3, dtype=np.float32))
        i_diag = parse_vec(prim, "physics:diagonalInertia", np.zeros(3, dtype=np.float32))
        i_rot = parse_quat(prim, "physics:principalAxes", wp.quat_identity())

        # parse children
        if type_name == "Xform":
            if prim.IsInstance():
                proto = prim.GetPrototype()
                for child in proto.GetChildren():
                    parse_prim(child, xform, scale, parent_body)
            else:
                for child in children_refs:
                    parse_prim(child, xform, scale, parent_body)
        elif type_name == "Scope":
            for child in children_refs:
                parse_prim(child, incoming_xform, incoming_scale, parent_body)
        elif type_name in shape_types:
            # parse shapes
            shape_params = dict(
                ke=default_ke, kd=default_kd, kf=default_kf, mu=default_mu, restitution=default_restitution
            )
            if material is not None:
                if "restitution" in material:
                    shape_params["restitution"] = material["restitution"]
                if "dynamicFriction" in material:
                    shape_params["mu"] = material["dynamicFriction"]

            if has_attribute(prim, "doubleSided") and not prim.GetAttribute("doubleSided").Get():
                print(f"Warning: treating {path} as double-sided because single-sided collisions are not supported.")

            if type_name == "Cube":
                size = parse_float(prim, "size", 2.0)
                if has_attribute(prim, "extents"):
                    extents = parse_vec(prim, "extents") * scale
                    # TODO position geom at extents center?
                    geo_pos = 0.5 * (extents[0] + extents[1])
                    extents = extents[1] - extents[0]
                else:
                    extents = scale * size
                shape_id = builder.add_shape_box(
                    body_id,
                    geo_tf.p,
                    geo_tf.q,
                    hx=extents[0] / 2,
                    hy=extents[1] / 2,
                    hz=extents[2] / 2,
                    density=density,
                    thickness=default_thickness,
                    **shape_params,
                )
            elif type_name == "Sphere":
                if not (scale[0] == scale[1] == scale[2]):
                    print("Warning: Non-uniform scaling of spheres is not supported.")
                if has_attribute(prim, "extents"):
                    extents = parse_vec(prim, "extents") * scale
                    # TODO position geom at extents center?
                    geo_pos = 0.5 * (extents[0] + extents[1])
                    extents = extents[1] - extents[0]
                    if not (extents[0] == extents[1] == extents[2]):
                        print("Warning: Non-uniform extents of spheres are not supported.")
                    radius = extents[0]
                else:
                    radius = parse_float(prim, "radius", 1.0) * scale[0]
                shape_id = builder.add_shape_sphere(
                    body_id, geo_tf.p, geo_tf.q, radius, density=density, **shape_params
                )
            elif type_name == "Plane":
                normal_str = parse_generic(prim, "axis", "Z").upper()
                geo_rot = geo_tf.q
                if normal_str != "Y":
                    normal = str2axis(normal_str)
                    c = np.cross(normal, (0.0, 1.0, 0.0))
                    angle = np.arcsin(np.linalg.norm(c))
                    axis = c / np.linalg.norm(c)
                    geo_rot = wp.mul(geo_rot, wp.quat_from_axis_angle(axis, angle))
                width = parse_float(prim, "width", 0.0) * scale[0]
                length = parse_float(prim, "length", 0.0) * scale[1]
                shape_id = builder.add_shape_plane(
                    body=body_id,
                    pos=geo_tf.p,
                    rot=geo_rot,
                    width=width,
                    length=length,
                    thickness=default_thickness,
                    **shape_params,
                )
            elif type_name == "Capsule":
                axis_str = parse_generic(prim, "axis", "Z").upper()
                radius = parse_float(prim, "radius", 0.5) * scale[0]
                half_height = parse_float(prim, "height", 2.0) / 2 * scale[1]
                assert not has_attribute(prim, "extents"), "Capsule extents are not supported."
                shape_id = builder.add_shape_capsule(
                    body_id,
                    geo_tf.p,
                    geo_tf.q,
                    radius,
                    half_height,
                    density=density,
                    up_axis="XYZ".index(axis_str),
                    **shape_params,
                )
            elif type_name == "Cylinder":
                axis_str = parse_generic(prim, "axis", "Z").upper()
                radius = parse_float(prim, "radius", 0.5) * scale[0]
                half_height = parse_float(prim, "height", 2.0) / 2 * scale[1]
                assert not has_attribute(prim, "extents"), "Cylinder extents are not supported."
                shape_id = builder.add_shape_cylinder(
                    body_id,
                    geo_tf.p,
                    geo_tf.q,
                    radius,
                    half_height,
                    density=density,
                    up_axis="XYZ".index(axis_str),
                    **shape_params,
                )
            elif type_name == "Cone":
                axis_str = parse_generic(prim, "axis", "Z").upper()
                radius = parse_float(prim, "radius", 0.5) * scale[0]
                half_height = parse_float(prim, "height", 2.0) / 2 * scale[1]
                assert not has_attribute(prim, "extents"), "Cone extents are not supported."
                shape_id = builder.add_shape_cone(
                    body_id,
                    geo_tf.p,
                    geo_tf.q,
                    radius,
                    half_height,
                    density=density,
                    up_axis="XYZ".index(axis_str),
                    **shape_params,
                )
            elif type_name == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
                indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.float32)
                counts = mesh.GetFaceVertexCountsAttr().Get()
                faces = []
                face_id = 0
                for count in counts:
                    if count == 3:
                        faces.append(indices[face_id : face_id + 3])
                    elif count == 4:
                        faces.append(indices[face_id : face_id + 3])
                        faces.append(indices[[face_id, face_id + 2, face_id + 3]])
                    else:
                        # assert False, f"Error while parsing USD mesh {path}: encountered polygon with {count} vertices, but only triangles and quads are supported."
                        continue
                    face_id += count
                m = wp.sim.Mesh(points, np.array(faces, dtype=np.int32).flatten())
                shape_id = builder.add_shape_mesh(
                    body_id,
                    geo_tf.p,
                    geo_tf.q,
                    scale=scale,
                    mesh=m,
                    density=density,
                    thickness=default_thickness,
                    **shape_params,
                )
            else:
                print(f"Warning: Unsupported geometry type {type_name} at {path}.")
                return

            path_body_map[path] = body_id
            path_shape_map[path] = shape_id

            if prim.HasRelationship("physics:filteredPairs"):
                other_paths = prim.GetRelationship("physics:filteredPairs").GetTargets()
                for other_path in other_paths:
                    path_collision_filters.add((path, str(other_path)))

            if "PhysicsCollisionAPI" not in schemas or not parse_generic(prim, "physics:collisionEnabled", True):
                no_collision_shapes.add(shape_id)

        else:
            print(f"Warning: encountered unsupported prim type {type_name}")

        # update mass properties of rigid bodies in cases where properties are defined with higher precedence
        if body_id >= 0:
            com = parse_vec(prim, "physics:centerOfMass")
            if com is not None:
                # overwrite COM
                builder.body_com[body_id] = com * scale

            if mass is not None and not (is_rigid_body and mass == 0.0):
                mass_ratio = mass / builder.body_mass[body_id]
                # mass has precedence over density, so we overwrite the mass computed from density
                builder.body_mass[body_id] = mass * mass_unit
                if mass > 0.0:
                    builder.body_inv_mass[body_id] = 1.0 / builder.body_mass[body_id]
                else:
                    builder.body_inv_mass[body_id] = 0.0
                # update inertia
                builder.body_inertia[body_id] *= mass_ratio
                if builder.body_inertia[body_id].any():
                    builder.body_inv_inertia[body_id] = np.linalg.inv(builder.body_inertia[body_id])
                else:
                    builder.body_inv_inertia[body_id] = np.zeros((3, 3), dtype=np.float32)

            if np.linalg.norm(i_diag) > 0.0:
                rot = np.array(wp.quat_to_matrix(i_rot), dtype=np.float32).reshape(3, 3)
                inertia = rot @ np.diag(i_diag) @ rot.T
                builder.body_inertia[body_id] = inertia
                if inertia.any():
                    builder.body_inv_inertia[body_id] = np.linalg.inv(inertia)
                else:
                    builder.body_inv_inertia[body_id] = np.zeros((3, 3), dtype=np.float32)

    parse_prim(
        stage.GetDefaultPrim(), incoming_xform=wp.transform(), incoming_scale=np.ones(3, dtype=np.float32) * linear_unit
    )

    shape_count = len(builder.shape_geo_type)

    # apply collision filters now that we have added all shapes
    for path1, path2 in path_collision_filters:
        shape1 = path_shape_map[path1]
        shape2 = path_shape_map[path2]
        builder.shape_collision_filter_pairs.add((shape1, shape2))

    # apply collision filters to all shapes that have no collision
    for shape_id in no_collision_shapes:
        for other_shape_id in range(shape_count):
            if other_shape_id != shape_id:
                builder.shape_collision_filter_pairs.add((shape_id, other_shape_id))

    # return stage parameters
    return {
        "fps": stage.GetFramesPerSecond(),
        "duration": stage.GetEndTimeCode() - stage.GetStartTimeCode(),
        "up_axis": UsdGeom.GetStageUpAxis(stage).lower(),
    }
