"""Helper functions to parse primitves and meshes coming from an 'import USD prim data'.

Use shapeinfos_from_bundle for initial parsing and update_shapeinfos_from_bundle for subsequent updates

Supported types are mesh, sphere, box, and capsule
"""

import math
import numpy as np
import warp as wp
from warp.types import transform

from pxr import Gf

# update mesh data given two sets of collider positions
# computes velocities and transforms points to world-space
@wp.kernel
def transform_mesh(collider_current: wp.array(dtype=wp.vec3),
                   collider_previous: wp.array(dtype=wp.vec3),
                   xform_current: wp.mat44,
                   xform_previous: wp.mat44,
                   mesh_points: wp.array(dtype=wp.vec3),
                   mesh_velocities: wp.array(dtype=wp.vec3),
                   dt: float,
                   alpha: float):

    tid = wp.tid()

    local_p1 = collider_current[tid]
    local_p0 = collider_previous[tid]

    world_p1 = wp.transform_point(xform_current, local_p1)
    world_p0 = wp.transform_point(xform_previous, local_p0)

    p = world_p1 * alpha + world_p0 * (1.0 - alpha)
    v = (world_p1 - world_p0) / dt

    mesh_points[tid] = p
    mesh_velocities[tid] = v


def triangulate(counts, indices):
    num_tris = np.sum(np.subtract(counts, 2))
    num_tri_vtx = num_tris * 3
    tri_indices = np.zeros(num_tri_vtx, dtype=int)
    ctr = 0
    wedgeIdx = 0
    for nb in counts:
        for i in range(nb - 2):
            tri_indices[ctr] = indices[wedgeIdx]
            tri_indices[ctr + 1] = indices[wedgeIdx + i + 1]
            tri_indices[ctr + 2] = indices[wedgeIdx + i + 2]
            ctr += 3
        wedgeIdx += nb
    return tri_indices


class ShapeInfo:
    def __init__(self):
        self.type = None
        self.path = ""
        self.pos = np.zeros(3)
        self.rot = wp.quat_identity()
        self.transform = wp.transform(self.pos, self.rot)
        self.scale = np.ones(4)
        self.mesh = None
        self.body = -1

        self.density = 1000.0
        self.ke = 1.e+5
        self.kd = 1000.0
        self.kf = 1000.0
        self.mu = 0.5

    def update_transform(self):
        self.transform = wp.transform(self.pos, self.rot)

    def add_to(self, builder: wp.sim.ModelBuilder):
        builder._add_shape(self.body, self.pos, self.rot, self.type, self.scale, self.mesh, self.density, self.ke, self.kd, self.kf, self.mu)

def _bundle_to_priminfo(bundle):
    names_of_interest = {
        'Mesh': {"faceVertexCounts", "faceVertexIndices", "points", "xformOp", },
        'Sphere': {"radius", "xformOp", },
        'Cube': {"size", "xformOp", },
        'Capsule': {"axis", "height", "radius", "xformOp", },
    }

    prim_types = bundle.attribute_by_name("__primTypes").value
    prim_paths = bundle.attribute_by_name("__primPaths").value
    priminfos = []
    transforms = []
    for type, path in zip(prim_types, prim_paths):
        priminfos.append({'type': type, 'path': path})

    for attr in bundle.attributes:
        if attr.name == "transform":
            transforms = attr.value
            continue

        comps = attr.name.split(":")
        if "primTime" != comps[0] and "prim" == comps[0][0:4]:
            id = int(comps[0][4:])
            if prim_types[id] not in names_of_interest:
                continue

            if comps[1] in names_of_interest[prim_types[id]]:
                prim_attr_name = comps[2] if comps[1] == "xformOp" else comps[1]
                priminfos[id][prim_attr_name] = attr.value
    
    assert len(transforms) == len(priminfos)
    for prim, xform in zip(priminfos, transforms):
        prim["transform"] = xform

    return priminfos


def _transform_from_prim(prim):
    return Gf.Matrix4d(prim["transform"].reshape(4,4))


def shapeinfos_from_bundle(bundle, device):
    """Helper function to describe USD primitives in a bundle as Warp shapes.
    Returns a list of ShapeInfos, each having all the parameters needed by ModelBuilder._add_shape

    Args:
        bundle: output of an 'import USD prim data' node from omni.graph.io
    """

    priminfos = _bundle_to_priminfo(bundle)

    shape_infos = []
    for prim in priminfos:
        try:
            shape = ShapeInfo()

            if prim["type"] == "Mesh":
                shape.type = wp.sim.GEO_MESH

                points = prim["points"]
                indices = prim["faceVertexIndices"]
                counts = prim["faceVertexCounts"]
                xform = _transform_from_prim(prim)

                shape.current_positions = wp.array(points, dtype=wp.vec3, device=device)
                shape.previous_positions = wp.array(points, dtype=wp.vec3, device=device)
                shape.current_transform = xform.GetTranspose()

                world_positions = []
                for i in range(len(points)):
                    world_positions.append(xform.Transform(Gf.Vec3f(tuple(points[i]))))

                shape.mesh = wp.sim.Mesh(
                    world_positions,
                    triangulate(counts, indices),
                    compute_inertia=False)
            else:
                rotXYZ = np.array(prim["rotateXYZ"]) * math.pi / 180.0
                shape.pos = np.array(prim["translate"])
                shape.rot = wp.quat_rpy(rotXYZ[0], rotXYZ[1], rotXYZ[2])

                if prim["type"] == "Sphere":
                    shape.type = wp.sim.GEO_SPHERE
                    shape.scale = (prim["radius"], 0.0, 0.0, 0.0)
                elif prim["type"] == "Cube":
                    shape.type = wp.sim.GEO_BOX
                    shape.scale = np.append(prim["scale"] * (0.5 * prim["size"]), 0.0)
                elif prim["type"] == "Capsule":
                    shape.type = wp.sim.GEO_CAPSULE
                    shape.scale = (prim["radius"], prim["height"] * 0.5, 0.0, 0.0)
                    if prim["axis"] == "Y":
                        shape.rot = wp.quat_multiply(shape.rot, wp.quat_rpy(0, 0, -math.pi / 2))
                    if prim["axis"] == "Z":
                        shape.rot = wp.quat_multiply(shape.rot, wp.quat_rpy(0, math.pi / 2, 0))
                else:
                    print(f"Unsupported primitive type {prim['type']} - ignoring")
                    continue

            shape.path = prim["path"]
            shape.update_transform()
            shape_infos.append(shape)
        except KeyError as err:
            print(f"[shapeinfos_from_bundle] Missing attribute {err.args[0]} for {prim['path']}")

    return shape_infos

def update_shapeinfos_from_bundle(shapeinfos, bundle, dt, device):
    """
    """
    priminfos = _bundle_to_priminfo(bundle)
    prim_dict = {}
    for prim in priminfos:
        prim_dict[prim["path"]] = prim

    for shape in shapeinfos:
        try:
            prim = prim_dict[shape.path]
        except KeyError as err:
            print(f"\"{err.args[0]}\" is no longer on bundle. Skipping update")
            continue

        try:
            if shape.type == wp.sim.GEO_MESH:
                # everywhere: if dirty
                shape.previous_transform = shape.current_transform
                shape.current_transform = _transform_from_prim(prim).GetTranspose()

                shape.current_positions, shape.previous_positions = shape.previous_positions, shape.current_positions
                points_host = wp.array(prim["points"], dtype=wp.vec3, copy=False, device="cpu")
                wp.copy(shape.current_positions, points_host)
                alpha = 0
                wp.launch(
                    kernel=transform_mesh,
                    dim=len(shape.mesh.vertices),
                    inputs=[shape.current_positions,
                            shape.previous_positions,
                            shape.current_transform,
                            shape.previous_transform,
                            shape.mesh.mesh.points,
                            shape.mesh.mesh.velocities,
                            dt,
                            alpha],
                    device=device)
                shape.mesh.mesh.refit()
            else:
                rotXYZ = np.array(prim["rotateXYZ"]) * math.pi / 180.0
                shape.pos = np.array(prim["translate"])
                shape.rot = wp.quat_rpy(rotXYZ[0], rotXYZ[1], rotXYZ[2])
                if shape.type == wp.sim.GEO_SPHERE:
                    shape.scale = (prim["radius"], 0.0, 0.0, 0.0)
                elif shape.type == wp.sim.GEO_BOX:
                    shape.scale = np.append(prim["scale"] * (0.5 * prim["size"]), 0.0)
                elif shape.type == wp.sim.GEO_CAPSULE:
                    shape.scale = (prim["radius"], prim["height"] * 0.5, 0.0, 0.0)
                    if prim["axis"] == "Y":
                        shape.rot = wp.quat_multiply(shape.rot, wp.quat_rpy(0, 0, -math.pi / 2))
                    if prim["axis"] == "Z":
                        shape.rot = wp.quat_multiply(shape.rot, wp.quat_rpy(0, math.pi / 2, 0))
                shape.update_transform()
        except KeyError as err:
            print(f"[update_shapeinfos_from_bundle] Missing attribute {err.args[0]} for {prim['path']}")
