# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This is the implementation of the OGN node defined in OgnProceduralVolume.ogn
"""

# Array or tuple values are accessed as numpy arrays so you probably need this import
import math

import numpy as np
import warp as wp

import omni.graph.core as og


@wp.func
def sdf_plane(p: wp.vec3, plane: wp.vec4):
    return plane[0]*p[0] + plane[1]*p[1] + plane[2]*p[2] + plane[3]

# signed sphere 
@wp.func
def sdf_sphere(p: wp.vec3, r: float):
    return wp.length(p) - r

# signed box 
@wp.func
def sdf_box(upper: wp.vec3, p: wp.vec3):

    qx = wp.abs(p[0])-upper[0]
    qy = wp.abs(p[1])-upper[1]
    qz = wp.abs(p[2])-upper[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))
    
    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)

# union 
@wp.func
def op_union(d1: float, d2: float):
    return wp.min(d1, d2)


@wp.func
def op_smooth_union(d1: float, d2: float, k: float):

    a = d1
    b = d2

    h = wp.clamp(0.5+0.5*(b-a)/k, 0.0, 1.0 )
    return wp.lerp(b, a, h) - k*h*(1.0-h)



# subtraction
@wp.func
def op_subtract(d1: float, d2: float):
    return wp.max(-d1, d2)

# intersection
@wp.func
def op_intersect(d1: float, d2: float):
    return wp.max(d1, d2)
    

@wp.kernel
def make_field(field: wp.array3d(dtype=float),
               center: wp.vec3,
               radius: float,
               time: float):

    i, j, k = wp.tid()

    p = wp.vec3(float(i), float(j), float(k))

    rng = wp.rand_init(42)
    noise = wp.noise(rng, wp.vec4(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5, 0.0)*0.25)

    center = center - wp.vec3(0.0, wp.sin(time), 0.0)*30.0

    sphere = 2.0*noise + wp.length(p - center) - radius
    box = sdf_box(wp.vec3(16.0, 32.0, 16.0), p-center)
    plane = sdf_plane(p, wp.vec4(0.0, 1.0, 0.0, 0.0))

    d = op_intersect(sphere, box)
    d = op_smooth_union(d, plane, 32.0)

    field[i,j,k] = d


def add_bundle_data(bundle, name, data, type):
    
    attr = bundle.attribute_by_name(name)
    if attr is None:
        attr = bundle.insert((type, name))

    attr.cpu_value = data


class OgnProceduralVolumeState:

    def __init__(self):
        self.field = None
        self.dim = None

class OgnProceduralVolume:

    @staticmethod
    def internal_state():

        return OgnProceduralVolumeState()

    """
    """
    @staticmethod
    def compute(db) -> bool:
        """Run simulation"""

        if db.inputs.execIn:

            state = db.internal_state
            time = db.inputs.time
            dim = (db.inputs.dim_x, db.inputs.dim_y, db.inputs.dim_z)

            with wp.ScopedDevice("cuda:0"):
               
                if state.dim != dim:
                    state.field = wp.zeros(shape=dim, dtype=float)
                    state.dim = dim

                wp.launch(make_field, dim=state.field.shape, inputs=[state.field, wp.vec3(dim[0]/2, dim[1]/4, dim[2]/2), dim[0]/4, time])

                add_bundle_data(db.outputs.volume, name="dim_x", data=db.inputs.dim_x, type=og.Type(og.BaseDataType.INT, tuple_count=1, array_depth=0))
                add_bundle_data(db.outputs.volume, name="dim_y", data=db.inputs.dim_y, type=og.Type(og.BaseDataType.INT, tuple_count=1, array_depth=0))
                add_bundle_data(db.outputs.volume, name="dim_z", data=db.inputs.dim_z, type=og.Type(og.BaseDataType.INT, tuple_count=1, array_depth=0))
                add_bundle_data(db.outputs.volume, name="data", data=state.field.ptr, type=og.Type(og.BaseDataType.UINT64, tuple_count=1, array_depth=0))

            db.outputs.execOut = db.inputs.execIn

        return True
