# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This is the implementation of the OGN node defined in OgnMarchingCubes.ogn
"""

# Array or tuple values are accessed as numpy arrays so you probably need this import
import math

import numpy as np
import warp as wp


class OgnMarchingCubesState:

    def __init__(self):
        self.mc = None


class OgnMarchingCubes:

    @staticmethod
    def internal_state():

        return OgnMarchingCubesState()

    """
    """
    @staticmethod
    def compute(db) -> bool:
        """Run simulation"""

        if db.inputs.execIn:

            state = db.internal_state
        
            with wp.ScopedDevice("cuda:0"):

                dim = (db.inputs.volume.attribute_by_name("dim_x").value,
                       db.inputs.volume.attribute_by_name("dim_y").value,
                       db.inputs.volume.attribute_by_name("dim_z").value) 

                if state.mc == None:
                    state.mc = wp.MarchingCubes(dim[0], dim[1], dim[2], db.inputs.max_vertices, db.inputs.max_triangles)

                # resize in case any dimensions changed    
                state.mc.resize(dim[0], dim[1], dim[2], db.inputs.max_vertices, db.inputs.max_triangles)

                # alias the incoming memory to a Warp array
                ptr = db.inputs.volume.attribute_by_name("data").value
                field = wp.array(ptr=ptr, dtype=float, shape=dim, owner=False)

                with wp.ScopedTimer("Surface Extraction", active=False):
                    state.mc.surface(field, db.inputs.threshold)

                num_verts = len(state.mc.verts)
                num_tris = int(len(state.mc.indices)/3)

                # print(f"{num_verts}, {num_tris}")

                db.outputs.points_size = num_verts
                db.outputs.points[:] = state.mc.verts.numpy()

                db.outputs.faceVertexCounts_size = num_tris
                db.outputs.faceVertexCounts[:] = [3]*num_tris

                db.outputs.faceVertexIndices_size = num_tris*3
                db.outputs.faceVertexIndices[:] = state.mc.indices.numpy()

            db.outputs.execOut = db.inputs.execIn

        return True
