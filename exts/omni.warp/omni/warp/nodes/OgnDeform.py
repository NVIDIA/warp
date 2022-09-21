# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This is the implementation of the OGN node defined in OgnDeform.ogn
"""

import math
import time

import numpy as np
import warp as wp

@wp.kernel
def deform(points_in: wp.array(dtype=wp.vec3),
            points_out: wp.array(dtype=wp.vec3),
            time: float):

    # get thread-id
    tid = wp.tid()

    # sine-wave deformer
    points_out[tid] = points_in[tid] + wp.vec3(0.0, wp.sin(time + points_in[tid][0]*0.1)*10.0, 0.0)
    

class OgnDeform:

    @staticmethod
    def compute(db) -> bool:

        with wp.ScopedDevice("cuda:0"):

            if len(db.inputs.points):

                # convert node inputs to a GPU array
                points_in = wp.array(db.inputs.points, dtype=wp.vec3)
                points_out = wp.zeros_like(points_in)

                # launch deformation kernel
                wp.launch(kernel=deform, dim=len(points_in), inputs=[points_in, points_out, db.inputs.time])

                # write node outputs
                db.outputs.points = points_out.numpy()
