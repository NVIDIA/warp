"""
This is the implementation of the OGN node defined in OgnParticleVolume.ogn
"""

# Array or tuple values are accessed as numpy arrays so you probably need this import
import math

import numpy as np

from pxr import Usd, UsdGeom, Gf, Sdf

import omni.timeline
import omni.appwindow
import omni.usd

# helper to get the transform for a bundle prim
def read_transform_bundle(bundle):
    timeline =  omni.timeline.get_timeline_interface()
    time = timeline.get_current_time()*timeline.get_time_codes_per_seconds()

    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Xformable(stage.GetPrimAtPath(bundle.bundle.get_prim_path()))
    return prim.ComputeLocalToWorldTransform(time)

def read_bounds_bundle(bundle):
    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Xformable(stage.GetPrimAtPath(bundle.bundle.get_prim_path()))
    
    return prim.ComputeLocalBound(0.0, purpose1="default")

class OgnParticleVolumeState:

    def __init__(self):

        self.initialized = False

class OgnParticleVolume:

    @staticmethod
    def internal_state():

        return OgnParticleVolumeState()

    """
    """
    @staticmethod
    def compute(db) -> bool:
        """Run simulation"""

        state = db.internal_state

        if db.inputs.volume.valid and (state.initialized == False or db.inputs.execIn):

            bounds = read_bounds_bundle(db.inputs.volume)
            (b, r, scale, u, t, p) = bounds.GetMatrix().Factor()
            size = bounds.GetRange().GetSize()

            if (db.inputs.spacing <= 0.0):
                print("Spacing must be positive")
                return False

            dim_x = int(size[0]/db.inputs.spacing)
            dim_y = int(size[1]/db.inputs.spacing)
            dim_z = int(size[2]/db.inputs.spacing)

            print(f"Creating particle grid with dim {dim_x}x{dim_y}x{dim_z}, bounds {size[0]}, {size[1]}, {size[2]}, spacing {db.inputs.spacing}")

            if (dim_x*dim_y*dim_z > 1024*1024):
                print("Trying to create particle volume with > 1M particles, increase spacing or geometry size")
                return False

            def particle_grid(dim_x, dim_y, dim_z, lower, radius, jitter):
                points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
                points_t = np.array((points[0], points[1], points[2])).T*radius + np.array(lower)
                points_t = points_t + np.random.rand(*points_t.shape)*radius*jitter - radius*jitter*0.5
                
                return points_t.reshape((-1, 3))

            points_local = particle_grid(dim_x, dim_y, dim_z, bounds.GetRange().GetMin(), db.inputs.spacing, db.inputs.spacing_jitter)
            points_world = []

            # transform to world space
            xform = read_transform_bundle(db.inputs.volume)

            for p in points_local:
                points_world.append(xform.Transform(Gf.Vec3f(p[0], p[1], p[2])))

            point_count = len(points_world)

            state.initialized = True

            db.outputs.positions_size = point_count
            db.outputs.positions[:] = points_world

            db.outputs.protoIndices_size = point_count
            db.outputs.protoIndices[:] = np.zeros(point_count)

        return True
