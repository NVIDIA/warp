# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Simple 2D wave-equation solver
"""

import math

import numpy as np
import warp as wp
import ctypes

import omni.timeline
import omni.usd
import omni.warp

from pxr import Usd, UsdGeom, Gf, Sdf

MAX_COLLIDERS = 4

# helper to read a USD xform out of graph inputs
def read_transform(input):
    xform = Gf.Matrix4d(input.reshape((4,4)))
    return xform

def read_xformable_bundle(bundle):
    stage = omni.usd.get_context().get_stage()
    prim_path = bundle.attribute_by_name("sourcePrimPath").value
    prim = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))

    return prim

# helper to get the transform for a bundle prim
def read_transform_bundle(bundle):

    timeline =  omni.timeline.get_timeline_interface()
    time = timeline.get_current_time()*timeline.get_time_codes_per_seconds()

    prim = read_xformable_bundle(bundle)
    return prim.ComputeLocalToWorldTransform(time)

def read_bounds_bundle(bundle):

    prim = read_xformable_bundle(bundle)
    return prim.ComputeLocalBound(0.0, purpose1="default")

def translate_bundle(bundle, pos):

    prim = read_xformable_bundle(bundle)   
    xform_ops = prim.GetOrderedXformOps()
    p = xform_ops[0].Get()

    xform_ops[0].Set(p + pos)



#kernel definitions
@wp.func
def sample(f: wp.array(dtype=float),
           x: int,
           y: int,
           width: int,
           height: int):

    # clamp texture coords
    x = wp.clamp(x, 0, width-1)
    y = wp.clamp(y, 0, height-1)
    
    s = f[y*width + x]
    return s

@wp.func
def laplacian(f: wp.array(dtype=float),
              x: int,
              y: int,
              width: int,
              height: int):
    
    ddx = sample(f, x+1, y, width, height) - 2.0*sample(f, x,y, width, height) + sample(f, x-1, y, width, height)
    ddy = sample(f, x, y+1, width, height) - 2.0*sample(f, x,y, width, height) + sample(f, x, y-1, width, height)

    return (ddx + ddy)

@wp.kernel
def wave_displace(hcurrent: wp.array(dtype=float),
                  hprevious: wp.array(dtype=float),
                  width: int,
                  height: int,
                  center_x: float,
                  center_y: float,
                  r: float,
                  mag: float,
                  t: float):

    tid = wp.tid()

    x = tid%width
    y = tid//width

    dx = float(x) - center_x
    dy = float(y) - center_y

    dist_sq = float(dx*dx + dy*dy)

    if (dist_sq < r*r):
        h = mag*wp.sin(t)
         
        hcurrent[tid]= h
        hprevious[tid] = h

@wp.kernel
def wave_solve(hprevious: wp.array(dtype=float),
               hcurrent: wp.array(dtype=float),
               width: int,
               height: int,
               inv_cell: float,
               k_speed: float,
               k_damp: float,
               dt: float):

    tid = wp.tid()

    x = tid%width
    y = tid//width

    l = laplacian(hcurrent, x, y, width, height)*inv_cell*inv_cell

    # integrate 
    h1 = hcurrent[tid]
    h0 = hprevious[tid]
    
    h = 2.0*h1 - h0 + dt*dt*(k_speed*l - k_damp*(h1-h0))

    # buffers get swapped each iteration
    hprevious[tid] = h


# simple kernel to apply height deltas to a vertex array
@wp.kernel
def grid_update(heights: wp.array(dtype=float),
                vertices: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    h = heights[tid]
    v = vertices[tid]

    v_new = wp.vec3(v[0], v[1], h)

    vertices[tid] = v_new



class OgnRippleState:

    def __init__(self):
        self.initialized = False

        self.sim_time = 0.0


class OgnRipple:
    """
    """

    @staticmethod
    def internal_state():

        return OgnRippleState()

    @staticmethod
    def compute(db) -> bool:
        """Compute the outputs from the current input"""

        timeline =  omni.timeline.get_timeline_interface()
        time = timeline.get_current_time()*timeline.get_time_codes_per_seconds()
        
        sim_fps = 60.0
        sim_substeps = 1
        sim_dt = (1.0/sim_fps)/sim_substeps
        
        grid_size = db.inputs.resolution

        # wave constants
        k_speed = db.inputs.speed
        k_damp = db.inputs.damp

        state = db.internal_state

        with wp.ScopedDevice("cuda:0"):

            # initialization
            if db.inputs.grid.valid and state.initialized is False:

                bounds = read_bounds_bundle(db.inputs.grid)
                (b, r, scale, u, t, p) = bounds.GetMatrix().Factor()
                size = bounds.GetRange().GetSize()

                state.sim_width = int(size[0]*scale[0]/grid_size)
                state.sim_height = int(size[1]*scale[1]/grid_size)

                print(f"Creating grid with dim {state.sim_width}, {state.sim_height}")

                state.sim_grid0 = wp.zeros(state.sim_width*state.sim_height, dtype=float)
                state.sim_grid1 = wp.zeros(state.sim_width*state.sim_height, dtype=float)

                state.sim_host = wp.zeros(state.sim_width*state.sim_height, dtype=float, device="cpu")
                
                state.verts_device =  wp.zeros(state.sim_width*state.sim_height, dtype=wp.vec3)
                state.verts_host =  wp.zeros(state.sim_width*state.sim_height, dtype=wp.vec3, device="cpu")

                # build grid mesh
                def grid_index(x, y, stride):
                    return y*stride + x

                #state.vertices = np.zeros(shape=(state.sim_width*state.sim_height, 3), dtype=np.float32)#state.verts_host.numpy().reshape((state.sim_width*state.sim_height, 3))
                state.vertices = state.verts_host.numpy().reshape((state.sim_width*state.sim_height, 3))
                state.indices = []
                state.counts = []

                for z in range(state.sim_height):
                    for x in range(state.sim_width):

                        center = np.array([0.5*state.sim_width*grid_size, 0.5*state.sim_height*grid_size, 0.0])
                        pos = (float(x)*grid_size, float(z)*grid_size, 0.0)

                        state.vertices[z*state.sim_width + x,:] = np.array(pos) - center

                        if (x > 0 and z > 0):
                            
                            state.indices.append(grid_index(x-1, z-1, state.sim_width))
                            state.indices.append(grid_index(x, z, state.sim_width))
                            state.indices.append(grid_index(x, z-1, state.sim_width))

                            state.indices.append(grid_index(x-1, z-1, state.sim_width))
                            state.indices.append(grid_index(x-1, z, state.sim_width))
                            state.indices.append(grid_index(x, z, state.sim_width))

                            state.counts.append(3)
                            state.counts.append(3)

                # initialize device vertices
                wp.copy(state.verts_device, state.verts_host)

                # todo: this is very slow to every frame, even when not connected, figure out how to improve this
                db.outputs.face_indices = state.indices
                db.outputs.face_counts = state.counts

                colliders = [db.inputs.collider_0,
                             db.inputs.collider_1,
                             db.inputs.collider_2,
                             db.inputs.collider_3]

                state.collider_vel = []
                state.collider_pos = []

                for i in range(MAX_COLLIDERS):
                    
                    state.collider_pos.append(Gf.Vec3d(0.0, 0.0, 0.0))
                    state.collider_vel.append(Gf.Vec3d(0.0, 0.0, 0.0))

                    if (colliders[i].valid):
                        state.collider_pos[-1] = read_transform_bundle(db.inputs.collider_0).ExtractTranslation()

                state.initialized = True

            if state.initialized:

                if timeline.is_playing():

                    def update_collider(collider, density, index):

                        if collider.valid:

                            grid_displace = db.inputs.displace
                            grid_xform = read_transform_bundle(db.inputs.grid).RemoveScaleShear()
                            collider_xform = read_transform_bundle(collider)

                            bounds = read_bounds_bundle(collider).ComputeAlignedBox()
                            radius = bounds.GetSize()[0]*0.5

                            # get new collider pos, if it is different from previous
                            # we assume it is being manipulated by the user and zero its velocity
                            collider_pos = collider_xform.ExtractTranslation()

                            # transform collider to grid local space
                            local_pos = grid_xform.GetInverse().Transform(collider_pos)

                            if ((local_pos - state.collider_pos[index]).GetLength() > 1.e-3):
                                state.collider_vel[index] = Gf.Vec3d(0.0, 0.0, 0.0)
                                state.collider_pos[index] = local_pos 


                            # create surface displacment around a point
                            cx = float(local_pos[0])/grid_size + state.sim_width*0.5
                            cz = float(local_pos[1])/grid_size + state.sim_height*0.5
                            cr = float(radius)/grid_size

                            # clamp coords to grid
                            cx = max(0, min(state.sim_width-1, cx))
                            cz = max(0, min(state.sim_height-1, cz))

                            # sample height
                            grid = state.sim_host.numpy()
                            height = grid[int(cz)*state.sim_width + int(cx)]

                            dt = 1.0/60.0

                            gravity =  Gf.Vec3d(0.0, 0.0, db.inputs.gravity)
                            buoyancy = Gf.Vec3d(0.0, 0.0, 0.0)
                            damp = Gf.Vec3d(0.0, 0.0, 0.0)

                            com = local_pos[2] - density

                            if (com < height):

                                # linear buoyancy force (approximates volume term by depth)
                                buoyancy = Gf.Vec3d(0.0, 0.0, float(height-com)*db.inputs.buoyancy)

                                # linear drag model 
                                v = state.collider_vel[index][2]
                                f = abs(v)*db.inputs.buoyancy_damp

                                # quadratic drag model
                                # v = state.collider_vel[index][2]
                                # f = v*v*db.inputs.buoyancy_damp
                                
                                # stability limit
                                if (f*dt > abs(v)):
                                    f = abs(v)/dt

                                # ensure drag opposes velocity
                                if (v > 0.0):
                                    f = -f

                                damp = Gf.Vec3d(0.0, 0.0, f)

                            else:
                                # disable displacment when body is above the water plane
                                grid_displace = 0.0

                            # integrate
                            if (db.inputs.buoyancy_enabled):

                                state.collider_vel[index] = state.collider_vel[index] + (damp + gravity + buoyancy)*dt
                                state.collider_pos[index] = state.collider_pos[index] + state.collider_vel[index]*dt

                                translate_bundle(collider, state.collider_vel[index]*dt)
                        
                            # apply displacement
                            if (grid_displace > 0.0):

                                wp.launch(
                                    kernel=wave_displace, 
                                    dim=state.sim_width*state.sim_height, 
                                    inputs=[state.sim_grid0, state.sim_grid1, state.sim_width, state.sim_height, cx, cz, cr, grid_displace, state.sim_time],  
                                    outputs=[])

                    # colliders
                    if time > db.inputs.delay:
                        update_collider(db.inputs.collider_0, db.inputs.density_0, 0)
                        update_collider(db.inputs.collider_1, db.inputs.density_1, 1)
                        update_collider(db.inputs.collider_2, db.inputs.density_2, 2)
                        update_collider(db.inputs.collider_3, db.inputs.density_3, 3)

                    # wave solve
                    for s in range(sim_substeps):
                        wp.launch(
                            kernel=wave_solve, 
                            dim=state.sim_width*state.sim_height, 
                            inputs=[state.sim_grid0, state.sim_grid1, state.sim_width, state.sim_height, 1.0/grid_size, k_speed, k_damp, sim_dt], 
                            outputs=[])

                        # swap grids
                        (state.sim_grid0, state.sim_grid1) = (state.sim_grid1, state.sim_grid0)

                        state.sim_time += sim_dt


                    # update grid vertices
                    wp.launch(kernel=grid_update,
                                dim=state.sim_width*state.sim_height,
                                inputs=[state.sim_grid0, state.verts_device],
                                outputs=[])

                # allocate output array
                db.outputs.vertices_size = len(state.verts_device)

                # copy points 
                points_out = omni.warp.from_omni_graph(db.outputs.vertices, dtype=wp.vec3)
                wp.copy(points_out, state.verts_device)

                return True

    
