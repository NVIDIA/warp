"""
Simple 2D wave-equation solver
"""

import math

import numpy as np
import warp as wp

import omni.timeline
import omni.appwindow
import omni.usd

from pxr import Usd, UsdGeom, Gf, Sdf


# helper to read a USD xform out of graph inputs
def read_transform(input):
    xform = Gf.Matrix4d(input.reshape((4,4)))
    return xform

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
    
    s = wp.load(f, y*width + x)
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

        wp.store(hcurrent, tid, h)
        wp.store(hprevious, tid, h)


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
    h1 = wp.load(hcurrent, tid)
    h0 = wp.load(hprevious, tid)
    
    h = 2.0*h1 - h0 + dt*dt*(k_speed*l - k_damp*(h1-h0))

    # buffers get swapped each iteration
    wp.store(hprevious, tid, h)


# simple kernel to apply height deltas to a vertex array
@wp.kernel
def grid_update(heights: wp.array(dtype=float),
                vertices: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    h = wp.load(heights, tid)
    v = wp.load(vertices, tid)

    v_new = wp.vec3(v[0], v[1], h)

    wp.store(vertices, tid, v_new)



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
 
        sim_fps = 60.0
        sim_substeps = 1
        sim_dt = (1.0/sim_fps)/sim_substeps
        
        grid_size = db.inputs.resolution
        grid_displace = db.inputs.displace

        # wave constants
        k_speed = db.inputs.speed
        k_damp = db.inputs.damp

        state = db.internal_state

        with wp.ScopedCudaGuard():

            # initialization
            if db.inputs.grid.valid and state.initialized is False:

                bounds = read_bounds_bundle(db.inputs.grid)
                (b, r, scale, u, t, p) = bounds.GetMatrix().Factor()
                size = bounds.GetRange().GetSize()

                state.sim_width = int(size[0]*scale[0]/grid_size)
                state.sim_height = int(size[1]*scale[1]/grid_size)

                print(f"Creating grid with dim {state.sim_width}, {state.sim_height}")

                state.sim_grid0 = wp.zeros(state.sim_width*state.sim_height, dtype=float, device="cuda")
                state.sim_grid1 = wp.zeros(state.sim_width*state.sim_height, dtype=float, device="cuda")

                state.sim_host = wp.zeros(state.sim_width*state.sim_height, dtype=float, device="cpu")
                
                state.verts_device =  wp.zeros(state.sim_width*state.sim_height, dtype=wp.vec3, device="cuda")
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

                state.initialized = True

            if state.initialized:

                if timeline.is_playing():

                    if db.inputs.collider.valid:

                        grid_xform = read_transform_bundle(db.inputs.grid).RemoveScaleShear()
                        collider_xform = read_transform_bundle(db.inputs.collider)

                        bounds = read_bounds_bundle(db.inputs.collider).ComputeAlignedBox()
                        radius = bounds.GetSize()[0]*0.5

                        # transform collider to grid local space
                        local_pos = grid_xform.GetInverse().Transform(collider_xform.ExtractTranslation())

                        # create surface displacment around a point
                        cx = float(local_pos[0])/grid_size + state.sim_width*0.5
                        cz = float(local_pos[1])/grid_size + state.sim_height*0.5
                        cr = float(radius)/grid_size
                    
                    else:
                        cx = 0.0
                        cz = 0.0
                        cr = 0.0

                    for s in range(sim_substeps):
                    
                        # apply displacement
                        wp.launch(
                            kernel=wave_displace, 
                            dim=state.sim_width*state.sim_height, 
                            inputs=[state.sim_grid0, state.sim_grid1, state.sim_width, state.sim_height, cx, cz, cr, grid_displace, state.sim_time],  
                            outputs=[],
                            device="cuda")


                        # integrate wave equation
                        wp.launch(
                            kernel=wave_solve, 
                            dim=state.sim_width*state.sim_height, 
                            inputs=[state.sim_grid0, state.sim_grid1, state.sim_width, state.sim_height, 1.0/grid_size, k_speed, k_damp, sim_dt], 
                            outputs=[],
                            device="cuda")

                        # swap grids
                        (state.sim_grid0, state.sim_grid1) = (state.sim_grid1, state.sim_grid0)

                        state.sim_time += sim_dt


                    # update grid vertices
                    wp.launch(kernel=grid_update,
                                dim=state.sim_width*state.sim_height,
                                inputs=[state.sim_grid0, state.verts_device],
                                outputs=[],
                                device="cuda")

                    # copy data back to host
                    wp.copy(dest=state.verts_host, src=state.verts_device)
                    wp.synchronize()

                # write outputs
                db.outputs.vertices_size = len(state.verts_host)
                db.outputs.vertices[:] = state.verts_host.numpy()

            return True
