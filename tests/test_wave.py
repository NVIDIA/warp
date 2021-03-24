# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import oglang as og

from pxr import Usd, UsdGeom, Gf, Sdf


# inline __device__ float Fetch(cudaTextureObject_t grid, int x, int y, int pitch)
# {
# 	const int coord = y*pitch + x;

# 	return tex1ogetch<float>(grid, coord);
# }

# inline __device__ float Laplacian(cudaTextureObject_t g, int x, int y, int pitch)
# {
# 	float ddx = Fetch(g, x+1, y, pitch) - 2.0f*Fetch(g, x,y, pitch) + Fetch(g, x-1, y, pitch);
# 	float ddy = Fetch(g, x, y+1, pitch) - 2.0f*Fetch(g, x,y, pitch) + Fetch(g, x, y-1, pitch);
	
# 	return (ddx + ddy);
# }

# __global__ void RippleAdd(int width, int height, int cx, int cy, float amplitude, float spread, float* current, float* previous, const int* mask)
# {
# 	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
# 	const int x = tid%width;
# 	const int y = tid/width;

# 	if (y < height)
# 	{
# 		// solid boundaries
# 		if (mask[tid]&0x000000ff != 0)
# 			return;

# 		int dx = x-cx;
# 		int dy = y-cy;
		
# 		float dSq = dx*dx + dy*dy;

# 		if (dSq < spread*spread)
# 		{
# 			const float v0 = current[tid];

# 			const float v = Lerp(v0, amplitude*SmoothStep(0.0f, 1.0f, 1.0f-sqrtf(dSq)/spread), 0.9f);
			
# 			current[tid] = v;
# 			previous[tid] = v;
# 		}
# 	}
# }

# __global__ void RippleSolve(int width, int height, cudaTextureObject_t hcurrent, cudaTextureObject_t hprevious, const int* mask, float* hnew, float kspeed, float kdamp, float dt)
# {
# 	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
# 	const int x = tid%width;
# 	const int y = tid/width;

# 	if (y < height)
# 	{
# 		// solid boundaries
# 		if (mask[tid]&0x000000ff != 0)
# 			return;

# 		// calculate the laplacian of height for each cell
# 		const float laplacian = Laplacian(hcurrent, x, y, width);

# 		// integrate 
# 		const float h1 = tex1ogetch<float>(hcurrent, tid);
# 		const float h0 = tex1ogetch<float>(hprevious, tid);
		
# 		const float h = (2.0f*h1 - h0 + kspeed*laplacian - kdamp*(h1-h0));

# 		hnew[tid] = h;
# 	}
# }

@og.func
def sample(f: og.array(float),
           x: int,
           y: int,
           width: int,
           height: int):

    # clamp texture coords
    x = og.clamp(x, 0, width-1)
    y = og.clamp(y, 0, height-1)
    
    s = og.load(f, y*width + x)
    return s

@og.func
def laplacian(f: og.array(float),
              x: int,
              y: int,
              width: int,
              height: int):
    
    ddx = sample(f, x+1, y, width, height) - 2.0*sample(f, x,y, width, height) + sample(f, x-1, y, width, height)
    ddy = sample(f, x, y+1, width, height) - 2.0*sample(f, x,y, width, height) + sample(f, x, y-1, width, height)

    return (ddx + ddy)

@og.func
def smoothstep(x: float):
    if x < 0.0:
        return 0.0
    
    if x > 1.0:
        return 1.0

    return 3.0*x*x - 2.0*x*x*x

@og.kernel
def wave_displace(hcurrent: og.array(float),
                  hprevious: og.array(float),
                  width: int,
                  height: int,
                  center_x: int,
                  center_y: int,
                  r: float,
                  mag: float,
                  t: float):

    tid = og.tid()

    x = tid%width
    y = tid//width

    dx = x - center_x
    dy = y - center_y

    dist_sq = float(dx*dx + dy*dy)

    if (dist_sq < r*r):

        #w = 1.0 - smoothstep(og.sqrt(dist_sq)/r)
        h = mag*og.sin(t)

        og.store(hcurrent, tid, h)
        og.store(hprevious, tid, h)


@og.kernel
def wave_solve(hprevious: og.array(float),
               hcurrent: og.array(float),
               width: int,
               height: int,
               k_speed: float,
               k_damp: float,
               dt: float):

    tid = og.tid()

    x = tid%width
    y = tid//width

    l = laplacian(hcurrent, x, y, width, height)

    # integrate 
    h1 = og.load(hcurrent, tid)
    h0 = og.load(hprevious, tid)
    
    h = (2.0*h1 - h0 + k_speed*l - k_damp*(h1-h0))

    # buffers get swapped each iteration
    og.store(hprevious, tid, h)


# params
sim_width = 128
sim_height = 128

sim_fps = 60.0
sim_substeps = 16
sim_duration = 5.0
sim_frames = int(sim_duration*sim_fps)
sim_dt = (1.0/sim_fps)/sim_substeps
sim_time = 0.0

# wave constants
k_speed = 0.0001
k_damp = 0.001

# set up grid for visualization
stage = Usd.Stage.CreateNew("tests/outputs/wave.usd")
stage.SetStartTimeCode(0.0)
stage.SetEndTimeCode(sim_duration*sim_fps)
stage.SetTimeCodesPerSecond(sim_fps)

grid = UsdGeom.Mesh.Define(stage, "/root")
grid_size = 0.1

vertices = []
indices = []
counts = []

def grid_index(x, y, stride):
    return y*stride + x

for z in range(sim_height):
    for x in range(sim_width):

        pos = Gf.Vec3f(float(x)*grid_size, 0.0, float(z)*grid_size) - Gf.Vec3f(float(sim_width)/2*grid_size, 0.0, float(sim_height)/2*grid_size)

        vertices.append(pos)
            
        if (x > 0 and z > 0):
            
            indices.append(grid_index(x-1, z-1, sim_width))
            indices.append(grid_index(x, z, sim_width))
            indices.append(grid_index(x, z-1, sim_width))

            indices.append(grid_index(x-1, z-1, sim_width))
            indices.append(grid_index(x-1, z, sim_width))
            indices.append(grid_index(x, z, sim_width))

            counts.append(3)
            counts.append(3)

grid.GetPointsAttr().Set(vertices, 0.0)
grid.GetFaceVertexIndicesAttr().Set(indices, 0.0)
grid.GetFaceVertexCountsAttr().Set(counts, 0.0)

# simulation context
context = og.Context("cpu")

# simulation grids
sim_grid0 = context.zeros(sim_width*sim_height, dtype=float)
sim_grid1 = context.zeros(sim_width*sim_height, dtype=float)

for i in range(sim_frames):

    # simulate
    for s in range(sim_substeps):

        #create surface displacment around a point

        cx = sim_width/2 + math.sin(sim_time)*sim_width/3
        cy = sim_height/2 + math.cos(sim_time)*sim_height/3

        context.launch(
            kernel=wave_displace, 
            dim=sim_width*sim_height, 
            inputs=[sim_grid0, sim_grid1, sim_width, sim_height, int(cx), int(cy), 10.0, 0.5, sim_time*2.0], 
            outputs=[])


        # integrate wave equation
        context.launch(
            kernel=wave_solve, 
            dim=sim_width*sim_height, 
            inputs=[sim_grid0, sim_grid1, sim_width, sim_height, k_speed, k_damp, sim_time], 
            outputs=[])

        # swap grids
        (sim_grid0, sim_grid1) = (sim_grid1, sim_grid0)

        sim_time += sim_dt


    # numpy view onto sim data
    sim_view = sim_grid0.numpy()

    # render
    for v in range(sim_width*sim_height):
        vertices[v] = Gf.Vec3f(vertices[v][0], float(sim_view[v]), vertices[v][2])
        

    grid.GetPointsAttr().Set(vertices, sim_time*sim_fps)


stage.Save()