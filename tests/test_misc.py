
import os
import sys
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import numpy as np
np.random.seed(532)



wp.config.mode = "release"
wp.config.verify_cuda = False

wp.init()

devices = wp.get_devices()
device = "cuda"
print_enabled = True


dt = 0.0001
lj_epsilon = 1.0
lj_sigma = 1.0
epsSigma6 = 24.0 * lj_epsilon * lj_sigma**6
epsSigma12 = 48.0 * lj_epsilon * lj_sigma**12
vel_scale = 400.0

num_points = 512
rho = 0.776
boxsize = (num_points / rho)**(1.0 / 3.0)


cell_radius = 4.0 * lj_sigma
query_radius = 3.0 * lj_sigma

dim_x = int(boxsize / cell_radius) + 1
dim_y = int(boxsize / cell_radius) + 1
dim_z = int(boxsize / cell_radius) + 1

period = wp.vec3(boxsize, boxsize, boxsize)




@wp.kernel
def lennard_jones(grid : wp.uint64,
                  points: wp.array(dtype=wp.vec3),
                  velocities: wp.array(dtype=wp.vec3),
                  radius: float,
                  dt: float,
                  epsSigma6: float,
                  epsSigma12: float,
                  period: wp.vec3
                  ):

    tid = wp.tid()
    
    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    
    # query point
    xi = points[i]
    vel = velocities[i]
    
    # construct query around point p
    query = wp.hash_grid_query(grid, xi, radius)
    index = int(0)
    
    force = wp.vec3(0.0, 0.0, 0.0)
    
    while(wp.hash_grid_query_next(query, index)):
        
        # periodic boundary conditions
        xj = points[index]
        
        xij = xi - xj
        rx = (-0.5 + wp.rint(xij[0] / period[0])) * period[0]
        ry = (-0.5 + wp.rint(xij[1] / period[1])) * period[1]
        rz = (-0.5 + wp.rint(xij[2] / period[2])) * period[2]
        xij -= wp.vec3(rx, ry, rz)
        
        # compute distance to point
        dij_sq = wp.dot(xij, xij)
        dij = wp.sqrt(dij_sq)
        # if within cutoff radius
        if (dij <= radius):
            # if farther than min distance 
            rd = 1.0 / 1e-1
            if (dij >= 1e-1):
                rd = 1.0 / dij
            
            rd7 = wp.pow(rd, 7.0)
            rd13 = wp.pow(rd, 13.0)

            f = epsSigma12 * rd13 - epsSigma6 * rd7
            fv = f * rd * xij
            force -= fv
            
    # integrate
    vel = vel + 0.5 * dt * force
    xi  = xi + dt * vel + 0.5 * dt * dt * force
    vel = vel + 0.5 * dt * force
    
    rx = (-0.5 + wp.rint(xi[0] / period[0])) * period[0]
    ry = (-0.5 + wp.rint(xi[1] / period[1])) * period[1]
    rz = (-0.5 + wp.rint(xi[2] / period[2])) * period[2]
    xi -= wp.vec3(rx, ry, rz)
    
    if (tid == 0):
        print(wp.vec3(xi[0], xi[1], xi[2]))

    points[i] = wp.vec3(xi[0], xi[1], xi[2])
    velocities[i] = wp.vec3(vel[0], vel[1], vel[2])
  


@wp.kernel
def simple(grid: wp.uint64):
    
   
    # if tid < 32:
    #     print(i)
    
    if wp.tid() == 0:
        # tid = wp.tid()
        # i = wp.hash_grid_point_id(grid, tid)
        #print(i)
        i = wp.hash_grid_point_id(grid, wp.tid())
        #query = wp.hash_grid_query(grid, wp.vec3(), 1.0)
        #print(i)
        #print("hello")
        print(i)


# Create the neighborlist
grid = wp.HashGrid(dim_x, dim_y, dim_z, device)

# points = np.random.rand(num_points, 3)*scale - np.array((scale, scale, scale))*0.5
# def particle_grid(dim_x, dim_y, dim_z, lower, radius, jitter):
#     points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
#     points_t = np.array((points[0], points[1], points[2])).T * radius * 2.0 + np.array(lower)
#     points_t = points_t + np.random.rand(*points_t.shape) * radius * jitter
#     return points_t.reshape((-1, 3))
# points = particle_grid(16, 16, 16, (0.0, 0.0, 0.0), cell_radius*0.25, 0.1)

Nx = int(np.ceil(num_points**0.33333))
Ny = int(np.ceil(num_points**0.33333))
Nz = int(np.ceil(num_points**0.33333))
dx = 0.9 * boxsize / Nx
dy = 0.9 * boxsize / Ny
dz = 0.9 * boxsize / Nz

points = np.zeros((num_points, 3), dtype=float)
counter = 0
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz): 
            points[counter] = np.array([(i + 1)*dx, (j + 1)*dy, (k + 1)*dz])
            counter += 1
            if counter==num_points:
                break
vels = np.random.normal(scale=vel_scale, size=(num_points, 3)) 

points_arr = wp.array(points, dtype=wp.vec3, device=device)
vels_arr = wp.array(vels, dtype=wp.vec3, device=device)


# with wp.ScopedTimer("grid build", active=print_enabled):
#     grid.build(points_arr, cell_radius)
#     wp.synchronize()
# with wp.ScopedTimer("grid query", active=print_enabled):
#     wp.launch(kernel=lennard_jones, dim=len(points), inputs=[grid.id, points_arr, vels_arr, query_radius, dt, epsSigma6, epsSigma12, period], device=device)
#     wp.synchronize()



# begin capture
grid.build(points_arr, cell_radius)

use_graph = True

if (use_graph):
    wp.capture_begin()

for i in range(10000):
    grid.build(points_arr, cell_radius)

    wp.launch(kernel=lennard_jones, dim=len(points), inputs=[grid.id, points_arr, vels_arr, query_radius, dt, epsSigma6, epsSigma12, period], device=device)

# end capture and return a graph object
if (use_graph):
    graph = wp.capture_end()
    wp.capture_launch(graph)


points_f = points_arr.numpy()
vels_f = vels_arr.numpy()
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points_f[:,0], points_f[:,1], points_f[:,2], s=1)
plt.show()




#pdb.set_trace()