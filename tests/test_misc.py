
import os
import sys
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import numpy as np
np.random.seed(532)


wp.config.mode = "release"
wp.config.verify_cuda = True

wp.init()

devices = wp.get_devices()
device = "cpu"
print_enabled = True

# num_points = 4096
dim_x = 128
dim_y = 128
dim_z = 128

dt = 0.0001
lj_epsilon = 1.0
lj_sigma = 1.0


epsSigma6 = 24.0 * lj_epsilon * lj_sigma**6
epsSigma12 = 48.0 * lj_epsilon * lj_sigma**12
period = wp.vec3(dim_x - 1, dim_y - 1, dim_z - 1)

scale = 150.0

cell_radius = 8.0
query_radius = 8.0



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
        rx = wp.rint(xij[0] / period[0]) * period[0]
        ry = wp.rint(xij[1] / period[1]) * period[1]
        rz = wp.rint(xij[2] / period[2]) * period[2]
        xij -= wp.vec3(rx, ry, rz)
        
        # compute distance to point
        dij = wp.length(xij)
        
        # if within cutoff radius
        if (dij <= radius):
            # if farther than min distance 
            rd = 1.0 / 1e-3
            if (dij >= 1e-3):
                rd = 1.0 / dij
            
            rd7 = wp.pow(rd, 7.0)
            rd13 = wp.pow(rd, 13.0)

            f = epsSigma12 * rd13 - epsSigma6 * rd7
            fv = f * rd * xij
            force -= fv
            
    # integrate
    vel += 0.5 * dt * force
    xi += dt * vel + 0.5 * dt * dt * force
    vel += 0.5 * dt * force
   
    points[i] = wp.vec3(xi[0], xi[1], xi[2])
    velocities[i] = wp.vec3(vel[0], vel[1], vel[2])





# Create the neighborlist
grid = wp.HashGrid(dim_x, dim_y, dim_z, device)

# points = np.random.rand(num_points, 3)*scale - np.array((scale, scale, scale))*0.5
def particle_grid(dim_x, dim_y, dim_z, lower, radius, jitter):
    points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
    points_t = np.array((points[0], points[1], points[2])).T * radius * 2.0 + np.array(lower)
    points_t = points_t + np.random.rand(*points_t.shape) * radius * jitter
    return points_t.reshape((-1, 3))

points = particle_grid(16, 32, 16, (0.0, 0.3, 0.0), cell_radius*0.25, 0.1)
points_arr = wp.array(points, dtype=wp.vec3, device=device)

vels = np.zeros_like(points)
vels_arr = wp.array(vels, dtype=wp.vec3, device=device)



with wp.ScopedTimer("grid build", active=print_enabled):
    grid.build(points_arr, cell_radius)
    wp.synchronize()


with wp.ScopedTimer("grid query", active=print_enabled):
    wp.launch(kernel=lennard_jones, dim=len(points), inputs=[grid.id, points_arr, vels_arr, query_radius, dt, epsSigma6, epsSigma12, period], device=device)
    wp.synchronize()



#pdb.set_trace()







# begin capture
#wp.capture_begin()

for i in range(100):
    grid.build(points_arr, cell_radius)
    wp.synchronize()
    wp.launch(kernel=lennard_jones, dim=len(points), inputs=[grid.id, query_radius, points_arr, vels_arr, dt, epsSigma6, epsSigma12, period], device=device)
    wp.synchronize()

# end capture and return a graph object
#graph = wp.capture_end()
#wp.capture_launch(graph)



points = points_arr.numpy()
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
# plt.show()


#pdb.set_trace()