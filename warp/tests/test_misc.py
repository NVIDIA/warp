import numpy as np
import open3d as o3d

import torch
import warp as wp

wp.config.mode = "debug"
wp.init()
device = "cuda"

@wp.kernel
def raycast_kernel(mesh: wp.uint64,
                   ray_starts: wp.array(dtype=wp.vec3),
                   ray_directions: wp.array(dtype=wp.vec3),
                   ray_hits: wp.array(dtype=wp.vec3),
                   count: int):

    #tid = wp.tid()

    t = float(0.0)                   # hit distance along ray
    u = float(0.0)                   # hit face barycentric u
    v = float(0.0)                   # hit face barycentric v
    sign = float(0.0)                # hit face sign
    n = wp.vec3()                    # hit face normal
    f = int(0)                       # hit face index
    max_dist = 1e6                   # max raycast disance


    # ray cast against the mesh
    for tid in range(count):
        if wp.mesh_query_ray(mesh, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f):
            ray_hits[tid] = ray_starts[tid] + t * ray_directions[tid]


def ray_cast(mesh, ray_starts, ray_directions):
    n = len(ray_starts)
    ray_starts = wp.array(ray_starts.cpu().numpy(), shape=(n,), dtype=wp.vec3, device=device)
    ray_directions = wp.array(ray_directions.cpu().numpy(), shape=(n,), dtype=wp.vec3, device=device)
    ray_hits = wp.zeros((n, ), dtype=wp.vec3, device=device)
    wp.launch(kernel=raycast_kernel,
              dim=1,
              inputs=[mesh.id, ray_starts, ray_directions, ray_hits, n],
              device=device)
    wp.synchronize()
    return ray_hits.numpy()


# Create raycast starts and directions
xx, yy = torch.meshgrid(torch.arange(0.1, 0.4, 0.01, device=device), torch.arange(0.1, 0.4, 0.01, device=device))
xx = xx.flatten().view(-1, 1)
yy = yy.flatten().view(-1, 1)
zz = torch.ones_like(xx)
ray_starts = torch.cat((xx, yy, zz), dim=1)
ray_dirs = torch.zeros_like(ray_starts)
ray_dirs[:, 2] = -1.0

# Create simple square mesh
vertices = np.array([[0., 0., 0.],
                     [0., 0.5, 0.],
                     [0.5, 0., 0.],
                     [0.5, 0.5, 0.]], dtype=np.float32)

triangles = np.array([[1, 0, 2],
                     [1, 2, 3]], dtype=np.int32)

o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)

mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=device),
               indices=wp.array(triangles.flatten(), dtype=int, device=device))

ray_hits = ray_cast(mesh, ray_starts, ray_dirs)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ray_hits)
o3d.visualization.draw_geometries([o3d_mesh, pcd])
