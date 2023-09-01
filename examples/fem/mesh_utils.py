import numpy as np
import warp as wp


def gen_trimesh(res, bounds_lo: wp.vec2 = wp.vec2(0.0), bounds_hi: wp.vec2 = wp.vec2(1.0)):
    """Constructs a triangular mesh by diving each cell of a dense 2D grid into two triangles

    Args:
        res: Resolution of the grid along each dimension
        bounds_lo: Position of the lower bound of the axis-aligned grid
        bounds_up: Position of the upper bound of the axis-aligned grid

    Returns:
        Tuple of ndarrays: (Vertex positions, Triangle vertex indices)
    """


    Nx = res[0]
    Ny = res[1]

    x = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    y = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)

    positions = np.transpose(np.meshgrid(x, y, indexing="ij"), axes=(1, 2, 0)).reshape(-1, 2)

    cx, cy = np.meshgrid(np.arange(Nx, dtype=int), np.arange(Ny, dtype=int), indexing="ij")

    vidx = np.transpose(
        np.array(
            [
                (Ny + 1) * cx + cy,
                (Ny + 1) * (cx + 1) + cy,
                (Ny + 1) * (cx + 1) + (cy + 1),
                (Ny + 1) * cx + cy,
                (Ny + 1) * (cx + 1) + (cy + 1),
                (Ny + 1) * (cx) + (cy + 1),
            ]
        )
    ).reshape((-1, 3))

    return wp.array(positions, dtype=wp.vec2), wp.array(vidx, dtype=int)


def gen_tetmesh(res, bounds_lo: wp.vec3 = wp.vec3(0.0), bounds_hi: wp.vec3 = wp.vec3(1.0)):
    """Constructs a tetrahedral mesh by diving each cell of a dense 3D grid into five tetrahedrons

    Args:
        res: Resolution of the grid along each dimension
        bounds_lo: Position of the lower bound of the axis-aligned grid
        bounds_up: Position of the upper bound of the axis-aligned grid

    Returns:
        Tuple of ndarrays: (Vertex positions, Tetrahedron vertex indices)
    """

    Nx = res[0]
    Ny = res[1]
    Nz = res[2]

    x = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    y = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
    z = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij"), axes=(1, 2, 3, 0)).reshape(-1, 3)

    # Global node indices for each cell
    cx, cy, cz = np.meshgrid(
        np.arange(Nx, dtype=int), np.arange(Ny, dtype=int), np.arange(Nz, dtype=int), indexing="ij"
    )

    grid_vidx = np.array(
        [
            (Ny + 1) * (Nz + 1) * cx + (Nz + 1) * cy + cz,
            (Ny + 1) * (Nz + 1) * cx + (Nz + 1) * cy + cz + 1,
            (Ny + 1) * (Nz + 1) * cx + (Nz + 1) * (cy + 1) + cz,
            (Ny + 1) * (Nz + 1) * cx + (Nz + 1) * (cy + 1) + cz + 1,
            (Ny + 1) * (Nz + 1) * (cx + 1) + (Nz + 1) * cy + cz,
            (Ny + 1) * (Nz + 1) * (cx + 1) + (Nz + 1) * cy + cz + 1,
            (Ny + 1) * (Nz + 1) * (cx + 1) + (Nz + 1) * (cy + 1) + cz,
            (Ny + 1) * (Nz + 1) * (cx + 1) + (Nz + 1) * (cy + 1) + cz + 1,
        ]
    )

    # decompose grid cells into 5 tets
    tet_vidx = np.array(
        [
            [0, 1, 2, 4],
            [3, 2, 1, 7],
            [5, 1, 7, 4],
            [6, 7, 4, 2],
            [4, 1, 2, 7],
        ]
    )

    # Convert to 3d index coordinates
    vidx_coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    tet_coords = vidx_coords[tet_vidx]

    # Symmetry bits for each cell
    ox, oy, oz = np.meshgrid(
        np.arange(Nx, dtype=int) % 2, np.arange(Ny, dtype=int) % 2, np.arange(Nz, dtype=int) % 2, indexing="ij"
    )
    tet_coords = np.broadcast_to(tet_coords, shape=(*ox.shape, *tet_coords.shape))

    # Flip coordinates according to symmetry
    ox_bk = np.broadcast_to(ox.reshape(*ox.shape, 1, 1), tet_coords.shape[:-1])
    oy_bk = np.broadcast_to(oy.reshape(*oy.shape, 1, 1), tet_coords.shape[:-1])
    oz_bk = np.broadcast_to(oz.reshape(*oz.shape, 1, 1), tet_coords.shape[:-1])

    tet_coords_x = tet_coords[..., 0] ^ ox_bk
    tet_coords_y = tet_coords[..., 1] ^ oy_bk
    tet_coords_z = tet_coords[..., 2] ^ oz_bk

    # Back to local vertex indices
    corner_indices = 4 * tet_coords_x + 2 * tet_coords_y + tet_coords_z

    # Now go from cell-local to global node indices
    # There must be a nicer way than this, but for example purposes this works

    corner_indices = corner_indices.reshape(-1, 4)

    grid_vidx = grid_vidx.reshape((8, -1, 1))
    grid_vidx = np.broadcast_to(grid_vidx, shape=(8, grid_vidx.shape[1], 5))
    grid_vidx = grid_vidx.reshape((8, -1))

    node_indices = np.arange(corner_indices.shape[0])
    tet_grid_vidx = np.transpose(
        [
            grid_vidx[corner_indices[:, 0], node_indices],
            grid_vidx[corner_indices[:, 1], node_indices],
            grid_vidx[corner_indices[:, 2], node_indices],
            grid_vidx[corner_indices[:, 3], node_indices],
        ]
    )

    return wp.array(positions, dtype=wp.vec3), wp.array(tet_grid_vidx, dtype=int)
