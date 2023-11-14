import numpy as np
import warp as wp

from warp.fem.utils import grid_to_tets, grid_to_tris, grid_to_quads, grid_to_hexes


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

    vidx = grid_to_tris(Nx, Ny)

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

    vidx = grid_to_tets(Nx, Ny, Nz)

    return wp.array(positions, dtype=wp.vec3), wp.array(vidx, dtype=int)


def gen_quadmesh(res, bounds_lo: wp.vec2 = wp.vec2(0.0), bounds_hi: wp.vec2 = wp.vec2(1.0)):
    """Constructs a quadrilateral mesh from a dense 2D grid

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

    vidx = grid_to_quads(Nx, Ny)

    return wp.array(positions, dtype=wp.vec2), wp.array(vidx, dtype=int)


def gen_hexmesh(res, bounds_lo: wp.vec3 = wp.vec3(0.0), bounds_hi: wp.vec3 = wp.vec3(1.0)):
    """Constructs a quadrilateral mesh from a dense 2D grid

    Args:
        res: Resolution of the grid along each dimension
        bounds_lo: Position of the lower bound of the axis-aligned grid
        bounds_up: Position of the upper bound of the axis-aligned grid

    Returns:
        Tuple of ndarrays: (Vertex positions, Triangle vertex indices)
    """

    Nx = res[0]
    Ny = res[1]
    Nz = res[2]

    x = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    y = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
    z = np.linspace(bounds_lo[1], bounds_hi[1], Nz + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij"), axes=(1, 2, 3, 0)).reshape(-1, 3)

    vidx = grid_to_hexes(Nx, Ny, Nz)

    return wp.array(positions, dtype=wp.vec3), wp.array(vidx, dtype=int)

