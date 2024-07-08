from typing import Any, Optional, Set, Tuple

import numpy as np

import warp as wp
import warp.fem as fem
from warp.optim.linear import LinearOperator, aslinearoperator, preconditioner
from warp.sparse import BsrMatrix, bsr_get_diag, bsr_mv, bsr_transposed

__all__ = [
    "gen_hexmesh",
    "gen_quadmesh",
    "gen_tetmesh",
    "gen_trimesh",
    "bsr_cg",
    "bsr_solve_saddle",
    "SaddleSystem",
    "invert_diagonal_bsr_matrix",
    "Plot",
]


#
# Mesh utilities
#


def gen_trimesh(res, bounds_lo: Optional[wp.vec2] = None, bounds_hi: Optional[wp.vec2] = None):
    """Constructs a triangular mesh by diving each cell of a dense 2D grid into two triangles

    Args:
        res: Resolution of the grid along each dimension
        bounds_lo: Position of the lower bound of the axis-aligned grid
        bounds_up: Position of the upper bound of the axis-aligned grid

    Returns:
        Tuple of ndarrays: (Vertex positions, Triangle vertex indices)
    """

    if bounds_lo is None:
        bounds_lo = wp.vec2(0.0)

    if bounds_hi is None:
        bounds_hi = wp.vec2(1.0)

    Nx = res[0]
    Ny = res[1]

    x = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    y = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)

    positions = np.transpose(np.meshgrid(x, y, indexing="ij"), axes=(1, 2, 0)).reshape(-1, 2)

    vidx = fem.utils.grid_to_tris(Nx, Ny)

    return wp.array(positions, dtype=wp.vec2), wp.array(vidx, dtype=int)


def gen_tetmesh(res, bounds_lo: Optional[wp.vec3] = None, bounds_hi: Optional[wp.vec3] = None):
    """Constructs a tetrahedral mesh by diving each cell of a dense 3D grid into five tetrahedrons

    Args:
        res: Resolution of the grid along each dimension
        bounds_lo: Position of the lower bound of the axis-aligned grid
        bounds_up: Position of the upper bound of the axis-aligned grid

    Returns:
        Tuple of ndarrays: (Vertex positions, Tetrahedron vertex indices)
    """

    if bounds_lo is None:
        bounds_lo = wp.vec3(0.0)

    if bounds_hi is None:
        bounds_hi = wp.vec3(1.0)

    Nx = res[0]
    Ny = res[1]
    Nz = res[2]

    x = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    y = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
    z = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij"), axes=(1, 2, 3, 0)).reshape(-1, 3)

    vidx = fem.utils.grid_to_tets(Nx, Ny, Nz)

    return wp.array(positions, dtype=wp.vec3), wp.array(vidx, dtype=int)


def gen_quadmesh(res, bounds_lo: Optional[wp.vec2] = None, bounds_hi: Optional[wp.vec2] = None):
    """Constructs a quadrilateral mesh from a dense 2D grid

    Args:
        res: Resolution of the grid along each dimension
        bounds_lo: Position of the lower bound of the axis-aligned grid
        bounds_up: Position of the upper bound of the axis-aligned grid

    Returns:
        Tuple of ndarrays: (Vertex positions, Triangle vertex indices)
    """
    if bounds_lo is None:
        bounds_lo = wp.vec2(0.0)

    if bounds_hi is None:
        bounds_hi = wp.vec2(1.0)

    Nx = res[0]
    Ny = res[1]

    x = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    y = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)

    positions = np.transpose(np.meshgrid(x, y, indexing="ij"), axes=(1, 2, 0)).reshape(-1, 2)

    vidx = fem.utils.grid_to_quads(Nx, Ny)

    return wp.array(positions, dtype=wp.vec2), wp.array(vidx, dtype=int)


def gen_hexmesh(res, bounds_lo: Optional[wp.vec3] = None, bounds_hi: Optional[wp.vec3] = None):
    """Constructs a quadrilateral mesh from a dense 2D grid

    Args:
        res: Resolution of the grid along each dimension
        bounds_lo: Position of the lower bound of the axis-aligned grid
        bounds_up: Position of the upper bound of the axis-aligned grid

    Returns:
        Tuple of ndarrays: (Vertex positions, Triangle vertex indices)
    """

    if bounds_lo is None:
        bounds_lo = wp.vec3(0.0)

    if bounds_hi is None:
        bounds_hi = wp.vec3(1.0)

    Nx = res[0]
    Ny = res[1]
    Nz = res[2]

    x = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    y = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
    z = np.linspace(bounds_lo[1], bounds_hi[1], Nz + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij"), axes=(1, 2, 3, 0)).reshape(-1, 3)

    vidx = fem.utils.grid_to_hexes(Nx, Ny, Nz)

    return wp.array(positions, dtype=wp.vec3), wp.array(vidx, dtype=int)


#
# Bsr matrix utilities
#


def _get_linear_solver_func(method_name: str):
    from warp.optim.linear import bicgstab, cg, cr, gmres

    if method_name == "bicgstab":
        return bicgstab
    if method_name == "gmres":
        return gmres
    if method_name == "cr":
        return cr
    return cg


def bsr_cg(
    A: BsrMatrix,
    x: wp.array,
    b: wp.array,
    max_iters: int = 0,
    tol: float = 0.0001,
    check_every=10,
    use_diag_precond=True,
    mv_routine=None,
    quiet=False,
    method: str = "cg",
) -> Tuple[float, int]:
    """Solves the linear system A x = b using an iterative solver, optionally with diagonal preconditioning

    Args:
        A: system left-hand side
        x: result vector and initial guess
        b: system right-hand-side
        max_iters: maximum number of iterations to perform before aborting. If set to zero, equal to the system size.
        tol: relative tolerance under which to stop the solve
        check_every: number of iterations every which to evaluate the current residual norm to compare against tolerance
        use_diag_precond: Whether to use diagonal preconditioning
        mv_routine: Matrix-vector multiplication routine to use for multiplications with ``A``
        quiet: if True, do not print iteration residuals
        method: Iterative solver method to use, defaults to Conjugate Gradient

    Returns:
        Tuple (residual norm, iteration count)

    """

    if mv_routine is None:
        M = preconditioner(A, "diag") if use_diag_precond else None
    else:
        A = LinearOperator(A.shape, A.dtype, A.device, matvec=mv_routine)
        M = None

    func = _get_linear_solver_func(method_name=method)

    def print_callback(i, err, tol):
        print(f"{func.__name__}: at iteration {i} error = \t {err}  \t tol: {tol}")

    callback = None if quiet else print_callback

    end_iter, err, atol = func(
        A=A,
        b=b,
        x=x,
        maxiter=max_iters,
        tol=tol,
        check_every=check_every,
        M=M,
        callback=callback,
    )

    if not quiet:
        res_str = "OK" if err <= atol else "TRUNCATED"
        print(f"{func.__name__}: terminated after {end_iter} iterations with error = \t {err} ({res_str})")

    return err, end_iter


class SaddleSystem(LinearOperator):
    """Builds a linear operator corresponding to the saddle-point linear system [A B^T; B 0]

    If use_diag_precond` is ``True``,  builds the corresponding diagonal preconditioner `[diag(A); diag(B diag(A)^-1 B^T)]`
    """

    def __init__(
        self,
        A: BsrMatrix,
        B: BsrMatrix,
        Bt: Optional[BsrMatrix] = None,
        use_diag_precond: bool = True,
    ):
        if Bt is None:
            Bt = bsr_transposed(B)

        self._A = A
        self._B = B
        self._Bt = Bt

        self._u_dtype = wp.vec(length=A.block_shape[0], dtype=A.scalar_type)
        self._p_dtype = wp.vec(length=B.block_shape[0], dtype=B.scalar_type)
        self._p_byte_offset = A.nrow * wp.types.type_size_in_bytes(self._u_dtype)

        saddle_shape = (A.shape[0] + B.shape[0], A.shape[0] + B.shape[0])

        super().__init__(saddle_shape, dtype=A.scalar_type, device=A.device, matvec=self._saddle_mv)

        if use_diag_precond:
            self._preconditioner = self._diag_preconditioner()
        else:
            self._preconditioner = None

    def _diag_preconditioner(self):
        A = self._A
        B = self._B

        M_u = preconditioner(A, "diag")

        A_diag = bsr_get_diag(A)

        schur_block_shape = (B.block_shape[0], B.block_shape[0])
        schur_dtype = wp.mat(shape=schur_block_shape, dtype=B.scalar_type)
        schur_inv_diag = wp.empty(dtype=schur_dtype, shape=B.nrow, device=self.device)
        wp.launch(
            _compute_schur_inverse_diagonal,
            dim=B.nrow,
            device=A.device,
            inputs=[B.offsets, B.columns, B.values, A_diag, schur_inv_diag],
        )

        if schur_block_shape == (1, 1):
            # Downcast 1x1 mats to scalars
            schur_inv_diag = schur_inv_diag.view(dtype=B.scalar_type)

        M_p = aslinearoperator(schur_inv_diag)

        def precond_mv(x, y, z, alpha, beta):
            x_u = self.u_slice(x)
            x_p = self.p_slice(x)
            y_u = self.u_slice(y)
            y_p = self.p_slice(y)
            z_u = self.u_slice(z)
            z_p = self.p_slice(z)

            M_u.matvec(x_u, y_u, z_u, alpha=alpha, beta=beta)
            M_p.matvec(x_p, y_p, z_p, alpha=alpha, beta=beta)

        return LinearOperator(
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            matvec=precond_mv,
        )

    @property
    def preconditioner(self):
        return self._preconditioner

    def u_slice(self, a: wp.array):
        return wp.array(
            ptr=a.ptr,
            dtype=self._u_dtype,
            shape=self._A.nrow,
            strides=None,
            device=a.device,
            pinned=a.pinned,
            copy=False,
        )

    def p_slice(self, a: wp.array):
        return wp.array(
            ptr=a.ptr + self._p_byte_offset,
            dtype=self._p_dtype,
            shape=self._B.nrow,
            strides=None,
            device=a.device,
            pinned=a.pinned,
            copy=False,
        )

    def _saddle_mv(self, x, y, z, alpha, beta):
        x_u = self.u_slice(x)
        x_p = self.p_slice(x)
        z_u = self.u_slice(z)
        z_p = self.p_slice(z)

        if y.ptr != z.ptr and beta != 0.0:
            wp.copy(src=y, dest=z)

        bsr_mv(self._A, x_u, z_u, alpha=alpha, beta=beta)
        bsr_mv(self._Bt, x_p, z_u, alpha=alpha, beta=1.0)
        bsr_mv(self._B, x_u, z_p, alpha=alpha, beta=beta)


def bsr_solve_saddle(
    saddle_system: SaddleSystem,
    x_u: wp.array,
    x_p: wp.array,
    b_u: wp.array,
    b_p: wp.array,
    max_iters: int = 0,
    tol: float = 0.0001,
    check_every=10,
    quiet=False,
    method: str = "cg",
) -> Tuple[float, int]:
    """Solves the saddle-point linear system [A B^T; B 0] (x_u; x_p) = (b_u; b_p) using an iterative solver, optionally with diagonal preconditioning

    Args:
        saddle_system: Saddle point system
        x_u: primal part of the result vector and initial guess
        x_p: Lagrange multiplier part of the result vector and initial guess
        b_u: primal left-hand-side
        b_p: constraint left-hand-side
        max_iters: maximum number of iterations to perform before aborting. If set to zero, equal to the system size.
        tol: relative tolerance under which to stop the solve
        check_every: number of iterations every which to evaluate the current residual norm to compare against tolerance
        quiet: if True, do not print iteration residuals
        method: Iterative solver method to use, defaults to BiCGSTAB

    Returns:
        Tuple (residual norm, iteration count)

    """
    x = wp.empty(dtype=saddle_system.scalar_type, shape=saddle_system.shape[0], device=saddle_system.device)
    b = wp.empty_like(x)

    wp.copy(src=x_u, dest=saddle_system.u_slice(x))
    wp.copy(src=x_p, dest=saddle_system.p_slice(x))
    wp.copy(src=b_u, dest=saddle_system.u_slice(b))
    wp.copy(src=b_p, dest=saddle_system.p_slice(b))

    func = _get_linear_solver_func(method_name=method)

    def print_callback(i, err, tol):
        print(f"{func.__name__}: at iteration {i} error = \t {err}  \t tol: {tol}")

    callback = None if quiet else print_callback

    end_iter, err, atol = func(
        A=saddle_system,
        b=b,
        x=x,
        maxiter=max_iters,
        tol=tol,
        check_every=check_every,
        M=saddle_system.preconditioner,
        callback=callback,
    )

    if not quiet:
        res_str = "OK" if err <= atol else "TRUNCATED"
        print(f"{func.__name__}: terminated after {end_iter} iterations with absolute error = \t {err} ({res_str})")

    wp.copy(dest=x_u, src=saddle_system.u_slice(x))
    wp.copy(dest=x_p, src=saddle_system.p_slice(x))

    return err, end_iter


@wp.kernel
def _compute_schur_inverse_diagonal(
    B_offsets: wp.array(dtype=int),
    B_indices: wp.array(dtype=int),
    B_values: wp.array(dtype=Any),
    A_diag: wp.array(dtype=Any),
    P_diag: wp.array(dtype=Any),
):
    row = wp.tid()

    zero = P_diag.dtype(P_diag.dtype.dtype(0.0))

    schur = zero

    beg = B_offsets[row]
    end = B_offsets[row + 1]

    for b in range(beg, end):
        B = B_values[b]
        col = B_indices[b]
        Ai = wp.inverse(A_diag[col])
        S = B * Ai * wp.transpose(B)
        schur += S

    P_diag[row] = fem.utils.inverse_qr(schur)


def invert_diagonal_bsr_matrix(A: BsrMatrix):
    """Inverts each block of a block-diagonal mass matrix"""

    values = A.values
    if not wp.types.type_is_matrix(values.dtype):
        values = values.view(dtype=wp.mat(shape=(1, 1), dtype=A.scalar_type))

    wp.launch(
        kernel=_block_diagonal_invert,
        dim=A.nrow,
        inputs=[values],
        device=values.device,
    )


@wp.kernel
def _block_diagonal_invert(values: wp.array(dtype=Any)):
    i = wp.tid()
    values[i] = fem.utils.inverse_qr(values[i])


#
# Plot utilities
#


def _plot_grid_surface(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    node_positions = field.space.node_grid()

    # Make data.
    X = node_positions[0]
    Y = node_positions[1]
    Z = field.dof_values.numpy().reshape(X.shape)

    # Plot the surface.
    return axes.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


def _plot_tri_surface(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.tri.triangulation import Triangulation

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    node_positions = field.space.node_positions().numpy()

    triangulation = Triangulation(
        x=node_positions[:, 0], y=node_positions[:, 1], triangles=field.space.node_triangulation()
    )

    Z = field.dof_values.numpy()

    # Plot the surface.
    return axes.plot_trisurf(triangulation, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


def _plot_tri_mesh(field, axes=None, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.tri.triangulation import Triangulation

    if axes is None:
        fig, axes = plt.subplots()

    vtx_positions = field.space.node_positions().numpy()
    displacement = field.dof_values.numpy()

    X = vtx_positions[:, 0] + displacement[:, 0]
    Y = vtx_positions[:, 1] + displacement[:, 1]

    triangulation = Triangulation(x=X, y=Y, triangles=field.space.node_triangulation())

    # Plot the surface.
    return axes.triplot(triangulation, **kwargs)[0]


def _plot_scatter_surface(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y = field.space.node_positions().numpy().T

    # Make data.
    Z = field.dof_values.numpy().reshape(X.shape)

    # Plot the surface.
    return axes.scatter(X, Y, Z, c=Z, cmap=cm.coolwarm)


def _plot_surface(field, axes=None):
    if hasattr(field.space, "node_grid"):
        return _plot_grid_surface(field, axes)
    elif hasattr(field.space, "node_triangulation"):
        return _plot_tri_surface(field, axes)
    else:
        return _plot_scatter_surface(field, axes)


def _plot_grid_color(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if axes is None:
        fig, axes = plt.subplots()

    node_positions = field.space.node_grid()

    # Make data.
    X = node_positions[0]
    Y = node_positions[1]
    Z = field.dof_values.numpy().reshape(X.shape)

    # Plot the surface.
    return axes.pcolormesh(X, Y, Z, cmap=cm.coolwarm)


def _plot_velocities(field, axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots()

    node_positions = field.space.node_positions().numpy()

    # Make data.
    X = node_positions[:, 0]
    Y = node_positions[:, 1]

    vel = field.dof_values.numpy()
    u = np.ascontiguousarray(vel[:, 0])
    v = np.ascontiguousarray(vel[:, 1])

    u = u.reshape(X.shape)
    v = v.reshape(X.shape)

    return axes.quiver(X, Y, u, v)


def plot_grid_streamlines(field, axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots()

    node_positions = field.space.node_grid()

    # Make data.
    X = node_positions[0][:, 0]
    Y = node_positions[1][0, :]

    vel = field.dof_values.numpy()
    u = np.ascontiguousarray(vel[:, 0])
    v = np.ascontiguousarray(vel[:, 1])

    u = np.transpose(u.reshape(node_positions[0].shape))
    v = np.transpose(v.reshape(node_positions[0].shape))

    splot = axes.streamplot(X, Y, u, v, density=2)
    splot.axes = axes
    return splot


def plot_3d_scatter(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y, Z = field.space.node_positions().numpy().T

    # Make data.
    f = field.dof_values.numpy().reshape(X.shape)

    # Plot the surface.
    return axes.scatter(X, Y, Z, c=f, cmap=cm.coolwarm)


def plot_3d_velocities(field, axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y, Z = field.space.node_positions().numpy().T

    vel = field.dof_values.numpy()
    u = np.ascontiguousarray(vel[:, 0])
    v = np.ascontiguousarray(vel[:, 1])
    w = np.ascontiguousarray(vel[:, 2])

    u = u.reshape(X.shape)
    v = v.reshape(X.shape)
    w = w.reshape(X.shape)

    return axes.quiver(X, Y, Z, u, v, w, length=1.0 / X.shape[0], normalize=False)


class Plot:
    def __init__(self, stage=None, default_point_radius=0.01):
        self.default_point_radius = default_point_radius

        self._surfaces = {}
        self._surface_vectors = {}
        self._volumes = {}

        self._usd_renderer = None
        if stage is not None:
            try:
                from warp.render import UsdRenderer

                self._usd_renderer = UsdRenderer(stage)
            except Exception as err:
                print(f"Could not initialize UsdRenderer for stage '{stage}': {err}.")

    def begin_frame(self, time):
        if self._usd_renderer is not None:
            self._usd_renderer.begin_frame(time=time)

    def end_frame(self):
        if self._usd_renderer is not None:
            self._usd_renderer.end_frame()

    def add_surface(self, name: str, field: fem.DiscreteField):
        if self._usd_renderer is not None:
            points_2d = field.space.node_positions().numpy()
            values = field.dof_values.numpy()
            points_3d = np.hstack((points_2d, values.reshape(-1, 1)))

            if hasattr(field.space, "node_triangulation"):
                indices = field.space.node_triangulation()
                self._usd_renderer.render_mesh(name, points=points_3d, indices=indices)
            else:
                self._usd_renderer.render_points(name, points=points_3d, radius=self.default_point_radius)

        if name not in self._surfaces:
            field_clone = field.space.make_field(space_partition=field.space_partition)
            self._surfaces[name] = (field_clone, [])

        self._surfaces[name][1].append(field.dof_values.numpy())

    def add_surface_vector(self, name: str, field: fem.DiscreteField):
        if self._usd_renderer is not None:
            points_2d = field.space.node_positions().numpy()
            values = field.dof_values.numpy()
            points_3d = np.hstack((points_2d + values, np.zeros_like(points_2d[:, 0]).reshape(-1, 1)))

            if hasattr(field.space, "node_triangulation"):
                indices = field.space.node_triangulation()
                self._usd_renderer.render_mesh(name, points=points_3d, indices=indices)
            else:
                self._usd_renderer.render_points(name, points=points_3d, radius=self.default_point_radius)

        if name not in self._surface_vectors:
            field_clone = field.space.make_field(space_partition=field.space_partition)
            self._surface_vectors[name] = (field_clone, [])

        self._surface_vectors[name][1].append(field.dof_values.numpy())

    def add_volume(self, name: str, field: fem.DiscreteField):
        if self._usd_renderer is not None:
            points_3d = field.space.node_positions().numpy()
            values = field.dof_values.numpy()

            self._usd_renderer.render_points(name, points_3d, radius=values)

        if name not in self._volumes:
            field_clone = field.space.make_field(space_partition=field.space_partition)
            self._volumes[name] = (field_clone, [])

        self._volumes[name][1].append(field.dof_values.numpy())

    def plot(self, streamlines: Set[str] = None, displacement: str = None):
        if streamlines is None:
            streamlines = set()
        return self._plot_matplotlib(streamlines, displacement)

    def _plot_matplotlib(self, streamlines: Set[str], displacement: str):
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        if streamlines is None:
            streamlines = []

        def make_animation(ax, field, values, plot_func, num_frames: int):
            def animate(i):
                ax.clear()
                field.dof_values = values[i]
                return plot_func(field, axes=ax)

            return animation.FuncAnimation(
                ax.figure,
                animate,
                interval=30,
                blit=False,
                frames=len(values),
            )

        for _name, (field, values) in self._surfaces.items():
            field.dof_values = values[0]
            ax = _plot_surface(field).axes

            if len(values) > 1:
                _anim = make_animation(ax, field, values, plot_func=_plot_surface, num_frames=len(values))

        for name, (field, values) in self._surface_vectors.items():
            field.dof_values = values[0]
            if name == displacement:
                ax = _plot_tri_mesh(field).axes

                if len(values) > 1:
                    _anim = make_animation(ax, field, values, plot_func=_plot_tri_mesh, num_frames=len(values))
            elif name in streamlines and hasattr(field.space, "node_grid"):
                ax = plot_grid_streamlines(field).axes
                ax.set_axis_off()
            else:
                ax = _plot_velocities(field).axes

                if len(values) > 1:
                    _anim = make_animation(ax, field, values, plot_func=_plot_velocities, num_frames=len(values))

        for _name, (field, values) in self._volumes.items():
            field.dof_values = values[0]
            ax = plot_3d_scatter(field).axes

        plt.show()
