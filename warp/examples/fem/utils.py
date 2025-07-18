# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gc
from typing import Any, Dict, Optional, Tuple

import numpy as np

import warp as wp
import warp.fem as fem
from warp.context import assert_conditional_graph_support
from warp.optim.linear import LinearOperator, aslinearoperator, preconditioner
from warp.sparse import BsrMatrix, bsr_get_diag, bsr_mv, bsr_transposed

__all__ = [
    "Plot",
    "SaddleSystem",
    "bsr_cg",
    "bsr_solve_saddle",
    "gen_hexmesh",
    "gen_quadmesh",
    "gen_tetmesh",
    "gen_trimesh",
    "invert_diagonal_bsr_matrix",
]

# matrix inversion routines contain nested loops,
# default unrolling leads to code explosion
wp.set_module_options({"max_unroll": 6})

#
# Mesh utilities
#


def gen_trimesh(res, bounds_lo: Optional[wp.vec2] = None, bounds_hi: Optional[wp.vec2] = None):
    """Constructs a triangular mesh by diving each cell of a dense 2D grid into two triangles

    Args:
        res: Resolution of the grid along each dimension
        bounds_lo: Position of the lower bound of the axis-aligned grid
        bounds_hi: Position of the upper bound of the axis-aligned grid

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
        bounds_hi: Position of the upper bound of the axis-aligned grid

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
        bounds_hi: Position of the upper bound of the axis-aligned grid

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
        bounds_hi: Position of the upper bound of the axis-aligned grid

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
    z = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij"), axes=(1, 2, 3, 0)).reshape(-1, 3)

    vidx = fem.utils.grid_to_hexes(Nx, Ny, Nz)

    return wp.array(positions, dtype=wp.vec3), wp.array(vidx, dtype=int)


def gen_volume(res, bounds_lo: Optional[wp.vec3] = None, bounds_hi: Optional[wp.vec3] = None, device=None) -> wp.Volume:
    """Constructs a wp.Volume from a dense 3D grid

    Args:
        res: Resolution of the grid along each dimension
        bounds_lo: Position of the lower bound of the axis-aligned grid
        bounds_hi: Position of the upper bound of the axis-aligned grid
        device: Cuda device on which to allocate the grid
    """

    if bounds_lo is None:
        bounds_lo = wp.vec3(0.0)

    if bounds_hi is None:
        bounds_hi = wp.vec3(1.0)

    extents = bounds_hi - bounds_lo
    voxel_size = wp.cw_div(extents, wp.vec3(res))

    x = np.arange(res[0], dtype=int)
    y = np.arange(res[1], dtype=int)
    z = np.arange(res[2], dtype=int)

    ijk = np.transpose(np.meshgrid(x, y, z), axes=(1, 2, 3, 0)).reshape(-1, 3)
    ijk = wp.array(ijk, dtype=wp.vec3i, device=device)
    return wp.Volume.allocate_by_voxels(
        ijk, voxel_size=voxel_size, translation=bounds_lo + 0.5 * voxel_size, device=device
    )


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
    M: BsrMatrix = None,
    mv_routine_uses_multiple_cuda_contexts: bool = False,
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
        mv_routine_uses_multiple_cuda_contexts: Whether the matrix-vector multiplication routine uses multiple CUDA contexts,
          which prevents the use of conditional CUDA graphs.

    Returns:
        Tuple (residual norm, iteration count)

    """

    if M is not None:
        M = aslinearoperator(M)
    elif mv_routine is None:
        M = preconditioner(A, "diag") if use_diag_precond else None
    else:
        A = LinearOperator(A.shape, A.dtype, A.device, matvec=mv_routine)
        M = None

    func = _get_linear_solver_func(method_name=method)

    callback = None

    use_cuda_graph = A.device.is_cuda and not wp.config.verify_cuda
    capturable = use_cuda_graph and not mv_routine_uses_multiple_cuda_contexts

    if capturable:
        try:
            assert_conditional_graph_support()
        except RuntimeError:
            capturable = False

    if not quiet:
        if capturable:

            @wp.func_native(snippet=f'printf("%s: ", "{func.__name__}");')
            def print_method_name():
                pass

            @fem.cache.dynamic_kernel(suffix=f"{check_every}{func.__name__}")
            def device_cg_callback(
                cur_iter: wp.array(dtype=int),
                err_sq: wp.array(dtype=Any),
                atol_sq: wp.array(dtype=Any),
            ):
                if cur_iter[0] % check_every == 0:
                    print_method_name()
                    wp.printf(
                        "at iteration %d error = \t %f  \t tol: %f\n",
                        cur_iter[0],
                        wp.sqrt(err_sq[0]),
                        wp.sqrt(atol_sq[0]),
                    )

            if check_every > 0:
                callback = device_cg_callback
        else:

            def print_callback(i, err, tol):
                print(f"{func.__name__}: at iteration {i} error = \t {err}  \t tol: {tol}")

            callback = print_callback

    if use_cuda_graph:
        # Temporarily disable garbage collection
        # Garbage collection of externally-allocated objects during graph capture may lead to
        # invalid operations or memory access errors.
        gc.disable()

    end_iter, err, atol = func(
        A=A,
        b=b,
        x=x,
        maxiter=max_iters,
        tol=tol,
        check_every=0 if capturable else check_every,
        M=M,
        callback=callback,
        use_cuda_graph=use_cuda_graph,
    )

    if use_cuda_graph:
        gc.enable()

    if isinstance(end_iter, wp.array):
        end_iter = end_iter.numpy()[0]
        err = np.sqrt(err.numpy()[0])
        atol = np.sqrt(atol.numpy()[0])

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

    err, end_iter = bsr_cg(
        saddle_system,
        x,
        b,
        max_iters=max_iters,
        tol=tol,
        check_every=check_every,
        quiet=quiet,
        method=method,
        M=saddle_system.preconditioner,
    )

    wp.copy(dest=x_u, src=saddle_system.u_slice(x))
    wp.copy(dest=x_p, src=saddle_system.p_slice(x))

    return err, end_iter


@wp.kernel(enable_backward=False)
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


@wp.kernel(enable_backward=False)
def _block_diagonal_invert(values: wp.array(dtype=Any)):
    i = wp.tid()
    values[i] = fem.utils.inverse_qr(values[i])


#
# Plot utilities
#


class Plot:
    def __init__(self, stage=None, default_point_radius=0.01):
        self.default_point_radius = default_point_radius

        self._fields = {}

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

    def add_field(self, name: str, field: fem.DiscreteField):
        if self._usd_renderer is not None:
            self._render_to_usd(field)

        if name not in self._fields:
            field_clone = field.space.make_field(space_partition=field.space_partition)
            self._fields[name] = (field_clone, [])

        self._fields[name][1].append(field.dof_values.numpy())

    def _render_to_usd(self, name: str, field: fem.DiscreteField):
        points = field.space.node_positions().numpy()
        values = field.dof_values.numpy()

        if values.ndim == 2:
            if values.shape[1] == field.space.dimension:
                # use values as displacement
                points += values
            else:
                # use magnitude
                values = np.linalg.norm(values, axis=1)

        if field.space.dimension == 2:
            z = values if values.ndim == 1 else np.zeros((points.shape[0], 1))
            points = np.hstack((points, z))

            if hasattr(field.space, "node_triangulation"):
                indices = field.space.node_triangulation()
                self._usd_renderer.render_mesh(name, points=points, indices=indices)
            else:
                self._usd_renderer.render_points(name, points=points, radius=self.default_point_radius)
        elif values.ndim == 1:
            self._usd_renderer.render_points(name, points, radius=values)
        else:
            self._usd_renderer.render_points(name, points, radius=self.default_point_radius)

    def plot(self, options: Optional[Dict[str, Any]] = None, backend: str = "auto"):
        if options is None:
            options = {}

        if backend == "pyvista":
            return self._plot_pyvista(options)
        if backend == "matplotlib":
            return self._plot_matplotlib(options)

        # try both
        try:
            return self._plot_pyvista(options)
        except ModuleNotFoundError:
            try:
                return self._plot_matplotlib(options)
            except ModuleNotFoundError:
                wp.utils.warn("pyvista or matplotlib must be installed to visualize solution results")

    def _plot_pyvista(self, options: Dict[str, Any]):
        import pyvista
        import pyvista.themes

        grids = {}
        scales = {}
        markers = {}

        animate = False

        ref_geom = options.get("ref_geom", None)
        if ref_geom is not None:
            if isinstance(ref_geom, tuple):
                vertices, counts, indices = ref_geom
                offsets = np.cumsum(counts)
                ranges = np.array([offsets - counts, offsets]).T
                faces = np.concatenate(
                    [[count, *list(indices[beg:end])] for (count, (beg, end)) in zip(counts, ranges)]
                )
                ref_geom = pyvista.PolyData(vertices, faces)
            else:
                ref_geom = pyvista.PolyData(ref_geom)

        for name, (field, values) in self._fields.items():
            cells, types = field.space.vtk_cells()
            node_pos = field.space.node_positions().numpy()

            args = options.get(name, {})

            grid_scale = np.max(np.max(node_pos, axis=0) - np.min(node_pos, axis=0))
            value_range = self._get_field_value_range(values, args)
            scales[name] = (grid_scale, value_range)

            if node_pos.shape[1] == 2:
                node_pos = np.hstack((node_pos, np.zeros((node_pos.shape[0], 1))))

            grid = pyvista.UnstructuredGrid(cells, types, node_pos)
            grids[name] = grid

            if len(values) > 1:
                animate = True

        def set_frame_data(frame):
            for name, (field, values) in self._fields.items():
                if frame > 0 and len(values) == 1:
                    continue

                v = values[frame % len(values)]
                grid = grids[name]
                grid_scale, value_range = scales[name]
                field_args = options.get(name, {})

                marker = None

                if field.space.dimension == 2 and v.ndim == 2 and v.shape[1] == 2:
                    grid.point_data[name] = np.hstack((v, np.zeros((v.shape[0], 1))))
                else:
                    grid.point_data[name] = v

                if v.ndim == 2:
                    grid.point_data[name + "_mag"] = np.linalg.norm(v, axis=1)

                if "arrows" in field_args:
                    glyph_scale = field_args["arrows"].get("glyph_scale", 1.0)
                    glyph_scale *= grid_scale / max(1.0e-8, value_range[1] - value_range[0])
                    marker = grid.glyph(scale=name, orient=name, factor=glyph_scale)
                elif "contours" in field_args:
                    levels = field_args["contours"].get("levels", 10)
                    if type(levels) == int:
                        levels = np.linspace(*value_range, levels)
                    marker = grid.contour(isosurfaces=levels, scalars=name + "_mag" if v.ndim == 2 else name)
                elif field.space.dimension == 2:
                    z_scale = grid_scale / max(1.0e-8, value_range[1] - value_range[0])

                    if "streamlines" in field_args:
                        center = np.mean(grid.points, axis=0)
                        density = field_args["streamlines"].get("density", 1.0)
                        cell_size = 1.0 / np.sqrt(field.space.geometry.cell_count())

                        separating_distance = 0.5 / (30.0 * density * cell_size)
                        # Try with various sep distance until we get at least one line
                        while separating_distance * cell_size < 1.0:
                            lines = grid.streamlines_evenly_spaced_2D(
                                vectors=name,
                                start_position=center,
                                separating_distance=separating_distance,
                                separating_distance_ratio=0.5,
                                step_length=0.25,
                                compute_vorticity=False,
                            )
                            if lines.n_lines > 0:
                                break
                            separating_distance *= 1.25
                        marker = lines.tube(radius=0.0025 * grid_scale / density)
                    elif "arrows" in field_args:
                        glyph_scale = field_args["arrows"].get("glyph_scale", 1.0)
                        glyph_scale *= grid_scale / max(1.0e-8, value_range[1] - value_range[0])
                        marker = grid.glyph(scale=name, orient=name, factor=glyph_scale)
                    elif "displacement" in field_args:
                        grid.points[:, 0:2] = field.space.node_positions().numpy() + v
                    else:
                        # Extrude surface
                        z = v if v.ndim == 1 else grid.point_data[name + "_mag"]
                        grid.points[:, 2] = z * z_scale

                elif field.space.dimension == 3:
                    if "streamlines" in field_args:
                        center = np.mean(grid.points, axis=0)
                        density = field_args["streamlines"].get("density", 1.0)
                        cell_size = 1.0 / np.sqrt(field.space.geometry.cell_count())
                        lines = grid.streamlines(vectors=name, n_points=int(100 * density))
                        marker = lines.tube(radius=0.0025 * grid_scale / np.sqrt(density))
                    elif "displacement" in field_args:
                        grid.points = field.space.node_positions().numpy() + v

                if frame == 0:
                    if v.ndim == 1:
                        grid.set_active_scalars(name)
                    else:
                        grid.set_active_vectors(name)
                        grid.set_active_scalars(name + "_mag")
                    markers[name] = marker
                elif marker:
                    markers[name].copy_from(marker)

        set_frame_data(0)

        subplot_rows = options.get("rows", 1)
        subplot_shape = (subplot_rows, (len(grids) + subplot_rows - 1) // subplot_rows)

        plotter = pyvista.Plotter(shape=subplot_shape, theme=pyvista.themes.DocumentProTheme())
        plotter.link_views()
        plotter.add_camera_orientation_widget()
        for index, (name, grid) in enumerate(grids.items()):
            plotter.subplot(index // subplot_shape[1], index % subplot_shape[1])
            grid_scale, value_range = scales[name]
            field = self._fields[name][0]
            marker = markers[name]
            if marker:
                if field.space.dimension == 2:
                    plotter.add_mesh(marker, show_scalar_bar=False)
                    plotter.add_mesh(grid, opacity=0.25, clim=value_range)
                    plotter.view_xy()
                else:
                    plotter.add_mesh(marker)
            elif field.space.geometry.cell_dimension == 3:
                plotter.add_mesh_clip_plane(grid, show_edges=True, clim=value_range, assign_to_axis="z")
            else:
                plotter.add_mesh(grid, show_edges=True, clim=value_range)

            if ref_geom:
                plotter.add_mesh(ref_geom)

        plotter.show(interactive_update=animate)

        frame = 0
        while animate and not plotter.iren.interactor.GetDone():
            frame += 1
            set_frame_data(frame)
            plotter.update()

    def _plot_matplotlib(self, options: Dict[str, Any]):
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        from matplotlib import cm

        def make_animation(fig, ax, cax, values, draw_func):
            def animate(i):
                cs = draw_func(ax, values[i])

                cax.cla()
                fig.colorbar(cs, cax)

                return cs

            return animation.FuncAnimation(
                ax.figure,
                animate,
                interval=30,
                blit=False,
                frames=len(values),
            )

        def make_draw_func(field, args, plot_func, plot_opts):
            def draw_fn(axes, values):
                axes.clear()

                field.dof_values = values
                cs = plot_func(field, axes=axes, **plot_opts)

                if "xlim" in args:
                    axes.set_xlim(*args["xlim"])
                if "ylim" in args:
                    axes.set_ylim(*args["ylim"])

                return cs

            return draw_fn

        anims = []

        field_count = len(self._fields)
        subplot_rows = options.get("rows", 1)
        subplot_shape = (subplot_rows, (field_count + subplot_rows - 1) // subplot_rows)

        for index, (name, (field, values)) in enumerate(self._fields.items()):
            args = options.get(name, {})
            v = values[0]

            plot_fn = None
            plot_3d = False
            plot_opts = {"cmap": cm.viridis}

            plot_opts["clim"] = self._get_field_value_range(values, args)

            if field.space.dimension == 2:
                if "contours" in args:
                    plot_opts["levels"] = args["contours"].get("levels", None)
                    plot_fn = _plot_contours
                elif v.ndim == 2 and v.shape[1] == 2:
                    if "displacement" in args:
                        plot_fn = _plot_displaced_tri_mesh
                    elif "streamlines" in args:
                        plot_opts["density"] = args["streamlines"].get("density", 1.0)
                        plot_fn = _plot_streamlines
                    elif "arrows" in args:
                        plot_opts["glyph_scale"] = args["arrows"].get("glyph_scale", 1.0)
                        plot_fn = _plot_quivers

                if plot_fn is None:
                    plot_fn = _plot_surface
                    plot_3d = True

            elif field.space.dimension == 3:
                if "arrows" in args or "streamlines" in args:
                    plot_opts["glyph_scale"] = args.get("arrows", {}).get("glyph_scale", 1.0)
                    plot_fn = _plot_quivers_3d
                elif field.space.geometry.cell_dimension == 2:
                    plot_fn = _plot_surface
                else:
                    plot_fn = _plot_3d_scatter
                plot_3d = True

            subplot_kw = {"projection": "3d"} if plot_3d else {}
            axes = plt.subplot(*subplot_shape, index + 1, **subplot_kw)

            if not plot_3d:
                axes.set_aspect("equal")

            draw_fn = make_draw_func(field, args, plot_func=plot_fn, plot_opts=plot_opts)
            cs = draw_fn(axes, values[0])

            fig = plt.gcf()
            cax = fig.colorbar(cs).ax

            if len(values) > 1:
                anims.append(make_animation(fig, axes, cax, values, draw_func=draw_fn))

        plt.show()

    @staticmethod
    def _get_field_value_range(values, field_options: Dict[str, Any]):
        value_range = field_options.get("clim", None)
        if value_range is None:
            value_range = (
                min(np.min(_value_or_magnitude(v)) for v in values),
                max(np.max(_value_or_magnitude(v)) for v in values),
            )

        return value_range


def _value_or_magnitude(values: np.ndarray):
    if values.ndim == 1:
        return values
    return np.linalg.norm(values, axis=-1)


def _field_triangulation(field):
    from matplotlib.tri import Triangulation

    node_positions = field.space.node_positions().numpy()
    return Triangulation(x=node_positions[:, 0], y=node_positions[:, 1], triangles=field.space.node_triangulation())


def _plot_surface(field, axes, **kwargs):
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Normalize

    C = _value_or_magnitude(field.dof_values.numpy())

    positions = field.space.node_positions().numpy().T
    if field.space.dimension == 3:
        X, Y, Z = positions
    else:
        X, Y = positions
        Z = C
        axes.set_zlim(kwargs["clim"])

    if hasattr(field.space, "node_grid"):
        X, Y = field.space.node_grid()
        C = C.reshape(X.shape)
        return axes.plot_surface(X, Y, C, linewidth=0.1, antialiased=False, **kwargs)

    if hasattr(field.space, "node_triangulation"):
        triangulation = _field_triangulation(field)

        if field.space.dimension == 3:
            plot = axes.plot_trisurf(triangulation, Z, linewidth=0.1, antialiased=False)
            # change colors -- recompute color map manually
            vmin, vmax = kwargs["clim"]
            norm = Normalize(vmin=vmin, vmax=vmax)
            values = np.mean(C[triangulation.triangles], axis=1)
            colors = get_cmap(kwargs["cmap"])(norm(values))
            plot.set_norm(norm)
            plot.set_fc(colors)
        else:
            plot = axes.plot_trisurf(triangulation, C, linewidth=0.1, antialiased=False, **kwargs)

        return plot

    # scatter
    return axes.scatter(X, Y, Z, c=C, **kwargs)


def _plot_displaced_tri_mesh(field, axes, **kwargs):
    triangulation = _field_triangulation(field)

    displacement = field.dof_values.numpy()
    triangulation.x += displacement[:, 0]
    triangulation.y += displacement[:, 1]

    Z = _value_or_magnitude(displacement)

    # Plot the surface.
    cs = axes.tripcolor(triangulation, Z, **kwargs)
    axes.triplot(triangulation, lw=0.1)

    return cs


def _plot_quivers(field, axes, clim=None, glyph_scale=1.0, **kwargs):
    X, Y = field.space.node_positions().numpy().T

    vel = field.dof_values.numpy()
    u = vel[:, 0].reshape(X.shape)
    v = vel[:, 1].reshape(X.shape)

    return axes.quiver(X, Y, u, v, _value_or_magnitude(vel), scale=1.0 / glyph_scale, **kwargs)


def _plot_quivers_3d(field, axes, clim=None, cmap=None, glyph_scale=1.0, **kwargs):
    X, Y, Z = field.space.node_positions().numpy().T

    vel = field.dof_values.numpy()

    colors = cmap((_value_or_magnitude(vel) - clim[0]) / (clim[1] - clim[0]))

    u = vel[:, 0].reshape(X.shape) / (clim[1] - clim[0])
    v = vel[:, 1].reshape(X.shape) / (clim[1] - clim[0])
    w = vel[:, 2].reshape(X.shape) / (clim[1] - clim[0])

    return axes.quiver(X, Y, Z, u, v, w, colors=colors, length=glyph_scale, clim=clim, cmap=cmap, **kwargs)


def _plot_streamlines(field, axes, clim=None, **kwargs):
    import matplotlib.tri as tr

    triangulation = _field_triangulation(field)

    vel = field.dof_values.numpy()

    itp_vx = tr.CubicTriInterpolator(triangulation, vel[:, 0])
    itp_vy = tr.CubicTriInterpolator(triangulation, vel[:, 1])

    X, Y = np.meshgrid(
        np.linspace(np.min(triangulation.x), np.max(triangulation.x), 100),
        np.linspace(np.min(triangulation.y), np.max(triangulation.y), 100),
    )

    u = itp_vx(X, Y)
    v = itp_vy(X, Y)
    C = np.sqrt(u * u + v * v)

    plot = axes.streamplot(X, Y, u, v, color=C, **kwargs)
    return plot.lines


def _plot_contours(field, axes, clim=None, **kwargs):
    triangulation = _field_triangulation(field)

    Z = _value_or_magnitude(field.dof_values.numpy())

    tc = axes.tricontourf(triangulation, Z, **kwargs)
    axes.tricontour(triangulation, Z, **kwargs)
    return tc


def _plot_3d_scatter(field, axes, **kwargs):
    X, Y, Z = field.space.node_positions().numpy().T

    f = _value_or_magnitude(field.dof_values.numpy()).reshape(X.shape)

    return axes.scatter(X, Y, Z, c=f, **kwargs)
