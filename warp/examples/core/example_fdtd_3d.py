# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An example implementation of an FDTD solver.

A point source on the surface of a 3-D Luneburg lens launches a curved
wavefront into the lens. The lens straightens the wavefront into a parallel
beam that leaves the opposite side.

The example uses the finite-difference time-domain (FDTD) method on a Yee
staggered grid. It marches Maxwell's curl equations with a leapfrog scheme and
a matched loss rate ``gamma``:

    mu_0 * dH/dt = -curl E - gamma * mu_0 * H
    eps_0 * eps_r * dE/dt = curl H - gamma * eps_0 * eps_r * E - J

The three 2-D coordinate-plane views of a 3-D Yee cell are:

        x-y plane                 x-z plane                 y-z plane
           y                         z                         z
           ^                         ^                         ^
           |                         |                         |
      o---E_x---o               o---E_x---o               o---E_y---o
      |         |               |         |               |         |
      E_y H_z   E_y             E_z  H_y  E_z             E_z  H_x  E_z
      |         |               |         |               |         |
      o---E_x---o ---> x        o---E_x---o ---> x        o---E_y---o ---> y

The temporal staggering is as follows:

    time       (n - 1/2) dt          n dt          (n + 1/2) dt        (n + 1) dt
               ------------------------------------------------------------------>

    H          H^(n-1/2)  -- update_h using E^n --> H^(n+1/2)

    E                                  E^n  -- update_e using H^(n+1/2) --> E^(n+1)

The ``update_e`` kernel keeps the tangential electric field components at zero on the six
outer faces. The ``gamma`` profile attenuates outgoing waves toward zero before they reach
those faces, reducing boundary reflections. It ramps from zero at the inner edge of the attenuation
layer to ``GAMMA_MAX`` at the wall and vanishes elsewhere.

The lens is a graded-index sphere of radius ``R_LENS`` centered on the origin,

    eps_r(r) = n(r)^2 = 2 - (r / R_LENS)^2    for r <= R_LENS

the Luneburg profile. A soft current source on the ``-z`` surface launches a curved
wavefront into the lens. The graded index refracts it into a parallel beam leaving the
``+z`` surface, visible in the rendered ``E_x`` center plane. ``J`` is ramped up from rest.

The scheme is second-order accurate in space and time and stable for
``dt <= dx / (c * sqrt(3))`` in 3-D. Units are normalized so that
``c = eps_0 = mu_0 = lambda_0 = f_0 = T_0 = 1``, hence a vacuum impedance of 1.
The cubic domain is fixed at ``HALF_WIDTH`` wavelengths half-extent; ``--resolution``
sets how finely it is gridded and every other grid quantity derives from it.

Requires Matplotlib for interactive visualization (use --headless to run without it).

Usage:
    python example_fdtd_3d.py
"""

import math

import numpy as np

import warp as wp

wp.config.enable_backward = False

# physical inputs, in normalized units (c = mu_0 = eps_0 = lambda_0 = f_0 = T_0 = 1)
HALF_WIDTH = 4.25  # half-extent of the cubic domain
R_LENS = 3.0  # lens radius in vacuum wavelengths
L_ABSORBER = 2.0 / 3.0  # depth of the attenuation layer lining each wall
E0 = 1.0  # source amplitude, tuned at the reference grid spacing DX_REF
DX_REF = 1.0 / 30.0  # grid spacing at which E0 and PLOT_E_SCALE were tuned
SOURCE_RAMP = 3.0  # periods over which the source envelope rises from rest
SIM_DURATION = 36.0  # default headless run length, in source periods
DIAG_INTERVAL = 100  # timesteps between electromagnetic energy diagnostics
PLOT_E_SCALE = 2.5e-4  # colorbar half-range for the E_x frames; SOURCE_SCALE keeps it valid at every resolution
GAMMA_MAX = -2.0 * math.log(1.0e-3) / L_ABSORBER

# resolution and derived quantities
N_GRID = 128  # stored nodes per axis across the fixed domain
DX = 2.0 * HALF_WIDTH / (N_GRID - 1)
DT = 0.95 * DX / math.sqrt(3.0)  # 3-D CFL limit with a 0.95 safety factor
SOURCE_SCALE = (DX_REF / DX) ** 2  # holds the dipole moment, fixed across resolutions
SOURCE_IJ = (N_GRID - 1) // 2
SOURCE_K = round((HALF_WIDTH - R_LENS) / DX)


@wp.func
def loss_coeffs(gamma: float) -> tuple[float, float]:
    """Return the matched-loss coefficients for a field component at its position on the Yee grid."""
    gamma_dt_half = 0.5 * DT * gamma
    return (1.0 - gamma_dt_half) / (1.0 + gamma_dt_half), 1.0 / (1.0 + gamma_dt_half)


@wp.func
def inv_eps_r(x: float, y: float, z: float) -> float:
    """Return the inverse relative permittivity of the Luneburg lens at a point given by (x, y, z)."""
    r_sq = x * x + y * y + z * z

    eps_r = float(1.0)
    if r_sq < R_LENS * R_LENS:
        eps_r = 2.0 - r_sq / (R_LENS * R_LENS)

    return 1.0 / eps_r


@wp.func
def source_amplitude(timestep_index: int) -> float:
    """Return the scaled point-source term ``DX * J_x`` sampled at ``t^{n+1/2}``.

    ``SOURCE_SCALE`` keeps the dipole moment ``J_x * DX^3``, and with it the radiated
    field and ``PLOT_E_SCALE``, independent of resolution.
    """
    t = (float(timestep_index) + 0.5) * DT
    return -E0 * SOURCE_SCALE * wp.sin(2.0 * wp.pi * t) * (1.0 - wp.exp(-((t / SOURCE_RAMP) ** 2.0)))


@wp.kernel
def update_h(
    e_x: wp.array3d[float],
    e_y: wp.array3d[float],
    e_z: wp.array3d[float],
    gamma_integer: wp.array[float],
    gamma_half_offset: wp.array[float],
    h_x: wp.array3d[float],
    h_y: wp.array3d[float],
    h_z: wp.array3d[float],
):
    """Advance ``H`` from ``t = (n - 1/2) dt`` to ``t = (n + 1/2) dt`` using ``E^n``.

    The representative component update (the others follow by cyclic permutation) is::

        H_x^{n+1/2}|_{i, j+1/2, k+1/2} = a * H_x^{n-1/2}
            - (b * dt / dx) * [(E_z|_{j+1} - E_z|_j) - (E_y|_{k+1} - E_y|_k)]

    with the matched-loss coefficients of ``loss_coeffs`` sampled from the 1-D profiles
    ``gamma_integer`` and ``gamma_half_offset`` at each component's staggered coordinates
    (the grid is cubic, so all three axes index the same two profiles).
    """
    i, j, k = wp.tid()

    dt_dx = DT / DX
    # last index whose half-offset indices are still inside the walls
    last_staggered = N_GRID - 2

    # H_x at (i, j+1/2, k+1/2)
    if j <= last_staggered and k <= last_staggered:
        a_x, b_x = loss_coeffs(gamma_integer[i] + gamma_half_offset[j] + gamma_half_offset[k])
        curl_x = (e_z[i, j + 1, k] - e_z[i, j, k]) - (e_y[i, j, k + 1] - e_y[i, j, k])
        h_x[i, j, k] = a_x * h_x[i, j, k] - b_x * dt_dx * curl_x

    # H_y at (i+1/2, j, k+1/2)
    if i <= last_staggered and k <= last_staggered:
        a_y, b_y = loss_coeffs(gamma_half_offset[i] + gamma_integer[j] + gamma_half_offset[k])
        curl_y = (e_x[i, j, k + 1] - e_x[i, j, k]) - (e_z[i + 1, j, k] - e_z[i, j, k])
        h_y[i, j, k] = a_y * h_y[i, j, k] - b_y * dt_dx * curl_y

    # H_z at (i+1/2, j+1/2, k)
    if i <= last_staggered and j <= last_staggered:
        a_z, b_z = loss_coeffs(gamma_half_offset[i] + gamma_half_offset[j] + gamma_integer[k])
        curl_z = (e_y[i + 1, j, k] - e_y[i, j, k]) - (e_x[i, j + 1, k] - e_x[i, j, k])
        h_z[i, j, k] = a_z * h_z[i, j, k] - b_z * dt_dx * curl_z


@wp.kernel
def update_e(
    h_x: wp.array3d[float],
    h_y: wp.array3d[float],
    h_z: wp.array3d[float],
    gamma_integer: wp.array[float],
    gamma_half_offset: wp.array[float],
    timestep_counter: wp.array[wp.int32],
    e_x: wp.array3d[float],
    e_y: wp.array3d[float],
    e_z: wp.array3d[float],
):
    """Advance ``E`` from ``t = n dt`` to ``t = (n + 1) dt`` using ``H^{n+1/2}``.

    The representative component update (the others follow by cyclic permutation) is::

        E_x^{n+1}|_{i+1/2, j, k} = a * E_x^n + (b * dt / (eps_r * dx))
            * [(H_z|_{j+1/2} - H_z|_{j-1/2}) - (H_y|_{k+1/2} - H_y|_{k-1/2}) - dx * J_x]

    where ``eps_r`` is the Luneburg profile of ``inv_eps_r``, evaluated analytically. A
    point source drives ``J_x`` at the single node ``(SOURCE_IJ, SOURCE_IJ, SOURCE_K)``,
    with ``source_amplitude`` supplying ``dx * J_x``.

    """
    i, j, k = wp.tid()

    timestep_index = timestep_counter[0]

    dt_dx = DT / DX

    last_staggered = N_GRID - 2
    last_interior = N_GRID - 2

    x_node = float(i) * DX - HALF_WIDTH
    y_node = float(j) * DX - HALF_WIDTH
    z_node = float(k) * DX - HALF_WIDTH
    half_dx = 0.5 * DX

    # E_x at (i+1/2, j, k), carrying the point source
    if i <= last_staggered and 1 <= j <= last_interior and 1 <= k <= last_interior:
        a_x, b_x = loss_coeffs(gamma_half_offset[i] + gamma_integer[j] + gamma_integer[k])
        curl_x = (h_z[i, j, k] - h_z[i, j - 1, k]) - (h_y[i, j, k] - h_y[i, j, k - 1])
        if i == SOURCE_IJ and j == SOURCE_IJ and k == SOURCE_K:
            curl_x -= source_amplitude(timestep_index)
        inv_eps_x = inv_eps_r(x_node + half_dx, y_node, z_node)
        new_e_x = a_x * e_x[i, j, k] + b_x * dt_dx * inv_eps_x * curl_x
        e_x[i, j, k] = new_e_x

    # E_y at (i, j+1/2, k)
    if 1 <= i <= last_interior and j <= last_staggered and 1 <= k <= last_interior:
        a_y, b_y = loss_coeffs(gamma_integer[i] + gamma_half_offset[j] + gamma_integer[k])
        curl_y = (h_x[i, j, k] - h_x[i, j, k - 1]) - (h_z[i, j, k] - h_z[i - 1, j, k])
        inv_eps_y = inv_eps_r(x_node, y_node + half_dx, z_node)
        new_e_y = a_y * e_y[i, j, k] + b_y * dt_dx * inv_eps_y * curl_y
        e_y[i, j, k] = new_e_y

    # E_z at (i, j, k+1/2)
    if 1 <= i <= last_interior and 1 <= j <= last_interior and k <= last_staggered:
        a_z, b_z = loss_coeffs(gamma_integer[i] + gamma_integer[j] + gamma_half_offset[k])
        curl_z = (h_y[i, j, k] - h_y[i - 1, j, k]) - (h_x[i, j, k] - h_x[i, j - 1, k])
        inv_eps_z = inv_eps_r(x_node, y_node, z_node + half_dx)
        new_e_z = a_z * e_z[i, j, k] + b_z * dt_dx * inv_eps_z * curl_z
        e_z[i, j, k] = new_e_z


@wp.kernel
def compute_em_energy(
    e_x: wp.array3d[float],
    e_y: wp.array3d[float],
    e_z: wp.array3d[float],
    h_x: wp.array3d[float],
    h_y: wp.array3d[float],
    h_z: wp.array3d[float],
    electric_energy_sum: wp.array[float],
    magnetic_energy_sum: wp.array[float],
):
    """Accumulate ``eps_r * E^2`` and ``H^2`` over the Yee grid."""
    i, j, k = wp.tid()

    x_node = float(i) * DX - HALF_WIDTH
    y_node = float(j) * DX - HALF_WIDTH
    z_node = float(k) * DX - HALF_WIDTH
    half_dx = 0.5 * DX

    # Note: DO NOT interpret `electric_energy_density` below as the energy density at a point `i, j, k` in space.
    # The ``compute_em_energy`` integrates each component over the entire domain at its own respective Yee cell locations.
    # It is valid here to do so because we are ultimately reporting a single summed electrical and magnetic energy value
    # over the entire domain.
    electric_energy_density = (
        e_x[i, j, k] * e_x[i, j, k] / inv_eps_r(x_node + half_dx, y_node, z_node)
        + e_y[i, j, k] * e_y[i, j, k] / inv_eps_r(x_node, y_node + half_dx, z_node)
        + e_z[i, j, k] * e_z[i, j, k] / inv_eps_r(x_node, y_node, z_node + half_dx)
    )
    magnetic_energy_density = h_x[i, j, k] * h_x[i, j, k] + h_y[i, j, k] * h_y[i, j, k] + h_z[i, j, k] * h_z[i, j, k]

    # `tile_sum` reduces per-thread values to one sum per block. Each
    # `wp.tile_atomic_add` then issues one global atomic per block instead
    # of one global atomic per thread (which would have been the case with wp.atomic_add()).
    electric_tile_sum = wp.tile_sum(wp.tile(electric_energy_density))
    magnetic_tile_sum = wp.tile_sum(wp.tile(magnetic_energy_density))
    wp.tile_atomic_add(electric_energy_sum, electric_tile_sum)
    wp.tile_atomic_add(magnetic_energy_sum, magnetic_tile_sum)


@wp.kernel
def increment_timestep(timestep_counter: wp.array[wp.int32]):
    timestep_counter[0] = timestep_counter[0] + 1


@wp.kernel
def extract_e_x_center_plane(e_x: wp.array3d[float], y_index: int, e_x_plane: wp.array2d[float]):
    i, k = wp.tid()
    e_x_plane[i, k] = e_x[i, y_index, k]


def build_loss_profiles() -> tuple[wp.array[float], wp.array[float]]:
    """Build the 1-D matched-loss rate profiles at grid nodes and half-cell offsets.

    The loss rate is additive per axis, ``gamma_q(d) = GAMMA_MAX * (d / L_ABSORBER)^3``,
    where ``d`` runs from zero at the inner edge of the attenuation layer to
    ``L_ABSORBER`` at the wall. The grid is cubic and uniform, so the three axes share the
    same two profiles.
    """

    def gamma_profile(coords: np.ndarray) -> np.ndarray:
        depth = np.clip((np.abs(coords) - (HALF_WIDTH - L_ABSORBER)) / L_ABSORBER, 0.0, 1.0)
        return GAMMA_MAX * depth**3

    node_coords = (np.arange(N_GRID) - (N_GRID - 1) / 2.0) * DX
    half_coords = node_coords + DX / 2.0

    gamma_integer = wp.array(gamma_profile(node_coords).astype(np.float32), dtype=float)
    gamma_half_offset = wp.array(gamma_profile(half_coords).astype(np.float32), dtype=float)
    return gamma_integer, gamma_half_offset


class Example:
    """FDTD solver for a point source collimated by a Luneburg lens.

    The six field components are stored as ``(N_GRID, N_GRID, N_GRID)`` fp32 arrays.
    Simulation parameters (``N_GRID``, ``DX``, ``DT``, etc.) are defined as
    module-level constants and captured directly by Warp kernels.
    """

    def __init__(self) -> None:
        self.steps_per_frame = 1  # maximum timesteps per rendered frame

        # use CUDA graph if GPU is available
        self.use_cuda_graph = wp.get_device().is_cuda
        self.graph = None

        field_shape = (N_GRID, N_GRID, N_GRID)
        self.e_x = wp.zeros(field_shape, dtype=float)
        self.e_y = wp.zeros(field_shape, dtype=float)
        self.e_z = wp.zeros(field_shape, dtype=float)
        self.h_x = wp.zeros(field_shape, dtype=float)
        self.h_y = wp.zeros(field_shape, dtype=float)
        self.h_z = wp.zeros(field_shape, dtype=float)
        self.e_x_center_plane_buffer = wp.empty((N_GRID, N_GRID), dtype=float)

        self.electric_energy_sum = wp.zeros(1, dtype=float)
        self.magnetic_energy_sum = wp.zeros(1, dtype=float)

        # global timestep index counter on device to make CUDA graph capture possible
        self.timestep_counter = wp.zeros(1, dtype=wp.int32)

        # the three axes index the same integer and half-offset profiles for a cubic uniform grid
        self.gamma_integer, self.gamma_half_offset = build_loss_profiles()

        points_per_wavelength = (N_GRID - 1) / (2.0 * HALF_WIDTH)
        print(
            f"FDTD Luneburg lens: {N_GRID}^3 grid, dx = {DX:.6f} "
            f"({points_per_wavelength:.1f} points per wavelength), dt = {DT:.6f}"
        )

    def _check_em_energy(self, timestep: int) -> None:
        """Compute and report electromagnetic energy, raising if it is not finite."""
        self.electric_energy_sum.zero_()
        self.magnetic_energy_sum.zero_()
        wp.launch(
            compute_em_energy,
            dim=(N_GRID, N_GRID, N_GRID),
            inputs=[self.e_x, self.e_y, self.e_z, self.h_x, self.h_y, self.h_z],
            outputs=[self.electric_energy_sum, self.magnetic_energy_sum],
        )

        w_e = 0.5 * DX**3 * float(self.electric_energy_sum.numpy()[0])
        w_h = 0.5 * DX**3 * float(self.magnetic_energy_sum.numpy()[0])
        if not (np.isfinite(w_e) and np.isfinite(w_h)):
            raise RuntimeError(f"Non-finite electromagnetic energy at timestep {timestep}.")
        print(f"step {timestep}, t = {timestep * DT:.6f}, W_E = {w_e:.6e}, W_H = {w_h:.6e}", flush=True)

    def step(self) -> None:
        """Advance ``H`` from curl ``E``, then ``E`` from curl ``H`` in leapfrog fashion."""
        wp.launch(
            update_h,
            dim=(N_GRID, N_GRID, N_GRID),
            inputs=[self.e_x, self.e_y, self.e_z, self.gamma_integer, self.gamma_half_offset],
            outputs=[self.h_x, self.h_y, self.h_z],
        )

        wp.launch(
            update_e,
            dim=(N_GRID, N_GRID, N_GRID),
            inputs=[
                self.h_x,
                self.h_y,
                self.h_z,
                self.gamma_integer,
                self.gamma_half_offset,
                self.timestep_counter,
            ],
            outputs=[self.e_x, self.e_y, self.e_z],
        )
        # Advance the device-side timestep counter after an electric-field update.
        wp.launch(increment_timestep, dim=1, inputs=[self.timestep_counter])

    def e_x_center_plane(self) -> np.ndarray:
        """Return ``E_x`` on the x-z plane closest to y = 0."""
        j_center = (N_GRID - 1) // 2
        wp.launch(
            extract_e_x_center_plane,
            dim=(N_GRID, N_GRID),
            inputs=[self.e_x, j_center],
            outputs=[self.e_x_center_plane_buffer],
        )
        return self.e_x_center_plane_buffer.numpy()

    def step_and_render_frame(self, frame_num: int, img=None, num_steps: int | None = None) -> tuple:
        """Advance up to ``steps_per_frame`` timesteps and update the Matplotlib image."""
        first_timestep = frame_num * self.steps_per_frame
        steps_this_frame = self.steps_per_frame
        if num_steps is not None:
            steps_this_frame = min(steps_this_frame, num_steps - first_timestep)
        self.timestep_counter.fill_(first_timestep)

        for substep in range(steps_this_frame):
            timestep_index = first_timestep + substep
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.step()
            if (timestep_index + 1) % DIAG_INTERVAL == 0:
                self._check_em_energy(timestep_index + 1)

        if img is not None:
            sim_timestep = first_timestep + steps_this_frame
            img.set_array(self.e_x_center_plane())
            img.axes.set_title(f"Luneburg lens, {N_GRID}$^3$, $E_x$ at timestep {sim_timestep}")

        return (img,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="3-D FDTD simulation of a point source collimated by a Luneburg lens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the default Warp device.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=N_GRID,
        help="Grid resolution: nodes per axis across the fixed 8.5-wavelength domain. "
        "Refining the grid leaves the physical extent, lens, and attenuation-layer depth unchanged.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Total number of leapfrog timesteps. Defaults to unlimited in interactive mode and to "
        "SIM_DURATION source periods, rounded to whole steps, in headless mode.",
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=20,
        help="Maximum number of leapfrog timesteps per rendered frame.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    if args.steps_per_frame < 1:
        parser.error(f"--steps-per-frame must be at least 1, got {args.steps_per_frame}")
    if args.resolution < 48:
        parser.error(
            f"--resolution {args.resolution} is too coarse: below 48 nodes per axis the absorbing "
            f"layer spans fewer than four cells and the grid falls under six points per wavelength."
        )

    # Rebind the resolution-derived module constants before the first kernel launch.
    N_GRID = args.resolution
    DX = 2.0 * HALF_WIDTH / (N_GRID - 1)
    DT = 0.95 * DX / math.sqrt(3.0)
    SOURCE_SCALE = (DX_REF / DX) ** 2
    SOURCE_IJ = (N_GRID - 1) // 2
    SOURCE_K = round((HALF_WIDTH - R_LENS) / DX)

    # dt tracks dx, so the headless default step count derives from a fixed physical duration
    num_steps = args.num_steps
    if num_steps is None and args.headless:
        num_steps = round(SIM_DURATION / DT)

    with wp.ScopedDevice(args.device):
        example = Example()
        example.steps_per_frame = args.steps_per_frame

        if example.use_cuda_graph:
            # capture one leapfrog timestep in a CUDA graph
            with wp.ScopedCapture() as capture:
                example.step()
            example.graph = capture.graph

        if args.headless:
            import time

            num_frames = (num_steps + args.steps_per_frame - 1) // args.steps_per_frame

            wp.synchronize_device()
            start_time = time.perf_counter()

            for i in range(num_frames):
                example.step_and_render_frame(frame_num=i, num_steps=num_steps)

            wp.synchronize_device()
            stop_time = time.perf_counter()

            elapsed = stop_time - start_time
            mcells_per_s = num_steps * N_GRID**3 / elapsed / 1e6
            print(f"{num_steps} timesteps on a {N_GRID}^3 grid: {elapsed:.3f} s, {mcells_per_s:.1f} Mcell-updates/s")
        else:
            import matplotlib.animation as anim
            import matplotlib.pyplot as plt

            num_frames = (
                (num_steps + args.steps_per_frame - 1) // args.steps_per_frame if num_steps is not None else None
            )

            fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
            ax.set_title(f"Luneburg lens, {N_GRID}$^3$, $E_x$ at timestep 0")
            ax.set_xlabel("z")
            ax.set_ylabel("x")

            img = ax.imshow(
                example.e_x_center_plane(),
                origin="lower",
                cmap="seismic",
                interpolation="antialiased",
                vmin=-PLOT_E_SCALE,
                vmax=PLOT_E_SCALE,
                extent=[-HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH],
            )
            fig.colorbar(img, ax=ax, label="$E_x$")
            ax.add_patch(plt.Circle((0.0, 0.0), R_LENS, fill=False, color="0.3", lw=0.8, ls="--"))

            seq = anim.FuncAnimation(
                fig,
                example.step_and_render_frame,
                fargs=(img, num_steps),
                frames=num_frames,
                init_func=lambda: (),  # without this, Matplotlib delivers frame 0 twice and replays the source window
                blit=False,
                interval=1,
                repeat=False,
                cache_frame_data=False,
            )

            plt.show()
