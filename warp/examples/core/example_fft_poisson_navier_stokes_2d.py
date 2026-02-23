# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


######################################################################################
# Example 2-D Incompressible Turbulence in a Periodic Box
#
# Implements a 2-D Navier Stokes solver in a periodic box using the
# streamfunction-vorticity formulation. The Poisson equation that relates streamfunction
# to vorticity is solved in Fourier space using tile-based FFT in Warp.
# Timestepping is performed using strong stability preserving RK3 scheme.
#
# Requires matplotlib for interactive visualization (use --headless to run without it).
#
######################################################################################

import numpy as np

import warp as wp

# simulation parameters
N_GRID = 512
DOMAIN_SIZE = 2 * np.pi
DT = 0.001
RE = 1000.0

# derived grid constants (captured by Warp kernels)
H = DOMAIN_SIZE / N_GRID
INV_H2 = 1.0 / (H * H)
INV_2H = 1.0 / (2.0 * H)

# parameters for Warp's tiled-FFT functionality
TILE_M = 1
TILE_N = N_GRID
TILE_TRANSPOSE_DIM = 16
BLOCK_DIM = TILE_N // 2

wp.config.enable_backward = False


@wp.func
def factorial(n: wp.int32) -> wp.int32:
    """Compute factorial of ``n``."""
    result = wp.int32(1)
    for i in range(2, n + 1):
        result *= i
    return result


@wp.func
def energy_spectrum(k: wp.float32, s: wp.int32, k_peak: wp.float32) -> wp.float32:
    """Compute energy at wavenumber magnitude k.

    Follows San and Staples 2012 Computers and Fluids (page 49).
    https://www.sciencedirect.com/science/article/abs/pii/S0045793012001363.

    Args:
        k: Input wavenumber magnitude.
        s: Shape parameter of spectrum.
        k_peak: Wavenumber magnitude at which maximum of energy spectrum lies.

    Returns:
        Energy contained at wavenumber magnitude k.
    """
    s_factorial = wp.float32(factorial(s))
    s_float32 = wp.float32(s)
    a_s = (2.0 * s_float32 + 1.0) ** (s_float32 + 1.0) / (2.0**s_float32 * s_factorial)
    energy_k = (
        a_s
        / (2.0 * k_peak)
        * (k / k_peak) ** (2.0 * s_float32 + 1.0)
        * wp.exp(-(s_float32 + 0.5) * (k / k_peak) ** 2.0)
    )
    return energy_k


@wp.func
def phase_randomizer(
    zeta: wp.array2d(dtype=wp.float32), eta: wp.array2d(dtype=wp.float32), i: int, j: int
) -> wp.float32:
    """Calculate value of the random phase at index (i, j).

    Follows San and Staples 2012 to return phase value in any quadrant based on
    the values of eta and zeta in the first quadrant.

    Args:
        zeta: First phase function.
        eta: Second phase function
        i: rowwise index on the 2-D simulation domain.
        j: columnwise index on the 2-D simulation domain

    Returns:
        Value of the random phase in any quadrant.
    """
    n_half = N_GRID // 2

    # first quadrant
    if i < n_half and j < n_half:
        return zeta[i, j] + eta[i, j]
    # second quadrant
    if i >= n_half and j < n_half:
        return -zeta[N_GRID - i, j] + eta[N_GRID - i, j]
    # third quadrant
    if i >= n_half and j >= n_half:
        return -zeta[N_GRID - i, N_GRID - j] - eta[N_GRID - i, N_GRID - j]
    # fourth quadrant
    return zeta[i, N_GRID - j] - eta[i, N_GRID - j]


@wp.func
def cyclic_index(idx: wp.int32, n: wp.int32) -> wp.int32:
    """Map any index to [0, n-1] for periodic boundary conditions."""
    ret_idx = idx % n
    if ret_idx < 0:
        ret_idx += n
    return ret_idx


@wp.kernel
def decaying_turbulence_initializer(
    k_peak: wp.float32,
    s: wp.int32,
    k_mag: wp.array2d(dtype=wp.float32),
    zeta: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    omega_hat_init: wp.array2d(dtype=wp.vec2f),
):
    """Initialize the vorticity field in Fourier space for decaying turbulence.

    Args:
        k_peak: Peak wavenumber of the energy spectrum.
        s: Shape parameter of the energy spectrum.
        k_mag: Wavenumber magnitude array.
        zeta: First phase function for phase randomization.
        eta: Second phase function for phase randomization.
        omega_hat_init: Output vorticity field in Fourier space.
    """
    i, j = wp.tid()

    amplitude = wp.sqrt((k_mag[i, j] / wp.pi) * energy_spectrum(k_mag[i, j], s, k_peak))
    phase = phase_randomizer(zeta, eta, i, j)
    omega_hat_init[i, j] = wp.vec2f(amplitude * wp.cos(phase), amplitude * wp.sin(phase))


@wp.kernel
def rk3_update(
    coeff0: float,
    coeff1: float,
    coeff2: float,
    omega_start: wp.array2d(dtype=float),
    omega_curr: wp.array2d(dtype=float),
    psi: wp.array2d(dtype=float),
    omega_next: wp.array2d(dtype=float),
):
    """Perform a single substep of SSP-RK3.

    Module-level constants ``N_GRID``, ``RE``, ``DT``, ``INV_H2``, and ``INV_2H``
    are captured automatically by Warp.

    Args:
        coeff0: SSP-RK3 coefficient for omega_start.
        coeff1: SSP-RK3 coefficient for omega_curr.
        coeff2: SSP-RK3 coefficient for RHS.
        omega_start: Vorticity field at the beginning of the time step (read-only).
        omega_curr: Vorticity field at the current substep (read-only stencil source).
        psi: Stream function field (read-only).
        omega_next: Output vorticity field for this substep (write-only).
    """
    i, j = wp.tid()

    left = cyclic_index(i - 1, N_GRID)
    right = cyclic_index(i + 1, N_GRID)
    top = cyclic_index(j + 1, N_GRID)
    bottom = cyclic_index(j - 1, N_GRID)

    laplacian = (
        omega_curr[right, j] + omega_curr[left, j] + omega_curr[i, top] + omega_curr[i, bottom] - 4.0 * omega_curr[i, j]
    ) * INV_H2

    # Advection via the Jacobian J(omega, psi) = d(omega)/dx * d(psi)/dy - d(omega)/dy * d(psi)/dx
    j1 = ((omega_curr[right, j] - omega_curr[left, j]) * INV_2H) * ((psi[i, top] - psi[i, bottom]) * INV_2H)
    j2 = ((omega_curr[i, top] - omega_curr[i, bottom]) * INV_2H) * ((psi[right, j] - psi[left, j]) * INV_2H)

    # RHS of vorticity equation: viscous diffusion + advection
    rhs = (1.0 / RE) * laplacian + j2 - j1

    omega_next[i, j] = coeff0 * omega_start[i, j] + coeff1 * omega_curr[i, j] + coeff2 * DT * rhs


@wp.kernel
def copy_float_to_vec2(omega: wp.array2d(dtype=wp.float32), omega_complex: wp.array2d(dtype=wp.vec2f)):
    """Pack a real array into a complex array with zero imaginary part."""
    i, j = wp.tid()
    omega_complex[i, j] = wp.vec2f(omega[i, j], 0.0)


@wp.kernel
def fft_tiled(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Perform 1-D FFT on each row using ``wp.tile_fft``."""
    i, _, _ = wp.tid()
    row_tile = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_fft(row_tile)
    wp.tile_store(y, row_tile, offset=(i * TILE_M, 0))


@wp.kernel
def ifft_tiled(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Perform 1-D inverse FFT on each row using ``wp.tile_ifft``."""
    i, _, _ = wp.tid()
    row_tile = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_ifft(row_tile)
    wp.tile_store(y, row_tile, offset=(i * TILE_M, 0))


@wp.kernel
def tiled_transpose(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Transpose a 2-D complex array using tiled shared-memory loads."""
    i, j = wp.tid()
    input_tile = wp.tile_load(
        x,
        shape=(TILE_TRANSPOSE_DIM, TILE_TRANSPOSE_DIM),
        offset=(i * TILE_TRANSPOSE_DIM, j * TILE_TRANSPOSE_DIM),
        storage="shared",
    )
    output_tile = wp.tile_transpose(input_tile)
    wp.tile_store(y, output_tile, offset=(j * TILE_TRANSPOSE_DIM, i * TILE_TRANSPOSE_DIM))


@wp.kernel
def multiply_k2_inverse(
    inv_k_sq: wp.array2d(dtype=wp.float32), omega_hat: wp.array2d(dtype=wp.vec2f), psi_hat: wp.array2d(dtype=wp.vec2f)
):
    """Solve Poisson equation in Fourier space.

    Args:
        inv_k_sq: Precomputed 1/|k|^2 array.
        omega_hat: Fourier transform of vorticity.
        psi_hat: Output Fourier transform of stream function.
    """
    i, j = wp.tid()
    psi_hat[i, j] = omega_hat[i, j] * inv_k_sq[i, j]


@wp.kernel
def extract_real_and_normalize(
    divisor: wp.float32,
    complex_array: wp.array2d(dtype=wp.vec2f),
    real_array: wp.array2d(dtype=wp.float32),
):
    """Extract and normalize the real part of a complex array."""
    i, j = wp.tid()
    real_array[i, j] = complex_array[i, j].x / divisor


class Example:
    """Implement 2-D flow in a periodic box using vorticity-streamfunction formulation.

    Simulation parameters (``N_GRID``, ``RE``, ``DT``, etc.) are defined as
    module-level constants and captured directly by Warp kernels.
    """

    def __init__(self) -> None:
        # SSP-RK3 coefficients
        self.rk3_coeffs = [
            [1.0, 0.0, 1.0],
            [3.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0],
            [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
        ]

        # use CUDA graph if GPU is available
        self.use_cuda_graph = wp.get_device().is_cuda
        self.graph = None
        self.num_steps = None
        self.sim_substeps = 1

        # initialize fields
        self._init_fields()

    def _init_fields(self) -> None:
        """Initialize all the required variables for the simulation."""
        # allocate warp arrays for vorticity and stream-function
        self.omega_start = wp.zeros((N_GRID, N_GRID), dtype=wp.float32)
        self.omega_curr = wp.zeros((N_GRID, N_GRID), dtype=wp.float32)
        self.omega_next = wp.zeros((N_GRID, N_GRID), dtype=wp.float32)
        self.psi = wp.zeros((N_GRID, N_GRID), dtype=wp.float32)

        # precompute 1/k^2 for spectral Poisson solver (avoid division by zero at k=0)
        k = np.fft.fftfreq(N_GRID, d=1.0 / N_GRID)
        kx, ky = np.meshgrid(k, k)
        k2 = kx**2 + ky**2
        inv_k_sq = np.zeros_like(k2)
        nonzero = k2 != 0
        inv_k_sq[nonzero] = 1.0 / k2[nonzero]
        self.inv_k_sq = wp.array2d(inv_k_sq.astype(np.float32), dtype=wp.float32)

        # allocate temporary warp arrays for spectral Poisson solver
        self.omega_complex = wp.zeros((N_GRID, N_GRID), dtype=wp.vec2f)
        self.fft_temp_1 = wp.zeros((N_GRID, N_GRID), dtype=wp.vec2f)
        self.fft_temp_2 = wp.zeros((N_GRID, N_GRID), dtype=wp.vec2f)

        # compute initial vorticity distribution for decaying turbulence
        k_mag_np = np.sqrt(k**2 + k[:, np.newaxis] ** 2)
        k_mag = wp.array2d(k_mag_np, dtype=wp.float32)

        rng = np.random.default_rng(42)
        zeta_np = 2 * np.pi * rng.random((N_GRID // 2 + 1, N_GRID // 2 + 1))
        eta_np = 2 * np.pi * rng.random((N_GRID // 2 + 1, N_GRID // 2 + 1))
        zeta = wp.array2d(zeta_np, dtype=wp.float32)
        eta = wp.array2d(eta_np, dtype=wp.float32)

        # set parameters for energy spectrum
        k_peak = 12.0
        s = 3

        wp.launch(
            decaying_turbulence_initializer,
            dim=(N_GRID, N_GRID),
            inputs=[k_peak, s, k_mag, zeta, eta],
            outputs=[self.omega_complex],
        )

        # compute 2-D IFFT of omega_complex
        self._fft_2d(ifft_tiled, self.omega_complex, self.fft_temp_1)

        # extract and scale real part to get initial vorticity field
        # note that we do not scale the field by N_GRID^2 here, N_GRID^2 is needed
        # only when we do the full roundtrip of field -> FFT -> IFFT -> field
        wp.launch(
            extract_real_and_normalize,
            dim=(N_GRID, N_GRID),
            inputs=[1.0, self.fft_temp_1],
            outputs=[self.omega_start],
        )

        # copy initial vorticity into omega_curr (must be a data copy, not a reference alias)
        wp.copy(self.omega_curr, self.omega_start)

        # solve initial Poisson equation to get psi from initial vorticity field
        self._solve_poisson()

    def _fft_2d(self, kernel, src, dst):
        """2-D FFT/IFFT via row-wise 1-D transform, transpose, row-wise 1-D transform.

        A 2-D FFT separates into 1-D transforms along each axis. Rather than
        implementing a column kernel, we transpose so that columns become rows,
        then reuse the same row-wise kernel for the second axis.
        """
        wp.launch_tiled(kernel, dim=[N_GRID, 1], inputs=[src], outputs=[self.fft_temp_1], block_dim=BLOCK_DIM)
        wp.launch_tiled(
            tiled_transpose,
            dim=(N_GRID // TILE_TRANSPOSE_DIM, N_GRID // TILE_TRANSPOSE_DIM),
            inputs=[self.fft_temp_1],
            outputs=[self.fft_temp_2],
            block_dim=TILE_TRANSPOSE_DIM * TILE_TRANSPOSE_DIM,
        )
        wp.launch_tiled(kernel, dim=[N_GRID, 1], inputs=[self.fft_temp_2], outputs=[dst], block_dim=BLOCK_DIM)

    def _solve_poisson(self) -> None:
        """Solve the Poisson equation: psi = IFFT( FFT(omega_curr) / |k|^2 )."""
        wp.launch(copy_float_to_vec2, dim=(N_GRID, N_GRID), inputs=[self.omega_curr], outputs=[self.omega_complex])
        self._fft_2d(fft_tiled, self.omega_complex, self.fft_temp_1)
        wp.launch(
            multiply_k2_inverse,
            dim=(N_GRID, N_GRID),
            inputs=[self.inv_k_sq, self.fft_temp_1],
            outputs=[self.fft_temp_2],
        )
        self._fft_2d(ifft_tiled, self.fft_temp_2, self.fft_temp_1)
        # Warp's tile_ifft uses the unnormalized convention, so divide by N^2
        wp.launch(
            extract_real_and_normalize,
            dim=(N_GRID, N_GRID),
            inputs=[N_GRID * N_GRID, self.fft_temp_1],
            outputs=[self.psi],
        )

    def step(self) -> None:
        """Advance simulation by one timestep using SSP-RK3."""
        for stage_coeff in self.rk3_coeffs:
            c0, c1, c2 = stage_coeff
            wp.launch(
                rk3_update,
                dim=(N_GRID, N_GRID),
                inputs=[c0, c1, c2, self.omega_start, self.omega_curr, self.psi],
                outputs=[self.omega_next],
            )

            # swap so omega_curr holds the updated field for the Poisson solve
            self.omega_curr, self.omega_next = self.omega_next, self.omega_curr

            # update streamfunction from new vorticity (in omega_curr)
            self._solve_poisson()

        # copy omega_curr to omega_start for next timestep
        wp.copy(self.omega_start, self.omega_curr)

    def step_and_render_frame(self, frame_num: int, img=None) -> tuple:
        """Advance simulation by ``sim_substeps`` timesteps and update the matplotlib image.

        Args:
            frame_num: Current frame number (required by FuncAnimation).
            img: Matplotlib image object to update.

        Returns:
            Tuple containing the updated image for blitting.
        """

        for _ in range(self.sim_substeps):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.step()

        sim_step = (frame_num + 1) * self.sim_substeps

        if img is not None:
            img.set_array(self.omega_curr.numpy().T)
            img.axes.set_title(f"2-D Incompressible Turbulence in a Periodic Box  —  step {sim_step}")

        if self.num_steps is not None:
            print(f"\rStep {sim_step} / {self.num_steps}", end="")
        else:
            print(f"\rStep {sim_step}", end="")

        return (img,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="2-D Incompressible Turbulence in a Periodic Box Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the default Warp device.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Total number of simulation time-steps. "
        "Defaults to unlimited in interactive mode and 10000 in headless mode.",
    )
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=100,
        help="Number of simulation time-steps per rendered frame.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example()
        example.sim_substeps = args.sim_substeps

        if example.use_cuda_graph:
            # capture first step in a CUDA graph
            with wp.ScopedCapture() as capture:
                example.step()
            example.graph = capture.graph

        if args.headless:
            num_steps = args.num_steps if args.num_steps is not None else 10000
            example.num_steps = num_steps
            num_frames = num_steps // args.sim_substeps
            for i in range(num_frames):
                example.step_and_render_frame(frame_num=i)
            print()
        else:
            import matplotlib.animation as anim
            import matplotlib.pyplot as plt

            example.num_steps = args.num_steps  # None means unlimited
            num_frames = args.num_steps // args.sim_substeps if args.num_steps is not None else None

            fig, ax = plt.subplots()
            ax.set_title("2-D Incompressible Turbulence in a Periodic Box  —  step 0")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            img = ax.imshow(
                example.omega_curr.numpy().T,
                origin="lower",
                cmap="twilight",
                animated=True,
                interpolation="antialiased",
                vmin=-15,
                vmax=15,
                extent=[0, DOMAIN_SIZE, 0, DOMAIN_SIZE],
            )
            fig.colorbar(img, ax=ax, label="ω (vorticity)")

            seq = anim.FuncAnimation(
                fig,
                example.step_and_render_frame,
                fargs=(img,),
                frames=num_frames,
                blit=False,
                interval=1,
                repeat=False,
                cache_frame_data=False,
            )

            plt.show()
