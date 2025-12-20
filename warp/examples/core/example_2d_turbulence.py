# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Example 2-D Fluid Flow in a Periodic Box

# Shows how to build a 2-D flow solver in a periodic box using the streamfuntion-
# vorticity formulation. We solve pressure Poission equation in Fourier space
# using tile-based FFT in Warp. For time-stepping, we use strong-stability preserving
# RK3 scheme.

######################################################################################

import math

import numpy as np

import warp as wp

# Box length and resolution of the flow domain
N_GRID = 512
LEN = 2 * np.pi
DT = 0.001
RE = 500.0

# Parameters for Warp's tiled-FFT functionality
TILE_M = 1
TILE_N = N_GRID
BLOCK_DIM = TILE_N // 2


# -----------------------------------------------------------------------------
# Functions to define initial conditions for the solver. Taylor-Green vortex
# and decaying turbulence initial conditions are available.
# -----------------------------------------------------------------------------
def random_phase_field(k):
    """From https://github.com/marinlauber/2D-Turbulence-Python under MIT license

    Given an array of wavenumber, generates a random phase field in the 2-D domain

    Args:
        k: Wavenumber array of length N_GRID

    Returns:
        2D random phase field on a (N_GRID, N_GRID)

    """
    N = len(k)
    N_half = N // 2

    # define phase array
    xi = np.zeros((N, N))

    # compute phase field in k space, need more points because of how
    # python organises wavenumbers
    rng = np.random.default_rng()
    zeta = 2 * np.pi * rng.random((N_half + 1, N_half + 1))
    eta = 2 * np.pi * rng.random((N_half + 1, N_half + 1))

    # quadrant \xi(kx,ky) = \zeta(kx,ky) + \eta(kx,ky)
    xi[:N_half, :N_half] = zeta[:-1, :-1] + eta[:-1, :-1]

    # quadrant \xi(-kx,ky) = -\zeta(kx,ky) + \eta(kx,ky)
    xi[N_half:, :N_half] = np.flip(-zeta[1:, :-1] + eta[1:, :-1], 0)

    # quadrant \xi(-kx,-ky) = -\zeta(kx,ky) - \eta(kx,ky)
    xi[N_half:, N_half:] = np.flip(-zeta[1:, 1:] - eta[1:, 1:])

    # quadrant \xi(kx,-ky) = \zeta(kx,ky) - \eta(kx,ky)
    xi[:N_half, N_half:] = np.flip(zeta[:-1, 1:] - eta[:-1, 1:], 1)

    return np.exp(1j * xi)


def energy_spectrum(k, s=3, kp=12):
    """From https://github.com/marinlauber/2D-Turbulence-Python under MIT license

    Returns energy contained in a wavenumber array k based on a pre-defined energy spectrum

    Args:
        k: Wavenumber vector of size N_GRID
        s, kp: Energy spectrum parameters

    Returns:
        E: Energy on N_GRID wavenumber vectors given by k

    """

    # normalise the spectrum
    a_s = (2 * s + 1) ** (s + 1) / (2**s * math.factorial(s))

    # compute sectrum at this wave number
    E = a_s / (2 * kp) * (k / kp) ** (2 * s + 1) * np.exp(-(s + 0.5) * (k / kp) ** 2)

    return E


def decaying_turbulence(coords):
    """Generate initial vorticity field for decaying turbulence simulation.

    Taken from https://github.com/marinlauber/2D-Turbulence-Python
    Also check "High-order methods for decaying two-dimensional homogeneous isotropic turbulence"

    Creates a random initial vorticity field
    Args:
        coords: 1D array of grid coordinates of size N_GRID

    Returns:
        2D array of initial vorticity values on (N_GRID, N_GRID)

    """
    N = len(coords)
    k = np.fft.fftfreq(N, d=1.0 / N)

    # wavenumber magnitude on 2D grid
    k_mag = np.sqrt(k**2 + k[:, np.newaxis] ** 2)
    w_hat = np.sqrt((k_mag / np.pi) * energy_spectrum(k_mag))

    # randomize the vorticity field while preserving energy at any given wavenumber
    w_hat = w_hat * random_phase_field(k)

    # obtain vorticity field in real space from Fourier space
    w = np.fft.ifft2(w_hat) * N * N

    return np.real(w).astype(np.float32)


def taylor_green(coords, Re, kappa=6.0, t=0.0):
    """Taylor-Green vortex initial condition

    Useful for validation as it has a known analytical solution in 2-D.

    Args:
        coords: 1D array of grid coordinates of size N_GRID
        Re: Reynolds number
        kappa: Wavenumber (default 6.0) of the flow field
        t: Time (default 0.0)
    Returns:
        2D array of vorticity values on (N_GRID, N_GRID)

    """
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    omega = 2 * kappa * np.sin(kappa * X) * np.sin(kappa * Y) * np.exp(-2 * kappa**2 * t / Re)

    return omega.astype(np.float32)


# -----------------------------------------------------------------------------
# Warp helper functions for periodicity imposition, calculating advection components,
# and calculating diffusion components.
# -----------------------------------------------------------------------------
@wp.func
def cyclic_index(idx: wp.int32, N: wp.int32):
    """Map an index to [0, N-1] for periodic boundary conditions.

    Args:
        idx: Input index that may be outside the valid range
        N: Grid size defining the periodic domain

    Returns:
        Index wrapped to the range [0, N-1]
    """
    ret_idx = idx % N
    if ret_idx < 0:
        ret_idx += N
    return ret_idx


@wp.func
def advection(
    omega_left: wp.float32,
    omega_right: wp.float32,
    omega_top: wp.float32,
    omega_down: wp.float32,
    psi_left: wp.float32,
    psi_right: wp.float32,
    psi_top: wp.float32,
    psi_down: wp.float32,
    h: wp.float32,
):
    """Calculate the advection term using central finite differences

    Args:
        omega_left: Vorticity at (i-1, j)
        omega_right: Vorticity at (i+1, j)
        omega_top: Vorticity at (i, j+1)
        omega_down: Vorticity at (i, j-1)
        psi_left: Stream function at (i-1, j)
        psi_right: Stream function at (i+1, j)
        psi_top: Stream function at (i, j+1)
        psi_down: Stream function at (i, j-1)
        h: Grid spacing

    Returns:
        Advection term value at grid point (i, j)
    """
    inv_2h = 1.0 / (2.0 * h)
    term_1 = ((omega_right - omega_left) * inv_2h) * ((psi_top - psi_down) * inv_2h)
    term_2 = ((omega_top - omega_down) * inv_2h) * ((psi_right - psi_left) * inv_2h)
    return term_2 - term_1


@wp.func
def diffusion(
    omega_left: wp.float32,
    omega_right: wp.float32,
    omega_center: wp.float32,
    omega_down: wp.float32,
    omega_top: wp.float32,
    h: wp.float32,
):
    """Calculate the Laplacian for viscous diffusion using central differences

    Args:
        omega_left: Vorticity at (i-1, j)
        omega_right: Vorticity at (i+1, j)
        omega_center: Vorticity at (i, j)
        omega_down: Vorticity at (i, j-1)
        omega_top: Vorticity at (i, j+1)
        h: Grid spacing

    Returns:
        Laplacian of vorticity at grid point (i, j)
    """
    inv_h2 = 1.0 / (h * h)
    # combines both the diffusion terms in the x and y direction together
    laplacian = (omega_right + omega_left + omega_top + omega_down - 4.0 * omega_center) * inv_h2
    return laplacian


# -----------------------------------------------------------------------------
# Warp kernels for SSP-RK3 timestepping
# -----------------------------------------------------------------------------
@wp.kernel
def viscous_advection_rk3_kernel(
    N: int,
    h: float,
    Re: float,
    dt: float,
    coeff0: float,
    coeff1: float,
    coeff2: float,
    omega_0: wp.array2d(dtype=float),
    omega_1: wp.array2d(dtype=float),
    psi: wp.array2d(dtype=float),
    rhs: wp.array2d(dtype=float),
):
    """Computes RHS viscous diffusion and advection terms, then performs a single substep of SSP-RK3

    Args:
        N: Grid size
        h: Grid spacing
        Re: Reynolds number
        dt: Time step size
        coeff0, coeff1, coeff2: SSP-RK3 coefficients
        omega_0: Vorticity field at the beginning of the time step
        omega_1: Updated vorticity field at the end of the time step
        psi: Stream function field
        rhs: Temporarily stores diffusion + advection terms
    """

    i, j = wp.tid()

    # Obtain the neighboring indices for the [i, j]th cell
    left_idx = cyclic_index(i - 1, N)
    right_idx = cyclic_index(i + 1, N)
    top_idx = cyclic_index(j + 1, N)
    bottom_idx = cyclic_index(j - 1, N)

    # Viscous diffusion term
    rhs[i, j] = (1.0 / Re) * diffusion(
        omega_1[left_idx, j], omega_1[right_idx, j], omega_1[i, j], omega_1[i, bottom_idx], omega_1[i, top_idx], h
    )

    # Advection term
    rhs[i, j] += advection(
        omega_1[left_idx, j],
        omega_1[right_idx, j],
        omega_1[i, top_idx],
        omega_1[i, bottom_idx],
        psi[left_idx, j],
        psi[right_idx, j],
        psi[i, top_idx],
        psi[i, bottom_idx],
        h,
    )

    # Singular RK update
    omega_1[i, j] = coeff0 * omega_0[i, j] + coeff1 * omega_1[i, j] + coeff2 * dt * rhs[i, j]


# -----------------------------------------------------------------------------
# Helper kernels for pressure Poisson solver in the spectral space
# -----------------------------------------------------------------------------
@wp.kernel
def copy_float_to_vec2(
    omega: wp.array2d(dtype=wp.float32),
    omega_complex: wp.array2d(dtype=wp.vec2f),
):
    """Copy real vorticity to complex array with zero imaginary part

    Args:
        omega: Input real-valued vorticity array
        omega_complex: Output complex array where real part is omega, imaginary is 0
    """
    i, j = wp.tid()
    omega_complex[i, j] = wp.vec2f(omega[i, j], 0.0)


@wp.kernel
def fft_tiled(
    x: wp.array2d(dtype=wp.vec2f),
    y: wp.array2d(dtype=wp.vec2f),
):
    """Perform 1D FFT on each row using wp.tile_fft()

    Args:
        x: Input complex array of shape (N, N)
        y: Output complex array of shape (N, N) storing FFT results
    """
    i, _, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_fft(a)
    wp.tile_store(y, a, offset=(i * TILE_M, 0))


@wp.kernel
def ifft_tiled(
    x: wp.array2d(dtype=wp.vec2f),
    y: wp.array2d(dtype=wp.vec2f),
):
    """Perform 1D inverse FFT on each row using wp.tile_ifft()

    Args:
        x: Input complex array of shape (N, N)
        y: Output complex array of shape (N, N) storing IFFT results
    """
    i, _, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_ifft(a)
    wp.tile_store(y, a, offset=(i * TILE_M, 0))


@wp.kernel
def transpose(
    x: wp.array2d(dtype=wp.vec2f),
    y: wp.array2d(dtype=wp.vec2f),
):
    """Transpose a 2D array

    Args:
        x: Input complex array of shape (N, N)
        y: Output complex array storing the transpose of x
    """
    i, j = wp.tid()
    y[j, i] = x[i, j]


@wp.kernel
def multiply_k2_inverse(
    K2I: wp.array2d(dtype=wp.float32),
    omega_hat: wp.array2d(dtype=wp.vec2f),
    psi_hat: wp.array2d(dtype=wp.vec2f),
):
    """Solve Poisson equation in Fourier space

    Args:
        K2I: Precomputed 1/|k|^2 array
        omega_hat: Fourier transform of vorticity
        psi_hat: Output Fourier transform of stream function
    """
    i, j = wp.tid()
    psi_hat[i, j] = wp.vec2f(omega_hat[i, j].x * K2I[i, j], omega_hat[i, j].y * K2I[i, j])


@wp.kernel
def extract_real_and_normalize(
    N: int,
    complex_array: wp.array2d(dtype=wp.vec2f),
    real_array: wp.array2d(dtype=wp.float32),
):
    """Extract real part and normalize after inverse 2D FFT

    Args:
        N: Grid size for normalization factor (divides by N^2)
        complex_array: Input complex array from inverse FFT
        real_array: Output real array with normalized values
    """
    i, j = wp.tid()
    real_array[i, j] = complex_array[i, j].x / wp.float32(N * N)


class Example:
    """2D flow in a periodic box using vorticity-streamfunction formulation"""

    def __init__(
        self,
        N=N_GRID,
        Re=RE,
        dt=DT,
        L=LEN,
        initial_condition="decaying",
    ):
        """Initialize the 2D turbulence simulation

        Args:
            N: Square grid resolution (must match TILE_N for FFT)
            Re: Reynolds number
            dt: Time step size
            L: Physical domain size (default 2*pi for periodic)
            initial_condition: "decaying" or "taylor_green"
        """
        self.N = N
        self.L = L
        self.h = self.L / self.N
        self.Re = Re
        self.dt = dt

        # SSP-RK3 coefficients
        self._rk3_coeffs = [
            [1.0, 0.0, 0.0],
            [3.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0],
            [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
        ]

        # Initialize fields
        self._init_fields(initial_condition)

        # Store initial vorticity range for consistent colormap
        omega_init = self.omega_1.numpy()
        self._vmax = np.max(np.abs(omega_init)) * 1.2

    def _init_fields(self, initial_condition):
        """Initializes all the required variables for the simulation"""

        # Grid coordinates (same for x and y on square grid)
        coords = np.linspace(0, self.L, self.N, endpoint=False)

        # Initial vorticity field according to initial condition
        if initial_condition == "taylor_green":
            omega_np = taylor_green(coords, self.Re)
        else:
            omega_np = decaying_turbulence(coords)

        # Warp arrays (including buffers) for vorticity, streamfunction
        self.omega_0 = wp.array2d(omega_np, dtype=wp.float32)
        self.omega_1 = wp.array2d(omega_np, dtype=wp.float32)
        self.psi = wp.zeros((self.N, self.N), dtype=wp.float32)

        # Warp array for RHS of NS equation
        self.rhs = wp.zeros((self.N, self.N), dtype=wp.float32)

        # Warp arrays for spectral Poisson solver
        self.omega_complex = wp.zeros((self.N, self.N), dtype=wp.vec2f)
        self.fft_temp_in = wp.zeros((self.N, self.N), dtype=wp.vec2f)
        self.fft_temp_out = wp.zeros((self.N, self.N), dtype=wp.vec2f)

        # Precompute 1/k^2 for spectral Poisson solver (avoiding division by zero at k=0)
        k = np.fft.fftfreq(self.N, d=1.0 / self.N)
        KX, KY = np.meshgrid(k, k)
        K2 = KX**2 + KY**2
        K2I = np.zeros_like(K2)
        nonzero = K2 != 0
        K2I[nonzero] = 1.0 / K2[nonzero]
        self.K2I = wp.array2d(K2I.astype(np.float32), dtype=wp.float32)

        # Solve initial Poisson equation to get psi from initial vorticity field
        self._solve_poisson()

    def _fft_2d(self, fft_kernel, input_arr, output_arr):
        """Perform 2D FFT or IFFT using row-wise transform + transpose pattern

        Args:
            fft_kernel: Either fft_tiled or ifft_tiled
            input_arr: Input complex array
            output_arr: Output complex array
        """
        # Rowise FFT
        wp.launch_tiled(
            fft_kernel,
            dim=[self.N, 1],
            inputs=[input_arr, self.fft_temp_in],
            block_dim=BLOCK_DIM,
        )

        # Transpose
        wp.launch(
            transpose,
            dim=(self.N, self.N),
            inputs=[self.fft_temp_in],
            outputs=[self.fft_temp_out],
        )

        # Columnwise FFT
        wp.launch_tiled(
            fft_kernel,
            dim=[self.N, 1],
            inputs=[self.fft_temp_out, output_arr],
            block_dim=BLOCK_DIM,
        )

    def _solve_poisson(self):
        """Solve Laplacian(psi) = -omega using spectral method.
        Uses 2D FFT: psi_hat = omega_hat / k^2, then inverse FFT.
        The 2D FFT is computed as 1D FFT along rows, transpose, 1D FFT along rows.
        """

        # Convert vorticity from wp.float32 to wp.vec2f for FFT
        wp.launch(
            copy_float_to_vec2,
            dim=(self.N, self.N),
            inputs=[self.omega_0],
            outputs=[self.omega_complex],
        )

        # Forward FFT
        self._fft_2d(fft_tiled, self.omega_complex, self.fft_temp_in)

        # Multiply by 1/k^2 (solve Poisson in Fourier space)
        wp.launch(
            multiply_k2_inverse,
            dim=(self.N, self.N),
            inputs=[self.K2I, self.fft_temp_in],
            outputs=[self.fft_temp_out],
        )

        # Inverse FFT
        self._fft_2d(ifft_tiled, self.fft_temp_out, self.fft_temp_in)

        # Extract real part and normalize
        wp.launch(
            extract_real_and_normalize,
            dim=(self.N, self.N),
            inputs=[self.N, self.fft_temp_in],
            outputs=[self.psi],
        )

    def step(self):
        """Advance simulation by one time step using SSP-RK3"""
        # Three-stage SSP-RK3
        for stage in range(3):
            c0, c1, c2 = self._rk3_coeffs[stage]

            # Zero the RHS array
            self.rhs.zero_()

            # Compute RHS and update omega_1
            wp.launch(
                viscous_advection_rk3_kernel,
                dim=(self.N, self.N),
                inputs=[
                    self.N,
                    self.h,
                    self.Re,
                    self.dt,
                    c0,
                    c1,
                    c2,
                    self.omega_0,
                    self.omega_1,
                    self.psi,
                    self.rhs,
                ],
            )

            # Update streamfunction from new vorticity
            # (swap omega_0 and omega_1 references for Poisson solve)
            wp.copy(self.omega_0, self.omega_1)
            self._solve_poisson()

    def step_and_render_frame(self, frame_num=None, img=None):
        self.step()

        if img:
            img.set_array(self.omega_1.numpy().T)

        return (img,)


# -----------------------------------------------------------------------------
# Command Line Interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="2D Turbulence Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=10000, help="Total number of frames.")
    parser.add_argument("--initial_condition", choices=("decaying", "taylor_green"), default="decaying")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(initial_condition=args.initial_condition)

        if args.headless:
            for _ in range(args.num_frames):
                example.step()
        else:
            import matplotlib
            import matplotlib.animation as anim
            import matplotlib.pyplot as plt

            fig = plt.figure()

            img = plt.imshow(
                example.omega_1.numpy().T,
                origin="lower",
                cmap="RdBu_r",
                animated=True,
                interpolation="antialiased",
            )
            img.set_norm(matplotlib.colors.Normalize(-example._vmax, example._vmax))

            seq = anim.FuncAnimation(
                fig,
                example.step_and_render_frame,
                fargs=(img,),
                frames=args.num_frames,
                blit=True,
                interval=8,
                repeat=False,
            )

            plt.show()
