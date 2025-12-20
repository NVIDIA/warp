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

wp.set_module_options({"enable_backward": False})

# Length and resolution of the flow domain
N_GRID = 512
L = 2 * np.pi

# FFT compile-time tile constants
TILE_M = 1
TILE_N = N_GRID
BLOCK_DIM = TILE_N // 2


# -----------------------------------------------------------------------------
# Functions to define initial conditions for the solver. Taylor-Green vortex
# and decaying turbulence initial conditions are implemented.
# -----------------------------------------------------------------------------
def random_phase_field(k):
    """From https://github.com/marinlauber/2D-Turbulence-Python under MIT license

    Given an array of wavenumber, generates a random phase field in the 2-D domain

    Args:
        k: wavenumber array of length N_GRID

    Returns:
        2D random phase field on a (N_GRID, N_GRID)

    """

    N = len(k)
    N_half = N // 2

    # define phase array
    xi = np.zeros((N, N))

    # compute phase field in k space, need more points because of how
    # python organises wavenumbers
    zeta = 2 * np.pi * np.random.rand(N_half + 1, N_half + 1)
    eta = 2 * np.pi * np.random.rand(N_half + 1, N_half + 1)

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
        k: wavenumber vector of size N_GRID
        s, kp: energy spectrum parameters

    Returns:
        E: energy on N_GRID wavenumber vectors given by k

    """

    # normalise the spectrum
    a_s = (2 * s + 1) ** (s + 1) / (2**s * math.factorial(s))

    # compute sectrum at this wave number
    E = a_s / (2 * kp) * (k / kp) ** (2 * s + 1) * np.exp(-(s + 0.5) * (k / kp) ** 2)

    return E


def decaying_turbulence(coords):
    """Generate initial vorticity field for decaying turbulence simulation.

    Taken from https://github.com/marinlauber/2D-Turbulence-Python
    Also check San & Staples: "High-order methods for decaying two-dimensional homogeneous isotropic turbulence"

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
    """Taylor-Green vortex initial condition.

    Useful for validation as it has a known analytical solution in 2-D.

    Args:
        coords: 1D array of grid coordinates of size N_GRID
        Re: Reynolds number
        kappa: wavenumber (default 6.0) of the flow field

    Returns:
        2D array of vorticity values on (N_GRID, N_GRID)

    """

    X, Y = np.meshgrid(coords, coords, indexing="ij")
    omega = 2 * kappa * np.sin(kappa * X) * np.sin(kappa * Y) * np.exp(-2 * kappa**2 * 0 / Re)

    return omega.astype(np.float32)


# -----------------------------------------------------------------------------
# Warp helper functions for periodicity imposition, calculating advection components,
# and calculating diffusion components.
# -----------------------------------------------------------------------------
@wp.func
def cyclic_index(idx: wp.int32, N: wp.int32):
    """Maps an index to [0, N_GRID-1] for periodic boundary condition in x and y directions."""
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
    """Calculates the advection term in the RHS of 2-D N-S equations using
    central finite difference scheme df/dx ~ (f[i+1] - f[i-1]) / (2 * h)
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
    """Calculates the advection term in the RHS of 2-D N-S equations using
    central finite difference scheme d2f/dx2 ~ (f[i+1] - 2 * f[i] + f[i-1]) / (h * h)
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
    """Computes RHS viscous diffusion and advection terms, then performs SSP-RK3 update.

    Calculates RHS = (1/Re)*∇²ω + (∂ψ/∂y)(∂ω/∂x) - (∂ψ/∂x)(∂ω/∂y) using central differences,
    then perform one RK update of the vorticity: ω₁ = coeff0*ω₀ + coeff1*ω₁ + coeff2*dt*RHS.

    Args:
        N: Grid size (number of points in each direction)
        h: Grid spacing (uniform in x and y)
        Re: Reynolds number for viscous diffusion scaling
        dt: Time step size
        coeff0, coeff1, coeff2: RK coefficients
        omega_0: Vorticity field at the beginning of a timestep
        omega_1: Updated vorticity field at the end of RK stage
        psi: Stream function field
        rhs: Temporarily stores diffusion + advection terms

    """

    i, j = wp.tid()

    # Obtain the neighboring indices for [i, j]th cell
    left_idx = cyclic_index(i - 1, N)
    right_idx = cyclic_index(i + 1, N)
    top_idx = cyclic_index(j + 1, N)
    bottom_idx = cyclic_index(j - 1, N)

    # Viscous diffusion term
    rhs[i, j] = (1.0 / Re) * diffusion(
        omega_1[left_idx, j],
        omega_1[right_idx, j],
        omega_1[i, j],
        omega_1[i, bottom_idx],
        omega_1[i, top_idx],
        h,
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
    """Copy real vorticity to complex array (real part only)."""
    i, j = wp.tid()
    omega_complex[i, j] = wp.vec2f(omega[i, j], 0.0)


@wp.kernel
def fft_tiled(
    x: wp.array2d(dtype=wp.vec2f),
    y: wp.array2d(dtype=wp.vec2f),
):
    """TBD"""
    i, j, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_fft(a)
    wp.tile_store(y, a, offset=(i * TILE_M, 0))


@wp.kernel
def ifft_tiled(
    x: wp.array2d(dtype=wp.vec2f),
    y: wp.array2d(dtype=wp.vec2f),
):
    """TBD"""
    i, j, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_ifft(a)
    wp.tile_store(y, a, offset=(i * TILE_M, 0))


@wp.kernel
def transpose(
    x: wp.array2d(dtype=wp.vec2f),
    y: wp.array2d(dtype=wp.vec2f),
):
    """TBD"""
    i, j = wp.tid()
    y[j, i] = x[i, j]


@wp.kernel
def multiply_k2_inverse(
    K2I: wp.array2d(dtype=wp.float32),
    omega_hat: wp.array2d(dtype=wp.vec2f),
    psi_hat: wp.array2d(dtype=wp.vec2f),
):
    """Multiply by -1/k^2 to solve Poisson equation in Fourier space.

    Solves: Laplacian(psi) = -omega  =>  psi_hat = omega_hat / k^2
    """
    i, j = wp.tid()
    psi_hat[i, j] = wp.vec2f(omega_hat[i, j].x * K2I[i, j], omega_hat[i, j].y * K2I[i, j])


@wp.kernel
def extract_real_and_normalize(
    N: int,
    complex_array: wp.array2d(dtype=wp.vec2f),
    real_array: wp.array2d(dtype=wp.float32),
):
    """Extract real part and normalize after inverse 2D FFT."""
    i, j = wp.tid()
    real_array[i, j] = complex_array[i, j].x / wp.float32(N * N)


# -----------------------------------------------------------------------------
# Example Class
# -----------------------------------------------------------------------------
class Example:
    """2D Turbulence simulation using vorticity-streamfunction formulation.

    This example demonstrates:
    - Spectral Poisson solver using tiled FFT
    - SSP-RK3 time integration
    - Periodic boundary conditions on a square (N x N) grid
    - Visualization of vorticity field evolution
    """

    def __init__(
        self,
        quiet=False,
        N=512,
        Re=500.0,
        dt=0.001,
        L=2.0 * np.pi,
        initial_condition="decaying",
        seed=42,
        save_every=500,
        output_dir="output",
    ):
        """Initialize the 2D turbulence simulation.

        Args:
            quiet: If True, suppress iteration output
            N: Square grid resolution (must match TILE_N for FFT)
            Re: Reynolds number
            dt: Time step size
            L: Physical domain size (default 2*pi for periodic)
            initial_condition: "decaying" or "taylor_green"
            seed: Random seed for reproducible initial conditions
            save_every: Save data every N steps (0 to disable)
            output_dir: Directory for output files
        """
        self._quiet = quiet
        self.save_every = save_every
        self.output_dir = output_dir
        self.initial_condition_name = initial_condition

        # Data collection for validation
        self._frame_data_omega = []
        self._frame_data_psi = []
        self._times = []

        # Validate resolution matches FFT tile size
        if N != TILE_N:
            raise ValueError(f"Resolution must be {TILE_N} to match FFT tile size")

        # Grid parameters (square grid)
        self.N = N
        self.L = L
        self.h = self.L / self.N

        # Physical parameters
        self.Re = Re
        self.dt = dt

        # Time tracking
        self.current_time = 0.0
        self.current_frame = 0

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
        self._vmax = np.max(np.abs(omega_init)) * 1.2  # slight padding

    def _init_fields(self, initial_condition):
        """TBD"""
        # Set random seed for reproducibility

        # Grid coordinates (same for x and y on square grid)
        coords = np.linspace(0, self.L, self.N, endpoint=False)

        # Initial vorticity field
        if initial_condition == "taylor_green":
            omega_np = taylor_green(coords, self.Re)
        else:
            omega_np = decaying_turbulence(coords)

        # Warp arrays for vorticity and streamfunction
        self.omega_0 = wp.array2d(omega_np, dtype=wp.float32)
        self.omega_1 = wp.array2d(omega_np, dtype=wp.float32)
        self.psi = wp.zeros((self.N, self.N), dtype=wp.float32)
        self.rhs = wp.zeros((self.N, self.N), dtype=wp.float32)

        # FFT workspace arrays (complex)
        self.omega_complex = wp.zeros((self.N, self.N), dtype=wp.vec2f)
        self.fft_temp_in = wp.zeros((self.N, self.N), dtype=wp.vec2f)
        self.fft_temp_out = wp.zeros((self.N, self.N), dtype=wp.vec2f)

        # Precompute 1/k^2 for Poisson solver (avoiding division by zero at k=0)
        k = np.fft.fftfreq(self.N, d=1.0 / self.N)
        KX, KY = np.meshgrid(k, k)
        K2 = KX**2 + KY**2
        K2I = np.zeros_like(K2)
        nonzero = K2 != 0
        K2I[nonzero] = 1.0 / K2[nonzero]
        self.K2I = wp.array2d(K2I.astype(np.float32), dtype=wp.float32)

        # Solve initial Poisson equation to get psi from omega
        self._solve_poisson()

    def _solve_poisson(self):
        """Solve Laplacian(psi) = -omega using spectral method.

        Uses 2D FFT: psi_hat = omega_hat / k^2, then inverse FFT.
        The 2D FFT is computed as 1D FFT along rows, transpose, 1D FFT along rows.
        """
        # Convert vorticity from wp.float32 to wp.vec2f for FFT
        wp.launch(
            copy_float_to_vec2,
            dim=(self.N, self.N),
            inputs=[self.omega_0, self.omega_complex],
        )

        # Forward FFT: rows
        wp.launch_tiled(
            fft_tiled,
            dim=[self.N, 1],
            inputs=[self.omega_complex, self.fft_temp_in],
            block_dim=BLOCK_DIM,
        )

        # Transpose
        wp.launch(
            transpose,
            dim=(self.N, self.N),
            inputs=[self.fft_temp_in, self.fft_temp_out],
        )

        # Forward FFT: columns (now rows after transpose)
        wp.launch_tiled(
            fft_tiled,
            dim=[self.N, 1],
            inputs=[self.fft_temp_out, self.fft_temp_in],
            block_dim=BLOCK_DIM,
        )

        # Multiply by 1/k^2 (solve Poisson in Fourier space)
        wp.launch(
            multiply_k2_inverse,
            dim=(self.N, self.N),
            inputs=[self.K2I, self.fft_temp_in, self.fft_temp_out],
        )

        # Inverse FFT: rows
        wp.launch_tiled(
            ifft_tiled,
            dim=[self.N, 1],
            inputs=[self.fft_temp_out, self.fft_temp_in],
            block_dim=BLOCK_DIM,
        )

        # Transpose back
        wp.launch(
            transpose,
            dim=(self.N, self.N),
            inputs=[self.fft_temp_in, self.fft_temp_out],
        )

        # Inverse FFT: columns (now rows after transpose)
        wp.launch_tiled(
            ifft_tiled,
            dim=[self.N, 1],
            inputs=[self.fft_temp_out, self.fft_temp_in],
            block_dim=BLOCK_DIM,
        )

        # Extract real part and normalize
        wp.launch(
            extract_real_and_normalize,
            dim=(self.N, self.N),
            inputs=[self.N, self.fft_temp_in, self.psi],
        )

    def step(self):
        """Advance simulation by one time step using SSP-RK3."""
        self.current_frame += 1

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

        self.current_time += self.dt

        if not self._quiet and (self.current_frame % 100 == 0 or self.current_frame < 10):
            stats = self.get_stats()
            print(f"Frame {self.current_frame}: omega_min={stats['min']:.3f}, omega_max={stats['max']:.3f}")

    def get_stats(self):
        """Return min/max vorticity statistics."""
        omega_np = self.omega_1.numpy()
        return {"min": float(np.min(omega_np)), "max": float(np.max(omega_np))}

    def should_save(self):
        """Check if current frame should be saved."""
        return self.save_every > 0 and self.current_frame % self.save_every == 0

    def collect_frame(self):
        """Collect current frame data for later saving."""
        self._frame_data_omega.append(self.omega_1.numpy().copy())
        self._frame_data_psi.append(self.psi.numpy().copy())
        self._times.append(self.current_time)

    def save_data(self):
        """Save collected simulation data to npz file."""
        import os

        if not self._frame_data_omega:
            print("No data to save!")
            return None

        os.makedirs(self.output_dir, exist_ok=True)

        re_str = f"{self.Re:g}"
        filename = f"{self.output_dir}/{self.initial_condition_name}_{self.N}_{re_str}.npz"

        coords = np.linspace(0, self.L, self.N, endpoint=False)

        np.savez_compressed(
            filename,
            vorticity=np.array(self._frame_data_omega),
            stream_function=np.array(self._frame_data_psi),
            times=np.array(self._times),
            x=coords,
            y=coords,
            config_dict={
                "N": self.N,
                "L": self.L,
                "h": self.h,
                "DT": self.dt,
                "Re": self.Re,
            },
        )

        print(f"Saved {len(self._frame_data_omega)} frames to: {filename}")
        return filename

    def step_and_render_frame(self, frame_num=None, img=None, title=None):
        """Advance simulation and update plot for FuncAnimation.

        Args:
            frame_num: Frame number (provided by FuncAnimation)
            img: Matplotlib imshow object to update
            title: Matplotlib title object to update

        Returns:
            Tuple of artists that were modified
        """
        self.step()

        if img is not None:
            img.set_array(self.omega_1.numpy().T)

        if title is not None:
            title.set_text(f"t = {self.current_time:.3f}")

        return (img, title) if title else (img,)


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
    parser.add_argument(
        "--N",
        type=int,
        default=512,
        help=f"Square grid resolution (must be {TILE_N} for current FFT tile size).",
    )
    parser.add_argument("--Re", type=float, default=500.0, help="Reynolds number.")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step size.")
    parser.add_argument("--num_frames", type=int, default=20000, help="Total number of simulation steps.")
    parser.add_argument(
        "--initial_condition",
        choices=("decaying", "taylor_green"),
        default="taylor_green",
        help="Initial vorticity distribution.",
    )
    parser.add_argument("--save_every", type=int, default=500, help="Save data every N steps (0 to disable).")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for npz files.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress iteration output.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible initial conditions.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet,
            N=args.N,
            Re=args.Re,
            dt=args.dt,
            initial_condition=args.initial_condition,
            seed=args.seed,
            save_every=args.save_every,
            output_dir=args.output_dir,
        )

        if args.headless:
            # Collect initial frame
            if example.save_every > 0:
                example.collect_frame()

            for _ in range(args.num_frames):
                example.step()
                if example.should_save():
                    example.collect_frame()

            print(f"Simulation complete: {args.num_frames} steps, final time = {example.current_time:.3f}")

            if example.save_every > 0:
                example.save_data()
        else:
            import matplotlib
            import matplotlib.animation as anim
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))

            img = ax.imshow(
                example.omega_1.numpy().T,
                origin="lower",
                cmap="RdBu_r",
                animated=True,
                interpolation="antialiased",
                extent=[0, example.L, 0, example.L],
            )
            img.set_norm(matplotlib.colors.TwoSlopeNorm(vmin=-example._vmax, vcenter=0, vmax=example._vmax))

            plt.colorbar(img, ax=ax, label="Vorticity")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            title = ax.set_title(f"2D Turbulence (Re={example.Re}), t = 0.000")

            seq = anim.FuncAnimation(
                fig,
                example.step_and_render_frame,
                fargs=(img, title),
                frames=args.num_frames,
                blit=True,
                interval=1,
                repeat=False,
            )

            plt.tight_layout()
            plt.show()
