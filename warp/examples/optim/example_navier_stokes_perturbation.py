# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Perturbation Optimizer for 2-D Decaying Turbulence
#
# Uses Warp's autodiff to find the small vorticity perturbation that
# causes the largest divergence from a reference trajectory after a
# fixed number of timesteps.
#
# The solver integrates the vorticity-streamfunction form of the 2-D
# incompressible Navier-Stokes equations with periodic boundaries,
# using a spectral (FFT-based) Poisson solver and SSP-RK3 timestepping.
#
# Differentiable counterpart of `example_fft_poisson_navier_stokes_2d.py`.
#
###########################################################################

import numpy as np

import warp as wp
import warp.optim

try:
    import matplotlib
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# simulation parameters
N_GRID = 512
DOMAIN_SIZE = 2 * np.pi
DT = 0.001
RE = 1000.0

# derived grid constants (captured by Warp kernels)
H = DOMAIN_SIZE / N_GRID
INV_H2 = 1.0 / (H * H)
INV_2H = 1.0 / (2.0 * H)

# Parameters for Warp's tiled-FFT functionality
TILE_TRANSPOSE_DIM = 16
BLOCK_DIM = N_GRID // 2


@wp.struct
class PoissonFFTBuffers:
    """Complex-valued scratch buffers for the FFT-based Poisson solver.

    The 2-D FFT is decomposed as: row-wise 1-D FFT → transpose → column-wise
    1-D FFT, using three scratch arrays for intermediate results.  A fourth
    array (``psi_hat``) stores the stream function in Fourier space.
    """

    omega_complex: wp.array2d[wp.vec2f]
    scratch_1: wp.array2d[wp.vec2f]
    scratch_2: wp.array2d[wp.vec2f]
    scratch_3: wp.array2d[wp.vec2f]
    psi_hat: wp.array2d[wp.vec2f]


@wp.func
def factorial(n: int) -> int:
    """Compute factorial of ``n``."""
    result = int(1)
    for i in range(2, n + 1):
        result *= i
    return result


@wp.func
def energy_spectrum(k: float, s: int, k_peak: float) -> float:
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
    s_factorial = float(factorial(s))
    s_float32 = float(s)
    a_s = (2.0 * s_float32 + 1.0) ** (s_float32 + 1.0) / (2.0**s_float32 * s_factorial)
    energy_k = (
        a_s
        / (2.0 * k_peak)
        * (k / k_peak) ** (2.0 * s_float32 + 1.0)
        * wp.exp(-(s_float32 + 0.5) * (k / k_peak) ** 2.0)
    )
    return energy_k


@wp.func
def phase_randomizer(zeta: wp.array2d[float], eta: wp.array2d[float], i: int, j: int) -> float:
    """Calculate value of the random phase at index (i, j).

    Follows San and Staples 2012 to return phase value in any quadrant based on
    the values of eta and zeta in the first quadrant.

    Args:
        zeta: First phase function.
        eta: Second phase function.
        i: rowwise index on the 2-D simulation domain.
        j: columnwise index on the 2-D simulation domain.

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
def cyclic_index(idx: int, n: int) -> int:
    """Map any index to [0, n-1] for periodic boundary conditions."""
    ret_idx = idx % n
    if ret_idx < 0:
        ret_idx += n
    return ret_idx


@wp.kernel
def decaying_turbulence_initializer(
    k_peak: float,
    s: int,
    k_mag: wp.array2d[float],
    zeta: wp.array2d[float],
    eta: wp.array2d[float],
    omega_hat_init: wp.array2d[wp.vec2f],
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
def rk3_substep(
    coeff0: float,
    coeff1: float,
    coeff2: float,
    omega_0: wp.array2d[float],
    omega_in: wp.array2d[float],
    psi: wp.array2d[float],
    omega_out: wp.array2d[float],
):
    """Compute diffusion + advection RHS and apply one strong-stability-preserving RK3 sub-stage update."""
    i, j = wp.tid()

    left = cyclic_index(i - 1, N_GRID)
    right = cyclic_index(i + 1, N_GRID)
    top = cyclic_index(j + 1, N_GRID)
    bottom = cyclic_index(j - 1, N_GRID)

    laplacian = (
        omega_in[right, j] + omega_in[left, j] + omega_in[i, top] + omega_in[i, bottom] - 4.0 * omega_in[i, j]
    ) * INV_H2

    j1 = ((omega_in[right, j] - omega_in[left, j]) * INV_2H) * ((psi[i, top] - psi[i, bottom]) * INV_2H)
    j2 = ((omega_in[i, top] - omega_in[i, bottom]) * INV_2H) * ((psi[right, j] - psi[left, j]) * INV_2H)

    rhs_val = (1.0 / RE) * laplacian + j2 - j1

    omega_out[i, j] = coeff0 * omega_0[i, j] + coeff1 * omega_in[i, j] + coeff2 * DT * rhs_val


@wp.kernel
def copy_float_to_complex(omega: wp.array2d[float], omega_complex: wp.array2d[wp.vec2f]):
    """Pack a real array into a complex array with zero imaginary part."""
    i, j = wp.tid()
    omega_complex[i, j] = wp.vec2f(omega[i, j], 0.0)


# Tile kernels with different block_dim values need separate modules because block_dim is
# baked into the compiled module header.  The two FFT kernels share the same block_dim so
# they can share a module; tiled_transpose below has a different block_dim and uses its own.
@wp.kernel(module="dft_kernels")
def fft_tiled(x: wp.array2d[wp.vec2f], y: wp.array2d[wp.vec2f]):
    """Perform 1-D FFT on each row using ``wp.tile_fft``."""
    i, _, _ = wp.tid()
    row_tile = wp.tile_load(x, shape=(1, N_GRID), offset=(i, 0))
    wp.tile_fft(row_tile)
    wp.tile_store(y, row_tile, offset=(i, 0))


@wp.kernel(module="dft_kernels")
def ifft_tiled(x: wp.array2d[wp.vec2f], y: wp.array2d[wp.vec2f]):
    """Perform 1-D inverse FFT on each row using ``wp.tile_ifft``."""
    i, _, _ = wp.tid()
    row_tile = wp.tile_load(x, shape=(1, N_GRID), offset=(i, 0))
    wp.tile_ifft(row_tile)
    wp.tile_store(y, row_tile, offset=(i, 0))


@wp.kernel(module="unique")
def tiled_transpose(x: wp.array2d[wp.vec2f], y: wp.array2d[wp.vec2f]):
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
def multiply_k2_inverse(inv_k_sq: wp.array2d[float], omega_hat: wp.array2d[wp.vec2f], psi_hat: wp.array2d[wp.vec2f]):
    """Solve Poisson equation in Fourier space: psi_hat = omega_hat / ||k||^2."""
    i, j = wp.tid()
    psi_hat[i, j] = omega_hat[i, j] * inv_k_sq[i, j]


@wp.kernel
def extract_real_and_normalize(
    divisor: float,
    complex_array: wp.array2d[wp.vec2f],
    real_array: wp.array2d[float],
):
    """Extract and normalize the real part of a complex array."""
    i, j = wp.tid()
    real_array[i, j] = complex_array[i, j].x / divisor


@wp.kernel
def add_perturbation(
    omega_0: wp.array2d[float],
    delta_omega: wp.array2d[float],
    out: wp.array2d[float],
):
    """Add perturbation delta_omega to the base vorticity field omega_0."""
    i, j = wp.tid()
    out[i, j] = omega_0[i, j] + delta_omega[i, j]


@wp.kernel
def compute_neg_mse(actual: wp.array2d[float], target: wp.array2d[float], loss: wp.array[float]):
    """Accumulate negative MSE loss (negative because the optimizer minimizes, but we want to maximize divergence)."""
    i, j = wp.tid()
    diff = actual[i, j] - target[i, j]
    wp.atomic_add(loss, 0, -(diff * diff) / float(N_GRID * N_GRID))


@wp.kernel
def compute_delta_omega_norm_sq(arr: wp.array2d[float], delta_omega_norm_sq: wp.array[float]):
    """Compute ||delta_omega||^2."""
    i, j = wp.tid()
    wp.atomic_add(delta_omega_norm_sq, 0, arr[i, j] * arr[i, j])


@wp.kernel
def clamp_delta_omega_norm(
    delta_omega: wp.array2d[float], delta_omega_norm_sq: wp.array[float], epsilon: wp.array[float]
):
    """Rescale delta_omega so its L2 norm does not exceed epsilon."""
    i, j = wp.tid()
    eps = epsilon[0]
    norm = wp.sqrt(delta_omega_norm_sq[0])
    if norm > eps:
        delta_omega[i, j] = delta_omega[i, j] * (eps / norm)


class Example:
    """Optimal perturbation solver for 2-D decaying turbulence."""

    def __init__(
        self,
        spin_up_steps: int = 500,
        lead_steps: int = 80,
        epsilon_frac: float = 0.2,
        lr: float = 0.01,
    ) -> None:
        """Initialize solver, run non-differentiable spin-up, and allocate arrays for autodiff.

        Args:
            spin_up_steps: Non-differentiable steps to reach a statistically stationary state.
            lead_steps: Number of differentiable forward steps.
            epsilon_frac: Maximum allowable perturbation norm as a fraction of base vorticity norm.
            lr: Adam learning rate.
        """
        self.lead_steps = lead_steps
        self.use_cuda_graph = wp.get_device().is_cuda

        self.rk3_coeffs = [
            [1.0, 0.0, 1.0],
            [3.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0],
            [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
        ]

        # Precompute 1/||k||^2 for spectral Poisson solver
        k = np.fft.fftfreq(N_GRID, d=1.0 / N_GRID)
        kx, ky = np.meshgrid(k, k)
        k2 = kx**2 + ky**2
        inv_k_sq = np.zeros_like(k2)
        nonzero = k2 != 0
        inv_k_sq[nonzero] = 1.0 / k2[nonzero]
        self.inv_k_sq = wp.array2d(inv_k_sq.astype(np.float32), dtype=float)

        # Initialize IC, spin-up to omega_0, run forward to get reference field (non-differentiable)
        self.omega_0 = wp.zeros((N_GRID, N_GRID), dtype=float)
        self.y_star = wp.zeros_like(self.omega_0)
        self._init_fields(spin_up_steps, lead_steps)

        # Allocate per-(timestep, stage) arrays for differentiable forward
        self._allocate_per_step_arrays()

        # Initialize perturbation, optimizer, CUDA graph
        self._init_optimizer(epsilon_frac, lr)

    def _init_fields(self, spin_up_steps: int, lead_steps: int) -> None:
        """Spin-up to a statistically stationary state, then run forward to produce the reference trajectory.

        Sets ``self.omega_0`` (base vorticity) and ``self.y_star`` (unperturbed vorticity at lead time).
        """
        # Allocate scratch arrays (freed when the function returns)
        omega_start = wp.zeros((N_GRID, N_GRID), dtype=float)
        omega_curr = wp.zeros_like(omega_start)
        omega_next = wp.zeros_like(omega_start)
        psi = wp.zeros_like(omega_start)
        poisson_fft = self._allocate_poisson_fft_buffers()

        # Generate decaying turbulence initial condition
        k = np.fft.fftfreq(N_GRID, d=1.0 / N_GRID)
        k_mag_np = np.sqrt(k**2 + k[:, np.newaxis] ** 2)
        k_mag = wp.array2d(k_mag_np, dtype=float)

        rng = np.random.default_rng(42)
        zeta_np = 2 * np.pi * rng.random((N_GRID // 2 + 1, N_GRID // 2 + 1))
        eta_np = 2 * np.pi * rng.random((N_GRID // 2 + 1, N_GRID // 2 + 1))
        zeta = wp.array2d(zeta_np, dtype=float)
        eta = wp.array2d(eta_np, dtype=float)

        wp.launch(
            decaying_turbulence_initializer,
            dim=(N_GRID, N_GRID),
            inputs=[12.0, 3, k_mag, zeta, eta],  # k_peak=12.0, s=3
            outputs=[poisson_fft.omega_complex],
        )
        self._fft_2d(
            ifft_tiled,
            in_arr=poisson_fft.omega_complex,
            row_scratch=poisson_fft.scratch_1,
            transpose_scratch=poisson_fft.scratch_2,
            out_arr=poisson_fft.scratch_3,
        )

        # Extract real part (no N_GRID^2 normalization — only needed for FFT → IFFT roundtrips)
        wp.launch(
            extract_real_and_normalize,
            dim=(N_GRID, N_GRID),
            inputs=[1.0, poisson_fft.scratch_3],
            outputs=[omega_start],
        )

        # Capture a single spin-up step into a CUDA graph
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self._spin_up_step(omega_start, omega_curr, omega_next, psi, poisson_fft)
            spin_up_graph = capture.graph

        # Spin-up to omega_0
        for _ in range(spin_up_steps):
            if self.use_cuda_graph:
                wp.capture_launch(spin_up_graph)
            else:
                self._spin_up_step(omega_start, omega_curr, omega_next, psi, poisson_fft)

        # Save base vorticity
        wp.copy(self.omega_0, omega_start)

        # Run forward to get the unperturbed reference field
        for _ in range(lead_steps):
            if self.use_cuda_graph:
                wp.capture_launch(spin_up_graph)
            else:
                self._spin_up_step(omega_start, omega_curr, omega_next, psi, poisson_fft)

        # Save unperturbed reference vorticity at lead time
        wp.copy(self.y_star, omega_start)

    def _spin_up_step(
        self,
        omega_start: wp.array2d[float],
        omega_curr: wp.array2d[float],
        omega_next: wp.array2d[float],
        psi: wp.array2d[float],
        poisson_fft: PoissonFFTBuffers,
    ) -> None:
        """Run one non-differentiable timestep. After return, ``omega_start`` holds the updated field."""

        wp.copy(omega_curr, omega_start)
        for c0, c1, c2 in self.rk3_coeffs:
            self._solve_poisson(omega=omega_curr, psi=psi, fft_buffers=poisson_fft)
            wp.launch(
                rk3_substep,
                dim=(N_GRID, N_GRID),
                inputs=[c0, c1, c2, omega_start, omega_curr, psi],
                outputs=[omega_next],
            )
            omega_curr, omega_next = omega_next, omega_curr
        wp.copy(omega_start, omega_curr)

    @staticmethod
    def _allocate_poisson_fft_buffers(requires_grad: bool = False) -> PoissonFFTBuffers:
        """Allocate ``PoissonFFTBuffers`` struct with all arrays zeroed."""
        buffers = PoissonFFTBuffers()
        buffers.omega_complex = wp.zeros((N_GRID, N_GRID), dtype=wp.vec2f, requires_grad=requires_grad)
        buffers.scratch_1 = wp.zeros_like(buffers.omega_complex)
        buffers.scratch_2 = wp.zeros_like(buffers.omega_complex)
        buffers.scratch_3 = wp.zeros_like(buffers.omega_complex)
        buffers.psi_hat = wp.zeros_like(buffers.omega_complex)
        return buffers

    def _allocate_per_step_arrays(self) -> None:
        """Pre-allocate arrays for the differentiable forward pass.

        Warp's autodiff requires each operation to write to a unique array so
        that gradients can be replayed in reverse.  We therefore allocate
        separate omega, psi, and FFT-scratch arrays for every (timestep, RK stage)
        pair.
        """
        T = self.lead_steps

        self.omega_timesteps = [wp.zeros((N_GRID, N_GRID), dtype=float, requires_grad=True) for _ in range(T + 1)]

        # Per-stage arrays for each timestep, indexed as [timestep][stage].
        self.omega_ts = []
        self.psi_ts = []
        self.fft_ts = []

        for _ in range(T):
            stage_omega, stage_psi, stage_fft = [], [], []
            for _ in range(3):
                stage_omega.append(wp.zeros((N_GRID, N_GRID), dtype=float, requires_grad=True))
                stage_psi.append(wp.zeros((N_GRID, N_GRID), dtype=float, requires_grad=True))
                stage_fft.append(self._allocate_poisson_fft_buffers(requires_grad=True))
            self.omega_ts.append(stage_omega)
            self.psi_ts.append(stage_psi)
            self.fft_ts.append(stage_fft)

    def _init_optimizer(self, epsilon_frac: float, lr: float) -> None:
        """Initialize the vorticity perturbation, Adam optimizer, and CUDA graph."""
        omega_0_np = self.omega_0.numpy()
        omega_0_norm = float(np.sqrt(np.sum(omega_0_np**2)))
        epsilon = epsilon_frac * omega_0_norm
        self.epsilon = wp.array([epsilon], dtype=float)
        print(f"||omega_0|| = {omega_0_norm:.4f}, epsilon = {epsilon:.4f}")

        rng = np.random.default_rng(42)
        delta_omega_np = rng.standard_normal((N_GRID, N_GRID)).astype(np.float32)
        delta_omega_norm = float(np.sqrt(np.sum(delta_omega_np**2)))
        delta_omega_np *= (epsilon / 2.0) / delta_omega_norm  # start at half epsilon
        self.delta_omega = wp.array2d(delta_omega_np, dtype=float, requires_grad=True)

        self.optimizer = warp.optim.Adam([self.delta_omega.flatten()], lr=lr)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.tape = None
        self.delta_omega_norm_sq = wp.zeros(1, dtype=float)

        if self.use_cuda_graph:
            self.tape = wp.Tape()
            # Capture forward + backward pass into a replayable CUDA graph.
            with wp.ScopedCapture() as capture:
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)
            self.graph_forward_backward = capture.graph

            # Capture perturbation clamping + gradient reset.
            with wp.ScopedCapture() as capture:
                self.delta_omega_norm_sq.zero_()
                wp.launch(
                    compute_delta_omega_norm_sq,
                    dim=(N_GRID, N_GRID),
                    inputs=[self.delta_omega],
                    outputs=[self.delta_omega_norm_sq],
                )
                wp.launch(
                    clamp_delta_omega_norm,
                    dim=(N_GRID, N_GRID),
                    inputs=[self.delta_omega, self.delta_omega_norm_sq, self.epsilon],
                )
                self.tape.zero()
            self.graph_project_and_reset = capture.graph

    def _fft_2d(
        self,
        kernel: wp.Kernel,
        in_arr: wp.array2d[wp.vec2f],
        row_scratch: wp.array2d[wp.vec2f],
        transpose_scratch: wp.array2d[wp.vec2f],
        out_arr: wp.array2d[wp.vec2f],
    ) -> None:
        """Perform 2-D FFT or IFFT: row-wise 1-D transform → transpose → column-wise 1-D transform."""
        wp.launch_tiled(
            kernel,
            dim=[N_GRID, 1],
            inputs=[in_arr],
            outputs=[row_scratch],
            block_dim=BLOCK_DIM,
        )

        wp.launch_tiled(
            tiled_transpose,
            dim=(N_GRID // TILE_TRANSPOSE_DIM, N_GRID // TILE_TRANSPOSE_DIM),
            inputs=[row_scratch],
            outputs=[transpose_scratch],
            block_dim=TILE_TRANSPOSE_DIM * TILE_TRANSPOSE_DIM,
        )

        wp.launch_tiled(
            kernel,
            dim=[N_GRID, 1],
            inputs=[transpose_scratch],
            outputs=[out_arr],
            block_dim=BLOCK_DIM,
        )

    def _solve_poisson(self, omega: wp.array2d[float], psi: wp.array2d[float], fft_buffers: PoissonFFTBuffers) -> None:
        """Solve the Poisson equation ``nabla^2(psi) = -omega`` spectrally via FFT."""

        # Pack vorticity into complex form for FFT
        wp.launch(copy_float_to_complex, dim=(N_GRID, N_GRID), inputs=[omega], outputs=[fft_buffers.omega_complex])

        # Forward fft_buffers: omega → omega_hat (stored in fft.scratch_3)
        self._fft_2d(
            fft_tiled,
            in_arr=fft_buffers.omega_complex,
            row_scratch=fft_buffers.scratch_1,
            transpose_scratch=fft_buffers.scratch_2,
            out_arr=fft_buffers.scratch_3,
        )

        # Solve in Fourier space: psi_hat = omega_hat / ||k||^2
        wp.launch(
            multiply_k2_inverse,
            dim=(N_GRID, N_GRID),
            inputs=[self.inv_k_sq, fft_buffers.scratch_3],
            outputs=[fft_buffers.psi_hat],
        )

        # Inverse FFT: psi_hat → psi (reuses scratch_1-3; safe because FFT/IFFT
        # are linear so the adjoint doesn't need the forward intermediates)
        self._fft_2d(
            ifft_tiled,
            in_arr=fft_buffers.psi_hat,
            row_scratch=fft_buffers.scratch_1,
            transpose_scratch=fft_buffers.scratch_2,
            out_arr=fft_buffers.scratch_3,
        )

        # Extract and normalize the real part
        wp.launch(
            extract_real_and_normalize,
            dim=(N_GRID, N_GRID),
            inputs=[float(N_GRID * N_GRID), fft_buffers.scratch_3],
            outputs=[psi],
        )

    def forward(self) -> None:
        """Differentiable forward pass: perturb omega_0, integrate forward, compute loss."""

        # Add perturbation to base vorticity
        wp.launch(
            add_perturbation,
            dim=(N_GRID, N_GRID),
            inputs=[self.omega_0, self.delta_omega],
            outputs=[self.omega_timesteps[0]],
        )

        # Integrate forward with unique arrays per (timestep, stage) for autodiff
        for t in range(self.lead_steps):
            omega_t = self.omega_timesteps[t]

            for s, (c0, c1, c2) in enumerate(self.rk3_coeffs):
                omega_in = omega_t if s == 0 else self.omega_ts[t][s - 1]

                self._solve_poisson(omega=omega_in, psi=self.psi_ts[t][s], fft_buffers=self.fft_ts[t][s])

                wp.launch(
                    rk3_substep,
                    dim=(N_GRID, N_GRID),
                    inputs=[c0, c1, c2, omega_t, omega_in, self.psi_ts[t][s]],
                    outputs=[self.omega_ts[t][s]],
                )

            # Copy final RK3 stage result to the next timestep's input
            wp.copy(self.omega_timesteps[t + 1], self.omega_ts[t][2])

        # Compute negative MSE between perturbed and unperturbed vorticity at lead time
        self.loss.zero_()
        wp.launch(
            compute_neg_mse,
            dim=(N_GRID, N_GRID),
            inputs=[self.omega_timesteps[self.lead_steps], self.y_star],
            outputs=[self.loss],
        )

    def step(self) -> None:
        """Run one optimization iteration: forward pass, backward pass, Adam update, and norm clamping."""
        if self.use_cuda_graph:
            wp.capture_launch(self.graph_forward_backward)
        else:
            self.tape = wp.Tape()
            with self.tape:
                self.forward()
            self.tape.backward(self.loss)

        # Adam.step() is not graph-capturable because it increments a CPU-side
        # bias-correction counter each iteration.
        self.optimizer.step([self.delta_omega.grad.flatten()])

        if self.use_cuda_graph:
            wp.capture_launch(self.graph_project_and_reset)
        else:
            # Clamp perturbation norm to epsilon
            self.delta_omega_norm_sq.zero_()
            wp.launch(
                compute_delta_omega_norm_sq,
                dim=(N_GRID, N_GRID),
                inputs=[self.delta_omega],
                outputs=[self.delta_omega_norm_sq],
            )
            wp.launch(
                clamp_delta_omega_norm,
                dim=(N_GRID, N_GRID),
                inputs=[self.delta_omega, self.delta_omega_norm_sq, self.epsilon],
            )
            self.tape.zero()

    def init_plot(self):
        """Create the live optimization figure with two panels (delta_omega and perturbed vorticity)."""
        if matplotlib.rcParams["figure.raise_window"]:
            matplotlib.rcParams["figure.raise_window"] = False

        fig, (ax_delta_omega, ax_omega) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(
            "Optimal perturbation: find the initial vorticity perturbation that\n"
            "maximizes divergence from the unperturbed trajectory at lead time",
            fontsize=10,
        )

        delta_omega_np = self.delta_omega.numpy()
        vlim_delta_omega = max(float(np.abs(delta_omega_np).max()), 1e-8)
        self._img_delta_omega = ax_delta_omega.imshow(
            delta_omega_np.T, origin="lower", cmap="RdBu_r", vmin=-vlim_delta_omega, vmax=vlim_delta_omega
        )
        self._ax_delta_omega = ax_delta_omega
        ax_delta_omega.set_title("Optimal vorticity perturbation")
        ax_delta_omega.set_xticks([])
        ax_delta_omega.set_yticks([])

        omega_np = self.omega_timesteps[self.lead_steps].numpy()
        vlim_omega = max(float(np.abs(omega_np).max()), 1e-8)
        self._img_omega = ax_omega.imshow(omega_np.T, origin="lower", cmap="RdBu_r", vmin=-vlim_omega, vmax=vlim_omega)
        self._ax_omega = ax_omega
        ax_omega.set_title("Perturbed vorticity at lead time")
        ax_omega.set_xticks([])
        ax_omega.set_yticks([])

        fig.tight_layout()

    def update_plot(self, iteration: int):
        """Update the live plot with current perturbation and perturbed vorticity fields."""
        delta_omega_np = self.delta_omega.numpy()
        vlim_delta_omega = max(float(np.abs(delta_omega_np).max()), 1e-8)
        self._img_delta_omega.set_data(delta_omega_np.T)
        self._img_delta_omega.set_clim(-vlim_delta_omega, vlim_delta_omega)
        self._ax_delta_omega.set_title(f"Optimal vorticity perturbation — iter {iteration}")

        omega_np = self.omega_timesteps[self.lead_steps].numpy()
        vlim_omega = max(float(np.abs(omega_np).max()), 1e-8)
        self._img_omega.set_data(omega_np.T)
        self._img_omega.set_clim(-vlim_omega, vlim_omega)
        self._ax_omega.set_title(f"Perturbed vorticity at lead time — iter {iteration}")

        plt.pause(0.001)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Optimal perturbation solver for 2-D decaying turbulence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--spin-up-steps", type=int, default=500, help="Non-diff spin-up steps.")
    parser.add_argument(
        "--lead-steps",
        type=int,
        default=80,
        help="Differentiable forward steps (physical time = lead_steps * DT).",
    )
    parser.add_argument(
        "--epsilon-frac", type=float, default=0.2, help="Maximum ||delta_omega|| as fraction of ||omega_0||."
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate.")
    parser.add_argument("--train-iters", type=int, default=300, help="Total number of training iterations.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    # check visualization availability early (before training) so user can cancel if needed
    can_visualize = False
    if not args.headless:
        if not MATPLOTLIB_AVAILABLE:
            print(
                "Warning: matplotlib not found. Skipping visualization. "
                "Install matplotlib to enable visualization: pip install matplotlib",
                file=sys.stderr,
            )
        # matplotlib is available, check if backend supports interactive display
        elif matplotlib.get_backend().lower() == "agg":
            print(
                "Warning: No interactive matplotlib backend available. Skipping visualization. "
                "Install python3-tk (Linux) or PySide6 to enable visualization.",
                file=sys.stderr,
            )
        else:
            can_visualize = True

    with wp.ScopedDevice(args.device):
        example = Example(
            spin_up_steps=args.spin_up_steps,
            lead_steps=args.lead_steps,
            epsilon_frac=args.epsilon_frac,
            lr=args.lr,
        )

        wp.synchronize_device()
        if (device := wp.get_device()).is_cuda:
            print(f"Current memory usage: {wp.get_mempool_used_mem_current(device) / (1024**3):.4f} GiB")

        if can_visualize:
            example.init_plot()

        print(f"\nOptimizing for {args.train_iters} iterations...")
        for i in range(args.train_iters):
            example.step()

            if i % 10 == 0 or i == args.train_iters - 1:
                loss_val = example.loss.numpy()[0]
                delta_omega_norm = float(np.linalg.norm(example.delta_omega.numpy()))
                print(f"Iteration {i:05d}  loss: {loss_val:>12.6f}  perturbation norm: {delta_omega_norm:.4f}")

                if can_visualize:
                    example.update_plot(i)

        if can_visualize:
            plt.show()
