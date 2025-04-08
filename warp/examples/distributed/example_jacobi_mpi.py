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
"""An example implementation of a distributed Jacobi solver using MPI.

This example shows how to solve the Laplace equation using Jacobi iteration on
multiple GPUs using Warp and mpi4py. This example is based on the basic "mpi"
example from the Multi GPU Programming Models repository.

This example requires mpi4py and a CUDA-aware MPI implementation. We suggest
downloading and installing NVIDIA HPC-X, followed by installing mpi4py from its
source distribution: python -m pip install mpi4py

Usage:
    mpirun -n 2 python example_jacobi_mpi.py

References:
    https://github.com/NVIDIA/multi-gpu-programming-models
    https://developer.nvidia.com/networking/hpc-x
    https://github.com/mpi4py/mpi4py
"""

import math
import sys
from typing import Tuple

import numpy as np
from mpi4py import MPI

import warp as wp
import warp.context
from warp.types import warp_type_to_np_dtype

wp.config.quiet = True  # Suppress wp.init() output


tol = 1e-8
wptype = wp.float32  # Global precision setting, can set wp.float64 here for double precision
pi = wptype(math.pi)  # GitHub #485


def calc_default_device(mpi_comm: "MPI.Comm") -> warp.context.Device:
    """Return the device that should be used for the current rank.

    This function is used to ensure that multiple MPI ranks running on the same
    node are assigned to different GPUs.

    Args:
        mpi_comm: The MPI communicator.

    Returns:
        The Warp device that should be used for the current rank.

    Raises:
        RuntimeError: If the number of visible devices is less than the number of ranks on the node.
    """

    # Find the local rank and size
    local_mpi_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)

    local_size = local_mpi_comm.Get_size()
    local_rank = local_mpi_comm.Get_rank()

    num_cuda_devices = warp.get_cuda_device_count()

    if 1 < num_cuda_devices < local_size:
        raise RuntimeError(
            f"Number of visible devices ({num_cuda_devices}) is less than number of ranks on the node ({local_size})"
        )

    if 1 < num_cuda_devices:
        # Get the device based on local_rank
        return warp.get_cuda_device(local_rank)
    else:
        return warp.get_device()


def calc_decomp_1d(total_points: int, rank: int, total_ranks: int) -> Tuple[int, int]:
    """Calculate a 1-D decomposition to divide ``total_points`` among ``total_ranks`` domains.

    Returns a tuple containing the starting index of the decomposition followed
    by number of points in the domain.

    If ``total_points`` can not be evenly divided among ``total_ranks``,
    the first ``total_points % total_ranks`` domains will contain one additional
    point.
    """

    if rank < total_points % total_ranks:
        num_domain_points = total_points // total_ranks + 1
        start_index = rank * num_domain_points
    else:
        num_domain_points = total_points // total_ranks
        start_index = total_points - (total_ranks - rank) * num_domain_points

    return (start_index, num_domain_points)


@wp.kernel
def jacobi_update(
    a: wp.array2d(dtype=wptype),
    iy_start: int,
    iy_end: int,
    nx: int,
    calculate_norm: bool,
    a_new: wp.array2d(dtype=wptype),
    l2_norm: wp.array(dtype=wptype),
):
    i, j = wp.tid()

    # Convert from local thread indices to the indices used to access the arrays

    iy = i + iy_start
    ix = j + 1

    local_l2_norm = wptype(0.0)

    if iy < iy_end and ix < nx - 1:
        new_val = wptype(0.25) * (a[iy - 1, ix] + a[iy + 1, ix] + a[iy, ix - 1] + a[iy, ix + 1])
        a_new[iy, ix] = new_val

        if calculate_norm:
            residue = new_val - a[iy, ix]
            local_l2_norm = residue * residue

    if calculate_norm:
        t = wp.tile(local_l2_norm)
        s = wp.tile_sum(t)
        wp.tile_atomic_add(l2_norm, s)


@wp.kernel
def initialize_boundaries(
    nx: int,
    ny: int,
    offset: int,
    a: wp.array2d(dtype=wptype),
    a_new: wp.array2d(dtype=wptype),
):
    i = wp.tid()

    boundary_val = wp.sin(wptype(2.0) * pi * wptype(i + offset) / wptype(ny - 1))

    a[i, 0] = boundary_val
    a[i, nx - 1] = boundary_val
    a_new[i, 0] = boundary_val
    a_new[i, nx - 1] = boundary_val


def benchmark_single_gpu(nx: int, ny: int, iter_max: int, nccheck: int = 1, verbose: bool = False):
    """Compute the solution on a single GPU for performance and correctness comparisons.

    Args:
        nx: The number of points in the x-direction.
        ny: The number of points in the y-direction.
        iter_max: The maximum number of Jacobi iterations.
        nccheck: The number of iterations between norm checks. Defaults to 1.
        verbose: Whether to print verbose output. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - runtime (float): The execution time of the solution in seconds.
            - solution (warp.array2d): The solution as a Warp array on the host
              with dimensions ``(ny, nx)``.
    """

    a = wp.zeros((ny, nx), dtype=wptype)
    a_new = wp.zeros_like(a)

    l2_norm_d = wp.zeros((1,), dtype=wptype)
    l2_norm_h = wp.ones_like(l2_norm_d, device="cpu", pinned=True)

    compute_stream = wp.Stream()
    push_top_stream = wp.Stream()
    push_bottom_stream = wp.Stream()

    compute_done = wp.Event()
    push_top_done = wp.Event()
    push_bottom_done = wp.Event()

    iy_start = 1
    iy_end = ny - 1
    update_shape = (iy_end - iy_start, nx - 2)

    wp.launch(initialize_boundaries, dim=(ny,), inputs=[nx, ny, 0], outputs=[a, a_new])

    if verbose:
        print(
            f"Single GPU jacobi relaxation: {iter_max} iterations on {ny} x {nx} mesh with norm check every {nccheck}"
            " iterations"
        )

    iter = 0
    l2_norm = 1.0

    start_time = MPI.Wtime()

    while l2_norm > tol and iter < iter_max:
        calculate_norm = (iter % nccheck == 0) or (iter % 100 == 0)

        with wp.ScopedStream(compute_stream):
            l2_norm_d.zero_()

            compute_stream.wait_event(push_top_done)
            compute_stream.wait_event(push_bottom_done)

            wp.launch(
                jacobi_update,
                update_shape,
                inputs=[a, iy_start, iy_end, nx, calculate_norm],
                outputs=[a_new, l2_norm_d],
            )
            wp.record_event(compute_done)

        if calculate_norm:
            wp.copy(l2_norm_h, l2_norm_d, stream=compute_stream)

        # Apply periodic boundary conditions
        push_top_stream.wait_event(compute_done)
        wp.copy(a_new[0], a_new[iy_end - 1], stream=push_top_stream)
        push_top_stream.record_event(push_top_done)

        push_bottom_stream.wait_event(compute_done)
        wp.copy(a_new[iy_end], a_new[iy_start], stream=push_bottom_stream)
        push_bottom_stream.record_event(push_bottom_done)

        if calculate_norm:
            wp.synchronize_stream(compute_stream)

            l2_norm = math.sqrt(l2_norm_h.numpy()[0])

            if verbose and iter % 100 == 0:
                print(f"{iter:5d}, {l2_norm:.6f}")

        # Swap arrays
        a, a_new = a_new, a

        iter += 1

    wp.synchronize_device()
    stop_time = MPI.Wtime()

    a_ref_h = wp.empty((ny, nx), dtype=wptype, device="cpu")
    wp.copy(a_ref_h, a)

    return stop_time - start_time, a_ref_h


class Example:
    def __init__(
        self,
        nx: int = 16384,
        ny: int = 16384,
        iter_max: int = 1000,
        nccheck: int = 1,
        csv: bool = False,
    ):
        self.iter_max = iter_max
        self.nx = nx  # Global resolution
        self.ny = ny  # Global resolution
        self.nccheck = nccheck
        self.csv = csv

        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()

        # Set the default device on the current rank
        self.device = calc_default_device(self.mpi_comm)
        wp.set_device(self.device)

        # We need to disable memory pools for peer-to-peer transfers using MPI
        # wp.set_mempool_enabled(wp.get_cuda_device(), False)
        self.compute_stream = wp.Stream()
        self.compute_done = wp.Event()

        # Compute the solution on a single GPU for comparisons
        self.runtime_serial, self.a_ref_h = benchmark_single_gpu(
            self.nx, self.ny, self.iter_max, self.nccheck, not self.csv and self.mpi_rank == 0
        )

        # num_local_rows: Number of rows from the full (self.ny, self.nx) solution that
        #                 this rank will calculate (excludes halo regions)
        # iy_start_global: Allows us to go from a local index to a global index.

        # self.ny-2 rows are distributed among the ranks for comparison with single-GPU case,
        # which reserves the first and last rows for the boundary conditions
        iy_decomp_start, self.num_local_rows = calc_decomp_1d(self.ny - 2, self.mpi_rank, self.mpi_size)

        # Add 1 to get the global start index since the 1-D decomposition excludes the boundaries
        self.iy_start_global = iy_decomp_start + 1

        self.mpi_comm.Barrier()
        if not self.csv:
            print(
                f"Rank {self.mpi_rank} on device {wp.get_cuda_device().pci_bus_id}: "
                f"{self.num_local_rows} rows from y = {self.iy_start_global} to y = {self.iy_start_global + self.num_local_rows - 1}"
            )
        self.mpi_comm.Barrier()

        # Allocate local array (the +2 is for the halo layer on each side)
        self.a = wp.zeros((self.num_local_rows + 2, self.nx), dtype=wptype)
        self.a_new = wp.zeros_like(self.a)

        # Allocate host array for the final result
        self.a_h = wp.empty((self.ny, self.nx), dtype=wptype, device="cpu")

        self.l2_norm_d = wp.zeros((1,), dtype=wptype)
        self.l2_norm_h = wp.ones_like(self.l2_norm_d, device="cpu", pinned=True)

        # Boundary Conditions
        # - y-boundaries (iy=0 and iy=self.ny-1): Periodic
        # - x-boundaries (ix=0 and ix=self.nx-1): Dirichlet

        # Local Indices
        self.iy_start = 1
        self.iy_end = self.iy_start + self.num_local_rows  # Last owned row begins at [iy_end-1, 0]

        # Don't need to loop over the Dirichlet boundaries in the Jacobi iteration
        self.update_shape = (self.num_local_rows, self.nx - 2)

        # Used for inter-rank communication
        self.lower_neighbor = (self.mpi_rank + 1) % self.mpi_size
        self.upper_neighbor = self.mpi_rank - 1 if self.mpi_rank > 0 else self.mpi_size - 1

        # Apply Dirichlet boundary conditions to both a and a_new
        wp.launch(
            initialize_boundaries,
            dim=(self.num_local_rows + 2,),
            inputs=[self.nx, self.ny, self.iy_start_global - 1],
            outputs=[self.a, self.a_new],
        )

        # MPI Warmup
        wp.synchronize_device()

        for _mpi_warmup in range(10):
            self.apply_periodic_bc()
            self.a, self.a_new = self.a_new, self.a

        wp.synchronize_device()

        if not self.csv and self.mpi_rank == 0:
            print(
                f"Jacobi relaxation: {self.iter_max} iterations on {self.ny} x {self.nx} mesh with norm check "
                f"every {self.nccheck} iterations"
            )

    def apply_periodic_bc(self) -> None:
        """Apply periodic boundary conditions to the array.

        This function sends the first row of owned data to the lower neighbor
        and the last row of owned data to the upper neighbor.
        """
        # Send the first row of owned data to the lower neighbor
        self.mpi_comm.Sendrecv(
            self.a_new[self.iy_start], self.lower_neighbor, 0, self.a_new[self.iy_end], self.upper_neighbor, 0
        )
        # Send the last row of owned data to the upper neighbor
        self.mpi_comm.Sendrecv(
            self.a_new[self.iy_end - 1], self.upper_neighbor, 0, self.a_new[0], self.lower_neighbor, 0
        )

    def step(self, calculate_norm: bool) -> None:
        """Perform a single Jacobi iteration step."""
        with wp.ScopedStream(self.compute_stream):
            self.l2_norm_d.zero_()
            wp.launch(
                jacobi_update,
                self.update_shape,
                inputs=[self.a, self.iy_start, self.iy_end, self.nx, calculate_norm],
                outputs=[self.a_new, self.l2_norm_d],
            )
            wp.record_event(self.compute_done)

    def run(self) -> None:
        """Run the Jacobi relaxation on multiple GPUs using MPI and compare with single-GPU results."""
        iter = 0
        l2_norm = np.array([1.0], dtype=warp_type_to_np_dtype[wptype])

        start_time = MPI.Wtime()

        while l2_norm > tol and iter < self.iter_max:
            calculate_norm = (iter % self.nccheck == 0) or (not self.csv and iter % 100 == 0)

            self.step(calculate_norm)

            if calculate_norm:
                wp.copy(self.l2_norm_h, self.l2_norm_d, stream=self.compute_stream)

            wp.synchronize_event(self.compute_done)

            self.apply_periodic_bc()

            if calculate_norm:
                wp.synchronize_stream(self.compute_stream)

                self.mpi_comm.Allreduce(self.l2_norm_h.numpy(), l2_norm)
                l2_norm = np.sqrt(l2_norm)

                if not self.csv and self.mpi_rank == 0 and iter % 100 == 0:
                    print(f"{iter:5d}, {l2_norm[0]:.6f}")

            # Swap arrays
            self.a, self.a_new = self.a_new, self.a

            iter += 1

        wp.synchronize_device()
        stop_time = MPI.Wtime()

        result_correct = self.check_results(tol)
        global_result_correct = self.mpi_comm.allreduce(result_correct, op=MPI.MIN)

        if not global_result_correct:
            sys.exit(1)
        elif global_result_correct and self.mpi_rank == 0:
            if self.csv:
                print(
                    f"mpi, {self.nx}, {self.ny}, {self.iter_max}, {self.nccheck}, {self.mpi_size}, 1, "
                    f"{stop_time - start_time}, {self.runtime_serial}"
                )
            else:
                print(f"Num GPUs: {self.mpi_size}")
                print(
                    f"{self.ny}x{self.nx}: 1 GPU: {self.runtime_serial:8.4f} s, "
                    f"{self.mpi_size} GPUs {stop_time - start_time:8.4f} s, "
                    f"speedup: {self.runtime_serial / (stop_time - start_time):8.2f}, "
                    f"efficiency: {self.runtime_serial / (stop_time - start_time) / self.mpi_size * 100:8.2f}"
                )

    def check_results(self, tol: float = 1e-8) -> bool:
        """Returns ``True`` if multi-GPU result is within ``tol`` of the single-GPU result.

        Comparison is performed on the host in a serial manner.
        """
        result_correct = True

        wp.copy(
            self.a_h,
            self.a,
            dest_offset=self.iy_start_global * self.nx,
            src_offset=self.nx,
            count=self.num_local_rows * self.nx,
        )

        a_ref_np = self.a_ref_h.numpy()
        a_np = self.a_h.numpy()

        for iy in range(self.iy_start_global, self.iy_start_global + self.num_local_rows):
            if not result_correct:
                break
            for ix in range(1, self.nx - 1):
                if math.fabs(a_ref_np[iy, ix] - a_np[iy, ix]) > tol:
                    result_correct = False
                    print(
                        f"ERROR on rank {self.mpi_rank}: a[{iy},{ix}] = {a_np[iy, ix]} does not match "
                        f"{a_ref_np[iy, ix]} (reference)"
                    )
                    break

        return result_correct


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--itermax", type=int, default=1000, help="Maximum number of Jacobi iterations.")
    parser.add_argument("--nccheck", type=int, default=1, help="Check convergence every nccheck iterations.")
    parser.add_argument("--nx", type=int, default=16384, help="Total resolution in x.")
    parser.add_argument("--ny", type=int, default=16384, help="Total resolution in y.")
    parser.add_argument("-csv", action="store_true", help="Print results as CSV values.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display the final solution in a graphical window using matplotlib.",
    )

    args = parser.parse_known_args()[0]

    example = Example(args.nx, args.ny, args.itermax, args.nccheck, args.csv)

    example.run()

    if args.visualize:
        import matplotlib.pyplot as plt

        # Plot the final result
        plt.imshow(example.a.numpy(), cmap="viridis", origin="lower", vmin=-1, vmax=1)
        plt.colorbar(label="Value")
        plt.title(f"Rank {example.mpi_rank} Jacobi Iteration Result")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()
