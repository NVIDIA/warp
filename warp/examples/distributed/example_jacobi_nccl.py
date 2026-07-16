# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An example implementation of a distributed Jacobi solver using NCCL.

This example shows how to solve the Laplace equation using Jacobi iteration on
multiple GPUs with Warp. NCCL (through nccl4py) exchanges halo rows between
neighboring ranks, piggybacking on MPI for setup. This
example is based on the basic "mpi" example from the Multi GPU Programming Models
repository.

This example requires nccl4py, mpi4py, and an MPI implementation.
Install the nccl4py extra that matches your CUDA major version:
python -m pip install "nccl4py[cu12]" or python -m pip install "nccl4py[cu13]".

Usage:
    mpirun -n 2 python example_jacobi_nccl.py

References:
    https://github.com/NVIDIA/multi-gpu-programming-models
"""

import math
import sys

import nccl.core as nccl
import numpy as np
from mpi4py import MPI

import warp as wp

wp.config.log_level = wp.LOG_WARNING

TOL = 1e-8

# Iterations between residual reductions and progress prints.
PRINT_INTERVAL = 100


def calc_default_device(mpi_comm: "MPI.Comm") -> wp.Device:
    """Return the Warp device this rank should use.

    Splits the communicator into a node-local group and maps each local rank to a
    distinct CUDA device when several are visible. With a single visible device, all
    local ranks share it.

    Args:
        mpi_comm: The global MPI communicator.

    Returns:
        The Warp device assigned to this rank.

    Raises:
        RuntimeError: If no CUDA device is visible, or if more than one device is
            visible but the node has fewer devices than ranks.
    """
    local_mpi_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)

    local_size = local_mpi_comm.Get_size()
    local_rank = local_mpi_comm.Get_rank()

    num_cuda_devices = wp.get_cuda_device_count()

    if num_cuda_devices == 0:
        raise RuntimeError("This example requires at least one CUDA device, but Warp sees none.")

    if 1 < num_cuda_devices < local_size:
        raise RuntimeError(
            f"Number of visible devices ({num_cuda_devices}) is less than number of ranks on the node ({local_size})"
        )

    if num_cuda_devices > 1:
        return wp.get_cuda_device(local_rank)
    else:
        return wp.get_cuda_device()


def calc_decomp_1d(total_points: int, rank: int, total_ranks: int) -> tuple[int, int]:
    """Compute this rank's slab in a 1-D row decomposition.

    Distributes ``total_points`` rows across ``total_ranks`` as evenly as possible,
    giving the first ``total_points % total_ranks`` ranks one extra row each.

    Args:
        total_points: Total number of interior rows to distribute.
        rank: Rank whose slab is computed.
        total_ranks: Number of ranks sharing the rows.

    Returns:
        The ``(start_index, num_domain_points)`` pair for this rank.
    """
    if rank < (total_points % total_ranks):
        num_domain_points = total_points // total_ranks + 1
        start_index = num_domain_points * rank
    else:
        num_domain_points = total_points // total_ranks
        start_index = total_points - (total_ranks - rank) * num_domain_points

    return (start_index, num_domain_points)


@wp.kernel
def jacobi_update(
    a: wp.array2d[float],
    iy_start: int,
    iy_end: int,
    nx: int,
    calculate_norm: bool,
    a_new: wp.array2d[float],
    l2_norm: wp.array[float],
):
    """Apply one Jacobi sweep of the 5-point Laplace stencil and optionally reduce the residual.

    When ``calculate_norm`` is true, the block reduces the
    squared change with ``wp.tile_sum`` and atomically adds it to ``l2_norm``.

    Args:
        a: Solution field read by the stencil (read-only).
        iy_start: First owned row index; rows above it are halo rows.
        iy_end: One past the last owned row index.
        nx: Number of columns in the mesh.
        calculate_norm: Whether to accumulate the squared residual this sweep.
        a_new: Output solution field for this sweep (write-only).
        l2_norm: Single-element accumulator for the summed squared residual.
    """
    i, j = wp.tid()

    iy = i + iy_start
    ix = j + 1

    local_l2_norm = float(0.0)

    if iy < iy_end and ix < nx - 1:
        new_val = float(0.25) * (a[iy - 1, ix] + a[iy + 1, ix] + a[iy, ix - 1] + a[iy, ix + 1])
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
    a: wp.array2d[float],
    a_new: wp.array2d[float],
):
    """Set the fixed sinusoidal boundary columns of the solution field.

    Writes the left (``ix = 0``) and right (``ix = nx - 1``) columns of both ``a`` and
    ``a_new`` so the Dirichlet boundary stays fixed across iterations. ``offset`` maps
    a rank's local row to its global row.

    Args:
        nx: Number of columns in the mesh.
        ny: Number of rows in the global mesh.
        offset: Global row index of this rank's first local row.
        a: Solution field whose boundary columns are written.
        a_new: Intermediate solution field whose boundary columns are written.
    """
    i = wp.tid()
    boundary_val = wp.sin(float(2.0) * wp.pi * float(i + offset) / float(ny - 1))
    a[i, 0] = boundary_val
    a[i, nx - 1] = boundary_val
    a_new[i, 0] = boundary_val
    a_new[i, nx - 1] = boundary_val


def benchmark_single_gpu(nx: int, ny: int, iter_max: int, nccheck: int = 2, verbose: bool = False):
    """Solve the full problem on a single GPU for timing and as a correctness reference.

    Runs Jacobi relaxation on the entire ``ny`` x ``nx`` mesh with periodic boundaries
    in y until the residual falls below ``TOL`` or ``iter_max`` iterations elapse.

    Args:
        nx: Number of columns in the mesh.
        ny: Number of rows in the mesh.
        iter_max: Maximum number of Jacobi iterations.
        nccheck: Positive even number of iterations between residual checks.
        verbose: Whether to print progress on the calling rank.

    Returns:
        A ``(runtime_seconds, solution_host_array)`` pair, where the solution is copied
        to a host array for later comparison.
    """
    a = wp.zeros((ny, nx), dtype=float)
    a_new = wp.zeros_like(a)

    l2_norm_d = wp.zeros((1,), dtype=float)
    compute_stream = wp.Stream()

    iy_start = 1
    iy_end = ny - 1
    update_shape = (iy_end - iy_start, nx - 2)

    wp.launch(initialize_boundaries, dim=(ny,), inputs=[nx, ny, 0], outputs=[a, a_new])

    if verbose:
        print(
            f"Single GPU jacobi relaxation: {iter_max} iterations on {ny} x {nx} mesh, "
            f"reporting the residual every {PRINT_INTERVAL} iterations"
        )

    iter = 0
    l2_norm = 1.0

    start_time = MPI.Wtime()

    while l2_norm > TOL and iter < iter_max:
        calculate_norm = ((iter + 1) % nccheck == 0) or ((iter + 1) % PRINT_INTERVAL == 0)

        with wp.ScopedStream(compute_stream):
            l2_norm_d.zero_()

            wp.launch(
                jacobi_update,
                update_shape,
                inputs=[a, iy_start, iy_end, nx, calculate_norm],
                outputs=[a_new, l2_norm_d],
            )

            # Apply periodic boundary conditions
            wp.copy(a_new[0], a_new[iy_end - 1])
            wp.copy(a_new[iy_end], a_new[iy_start])

        if calculate_norm:
            l2_norm = math.sqrt(l2_norm_d.numpy()[0])

            if verbose and (iter + 1) % PRINT_INTERVAL == 0:
                print(f"{iter + 1:5d}, {l2_norm:.6f}")

        # Swap arrays
        a, a_new = a_new, a

        iter += 1

    wp.synchronize_device()
    stop_time = MPI.Wtime()

    a_ref_h = wp.empty((ny, nx), dtype=float, device="cpu")
    wp.copy(a_ref_h, a)

    return stop_time - start_time, a_ref_h


class Example:
    """Distributed Jacobi solver for the Laplace equation on a 1-D row decomposition.

    Each rank owns a contiguous slab of mesh rows plus one halo row above and below.
    A Warp kernel applies the 5-point stencil, NCCL exchanges the halo rows with the
    neighboring ranks, and the two-iteration ``step`` is captured into a CUDA graph.
    MPI coordinates setup and the convergence reduction. ``TOL`` is a module-level
    constant captured by the host loop.

    Args:
        nx: Total mesh resolution in x.
        ny: Total mesh resolution in y.
        iter_max: Maximum number of Jacobi iterations.
        nccheck: Positive even number of iterations between residual checks, applied to both the single-GPU
            reference and the distributed solver.
        csv: Whether to print results as CSV instead of human-readable text.
    """

    def __init__(
        self,
        nx: int = 16384,
        ny: int = 16384,
        iter_max: int = 1000,
        nccheck: int = 2,
        csv: bool = False,
    ):
        if iter_max % 2 != 0:
            raise ValueError(
                f"iter_max must be even because each step() call advances two Jacobi iterations; got {iter_max}."
            )
        if nccheck <= 0 or nccheck % 2 != 0:
            raise ValueError(
                f"nccheck must be a positive even integer because each step() call advances two Jacobi iterations; got {nccheck}."
            )
        self.iter_max = iter_max
        self.nx = nx
        self.ny = ny
        self.nccheck = nccheck
        self.csv = csv

        self.mpi_comm = MPI.COMM_WORLD
        self.rank = self.mpi_comm.Get_rank()
        self.size = self.mpi_comm.Get_size()

        self.device = calc_default_device(self.mpi_comm)
        wp.set_device(self.device)

        self.compute_stream = wp.Stream()

        # Capture the iteration loop into a CUDA graph when running on a GPU.
        self.use_cuda_graph = self.device.is_cuda
        self.graph = None

        self.runtime_serial, self.a_ref_h = benchmark_single_gpu(
            self.nx, self.ny, self.iter_max, self.nccheck, not self.csv and self.rank == 0
        )

        if self.size > self.ny - 2:
            raise ValueError(
                f"Cannot distribute {self.ny - 2} interior rows across {self.size} ranks; "
                f"each rank needs at least one interior row. Reduce the rank count or increase --ny."
            )
        iy_decomp_start, self.num_local_rows = calc_decomp_1d(self.ny - 2, self.rank, self.size)

        self.iy_start_global = iy_decomp_start + 1

        self.mpi_comm.Barrier()
        if not self.csv:
            print(
                f"Rank {self.rank} on device {wp.get_cuda_device().pci_bus_id}: "
                f"{self.num_local_rows} rows from y = {self.iy_start_global} to y = {self.iy_start_global + self.num_local_rows - 1}"
            )
        self.mpi_comm.Barrier()

        # Local array layout, num_local_rows + 2 rows tall.
        # Owned rows form the half-open range [iy_start, iy_end).
        # row 0                   -> top halo    (received from the upper neighbor)
        # row iy_end              -> bottom halo (received from the lower neighbor)
        # rows iy_start..iy_end-1 -> owned interior rows
        self.a = wp.zeros((self.num_local_rows + 2, self.nx), dtype=float)
        self.a_new = wp.zeros_like(self.a)

        self.a_h = wp.empty((self.ny, self.nx), dtype=float, device="cpu")

        self.l2_norm_d = wp.zeros((1,), dtype=float)

        self.iy_start = 1  # first owned row (row 0 is the top halo)
        self.iy_end = self.iy_start + self.num_local_rows  # one past the last owned row, i.e. the bottom halo row

        self.update_shape = (self.num_local_rows, self.nx - 2)

        self.lower_neighbor = (self.rank + 1) % self.size
        self.upper_neighbor = self.rank - 1 if self.rank > 0 else self.size - 1

        unique_id = nccl.get_unique_id() if self.rank == 0 else None  # Rank 0 generates a unique ID
        unique_id = self.mpi_comm.bcast(unique_id, root=0)  # Broadcast the ID to all ranks
        # Ensure that all processes have completed the MPI broadcast.
        # This can be required when combining MPI with NCCL.
        self.mpi_comm.Barrier()
        self.nccl_comm = nccl.Communicator.init(nranks=self.size, rank=self.rank, unique_id=unique_id)

        wp.launch(
            initialize_boundaries,
            dim=(self.num_local_rows + 2,),
            inputs=[self.nx, self.ny, self.iy_start_global - 1],
            outputs=[self.a, self.a_new],
        )

        if not self.csv and self.rank == 0:
            print(
                f"Jacobi relaxation: {self.iter_max} iterations on {self.ny} x {self.nx} mesh, "
                f"reporting the residual every {PRINT_INTERVAL} iterations"
            )

    def _exchange_halos(self, a: wp.array2d) -> None:
        """Exchange the top and bottom halo rows of ``a`` with neighbor ranks.

        Args:
            a: Local solution field whose halo rows (row ``0`` and row ``iy_end``) are
                filled from the neighboring ranks.
        """
        with nccl.group():
            # NCCL matches same-peer point-to-point ops by issue order. With size == 2
            # both neighbors are the same rank, so each send must pair with the recv from
            # the other neighbor; otherwise the top and bottom halos get swapped incorrectly.
            self.nccl_comm.send(a[self.iy_end - 1], self.lower_neighbor, stream=self.compute_stream)
            self.nccl_comm.recv(a[0], self.upper_neighbor, stream=self.compute_stream)

            self.nccl_comm.send(a[self.iy_start], self.upper_neighbor, stream=self.compute_stream)
            self.nccl_comm.recv(a[self.iy_end], self.lower_neighbor, stream=self.compute_stream)

    def _jacobi_iteration(self, src: wp.array2d, dst: wp.array2d, calculate_norm: bool) -> None:
        """Update ``dst`` from ``src`` with one Jacobi sweep, then exchange its halos.

        Args:
            src: Solution field read by the stencil this sweep.
            dst: Solution field written this sweep and then halo-exchanged.
            calculate_norm: Whether to accumulate the L2 residual into ``l2_norm_d``.
        """
        self.l2_norm_d.zero_()
        wp.launch(
            jacobi_update,
            self.update_shape,
            inputs=[src, self.iy_start, self.iy_end, self.nx, calculate_norm],
            outputs=[dst, self.l2_norm_d],
        )
        self._exchange_halos(dst)

    def step(self) -> None:
        """Advance two Jacobi iterations (a -> a_new -> a) on the compute stream.

        The two sweeps return the ``a`` and ``a_new`` buffers to their original roles, so the
        body replays consistently after CUDA graph capture. The residual is
        accumulated only on the second sweep and left in ``self.l2_norm_d`` for the
        host to read between graph replays.

        All work runs on ``self.compute_stream``.
        """
        with wp.ScopedStream(self.compute_stream, sync_enter=False):
            self._jacobi_iteration(self.a, self.a_new, calculate_norm=False)
            self._jacobi_iteration(self.a_new, self.a, calculate_norm=True)

    def run(self) -> None:
        """Iterate to convergence across all ranks, then compare against the single-GPU result."""
        iter = 0
        l2_norm = np.array([1.0], dtype=wp.dtype_to_numpy(wp.float32))

        # Capture the two-iteration step once, then replay it each loop turn.
        if self.use_cuda_graph:
            wp.synchronize_device()
            with wp.ScopedCapture(stream=self.compute_stream) as capture:
                self.step()
            self.graph = capture.graph

        wp.synchronize_device()
        start_time = MPI.Wtime()

        while l2_norm > TOL and iter < self.iter_max:
            if self.use_cuda_graph:
                wp.capture_launch(self.graph, stream=self.compute_stream)
            else:
                self.step()

            iter += 2

            # Check convergence on the nccheck grid in every mode
            should_check = (iter % self.nccheck == 0) or (not self.csv and iter % PRINT_INTERVAL == 0)
            if should_check:
                self.mpi_comm.Allreduce(self.l2_norm_d.numpy(), l2_norm)
                l2_norm = np.sqrt(l2_norm)
                if not self.csv and self.rank == 0 and iter % PRINT_INTERVAL == 0:
                    print(f"{iter:5d}, {l2_norm[0]:.6f}")

        wp.synchronize_device()
        stop_time = MPI.Wtime()

        result_correct = self.check_results(TOL)
        global_result_correct = self.mpi_comm.allreduce(result_correct, op=MPI.MIN)

        if not global_result_correct:
            sys.exit(1)
        elif global_result_correct and self.rank == 0:
            if self.csv:
                print(
                    f"nccl, {self.nx}, {self.ny}, {self.iter_max}, {self.nccheck}, {self.size}, 1, "
                    f"{stop_time - start_time}, {self.runtime_serial}"
                )
            else:
                print(f"Num GPUs: {self.size}")
                print(
                    f"{self.ny}x{self.nx}: 1 GPU: {self.runtime_serial:8.4f} s, "
                    f"{self.size} GPUs {stop_time - start_time:8.4f} s, "
                    f"speedup: {self.runtime_serial / (stop_time - start_time):8.2f}, "
                    f"efficiency: {self.runtime_serial / (stop_time - start_time) / self.size * 100:8.2f}"
                )

    def check_results(self, tol: float = 1e-8) -> bool:
        """Return ``True`` if the multi-GPU result is within ``tol`` of the single-GPU result.

        Comparison is performed on the host, one point at a time.

        Args:
            tol: Maximum allowed absolute difference at any interior point.
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

        # Compare only the interior columns of the rows this rank owns. argwhere returns
        # row-major indices, so the first entry is the same mismatch the original
        # point-by-point scan would have reported.
        rows = slice(self.iy_start_global, self.iy_start_global + self.num_local_rows)
        cols = slice(1, self.nx - 1)
        mismatches = np.argwhere(np.abs(a_ref_np[rows, cols] - a_np[rows, cols]) > tol)

        if mismatches.size > 0:
            local_iy, local_ix = mismatches[0]
            iy = self.iy_start_global + int(local_iy)
            ix = 1 + int(local_ix)
            result_correct = False
            print(
                f"ERROR on rank {self.rank}: a[{iy},{ix}] = {a_np[iy, ix]} does not match "
                f"{a_ref_np[iy, ix]} (reference)"
            )

        return result_correct


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--itermax", type=int, default=1000, help="Maximum number of Jacobi iterations.")
    parser.add_argument(
        "--nccheck", type=int, default=2, help="Check convergence every nccheck iterations; must be positive and even."
    )
    parser.add_argument("--nx", type=int, default=16384, help="Total resolution in x.")
    parser.add_argument("--ny", type=int, default=16384, help="Total resolution in y.")
    parser.add_argument("--csv", action="store_true", help="Print results as CSV values.")
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
        plt.title(f"Rank {example.rank} Jacobi Iteration Result")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()
