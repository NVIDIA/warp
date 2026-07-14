# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An example implementation of a distributed Jacobi solver using host-initiated NVSHMEM.

This example shows how to solve the Laplace equation using Jacobi iteration on
multiple GPUs with Warp. It uses the host-initiated nvshmem4py API to exchange
halo rows between neighboring ranks, piggybacking on MPI for bootstrap. The
Warp kernels contain no device-side NVSHMEM calls. This example is based on the
basic "mpi" example from the Multi GPU Programming Models repository and mirrors
``example_jacobi_nccl.py``; only the halo-exchange mechanics differ.

Warp arrays involved in halo exchange must live on the NVSHMEM symmetric heap, so
``a`` and ``a_new`` are allocated with ``nvshmem.core.memory.buffer`` and wrapped
as Warp arrays via their device pointers. NVSHMEM ``nvshmem_malloc`` is collective
and requires the same size on every PE, so each rank allocates the uniform maximum
slab size and views only its owned rows.

This example requires nvshmem4py, cuda-python, mpi4py, and an MPI implementation.
Install the nvshmem4py package that matches your CUDA major version:
python -m pip install nvshmem4py-cu12 or python -m pip install nvshmem4py-cu13.

Usage:
    mpirun -n 2 python example_jacobi_nvshmem.py

References:
    https://github.com/NVIDIA/multi-gpu-programming-models
"""

import math
import sys

import numpy as np
import nvshmem.core
import nvshmem.core.collective
import nvshmem.core.memory
import nvshmem.core.rma
from cuda.core import Buffer, Device
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
    A Warp kernel applies the 5-point stencil, NVSHMEM one-sided puts exchange the halo
    rows with the neighboring ranks, and the two-iteration ``step`` is captured into a
    CUDA graph. MPI coordinates setup and the convergence reduction. ``TOL`` is a
    module-level constant captured by the host loop.

    The solution fields live on the NVSHMEM symmetric heap. ``nvshmem_malloc`` is
    collective and requires an identical size on every PE, so every rank allocates the
    uniform maximum slab (``calc_decomp_1d(ny - 2, 0, size)`` rows, the most any rank
    receives) and wraps only its owned ``(num_local_rows + 2, nx)`` view as a Warp array.

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
            raise ValueError(f"iter_max must be even because each step advances two Jacobi iterations; got {iter_max}.")
        if nccheck <= 0 or nccheck % 2 != 0:
            raise ValueError(
                f"nccheck must be a positive even integer because each step() call advances two Jacobi iterations; "
                f"got {nccheck}."
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

        # NVSHMEM is bootstrapped over MPI on the same CUDA device Warp uses.
        cuda_dev = Device(self.device.ordinal)
        cuda_dev.set_current()
        nvshmem.core.init(device=cuda_dev, mpi_comm=self.mpi_comm, initializer_method="mpi")

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
        self.iy_start = 1  # first owned row (row 0 is the top halo)
        self.iy_end = self.iy_start + self.num_local_rows  # one past the last owned row, i.e. the bottom halo row

        # NVSHMEM symmetric allocations must be the same size on every PE. Rank 0 always
        # receives the most rows under calc_decomp_1d, so size every buffer to that slab.
        _, max_local_rows = calc_decomp_1d(self.ny - 2, 0, self.size)
        self._row_bytes = self.nx * np.dtype(np.float32).itemsize
        symmetric_bytes = (max_local_rows + 2) * self._row_bytes

        self.a, self.a_buf = self._alloc_symmetric((self.num_local_rows + 2, self.nx), symmetric_bytes)
        self.a_new, self.a_new_buf = self._alloc_symmetric((self.num_local_rows + 2, self.nx), symmetric_bytes)
        self.a.zero_()
        self.a_new.zero_()

        self.a_h = wp.empty((self.ny, self.nx), dtype=float, device="cpu")

        self.l2_norm_d = wp.zeros((1,), dtype=float)

        self.update_shape = (self.num_local_rows, self.nx - 2)

        self.lower_neighbor = (self.rank + 1) % self.size
        self.upper_neighbor = self.rank - 1 if self.rank > 0 else self.size - 1

        # A put into the upper neighbor's bottom halo targets that neighbor's iy_end,
        # which can differ by one row under the uneven decomposition. calc_decomp_1d is
        # deterministic, so derive the neighbor's slab directly instead of communicating.
        _, upper_num_local_rows = calc_decomp_1d(self.ny - 2, self.upper_neighbor, self.size)
        self._upper_halo_offset_bytes = (1 + upper_num_local_rows) * self._row_bytes

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

    def _alloc_symmetric(self, shape: tuple[int, int], nbytes: int) -> tuple[wp.array, object]:
        """Allocate an NVSHMEM symmetric buffer and wrap it as a Warp array.

        Every PE must call this with the same ``nbytes`` so the symmetric heap stays
        consistent across the NVSHMEM team. ``shape`` is this rank's logical view and may
        cover fewer bytes than ``nbytes``; the slack rows are unused.

        Args:
            shape: Logical ``(rows, cols)`` shape of the Warp array view.
            nbytes: Uniform symmetric allocation size in bytes, identical on every PE.

        Returns:
            A ``(warp_array, nvshmem_buffer)`` pair. The Warp array does not own the
            buffer; free it with ``nvshmem.core.memory.free``.
        """
        buf = nvshmem.core.memory.buffer(nbytes)
        arr_view = wp.array(ptr=int(buf.handle), shape=shape, dtype=float, device=self.device)
        return arr_view, buf

    def _exchange_halos(self, buf: object) -> None:
        """Exchange the top and bottom halo rows with neighbor ranks via NVSHMEM puts.

        Each rank pushes its edge rows directly into the neighboring ranks' halo rows,
        the one-sided counterpart of NCCL's matched send/recv:
          - Last owned row (``iy_end - 1``) -> lower neighbor's top halo (row 0)
          - First owned row (``iy_start``)  -> upper neighbor's bottom halo (its ``iy_end``)

        ``barrier_all`` completes the local puts and ensures every rank's halos are
        populated before the next sweep reads them.

        Args:
            buf: NVSHMEM symmetric buffer backing the array whose halos are exchanged.
        """
        base = int(buf.handle)

        # Push the last owned row into the lower neighbor's top halo (row 0).
        src = Buffer.from_handle(base + (self.iy_end - 1) * self._row_bytes, self._row_bytes)
        dst = Buffer.from_handle(base, self._row_bytes)
        nvshmem.core.rma.put(dst, src, self.lower_neighbor, self.compute_stream)

        # Push the first owned row into the upper neighbor's bottom halo (its iy_end).
        src = Buffer.from_handle(base + self.iy_start * self._row_bytes, self._row_bytes)
        dst = Buffer.from_handle(base + self._upper_halo_offset_bytes, self._row_bytes)
        nvshmem.core.rma.put(dst, src, self.upper_neighbor, self.compute_stream)

        # barrier_all completes the puts and synchronizes all PEs before the halos are read.
        nvshmem.core.collective.barrier_all(self.compute_stream)

    def _jacobi_iteration(self, src: wp.array2d, dst: wp.array2d, dst_buf: object, calculate_norm: bool) -> None:
        """Update ``dst`` from ``src`` with one Jacobi sweep, then exchange its halos.

        Args:
            src: Solution field read by the stencil this sweep.
            dst: Solution field written this sweep and then halo-exchanged.
            dst_buf: NVSHMEM symmetric buffer backing ``dst``.
            calculate_norm: Whether to accumulate the L2 residual into ``l2_norm_d``.
        """
        self.l2_norm_d.zero_()
        wp.launch(
            jacobi_update,
            self.update_shape,
            inputs=[src, self.iy_start, self.iy_end, self.nx, calculate_norm],
            outputs=[dst, self.l2_norm_d],
        )
        self._exchange_halos(dst_buf)

    def step(self) -> None:
        """Advance two Jacobi iterations (a -> a_new -> a) on the compute stream.

        The two sweeps return the ``a`` and ``a_new`` buffers to their original roles, so the
        body replays consistently after CUDA graph capture. The residual is
        accumulated only on the second sweep and left in ``self.l2_norm_d`` for the
        host to read between graph replays.

        All work runs on ``self.compute_stream``.
        """
        with wp.ScopedStream(self.compute_stream, sync_enter=False):
            self._jacobi_iteration(self.a, self.a_new, self.a_new_buf, calculate_norm=False)
            self._jacobi_iteration(self.a_new, self.a, self.a_buf, calculate_norm=True)

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
                    f"nvshmem, {self.nx}, {self.ny}, {self.iter_max}, {self.nccheck}, {self.size}, 1, "
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

    def finalize(self) -> None:
        """Free the symmetric buffers and finalize NVSHMEM."""
        nvshmem.core.memory.free(self.a_buf)
        nvshmem.core.memory.free(self.a_new_buf)
        nvshmem.core.finalize()


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

    example.finalize()
