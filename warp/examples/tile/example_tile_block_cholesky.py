# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

###########################################################################
# Example Tile Block Cholesky
#
# Shows how to write a kernel computing a blocked Cholesky factorization
# of a symmetric positive definite matrix using Warp Tile APIs.
#
###########################################################################

from functools import lru_cache

import numpy as np

import warp as wp

wp.set_module_options({"enable_backward": False})


@lru_cache(maxsize=None)
def create_blocked_cholesky_kernel(block_size: int):
    @wp.kernel
    def blocked_cholesky_kernel(
        A: wp.array(dtype=float, ndim=2),
        L: wp.array(dtype=float, ndim=2),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
    ):
        """
        Computes the Cholesky factorization of a symmetric positive definite matrix A in blocks.
        It returns a lower-triangular matrix L such that A = L L^T.

        A is assumed to support block reading.
        """
        tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()

        active_matrix_size = active_matrix_size_arr[0]

        # Round up active_matrix_size to next multiple of block_size
        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        # Process the matrix in blocks along its leading dimension.
        for k in range(0, n, block_size):
            end = k + block_size

            # Load current diagonal block A[k:end, k:end]
            # and update with contributions from previously computed blocks.
            A_kk_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(k, k), storage="shared")
            # The following if pads the matrix if it is not divisible by block_size
            if k + block_size > active_matrix_size:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block

                for i in range(num_iterations):
                    linear_index = tid_block + i * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = A_kk_tile[row, col]
                    if k + row >= active_matrix_size or k + col >= active_matrix_size:
                        value = wp.where(row == col, float(1), float(0))
                    A_kk_tile[row, col] = value

            if k > 0:
                for j in range(0, k, block_size):
                    L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j))
                    L_block_T = wp.tile_transpose(L_block)
                    L_L_T_block = wp.tile_matmul(L_block, L_block_T)
                    A_kk_tile -= L_L_T_block

            # Compute the Cholesky factorization for the block
            L_kk_tile = wp.tile_cholesky(A_kk_tile)
            wp.tile_store(L, L_kk_tile, offset=(k, k))

            # Process the blocks below the current block
            for i in range(end, n, block_size):
                A_ik_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(i, k), storage="shared")
                # The following if pads the matrix if it is not divisible by block_size
                if i + block_size > active_matrix_size or k + block_size > active_matrix_size:
                    num_tile_elements = block_size * block_size
                    num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block

                    for ii in range(num_iterations):
                        linear_index = tid_block + ii * num_threads_per_block
                        linear_index = linear_index % num_tile_elements
                        row = linear_index // block_size
                        col = linear_index % block_size
                        value = A_ik_tile[row, col]
                        if i + row >= active_matrix_size or k + col >= active_matrix_size:
                            value = wp.where(i + row == k + col, float(1), float(0))
                        A_ik_tile[row, col] = value

                if k > 0:
                    for j in range(0, k, block_size):
                        L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j))
                        L_2_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j))
                        L_T_tile = wp.tile_transpose(L_2_tile)
                        L_L_T_tile = wp.tile_matmul(L_tile, L_T_tile)
                        A_ik_tile -= L_L_T_tile

                t = wp.tile_transpose(A_ik_tile)
                tmp = wp.tile_lower_solve(L_kk_tile, t)
                sol_tile = wp.tile_transpose(tmp)

                wp.tile_store(L, sol_tile, offset=(i, k))

    return blocked_cholesky_kernel


@lru_cache(maxsize=None)
def create_blocked_cholesky_solve_kernel(block_size: int):
    @wp.kernel
    def blocked_cholesky_solve_kernel(
        L: wp.array(dtype=float, ndim=2),
        b: wp.array(dtype=float, ndim=2),
        x: wp.array(dtype=float, ndim=2),
        y: wp.array(dtype=float, ndim=2),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
    ):
        """
        Solves A x = b given the Cholesky factor L (A = L L^T) using
        blocked forward and backward substitution.

        b can be a vector or 2-D array with multiple right-hand sides.
        """

        active_matrix_size = active_matrix_size_arr[0]

        # Round up active_matrix_size to next multiple of block_size
        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        # Forward substitution: solve L y = b
        for i in range(0, n, block_size):
            i_end = i + block_size
            rhs_tile = wp.tile_load(b, shape=(block_size, 1), offset=(i, 0))
            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y, shape=(block_size, 1), offset=(j, 0))
                    Ly_block = wp.tile_matmul(L_block, y_block)
                    rhs_tile -= Ly_block
            L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i))
            y_tile = wp.tile_lower_solve(L_tile, rhs_tile)
            wp.tile_store(y, y_tile, offset=(i, 0))

        # Backward substitution: solve L^T x = y
        for i in range(n - block_size, -1, -block_size):
            i_start = i
            i_end = i_start + block_size
            rhs_tile = wp.tile_load(y, shape=(block_size, 1), offset=(i_start, 0))
            if i_end < n:
                for j in range(i_end, n, block_size):
                    L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(j, i_start))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x, shape=(block_size, 1), offset=(j, 0))
                    L_T_x_tile = wp.tile_matmul(L_T_tile, x_tile)
                    rhs_tile -= L_T_x_tile
            L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i_start, i_start))
            x_tile = wp.tile_upper_solve(wp.tile_transpose(L_tile), rhs_tile)
            wp.tile_store(x, x_tile, offset=(i_start, 0))

    return blocked_cholesky_solve_kernel


# TODO: Add batching support to solve multiple equation systems at once (one per thread block)
class BlockCholeskySolver:
    """
    A class for solving linear systems using the Cholesky factorization.
    """

    def __init__(self, max_num_equations: int, block_size=16, device="cuda"):
        # Round up max_num_equations to next multiple of block_size
        max_num_equations = ((max_num_equations + block_size - 1) // block_size) * block_size

        self.max_num_equations = max_num_equations
        self.device = device

        self.num_threads_per_block_factorize = 128
        self.num_threads_per_block_solve = 64
        self.active_matrix_size_int = -1

        self.block_size = block_size
        self.cholesky_kernel = create_blocked_cholesky_kernel(block_size)
        self.solve_kernel = create_blocked_cholesky_solve_kernel(block_size)

        # Allocate workspace arrays for factorization and solve
        self.L = wp.zeros(shape=(self.max_num_equations, self.max_num_equations), dtype=float, device=self.device)
        self.y = wp.zeros(shape=(self.max_num_equations, 1), dtype=float, device=self.device)  # temp memory
        self.active_matrix_size = wp.zeros(
            shape=(1,), dtype=int, device=self.device
        )  # array to hold active matrix size
        self.active_matrix_size_external = None

    def factorize(self, A: wp.array(dtype=float, ndim=2), num_active_equations: int):
        """
        Computes the Cholesky factorization of a symmetric positive definite matrix A in blocks.
        It returns a lower-triangular matrix L such that A = L L^T.
        """

        assert num_active_equations <= self.max_num_equations, (
            f"Number of active equations ({num_active_equations}) exceeds maximum allowed ({self.max_num_equations})"
        )

        padded_n = ((num_active_equations + self.block_size - 1) // self.block_size) * self.block_size

        # Verify input dimensions
        assert A.shape[0] == A.shape[1], "Matrix A must be square"
        assert A.shape[0] >= padded_n, f"Matrix A must be at least {padded_n}x{padded_n} to accommodate padding"

        self.active_matrix_size.zero_()
        wp.copy(self.active_matrix_size, wp.array([num_active_equations], dtype=int, device=self.device))

        self.factorize_dynamic(A, self.active_matrix_size)

        self.active_matrix_size_external = None
        self.active_matrix_size_int = num_active_equations

    def factorize_dynamic(self, A: wp.array(dtype=float, ndim=2), num_active_equations: wp.array(dtype=int, ndim=1)):
        """
        Computes the Cholesky factorization of a symmetric positive definite matrix A in blocks.
        It returns a lower-triangular matrix L such that A = L L^T.
        """

        self.active_matrix_size_external = num_active_equations
        self.active_matrix_size_int = -1

        wp.launch_tiled(
            self.cholesky_kernel,
            dim=1,
            inputs=[A, self.L, num_active_equations],
            block_dim=self.num_threads_per_block_factorize,
            device=self.device,
        )

    def solve(self, rhs: wp.array(dtype=float, ndim=2), result: wp.array(dtype=float, ndim=2)):
        """
        Solves A x = b given the Cholesky factor L (A = L L^T) using
        blocked forward and backward substitution.

        b can be a vector or 2-D array with multiple right-hand sides.
        """

        # Do safety checks but they can only be done if the matrix size is known on the host
        if self.active_matrix_size_int > 0:
            n = self.active_matrix_size_int
            padded_n = ((n + self.block_size - 1) // self.block_size) * self.block_size

            # Verify input dimensions
            assert rhs.shape[1] == 1, "Matrix b must be a column vector"
            assert rhs.shape[0] >= padded_n, f"Matrix b must be at least {padded_n}x1 to accommodate padding"

            assert result.shape[1] == 1, "Matrix result must be a column vector"
            assert result.shape[0] >= padded_n, f"Matrix result must be at least {padded_n}x1 to accommodate padding"

        if self.active_matrix_size_external is not None:
            matrix_size = self.active_matrix_size_external
        else:
            matrix_size = self.active_matrix_size

        # Then solve the system using blocked_cholesky_solve kernel
        wp.launch_tiled(
            self.solve_kernel,
            dim=1,
            inputs=[self.L, rhs, result, self.y, matrix_size],
            block_dim=self.num_threads_per_block_solve,
            device=self.device,
        )


class CholeskySolverNumPy:
    """
    A class for solving linear systems using the Cholesky factorization.
    """

    def __init__(self, max_num_equations: int):
        self.max_num_equations = max_num_equations
        self.num_active_equations = 0

        # Allocate workspace arrays for factorization and solve
        self.L = np.zeros((self.max_num_equations, self.max_num_equations))
        self.y = np.zeros((self.max_num_equations, 1))  # temp memory

    def factorize(self, A: np.ndarray, num_active_equations: int):
        """
        Computes the Cholesky factorization of a symmetric positive definite matrix A.
        It returns a lower-triangular matrix L such that A = L L^T.
        """
        assert num_active_equations <= self.max_num_equations, (
            f"Number of active equations ({num_active_equations}) exceeds maximum allowed ({self.max_num_equations})"
        )

        self.num_active_equations = num_active_equations

        # Verify input dimensions
        assert A.shape[0] == A.shape[1], "Matrix A must be square"
        assert A.shape[0] >= num_active_equations, (
            f"Matrix A must be at least {num_active_equations}x{num_active_equations}"
        )

        # Compute Cholesky factorization
        self.L[:num_active_equations, :num_active_equations] = np.linalg.cholesky(
            A[:num_active_equations, :num_active_equations]
        )

    def solve(self, rhs: np.ndarray, result: np.ndarray):
        """
        Solves A x = b given the Cholesky factor L (A = L L^T) using
        forward and backward substitution.

        b can be a vector or 2-D array with multiple right-hand sides.
        """
        assert self.num_active_equations <= self.max_num_equations, (
            f"Number of active equations ({self.num_active_equations}) exceeds maximum allowed ({self.max_num_equations})"
        )

        n = self.num_active_equations

        # Verify input dimensions
        assert rhs.shape[1] == 1, "Matrix b must be a column vector"
        assert rhs.shape[0] >= n, f"Matrix b must be at least {n}x1"

        assert result.shape[1] == 1, "Matrix result must be a column vector"
        assert result.shape[0] >= n, f"Matrix result must be at least {n}x1"

        # Forward substitution: L y = b
        self.y[:n] = np.linalg.solve(self.L[:n, :n], rhs[:n])

        # Backward substitution: L^T x = y
        result[:n] = np.linalg.solve(self.L[:n, :n].T, self.y[:n])


def test_cholesky_solver(n, warp_solver: BlockCholeskySolver, device: str = "cuda"):
    # Create a symmetric positive definite matrix
    rng = np.random.default_rng(0)
    A_full = rng.standard_normal((n, n))
    A_full = A_full @ A_full.T + n * np.eye(n)  # ensure SPD
    block_size = warp_solver.block_size

    # Pad matrix to make it divisible by block_size
    padded_n = ((n + block_size - 1) // block_size) * block_size
    padding = padded_n - n

    if padding > 0:
        # Pad the original matrix with random values while maintaining SPD
        A_padded = rng.standard_normal((padded_n, padded_n))
        A_padded[:n, :n] = A_full
        padding_block = rng.standard_normal((padding, padding))
        padding_block = padding_block @ padding_block.T + padding * np.eye(padding)
        A_padded[n:, n:] = padding_block
        A_padded[n:, :n] = rng.standard_normal((padding, n))
        A_padded[:n, n:] = A_padded[n:, :n].T  # Maintain symmetry
    else:
        A_padded = A_full

    # Create random RHS vector and pad
    b = rng.standard_normal(n)
    b_padded = rng.standard_normal(padded_n)
    b_padded[:n] = b

    print("\nSolving with NumPy:")
    # NumPy reference solution
    x = np.linalg.solve(A_full, b)
    L_full = np.linalg.cholesky(A_full)

    # Verify NumPy solution
    err = np.linalg.norm(A_full - L_full @ L_full.T)
    res_norm = np.linalg.norm(b - A_full @ x)
    print(f"Cholesky factorization error: {err:.3e}")
    print(f"Solution residual norm: {res_norm:.3e}")

    print("\nSolving with Warp kernels:")
    # Initialize Warp arrays
    A_wp = wp.array(A_padded, dtype=wp.float32, device=device)
    b_wp = wp.array(b_padded, dtype=wp.float32, device=device).reshape((padded_n, 1))
    x_wp = wp.zeros_like(b_wp)

    # Create and use the Cholesky solver
    warp_solver.factorize(A_wp, n)
    warp_solver.solve(b_wp, x_wp)
    wp.synchronize()

    # Get result back to CPU and verify
    x_warp = x_wp.numpy()[:n].squeeze()
    L_warp = warp_solver.L.numpy()

    # Verify Warp solution
    err_warp = np.linalg.norm(A_full - L_warp[:n, :n] @ L_warp[:n, :n].T)
    res_norm_warp = np.linalg.norm(b - A_full @ x_warp)
    diff_norm = np.linalg.norm(x - x_warp)

    print(f"Warp Cholesky factorization error: {err_warp:.3e}")
    print(f"Warp solution residual norm: {res_norm_warp:.3e}")
    print(f"Difference between CPU and GPU solutions: {diff_norm:.3e}")


@wp.kernel
def assign_int_kernel(arr: wp.array(dtype=int, ndim=1), value: int):
    """Assigns an integer value into the first element of an array"""
    arr[0] = value


def test_cholesky_solver_graph_capture():
    wp.clear_kernel_cache()

    max_equations = 1000

    # Create random SPD matrix A and random RHS b
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal((max_equations, max_equations))
    A_np = A_np @ A_np.T + np.eye(max_equations) * max_equations  # Make SPD
    b_np = rng.standard_normal((max_equations, 1))

    device = "cuda"

    with wp.ScopedDevice(device):
        warp_solver = BlockCholeskySolver(max_equations, block_size=32)

        # Create Warp arrays
        # Round up dimensions to next multiple of block size
        block_size = warp_solver.block_size
        padded_n = ((max_equations + block_size - 1) // block_size) * block_size

        # Create padded arrays initialized with zeros
        A_padded = np.zeros((padded_n, padded_n), dtype=np.float32)
        b_padded = np.zeros((padded_n, 1), dtype=np.float32)

        # Copy original data into padded arrays
        A_padded[:max_equations, :max_equations] = A_np
        b_padded[:max_equations, :] = b_np

        # Create Warp arrays from padded numpy arrays
        A_wp = wp.array(A_padded, dtype=wp.float32, ndim=2)
        b_wp = wp.array(b_padded, dtype=wp.float32, ndim=2)

        # Create result array
        x_wp = wp.zeros_like(b_wp)
        # Create array for equation system size
        n_wp = wp.array([1], dtype=wp.int32)

        # Create a stream for graph capture
        stream = wp.Stream(device)

        with wp.ScopedStream(stream):
            # Begin graph capture
            wp.capture_begin()
            try:
                # Loop through different system sizes
                for n in range(1, max_equations + 1):
                    # Update system size
                    wp.launch(assign_int_kernel, dim=1, inputs=[n_wp, n])

                    # Factorize A
                    warp_solver.factorize_dynamic(A_wp, n_wp)

                    # Solve system
                    warp_solver.solve(b_wp, x_wp)

            finally:
                # End graph capture
                graph = wp.capture_end()

            # Run the captured graph
            with wp.ScopedTimer("Launch graph", cuda_filter=wp.TIMING_GRAPH):
                wp.capture_launch(graph, stream=stream)

            wp.synchronize()
            print("Finished!")


if __name__ == "__main__":
    wp.clear_kernel_cache()

    test_graph_capture = False

    if test_graph_capture:
        test_cholesky_solver_graph_capture()

    else:
        device = "cpu"

        # Test equation sys  sizes
        sizes = [32, 70, 128, 192, 257, 320, 401, 1000]

        # Initialize solver once with max size
        warp_solver = BlockCholeskySolver(max(sizes), block_size=16, device=device)

        for n in sizes:
            print(f"\nTesting system size n = {n}")
            test_cholesky_solver(n, warp_solver, device)
