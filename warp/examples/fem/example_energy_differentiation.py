# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble an energy and its first and second variations from one integrand."""

import numpy as np

import warp as wp
import warp.fem as fem


@fem.integrand
def energy(s: fem.Sample, u: fem.Field):
    value = u(s)
    return 0.5 * value * value + 0.25 * value * value * value * value


def _to_dense(matrix):
    """Convert a BSR matrix to a dense array, using only its active entries."""
    nnz = matrix.nnz_sync()
    block_rows, block_cols = matrix.block_shape
    rows = matrix.uncompress_rows().numpy()[:nnz]
    columns = matrix.columns.numpy()[:nnz]
    values = matrix.values.numpy()[:nnz].reshape(nnz, block_rows, block_cols)

    # Scatter every block at once, with the block row and column expanded to scalar indices
    row_index = rows[:, None, None] * block_rows + np.arange(block_rows)[None, :, None]
    col_index = columns[:, None, None] * block_cols + np.arange(block_cols)[None, None, :]
    dense = np.zeros((matrix.nrow * block_rows, matrix.ncol * block_cols))
    np.add.at(dense, (row_index, col_index), values)
    return dense


def main():
    geometry = fem.Grid2D(res=wp.vec2i(2))
    domain = fem.Cells(geometry)
    space = fem.make_polynomial_space(geometry, degree=1)
    current = space.make_field()
    current.dof_values.fill_(0.5)
    test = fem.make_test(space, domain=domain)
    trial = fem.make_trial(space, domain=domain)
    fd_options = {"test_step": 0.01, "trial_step": 0.01}

    total = fem.integrate(energy, domain=domain, fields={"u": current})

    # Forward-mode jets give exact derivatives of the energy
    residual = fem.integrate(energy, fields={"u": fem.bind(current, test=test)}, differentiation="jets")
    tangent = fem.integrate(energy, fields={"u": fem.bind(current, test=test, trial=trial)}, differentiation="jets")

    # Centered finite differences use the same bindings and approximate the same derivatives
    fd_residual = fem.integrate(
        energy,
        fields={"u": fem.bind(current, test=test)},
        differentiation="finite_difference",
        differentiation_options=fd_options,
    )
    fd_tangent = fem.integrate(
        energy,
        fields={"u": fem.bind(current, test=test, trial=trial)},
        differentiation="finite_difference",
        differentiation_options=fd_options,
    )

    print(f"Energy: {total}")
    print(f"Residual: {residual.numpy()}")
    print(f"Tangent nonzero blocks: {tangent.nnz_sync()}")
    print(f"Finite-difference residual error: {np.max(np.abs(fd_residual.numpy() - residual.numpy())):.2e}")
    # Compare as dense matrices, since the raw value arrays may hold unused capacity
    error = np.max(np.abs(_to_dense(fd_tangent) - _to_dense(tangent)))
    print(f"Finite-difference tangent error: {error:.2e}")


if __name__ == "__main__":
    main()
