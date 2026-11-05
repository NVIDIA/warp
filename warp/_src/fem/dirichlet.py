# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import warp as wp
from warp._src.fem.linalg import array_axpy, symmetric_eigenvalues_qr
from warp._src.types import type_is_matrix, type_size
from warp.sparse import (
    BsrMatrix,
    bsr_assign,
    bsr_axpy,
    bsr_block_index,
    bsr_copy,
    bsr_mm,
    bsr_mv,
    bsr_row_index,
)


def normalize_dirichlet_projector(projector_matrix: BsrMatrix, fixed_value: wp.array | None = None):
    """
    Scale projector so that it becomes idempotent, and apply the same scaling to fixed_value if provided

    Only the diagonal blocks are normalized; off-diagonal blocks are left unchanged.
    """

    if projector_matrix.ncol != projector_matrix.nrow:
        raise ValueError("Projector must be a square block-diagonal matrix")

    # Cast blocks to matrix type if necessary
    projector_values = projector_matrix.values
    if not type_is_matrix(projector_values.dtype):
        projector_values = wp.array(
            data=None,
            ptr=projector_values.ptr,
            capacity=projector_values.capacity,
            device=projector_values.device,
            dtype=wp.types.matrix(shape=projector_matrix.block_shape, dtype=projector_matrix.scalar_type),
            shape=projector_values.shape[0],
        )

    if fixed_value is None:
        wp.launch(
            kernel=_normalize_dirichlet_projector_kernel,
            dim=projector_matrix.nrow,
            device=projector_values.device,
            inputs=[projector_matrix.offsets, projector_matrix.row_counts, projector_matrix.columns, projector_values],
        )

    else:
        if fixed_value.shape[0] != projector_matrix.nrow:
            raise ValueError("Fixed value array must be of length equal to the number of rows of blocks")

        if type_size(fixed_value.dtype) == 1:
            # array of scalars, convert to 1d array of vectors
            fixed_value = wp.array(
                data=None,
                ptr=fixed_value.ptr,
                capacity=fixed_value.capacity,
                device=fixed_value.device,
                dtype=wp.types.vector(length=projector_matrix.block_shape[0], dtype=projector_matrix.scalar_type),
                shape=fixed_value.shape[0],
            )

        wp.launch(
            kernel=_normalize_dirichlet_projector_and_values_kernel,
            dim=projector_matrix.nrow,
            device=projector_values.device,
            inputs=[
                projector_matrix.offsets,
                projector_matrix.row_counts,
                projector_matrix.columns,
                projector_values,
                fixed_value,
            ],
        )


def project_system_rhs(
    system_matrix: BsrMatrix, system_rhs: wp.array, projector_matrix: BsrMatrix, fixed_value: wp.array | None = None
):
    """Projects the right-hand-side of a linear system to enforce Dirichlet boundary conditions

    ``rhs = (I - projector) * ( rhs - system * projector * fixed_value) + projector * fixed_value``
    """

    rhs_tmp = wp.empty_like(system_rhs)
    rhs_tmp.assign(system_rhs)

    if fixed_value is None:
        system_rhs.zero_()
    else:
        bsr_mv(A=projector_matrix, x=fixed_value, y=system_rhs, alpha=1.0, beta=0.0)

    bsr_mv(A=system_matrix, x=system_rhs, y=rhs_tmp, alpha=-1.0, beta=1.0)

    # here rhs_tmp = system_rhs - system_matrix * projector * fixed_value
    # system_rhs = projector * fixed_value
    array_axpy(x=rhs_tmp, y=system_rhs, alpha=1.0, beta=1.0)
    bsr_mv(A=projector_matrix, x=rhs_tmp, y=system_rhs, alpha=-1.0, beta=1.0)


def project_system_matrix(
    system_matrix: BsrMatrix,
    projector_matrix: BsrMatrix,
    projector_structure: Literal["general", "block_diagonal"] = "general",
):
    """Projects the left-hand-side of a linear system to enforce Dirichlet boundary conditions

    ``system = (I - projector) * system * (I - projector) + projector``

    By default, the projection uses general sparse matrix products, which support any projector
    but allocate temporary storage and read back block counts. Dirichlet projectors assembled with
    ``assembly="nodal"`` are block-diagonal: every block of the result then only depends on the
    matching block of ``system_matrix`` and on the projector's diagonal blocks, and passing
    ``projector_structure="block_diagonal"`` projects ``system_matrix`` in place with a single
    kernel, keeping its sparsity pattern.

    Args:
        system_matrix: Square system matrix, modified in place.
        projector_matrix: Square projector matrix with the same block shape as ``system_matrix``.
        projector_structure: How to project the system matrix.

          - ``"general"`` uses general sparse matrix products, which also support projectors with
            off-diagonal blocks.
          - ``"block_diagonal"`` uses the single-kernel projection. It does not allocate or read
            back, so it can be used during graph capture. The projector must be block-diagonal,
            and ``system_matrix`` must store a diagonal block in every row where the projector is
            nonzero: off-diagonal projector blocks are ignored, and a row where the projector is
            nonzero but ``system_matrix`` has no diagonal block does not get its projector block
            added.
    """

    _validate_projector_structure(system_matrix, projector_matrix, projector_structure)

    if projector_structure == "block_diagonal":
        _project_system_matrix_block_diagonal(system_matrix, projector_matrix)
        return

    complement_system = bsr_copy(system_matrix)
    bsr_mm(x=projector_matrix, y=system_matrix, z=complement_system, alpha=-1.0, beta=1.0)

    bsr_assign(dest=system_matrix, src=complement_system)
    bsr_axpy(x=projector_matrix, y=system_matrix)
    bsr_mm(x=complement_system, y=projector_matrix, z=system_matrix, alpha=-1.0, beta=1.0)


def project_linear_system(
    system_matrix: BsrMatrix,
    system_rhs: wp.array,
    projector_matrix: BsrMatrix,
    fixed_value: wp.array | None = None,
    normalize_projector=True,
    projector_structure: Literal["general", "block_diagonal"] = "general",
):
    """
    Projects both the left-hand-side and right-hand-side of a linear system to enforce Dirichlet boundary conditions

    If normalize_projector is True, first apply scaling so that the projector_matrix is idempotent.
    ``projector_structure`` selects how the system matrix is projected; pass ``"block_diagonal"`` for
    the in-place, graph-capturable projection of block-diagonal projectors. See :func:`project_system_matrix`.
    """
    # Validate before the right-hand side and projector are modified
    _validate_projector_structure(system_matrix, projector_matrix, projector_structure)

    if normalize_projector:
        normalize_dirichlet_projector(projector_matrix, fixed_value)

    project_system_rhs(system_matrix, system_rhs, projector_matrix, fixed_value)
    project_system_matrix(system_matrix, projector_matrix, projector_structure=projector_structure)


def _validate_projector_structure(system_matrix: BsrMatrix, projector_matrix: BsrMatrix, projector_structure: str):
    if projector_structure not in ("block_diagonal", "general"):
        raise ValueError(f"Invalid projector structure '{projector_structure}'")

    if projector_structure == "block_diagonal" and (
        system_matrix.nrow != system_matrix.ncol
        or projector_matrix.nrow != system_matrix.nrow
        or projector_matrix.ncol != system_matrix.ncol
        or projector_matrix.block_shape != system_matrix.block_shape
        or projector_matrix.block_shape[0] != projector_matrix.block_shape[1]
    ):
        raise ValueError("Block-diagonal projection requires square matrices with matching shapes and square blocks")


def _project_system_matrix_block_diagonal(system_matrix: BsrMatrix, projector_matrix: BsrMatrix):
    if system_matrix.nnz == 0:
        return

    wp.launch(
        _project_system_matrix_block_diagonal_kernel,
        dim=system_matrix.nnz,
        device=system_matrix.device,
        inputs=[
            system_matrix.nrow,
            system_matrix.offsets,
            system_matrix.row_counts,
            system_matrix.columns,
            _as_matrix_values(system_matrix),
            projector_matrix.offsets,
            projector_matrix.row_counts,
            projector_matrix.columns,
            _as_matrix_values(projector_matrix),
        ],
    )


def _as_matrix_values(matrix: BsrMatrix) -> wp.array:
    """View the block values of a matrix as an array of matrices, including 1x1 blocks."""
    values = matrix.values
    if type_is_matrix(values.dtype):
        return values
    return wp.array(
        data=None,
        ptr=values.ptr,
        capacity=values.capacity,
        device=values.device,
        dtype=wp.types.matrix(shape=matrix.block_shape, dtype=matrix.scalar_type),
        shape=values.shape[0],
    )


@wp.kernel(enable_backward=False, module="unique")
def _project_system_matrix_block_diagonal_kernel(
    row_count: int,
    system_offsets: wp.array(dtype=int),
    system_row_counts: wp.array(dtype=int),
    system_columns: wp.array(dtype=int),
    system_values: wp.array(dtype=Any),
    projector_offsets: wp.array(dtype=int),
    projector_row_counts: wp.array(dtype=int),
    projector_columns: wp.array(dtype=int),
    projector_values: wp.array(dtype=Any),
):
    block = wp.tid()

    row = bsr_row_index(system_offsets, row_count, block, system_row_counts)
    if row == -1:
        return

    col = system_columns[block]
    value = system_values[block]

    # Same operations and order as the general path:
    # complement = (I - P) K, then result = complement + P - complement P
    complement = value
    row_projector_block = bsr_block_index(row, row, projector_offsets, projector_columns, projector_row_counts)
    if row_projector_block != -1:
        complement = value - type(value)(projector_values[row_projector_block]) * value

    result = complement
    col_projector_block = bsr_block_index(col, col, projector_offsets, projector_columns, projector_row_counts)
    if col_projector_block != -1:
        col_projector = type(value)(projector_values[col_projector_block])
        if row == col:
            result += col_projector
        result -= complement * col_projector

    system_values[block] = result


@wp.func
def _normalize_projector_and_value(A: Any, b: Any):
    # Do a modal decomposition of the left and right hand side,
    # Make lhs an orthogonal projection and apply corresponding scaling to right-hand side

    eps = wp.trace(A) * A.dtype(1.0e-6)

    diag, ev = symmetric_eigenvalues_qr(A, eps * eps)

    zero = A.dtype(0)
    A_unitary = type(A)(zero)
    b_normalized = type(b)(zero)

    for k in range(b.length):
        if diag[k] > eps:  # ignore small eigenmodes
            v = ev[k]
            A_unitary += wp.outer(v, v)
            b_normalized += wp.dot(v, b) / diag[k] * v

    return A_unitary, b_normalized


@wp.kernel
def _normalize_dirichlet_projector_and_values_kernel(
    offsets: wp.array(dtype=int),
    row_counts: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    block_values: wp.array(dtype=Any),
    fixed_values: wp.array(dtype=Any),
):
    row = wp.tid()

    diag = bsr_block_index(row, row, offsets, columns, row_counts)

    if diag != -1:
        P = block_values[diag]

        P_norm, v_norm = _normalize_projector_and_value(P, fixed_values[row])

        block_values[diag] = P_norm
        fixed_values[row] = v_norm


@wp.kernel
def _normalize_dirichlet_projector_kernel(
    offsets: wp.array(dtype=int),
    row_counts: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    block_values: wp.array(dtype=Any),
):
    row = wp.tid()

    diag = bsr_block_index(row, row, offsets, columns, row_counts)

    if diag != -1:
        P = block_values[diag]

        P_norm, _v_norm = _normalize_projector_and_value(P, type(P[0])())
        block_values[diag] = P_norm
