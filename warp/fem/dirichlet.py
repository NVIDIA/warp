from typing import Any, Optional

import warp as wp
from warp.fem.utils import array_axpy, symmetric_eigenvalues_qr
from warp.sparse import BsrMatrix, bsr_assign, bsr_axpy, bsr_copy, bsr_mm, bsr_mv
from warp.types import type_is_matrix, type_length


def normalize_dirichlet_projector(projector_matrix: BsrMatrix, fixed_value: Optional[wp.array] = None):
    """
    Scale projector so that it becomes idempotent, and apply the same scaling to fixed_value if provided
    """

    if projector_matrix.nrow < projector_matrix.nnz or projector_matrix.ncol != projector_matrix.nrow:
        raise ValueError("Projector must be a square diagonal matrix, with at most one non-zero block per row")

    # Cast blocks to matrix type if necessary
    projector_values = projector_matrix.values
    if not type_is_matrix(projector_values.dtype):
        projector_values = wp.array(
            data=None,
            ptr=projector_values.ptr,
            capacity=projector_values.capacity,
            device=projector_values.device,
            dtype=wp.mat(shape=projector_matrix.block_shape, dtype=projector_matrix.scalar_type),
            shape=projector_values.shape[0],
        )

    if fixed_value is None:
        wp.launch(
            kernel=_normalize_dirichlet_projector_kernel,
            dim=projector_matrix.nrow,
            device=projector_values.device,
            inputs=[projector_matrix.offsets, projector_matrix.columns, projector_values],
        )

    else:
        if fixed_value.shape[0] != projector_matrix.nrow:
            raise ValueError("Fixed value array must be of length equal to the number of rows of blocks")

        if type_length(fixed_value.dtype) == 1:
            # array of scalars, convert to 1d array of vectors
            fixed_value = wp.array(
                data=None,
                ptr=fixed_value.ptr,
                capacity=fixed_value.capacity,
                device=fixed_value.device,
                dtype=wp.vec(length=projector_matrix.block_shape[0], dtype=projector_matrix.scalar_type),
                shape=fixed_value.shape[0],
            )

        wp.launch(
            kernel=_normalize_dirichlet_projector_and_values_kernel,
            dim=projector_matrix.nrow,
            device=projector_values.device,
            inputs=[projector_matrix.offsets, projector_matrix.columns, projector_values, fixed_value],
        )


def project_system_rhs(
    system_matrix: BsrMatrix, system_rhs: wp.array, projector_matrix: BsrMatrix, fixed_value: Optional[wp.array] = None
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


def project_system_matrix(system_matrix: BsrMatrix, projector_matrix: BsrMatrix):
    """Projects the right-hand-side of a linear system to enforce Dirichlet boundary conditions

    ``system = (I - projector) * system * (I - projector) + projector``
    """

    complement_system = bsr_copy(system_matrix)
    bsr_mm(x=projector_matrix, y=system_matrix, z=complement_system, alpha=-1.0, beta=1.0)

    bsr_assign(dest=system_matrix, src=complement_system)
    bsr_axpy(x=projector_matrix, y=system_matrix)
    bsr_mm(x=complement_system, y=projector_matrix, z=system_matrix, alpha=-1.0, beta=1.0)


def project_linear_system(
    system_matrix: BsrMatrix,
    system_rhs: wp.array,
    projector_matrix: BsrMatrix,
    fixed_value: Optional[wp.array] = None,
    normalize_projector=True,
):
    """
    Projects both the left-hand-side and right-hand-side of a linear system to enforce Dirichlet boundary conditions

    If normalize_projector is True, first apply scaling so that the projector_matrix is idempotent
    """
    if normalize_projector:
        normalize_dirichlet_projector(projector_matrix, fixed_value)

    project_system_rhs(system_matrix, system_rhs, projector_matrix, fixed_value)
    project_system_matrix(system_matrix, projector_matrix)


@wp.func
def _normalize_projector_and_value(A: Any, b: Any):
    # Do a modal decomposition of the left and right hand side,
    # Make lhs an orthogonal projection and apply corresponding scaling to righ-hand-side

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
    columns: wp.array(dtype=int),
    block_values: wp.array(dtype=Any),
    fixed_values: wp.array(dtype=Any),
):
    row = wp.tid()

    beg = offsets[row]
    end = offsets[row + 1]

    if beg == end:
        return

    diag = wp.lower_bound(columns, beg, end, row)

    if diag < end and columns[diag] == row:
        P = block_values[diag]

        P_norm, v_norm = _normalize_projector_and_value(P, fixed_values[row])

        block_values[diag] = P_norm
        fixed_values[row] = v_norm


@wp.kernel
def _normalize_dirichlet_projector_kernel(
    offsets: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    block_values: wp.array(dtype=Any),
):
    row = wp.tid()

    beg = offsets[row]
    end = offsets[row + 1]

    if beg == end:
        return

    diag = wp.lower_bound(columns, beg, end, row)

    if diag < end and columns[diag] == row:
        P = block_values[diag]

        P_norm, v_norm = _normalize_projector_and_value(P, type(P[0])())
        block_values[diag] = P_norm
