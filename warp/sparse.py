import warp as wp
import warp.types
import warp.utils

from typing import Tuple, Any, Union


_struct_cache = dict()


class BsrMatrix:
    """Untyped base class for BSR and CSR matrices.

    Should not be constructed directly but through functions such as :func:`bsr_zeros`.

    Attributes:
        nrow (int): Number of rows of blocks
        ncol (int): Number of columns of blocks
        nnz (int):  Number of non-zero blocks: equal to `offsets[-1]`, cached on host for convenience
        offsets (wp.array(dtype=int)): Array of size at least 1 + nrows containing start and end offsets og blocks in each row
        columns (wp.array(dtype=int)): Array of size at least equal to nnz containing block column indices
        values (wp.array(dtype=dtype)): Array of size at least equal to nnz containing block values
    """

    @property
    def scalar_type(self) -> type:
        """Scalar type for each of the blocks' coefficients. FOr CSR matrices, this is equal to the block type"""
        return warp.types.type_scalar_type(self.values.dtype)

    @property
    def block_shape(self) -> Tuple[int, int]:
        """Shape of the individual blocks"""
        return getattr(self.values.dtype, "_shape_", (1, 1))

    @property
    def block_size(self) -> Tuple[int, int]:
        """Size of the individual blocks, i.e. number of rows per block times number of columsn per block"""
        return warp.types.type_length(self.values.dtype)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the matrix, i.e. number of rows/columns of blocks times number of rows/columsn per block"""
        block_shape = self.block_shape
        return (self.nrow * block_shape[0], self.ncol * block_shape[1])


def bsr_matrix_t(dtype: type):
    dtype = wp.types.type_to_warp(dtype)

    class BsrMatrixTyped(BsrMatrix):
        nrow: int
        """Number of rows of blocks"""
        ncol: int
        """Number of columns of blocks"""
        nnz: int
        """Number of non-zero blocks: equal to offsets[-1], cached on host for convenience"""
        offsets: wp.array(dtype=int)
        """Array of size at least 1 + nrows"""
        columns: wp.array(dtype=int)
        """Array of size at least equal to nnz"""
        values: wp.array(dtype=dtype)

    module = wp.get_module(BsrMatrix.__module__)

    if hasattr(dtype, "_shape_"):
        type_str = f"{warp.types.type_scalar_type(dtype).__name__}_{dtype._shape_[0]}_{dtype._shape_[1]}"
    else:
        type_str = dtype.__name__
    key = f"{BsrMatrix.__qualname__}_{type_str}"

    if key not in _struct_cache:
        _struct_cache[key] = wp.codegen.Struct(
            cls=BsrMatrixTyped,
            key=key,
            module=module,
        )

    return _struct_cache[key]


def bsr_zeros(
    rows_of_blocks: int, cols_of_blocks: int, block_type: type, device: wp.context.Devicelike = None
) -> BsrMatrix:
    """
    Constructs an empty BSR or CS matrix with the given shape
    """
    bsr = bsr_matrix_t(block_type)()

    bsr.nrow = rows_of_blocks
    bsr.ncol = cols_of_blocks
    bsr.nnz = 0
    bsr.columns = wp.empty(shape=(0,), dtype=int, device=device)
    bsr.values = wp.empty(shape=(0,), dtype=block_type, device=device)
    bsr.offsets = wp.zeros(shape=(bsr.nrow + 1,), dtype=int, device=device)

    return bsr


def _bsr_ensure_fits(bsr: BsrMatrix, nrow: int = None, nnz: int = None):
    if nrow is None:
        nrow = bsr.nrow
    if nnz is None:
        nnz = bsr.nnz

    if bsr.offsets.size < nrow + 1:
        bsr.offsets = wp.empty(shape=(nrow + 1,), dtype=int, device=bsr.offsets.device)
    if bsr.columns.size < nnz:
        bsr.columns = wp.empty(shape=(nnz,), dtype=int, device=bsr.columns.device)
    if bsr.values.size < nnz:
        bsr.values = wp.empty(shape=(nnz,), dtype=bsr.values.dtype, device=bsr.values.device)


def bsr_set_from_triplets(
    dest: BsrMatrix,
    rows: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    values: wp.array(dtype=Any),
):
    """
    Fills a BSR matrix `dest` with values defined by COO triplets `rows`, `columns`, `values`.

    Values must be either one-dimensional with data type identical to the `dest` matrix block times,
    or a 3d array with data type equal to the `dest` matrix scalar type.

    Previous blocks of `dest` are discarded.
    """

    if values.device != columns.device or values.device != rows.device or values.device != dest.values.device:
        raise ValueError("All arguments must reside on the same device")

    if values.shape[0] != rows.shape[0] or values.shape[0] != columns.shape[0]:
        raise ValueError("All triplet arrays must have the same length")

    # Accept either array1d(dtype) or contiguous array3d(scalar_type) as values
    if values.ndim == 1:
        if values.dtype != dest.values.dtype:
            raise ValueError("Values array type must correspond to that of dest matrix")
    elif values.ndim == 3:
        if values.shape[1:] != dest.block_shape:
            raise ValueError(
                f"Last two dimensions in values array ({values.shape[1:]}) shoudl correspond to matrix block shape {(dest.block_shape)})"
            )

        if warp.types.type_scalar_type(values.dtype) != dest.scalar_type:
            raise ValueError("Scalar type of values array should correspond to that of matrix")

        if not values.is_contiguous:
            raise ValueError("Multi-dimensional values array should be contiguous")
    else:
        raise ValueError("Number of dimension for values array should be 1 or 3")

    nnz = rows.shape[0]

    # Increase dest array sizes if needed
    _bsr_ensure_fits(dest, nnz=nnz)

    device = dest.values.device
    scalar_type = dest.scalar_type
    from warp.context import runtime

    if device.is_cpu:
        if scalar_type == wp.float32:
            native_func = runtime.core.bsr_matrix_from_triplets_float_host
        elif scalar_type == wp.float64:
            native_func = runtime.core.bsr_matrix_from_triplets_double_host
    else:
        if scalar_type == wp.float32:
            native_func = runtime.core.bsr_matrix_from_triplets_float_device
        elif scalar_type == wp.float64:
            native_func = runtime.core.bsr_matrix_from_triplets_double_device

    if not native_func:
        raise NotImplementedError(f"bsr_from_triplets not implemented for scalar type {scalar_type}")

    dest.nnz = native_func(
        dest.block_shape[0],
        dest.block_shape[1],
        dest.nrow,
        nnz,
        rows.ptr,
        columns.ptr,
        values.ptr,
        dest.offsets.ptr,
        dest.columns.ptr,
        dest.values.ptr,
    )


def bsr_assign(dest: BsrMatrix, src: BsrMatrix):
    """Copies the content of the `src` matrix to `dest`, possibly casting the block values."""

    if dest.values.device != src.values.device:
        raise ValueError("Source and destination matrices must reside on the same device")

    if dest.block_shape != src.block_shape:
        raise ValueError("Source and destination matrices must have the same block shape")

    dest.nrow = src.nrow
    dest.ncol = src.ncol
    dest.nnz = src.nnz

    _bsr_ensure_fits(dest)

    wp.copy(dest=dest.offsets, src=src.offsets, count=src.nrow + 1)
    wp.copy(dest=dest.columns, src=src.columns, count=src.nnz)

    warp.utils.array_cast(out_array=dest.values, in_array=src.values, count=src.nnz)


def bsr_copy(A: BsrMatrix, scalar_type=None):
    """Returns a copy of matrix A, possibly asting values to a new scalar type"""
    if scalar_type is None:
        block_type = A.values.dtype
    elif A.block_shape == (1, 1):
        block_type = scalar_type
    else:
        block_type = wp.types.matrix(shape=A.block_shape, dtype=scalar_type)

    copy = bsr_zeros(rows_of_blocks=A.nrow, cols_of_blocks=A.ncol, block_type=block_type, device=A.values.device)
    bsr_assign(dest=copy, src=A)
    return copy


def bsr_set_transpose(dest: BsrMatrix, src: BsrMatrix):
    """Assigns the transposed matrix `src` to matrix `dest`"""

    if dest.values.device != src.values.device:
        raise ValueError("All arguments must reside on the same device")

    if dest.scalar_type != src.scalar_type:
        raise ValueError("All arguments must have the same scalar type")

    if src.block_shape == (1, 1):
        transpose_block_shape = (1, 1)
    else:
        transpose_block_shape = src.block_shape[::-1]

    if dest.block_shape != transpose_block_shape:
        raise ValueError(f"Destination block shape must be {transpose_block_shape}")

    dest.nrow = src.ncol
    dest.ncol = src.nrow
    dest.nnz = src.nnz

    # Increase dest array sizes if needed
    _bsr_ensure_fits(dest)

    from warp.context import runtime

    if dest.values.device.is_cpu:
        if dest.scalar_type == wp.float32:
            native_func = runtime.core.bsr_transpose_float_host
        elif dest.scalar_type == wp.float64:
            native_func = runtime.core.bsr_transpose_double_host
    else:
        if dest.scalar_type == wp.float32:
            native_func = runtime.core.bsr_transpose_float_device
        elif dest.scalar_type == wp.float64:
            native_func = runtime.core.bsr_transpose_double_device

    if not native_func:
        raise NotImplementedError(f"bsr_set_transpose not implemented for scalar type {dest.scalar_type}")

    native_func(
        src.block_shape[0],
        src.block_shape[1],
        src.nrow,
        src.ncol,
        src.nnz,
        src.offsets.ptr,
        src.columns.ptr,
        src.values.ptr,
        dest.offsets.ptr,
        dest.columns.ptr,
        dest.values.ptr,
    )


def bsr_transposed(A: BsrMatrix):
    """Returns a copy of the transposed matrix `A`"""

    if A.block_shape == (1, 1):
        block_type = A.values.dtype
    else:
        block_type = wp.types.matrix(shape=A.block_shape[::-1], dtype=A.scalar_type)

    transposed = bsr_zeros(rows_of_blocks=A.ncol, cols_of_blocks=A.nrow, block_type=block_type, device=A.values.device)
    bsr_set_transpose(dest=transposed, src=A)
    return transposed


@wp.kernel
def _bsr_get_diag_kernel(
    A_offsets: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
    A_values: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    row = wp.tid()
    beg = A_offsets[row]
    end = A_offsets[row + 1]

    diag = wp.lower_bound(A_columns, beg, end, row)
    if A_columns[diag] == row:
        out[row] = A_values[diag]


def bsr_get_diag(A: BsrMatrix, out: wp.array = None):
    """Returns the block diagonal of a square sparse matrix"""
    if A.nrow != A.ncol:
        raise ValueError("bsr_get_diag is only available for square sparse matrices")

    if out is None:
        out = wp.zeros(shape=(A.nrow,), dtype=A.values.dtype, device=A.values.device)
    else:
        if out.dtype != A.values.dtype:
            raise ValueError(f"Output array must have type {A.values.dtype}")
        if out.device != A.values.device:
            raise ValueError(f"Output array must reside on device {A.values.device}")
        if out.shape[0] < A.nrow:
            raise ValueError(f"Output array must be of length at least {A.nrow}")

    wp.launch(
        kernel=_bsr_get_diag_kernel, dim=A.nrow, device=A.values.device, inputs=[A.offsets, A.columns, A.values, out]
    )

    return out


@wp.kernel
def _bsr_set_diag_kernel(
    A_offsets: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
):
    row = wp.tid()
    A_offsets[row + 1] = row + 1
    A_columns[row] = row

    if row == 0:
        A_offsets[0] = 0


def bsr_set_diag(A: BsrMatrix, diag: wp.array):
    """Sets A as a block-diagonal square matrix"""

    A.nrow = diag.shape[0]
    A.ncol = diag.shape[0]
    A.nnz = diag.shape[0]

    A.values = diag
    if A.columns.size < A.nrow:
        A.columns = wp.empty(shape=(A.nrow,), dtype=int, device=diag.device)
    if A.offsets.size < A.nrow + 1:
        A.offsets = wp.empty(shape=(A.nrow + 1,), dtype=int, device=diag.device)

    wp.launch(kernel=_bsr_set_diag_kernel, dim=A.nrow, device=A.values.device, inputs=[A.offsets, A.columns])


def bsr_diag(diag: wp.array):
    """Creates a square block-diagonal BSR matrix from the values array `diag`"""
    A = bsr_zeros(rows_of_blocks=diag.shape[0], cols_of_blocks=diag.shape[0], block_type=diag.dtype, device=diag.device)
    bsr_set_diag(A, diag)
    return A


@wp.kernel
def _bsr_get_block_row(dest_offset: int, bsr_offsets: wp.array(dtype=int), rows: wp.array(dtype=int)):
    i = wp.tid()

    row = wp.lower_bound(bsr_offsets, i + 1) - 1
    rows[dest_offset + i] = row


@wp.kernel
def _bsr_axpy_add_block(
    src_offset: int,
    scale: Any,
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    dst_offsets: wp.array(dtype=int),
    dst_columns: wp.array(dtype=int),
    src_values: wp.array(dtype=Any),
    dst_values: wp.array(dtype=Any),
):
    i = wp.tid()
    row = rows[i + src_offset]
    col = cols[i + src_offset]
    beg = dst_offsets[row]
    end = dst_offsets[row + 1]

    block = wp.lower_bound(dst_columns, beg, end, col)

    dst_values[block] = dst_values[block] + scale * src_values[i]


def bsr_axpy(x: BsrMatrix, y: BsrMatrix, alpha: float = 1.0, beta: float = 1.0):
    """
    Performs the operation `y := alpha * X + beta * y` on BSR matrices `x` and `y`
    """

    if y is None:
        y = bsr_zeros(x.nrow, x.ncol, block_type=x.block_type, device=x.values.device)
        beta = 0.0

    device = y.values.device

    if x.values.device != y.values.device:
        raise ValueError("All arguments must reside on the same device")

    if x.scalar_type != y.scalar_type or x.block_shape != y.block_shape:
        raise ValueError("Matrices must have the same block type")

    if x.nrow != y.nrow or x.ncol != y.ncol:
        raise ValueError("Matrices must have the same number of rows and columns")

    alpha = y.scalar_type(alpha)
    beta = y.scalar_type(beta)

    sum_nnz = x.nnz + y.nnz
    sum_rows = wp.empty(shape=(sum_nnz), dtype=int, device=device)
    sum_cols = wp.empty(shape=(sum_nnz), dtype=int, device=device)

    if y.nnz > 0:
        wp.copy(sum_cols, y.columns, 0, 0, y.nnz)
        wp.launch(kernel=_bsr_get_block_row, device=device, dim=y.nnz, inputs=[0, y.offsets, sum_rows])

    if x.nnz > 0:
        wp.copy(sum_cols, x.columns, y.nnz, 0, x.nnz)
        wp.launch(kernel=_bsr_get_block_row, device=device, dim=x.nnz, inputs=[y.nnz, x.offsets, sum_rows])

    # Increase dest array sizes if needed
    if y.columns.shape[0] < sum_nnz:
        y.columns = wp.empty(shape=(sum_nnz,), dtype=int, device=device)

    from warp.context import runtime

    if device.is_cpu:
        native_func = runtime.core.bsr_matrix_from_triplets_float_host
    else:
        native_func = runtime.core.bsr_matrix_from_triplets_float_device

    sum_nnz = native_func(
        y.block_shape[0],
        y.block_shape[1],
        y.nrow,
        sum_nnz,
        sum_rows.ptr,
        sum_cols.ptr,
        0,
        y.offsets.ptr,
        y.columns.ptr,
        0,
    )

    sum_values = wp.zeros(shape=(sum_nnz,), dtype=y.values.dtype, device=device)

    wp.launch(
        kernel=_bsr_axpy_add_block,
        device=device,
        dim=y.nnz,
        inputs=[0, beta, sum_rows, sum_cols, y.offsets, y.columns, y.values, sum_values],
    )
    wp.launch(
        kernel=_bsr_axpy_add_block,
        device=device,
        dim=x.nnz,
        inputs=[y.nnz, alpha, sum_rows, sum_cols, y.offsets, y.columns, x.values, sum_values],
    )

    y.values = sum_values
    y.nnz = sum_nnz

    return y


@wp.kernel
def _bsr_mm_count_coeffs(
    z_nnz: int,
    x_offsets: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    counts: wp.array(dtype=int),
):
    row = wp.tid()
    count = int(0)

    x_beg = x_offsets[row]
    x_end = x_offsets[row + 1]

    for x_block in range(x_beg, x_end):
        x_col = x_columns[x_block]
        count += y_offsets[x_col + 1] - y_offsets[x_col]

    counts[row + 1] = count

    if row == 0:
        counts[0] = z_nnz


@wp.kernel
def _bsr_mm_list_coeffs(
    x_offsets: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    mm_offsets: wp.array(dtype=int),
    mm_rows: wp.array(dtype=int),
    mm_cols: wp.array(dtype=int),
):
    row = wp.tid()
    mm_block = mm_offsets[row]

    x_beg = x_offsets[row]
    x_end = x_offsets[row + 1]

    for x_block in range(x_beg, x_end):
        x_col = x_columns[x_block]

        y_beg = y_offsets[x_col]
        y_end = y_offsets[x_col + 1]
        for y_block in range(y_beg, y_end):
            mm_cols[mm_block] = y_columns[y_block]
            mm_rows[mm_block] = row
            mm_block += 1


@wp.kernel
def _bsr_mm_compute_values(
    alpha: Any,
    x_offsets: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    x_values: wp.array(dtype=Any),
    y_offsets: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    y_values: wp.array(dtype=Any),
    mm_offsets: wp.array(dtype=int),
    mm_cols: wp.array(dtype=int),
    mm_values: wp.array(dtype=Any),
):
    row = wp.tid()
    mm_beg = mm_offsets[row]
    mm_end = mm_offsets[row + 1]

    x_beg = x_offsets[row]
    x_end = x_offsets[row + 1]
    for x_block in range(x_beg, x_end):
        x_col = x_columns[x_block]
        ax_val = alpha * x_values[x_block]

        y_beg = y_offsets[x_col]
        y_end = y_offsets[x_col + 1]

        for y_block in range(y_beg, y_end):
            mm_block = wp.lower_bound(mm_cols, mm_beg, mm_end, y_columns[y_block])
            mm_values[mm_block] = mm_values[mm_block] + ax_val * y_values[y_block]


_pinned_temp_count_buffer = {}


def _get_pinned_temp_count_buffer(device):
    device = str(device)
    if device not in _pinned_temp_count_buffer:
        _pinned_temp_count_buffer[device] = wp.empty(shape=(1,), dtype=int, pinned=True, device="cpu")

    return _pinned_temp_count_buffer[device]


def bsr_mm(x: BsrMatrix, y: BsrMatrix, z: BsrMatrix = None, alpha: float = 1.0, beta: float = 0.0):
    """
    Performs the operation `z := alpha * X * Y + beta * z` on BSR matrices `x`, `y` and `z`
    """

    if z is None:
        z_block_shape = (x.block_shape[0], y.block_shape[1])
        if z_block_shape == (1, 1):
            z_block_type = x.scalar_type
        else:
            z_block_type = wp.types.matrix(shape=z_block_shape, dtype=x.scalar_type)
        z = bsr_zeros(x.nrow, y.ncol, block_type=z_block_type, device=x.values.device)
        beta = 0.0

    if x.values.device != y.values.device or x.values.device != z.values.device:
        raise ValueError("All arguments must reside on the same device")

    if x.scalar_type != y.scalar_type or x.scalar_type != z.scalar_type:
        raise ValueError("Matrices must have the same scalar type")

    if x.block_shape[0] != z.block_shape[0] or y.block_shape[1] != z.block_shape[1]:
        raise ValueError("Incompatible blocks sizes for matrix multiplication")

    if x.nrow != z.nrow or z.ncol != y.ncol:
        raise ValueError("Incompatible number of rows/columns for matrix multiplication")

    device = z.values.device

    alpha = z.scalar_type(alpha)
    beta = z.scalar_type(beta)

    # Prefix sum of number of (unmerged) mm blocks per row
    mm_row_counts = wp.empty(shape=(z.nrow + 1,), dtype=int, device=device)
    wp.launch(
        kernel=_bsr_mm_count_coeffs,
        device=device,
        dim=z.nrow,
        inputs=[z.nnz, x.offsets, x.columns, y.offsets, mm_row_counts],
    )
    warp.utils.array_scan(mm_row_counts, mm_row_counts)

    # Get back total counts on host
    if device.is_cuda:
        mm_tot_count = _get_pinned_temp_count_buffer(device)
        wp.copy(dest=mm_tot_count, src=mm_row_counts, src_offset=z.nrow, count=1)
        wp.synchronize_stream(wp.get_stream())
        mm_nnz = int(mm_tot_count.numpy()[0])
    else:
        mm_nnz = int(mm_row_counts.numpy()[z.nrow])

    mm_rows = wp.empty(shape=(mm_nnz), dtype=int, device=device)
    mm_cols = wp.empty(shape=(mm_nnz), dtype=int, device=device)

    # Copy z rows columns
    wp.copy(mm_cols, z.columns, 0, 0, z.nnz)
    wp.launch(kernel=_bsr_get_block_row, device=device, dim=z.nnz, inputs=[0, z.offsets, mm_rows])

    # Fill unmerged mm blocks rows and columns
    wp.launch(
        kernel=_bsr_mm_list_coeffs,
        device=device,
        dim=z.nrow,
        inputs=[x.offsets, x.columns, y.offsets, y.columns, mm_row_counts, mm_rows, mm_cols],
    )

    # Increase dest array sizes if needed
    if z.columns.shape[0] < mm_nnz:
        z.columns = wp.empty(shape=(mm_nnz,), dtype=int, device=device)

    from warp.context import runtime

    if device.is_cpu:
        native_func = runtime.core.bsr_matrix_from_triplets_float_host
    else:
        native_func = runtime.core.bsr_matrix_from_triplets_float_device

    mm_nnz = native_func(
        z.block_shape[0],
        z.block_shape[1],
        z.nrow,
        mm_nnz,
        mm_rows.ptr,
        mm_cols.ptr,
        0,
        z.offsets.ptr,
        z.columns.ptr,
        0,
    )

    mm_values = wp.zeros(shape=(mm_nnz,), dtype=z.values.dtype, device=device)

    # Copy blocks from z
    wp.launch(
        kernel=_bsr_axpy_add_block,
        device=device,
        dim=z.nnz,
        inputs=[0, beta, mm_rows, mm_cols, z.offsets, z.columns, z.values, mm_values],
    )

    # Add mm blocks
    wp.launch(
        kernel=_bsr_mm_compute_values,
        device=device,
        dim=z.nrow,
        inputs=[alpha, x.offsets, x.columns, x.values, y.offsets, y.columns, y.values, z.offsets, z.columns, mm_values],
    )

    z.values = mm_values
    z.nnz = mm_nnz

    return z


@wp.kernel
def _bsr_mv_kernel(
    alpha: Any,
    A_offsets: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
    A_values: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    beta: Any,
    y: wp.array(dtype=Any),
):
    row = wp.tid()
    beg = A_offsets[row]
    end = A_offsets[row + 1]

    yr = y[row]
    v = yr - yr  # WAR to get zero with correct type
    for block in range(beg, end):
        v = v + A_values[block] * x[A_columns[block]]

    y[row] = beta * yr + alpha * v


def bsr_mv(A: BsrMatrix, x: wp.array, y: wp.array, alpha: float = 1.0, beta: float = 0.0):
    """
    Naive implementation of sparse matrix-vector product, `y := alpha * A * x + beta * y`.
    """
    alpha = A.scalar_type(alpha)
    beta = A.scalar_type(beta)

    # if A.scalar_type != x.dtype or A.scalar_type != y.dtype:
    #    raise ValueError("A, x and y must have the same data types")

    if A.values.device != x.device or A.values.device != y.device:
        raise ValueError("A, x and y must reide on the same device")

    if x.shape[0] != A.ncol:
        raise ValueError("Number of columns of A must match number of rows of x")
    if y.shape[0] != A.nrow:
        raise ValueError("Number of rows of A must match number of rows of y")

    # Promote scalar vectors to length-1 vecs
    block_shape = A.block_shape
    if block_shape != (1, 1):
        if block_shape[0] == 1:
            if y.dtype == A.scalar_type:
                y = y.view(dtype=wp.vec(length=1, dtype=A.scalar_type))
        if block_shape[1] == 1:
            if x.dtype == A.scalar_type:
                x = x.view(dtype=wp.vec(length=1, dtype=A.scalar_type))

    wp.launch(
        kernel=_bsr_mv_kernel,
        device=A.values.device,
        dim=A.nrow,
        inputs=[alpha, A.offsets, A.columns, A.values, x, beta, y],
    )
