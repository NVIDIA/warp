Sparse Matrices
===============

.. currentmodule:: warp.sparse

Warp includes a sparse linear algebra module :mod:`warp.sparse` that implements common sparse matrix operations for simulation. The module provides efficient implementations of Block Sparse Row (BSR) matrices, with Compressed Sparse Row (CSR) matrices supported as a special case with 1x1 block size.

Working with Sparse Matrices
----------------------------

Creating Sparse Matrices
~~~~~~~~~~~~~~~~~~~~~~~~

The most common way to create a sparse matrix is using coordinate (COO) format triplets:

.. code-block:: python

    import warp as wp
    from warp.sparse import bsr_from_triplets

    # Create a 4x4 sparse matrix with 2x2 blocks
    rows = wp.array([0, 1, 2], dtype=int)  # Row indices
    cols = wp.array([1, 2, 3], dtype=int)  # Column indices
    vals = wp.array([...], dtype=float)    # Block values

    # Create BSR matrix
    A = bsr_from_triplets(
        rows_of_blocks=2,      # Number of rows of blocks
        cols_of_blocks=2,      # Number of columns of blocks
        rows=rows,            # Row indices
        columns=cols,         # Column indices
        values=vals           # Block values
    )

You can also create special matrices:

.. code-block:: python

    from warp.sparse import bsr_zeros, bsr_identity, bsr_diag

    # Create empty matrix
    A = bsr_zeros(rows_of_blocks=4, cols_of_blocks=4, block_type=wp.float32)

    # Create identity matrix
    I = bsr_identity(rows_of_blocks=4, block_type=wp.float32)

    # Create diagonal matrix
    D = bsr_diag(diag=wp.array([1.0, 2.0, 3.0, 4.0]))



Block Sizes
~~~~~~~~~~~

BSR matrices support different block sizes. For example:

.. code-block:: python

    # 1x1 blocks (CSR format)
    A = bsr_zeros(4, 4, block_type=wp.float32)

    # 3x3 blocks
    A = bsr_zeros(2, 2, block_type=wp.mat33)

    # Rectangular block sizes
    A = bsr_zeros(2, 3, block_type=wp.mat23)

The module also provides functions to convert between different block sizes and scalar types:

.. code-block:: python

    from warp.sparse import bsr_copy, bsr_assign

    # Convert block size from 2x2 to 1x1 (BSR to CSR)
    A_csr = bsr_copy(A, block_shape=(1, 1))

    # Convert block size from 1x1 to 2x2 (CSR to BSR)
    A_bsr = bsr_copy(A, block_shape=(2, 2))

    # Convert scalar type from float32 to float64
    A_double = bsr_copy(A, scalar_type=wp.float64)

    # Convert both block size and scalar type
    A_new = bsr_copy(A, block_shape=(2, 2), scalar_type=wp.float64)

    # In-place conversion using bsr_assign
    B = bsr_zeros(rows_of_blocks=4, cols_of_blocks=4, block_type=wp.mat22)
    bsr_assign(src=A, dest=B)  # Converts A to 2x2 blocks and stores in B

.. note:: When converting block sizes, the source and destination block dimensions must be compatible:

          - The new block dimensions must either be multiples or exact dividers of the original dimensions
          - The total matrix dimensions must be evenly divisible by the new block size

For example:

.. code-block:: python

    # Valid conversions:
    A = bsr_zeros(4, 4, block_type=wp.mat22)  # 8x8 matrix with 2x2 blocks
    B = bsr_copy(A, block_shape=(1, 1))       # 8x8 matrix with 1x1 blocks
    C = bsr_copy(A, block_shape=(4, 4))       # 8x8 matrix with 4x4 blocks

    # Invalid conversion (will raise ValueError):
    D = bsr_copy(A, block_shape=(3, 3))       # 3x3 blocks don't divide 8x8 evenly


Non-Zero Block Count
~~~~~~~~~~~~~~~~~~~~

The number of non-zero blocks in a BSR matrix is computed on the device and not automatically synchronized to the host to avoid performance overhead and allow graph capture. 
The :attr:`BsrMatrix.nnz` attribute of a BSR matrix is always an upper bound for the stored block array size.
For compact matrices, the actual active count is stored on the device at ``offsets[nrow]``.

To get the exact count on host, you can explicitly synchronize using :meth:`BsrMatrix.nnz_sync`:

.. code-block:: python

    # The nnz attribute is an upper bound
    upper_bound = A.nnz

    # Get offsets[nrow] on host
    storage_nnz = A.nnz_sync()

.. note::

    The :meth:`BsrMatrix.nnz_sync` method ensures that any ongoing transfer
    of ``offsets[nrow]`` from the device offsets array to the host has completed
    and updates the nnz upper bound. For compact matrices, this is the active
    non-zero block count. For padded matrices, this is the total row-capacity
    storage size, not necessarily the active block count.

.. note::

    During CPU APIC graph capture, :meth:`BsrMatrix.nnz_sync` performs its host
    readback outside the recorded stream when the matrix topology has not been
    changed by the capture. If the same capture has already recorded a topology
    update for that matrix, :meth:`BsrMatrix.nnz_sync` raises
    :exc:`NotImplementedError`; read the count after replay instead.

If the number of non-zeros has been changed from outside of the :mod:`warp.sparse` builtin functions, for instance by direct modifications to the offsets array, use the :meth:`BsrMatrix.notify_nnz_changed` method to ensure consistency.

Row Capacity and Padded Topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, sparse constructors and topology-changing operations produce compact BSR/CSR matrices. In a compact matrix, row ``r`` stores its active blocks in ``offsets[r]:offsets[r + 1]``.

Warp also supports matrices with reserved row capacity. These matrices use ``row_counts`` to mark the active block count of each row:

.. code-block:: text

    offsets[row]                       : offsets[row] + row_counts[row]  active blocks
    offsets[row] + row_counts[row]     : offsets[row + 1]                slack capacity
    offsets[row]                       : offsets[row + 1]                total row capacity

The required invariant is:

.. code-block:: text

    0 <= row_counts[row] <= offsets[row + 1] - offsets[row]

For compact matrices, ``row_counts`` may be unset. In that case sparse operations infer active row counts from adjacent offsets. For padded matrices, :attr:`BsrMatrix.nnz` remains the storage upper bound, not necessarily the active block count. Slack entries are ignored by sparse operations, and their column and value data is not part of the matrix contract. Use ``row_counts`` to identify the active entries in padded rows.

Use ``row_capacity`` with :func:`bsr_zeros` to allocate an empty padded matrix with reserved storage:

.. code-block:: python

    # Reserve eight block slots per row.
    A = bsr_zeros(rows, cols, block_type=wp.float32, row_capacity=8)

    # Or reserve per-row capacity from a Warp array.
    row_capacity = wp.array([2, 0, 4], dtype=int, device=device)
    A = bsr_zeros(3, cols, block_type=wp.float32, device=device, row_capacity=row_capacity)

Providing ``row_capacity`` implies ``topology="padded"`` unless an explicit topology is passed. Passing ``row_capacity`` with ``topology="compact"`` raises ``ValueError``.
The convenience constructor :func:`bsr_from_triplets` always builds compact storage. To build COO triplets into reserved row capacity, allocate with :func:`bsr_zeros` using ``row_capacity`` and then call :func:`bsr_set_from_triplets` with ``topology="padded"``.
Padded matrices created during CUDA graph capture follow normal graph-capture allocation semantics.
Inside CUDA capture, use a uniform integer ``row_capacity`` or pass an explicit ``nnz_capacity`` with per-row
``row_capacity`` arrays; otherwise construction requires a host readback of the total row capacity.

Operations that may change topology accept a ``topology`` policy where supported:

.. code-block:: text

    compact  rebuild compact topology, discarding row padding
    masked   keep the current active topology and update values only
    padded   write into existing per-row capacity

The ``"compact"`` policy is the default for builders and unmasked arithmetic. The ``"padded"`` policy is an allocation contract: each destination row must already have enough capacity for the result.
For :func:`bsr_mm`, ``reuse_topology=True`` is a compact-topology optimization that reuses product topology data stored in ``work_arrays`` from a previous call; it is only supported with ``topology="compact"``.
The older ``masked=True`` arguments are deprecated; pass ``topology="masked"`` instead.

For example, a compact source can be copied into an already padded destination:

.. code-block:: python

    from warp.sparse import bsr_assign, bsr_set_zero, bsr_zeros

    dest = bsr_zeros(src.nrow, src.ncol, src.values.dtype, device=src.device, row_capacity=src.ncol)
    bsr_set_zero(dest, topology="padded")       # keep row capacity, clear active rows
    bsr_assign(dest=dest, src=src, topology="padded")

``bsr_set_zero(..., topology="padded")`` keeps row capacity only when the matrix size is unchanged. If a new size is provided, the resized empty matrix starts with no row capacity.

Padded topology-changing operations ignore row-capacity overflow by default and record status on the destination matrix.
The status is sticky until cleared, so clear it before a checked operation when the caller needs to detect
insufficient row capacity from that operation:

.. code-block:: python

    from warp.sparse import BSR_STATUS_SUCCESS, bsr_axpy, bsr_axpy_work_arrays

    work = bsr_axpy_work_arrays()
    y.clear_status()
    bsr_axpy(x, y, topology="padded", work_arrays=work)
    if y.status_sync() != BSR_STATUS_SUCCESS:
        raise RuntimeError(y.status_message())

Functions such as :func:`bsr_assign`, :func:`bsr_set_from_triplets`, and :func:`bsr_set_transpose` report status on their destination matrix the same way:

.. code-block:: python

    from warp.sparse import BSR_STATUS_SUCCESS, bsr_set_transpose

    dest.clear_status()
    bsr_set_transpose(dest, src, topology="padded")
    if dest.status_sync() != BSR_STATUS_SUCCESS:
        raise RuntimeError(dest.status_message())

Because :meth:`BsrMatrix.status_sync` reads device status on the host, it raises during live CUDA graph capture.
Call it after the graph launch instead, keeping the matrix alive beyond the capture; :meth:`BsrMatrix.clear_status` is device-side and may be called inside the capture.

Row-ordered candidate entries can be compressed with :func:`bsr_compress`. With ``inplace=True``, matrices whose scalar type is ``float32`` or ``float64`` have each active row range sorted, duplicate columns accumulated, optional numerical zero blocks pruned, and ``row_counts`` updated in place using native compression support without ``O(nnz)`` temporary allocation. This path is not differentiable. Pass ``topology="compact"`` to additionally pack active blocks into compact row storage:

.. code-block:: python

    from warp.sparse import bsr_compress

    bsr_compress(A, inplace=True)
    bsr_compress(A, inplace=True, topology="compact")

With the default ``inplace=False``, :func:`bsr_compress` rebuilds the topology natively from the active row and column data, then accumulates values with differentiable Warp kernels. By default it packs active entries into compact row storage. Pass ``topology="padded"`` to preserve per-row capacity; compression cannot increase the active block count, so this padded mode does not report row-capacity overflow. The API does not guarantee that existing storage buffers are preserved.

.. code-block:: python

    from warp.sparse import bsr_compress

    bsr_compress(A)

When entries are generated directly into existing padded BSR row capacity, call :func:`bsr_compress` with ``inplace=True`` to sort and coalesce each active row range.

Converting back to COO Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can convert a BSR matrix back to coordinate (COO) format using the matrix's properties:

.. code-block:: python

    # Get row indices from compressed format
    nnz = A.nnz_sync()
    rows = A.uncompress_rows()[:nnz]  # Returns array of row indices for each block
    
    # Get column indices and values
    cols = A.columns[:nnz]            # Column indices for each block
    vals = A.values[:nnz]             # Block values

    # Now you have the COO format:
    # - rows[i] is the row index of block i
    # - cols[i] is the column index of block i
    # - vals[i] is the value of block i

For matrices with row padding, :meth:`BsrMatrix.uncompress_rows` returns an array sized to :attr:`BsrMatrix.nnz` and uses ``-1`` for slack entries. Use :func:`bsr_compress` first when a compact COO export is required:

.. code-block:: python

    from warp.sparse import bsr_compress

    bsr_compress(A)
    nnz = A.nnz_sync()
    rows = A.uncompress_rows()[:nnz]
    cols = A.columns[:nnz]
    vals = A.values[:nnz]


Matrix Operations
~~~~~~~~~~~~~~~~~

The module supports common matrix operations with overloaded operators:

.. code-block:: python

    # Matrix addition
    C = A + B

    # Matrix subtraction
    C = A - B

    # Scalar multiplication
    C = 2.0 * A

    # Matrix multiplication
    C = A @ B

    # Matrix-vector multiplication
    y = A @ x

    # Transpose
    AT = A.transpose()

    # In-place operations
    A += B
    A -= B
    A *= 2.0
    A @= B

For more control about memory allocations, you may use the underlying lower-level functions:

.. code-block:: python

    from warp.sparse import bsr_mm, bsr_mv, bsr_axpy, bsr_scale

    # Matrix-matrix multiplication
    # C = alpha * A @ B + beta * C
    bsr_mm(
        A, B, C,           # Input and output matrices
        alpha=1.0,         # Scale factor for A @ B
        beta=0.0,          # Scale factor for C
    )

    # Matrix-vector multiplication
    # y = alpha * A @ x + beta * y
    bsr_mv(
        A, x, y,           # Input matrix, vector, and output vector
        alpha=1.0,         # Scale factor for A @ x
        beta=0.0,          # Scale factor for y
    )

    # Matrix addition (in-place)
    # y = alpha * x + beta * y
    bsr_axpy(
        x, y,              # Input matrix and output matrix (modified in-place)
        alpha=1.0,         # Scale factor for x
        beta=1.0,          # Scale factor for y
    )

    # Matrix scaling
    # A = alpha * A
    bsr_scale(
        A,                 # Input/output matrix
        alpha,             # Scale factor
    )


Kernel-level utilities
~~~~~~~~~~~~~~~~~~~~~~

The following Warp functions are available for use in kernels:

* :func:`warp.sparse.bsr_row_index`
* :func:`warp.sparse.bsr_block_index`

The ``bsr_row_index`` and ``bsr_block_index`` helpers search compact storage
rows by default. Pass ``row_counts`` to their row-capacity overloads to ignore
padded slack storage.


.. _iterative-linear-solvers:

Iterative Linear Solvers
------------------------

The :mod:`warp.optim.linear` module provides several iterative linear solvers with optional preconditioning:

- Conjugate Gradient (CG)
- Conjugate Residual (CR)
- Bi-Conjugate Gradient Stabilized (BiCGSTAB)
- Generalized Minimal Residual (GMRES)

.. code-block:: python

    from warp.optim.linear import cg

    # Solve Ax = b
    x = cg(A, b, x=x, maxiter=100, tol=1e-6)

While primarily intended for sparse matrices, these solvers also work with dense linear operators provided as 2D Warp arrays.
Custom operators can be implemented using the :class:`warp.optim.linear.LinearOperator` interface.

For a complete listing of all sparse matrix functions and their signatures, see the :doc:`../api_reference/warp_sparse` API reference.
