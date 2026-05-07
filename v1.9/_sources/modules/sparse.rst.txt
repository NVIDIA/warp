warp.sparse
===============================

.. currentmodule:: warp.sparse

Warp includes a sparse linear algebra module ``warp.sparse`` that implements common sparse matrix operations for simulation. The module provides efficient implementations of Block Sparse Row (BSR) matrices, with Compressed Sparse Row (CSR) matrices supported as a special case with 1x1 block size.

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
The `nnz` attribute of a BSR matrix is always an upper bound for the number of non-zero blocks, but the actual count is stored on the device at `offsets[nrow]`.

To get the exact count on host, you can explicitly synchronize using the `nnz_sync()` method:

.. code-block:: python

    # The nnz attribute is an upper bound
    upper_bound = A.nnz

    # Get the exact number of non-zero blocks
    exact_nnz = A.nnz_sync()

.. note:: The `nnz_sync()` method ensures that any ongoing transfer of the exact nnz number from the device offsets array to the host has completed and updates the nnz upper bound. This synchronization is only necessary when you need the exact count - for most operations, the upper bound is sufficient.

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

    # Matrix addition
    # C = alpha * A + beta * B
    bsr_axpy(
        A, B, C,           # Input and output matrices
        alpha=1.0,         # Scale factor for A
        beta=1.0,          # Scale factor for B
    )

    # Matrix scaling
    # A = alpha * A
    bsr_scale(
        A,                 # Input/output matrix
        alpha,             # Scale factor
    )


Reference
~~~~~~~~~
.. automodule:: warp.sparse
    :members:


.. _iterative-linear-solvers:

Iterative Linear Solvers
------------------------

.. currentmodule:: warp.optim.linear

Warp provides several iterative linear solvers with optional preconditioning:

- Conjugate Gradient (CG)
- Conjugate Residual (CR)
- Bi-Conjugate Gradient Stabilized (BiCGSTAB)
- Generalized Minimal Residual (GMRES)

.. code-block:: python

    from warp.optim.linear import cg

    # Solve Ax = b
    x = cg(A, b, max_iter=100, tol=1e-6)

.. note:: While primarily intended for sparse matrices, these solvers also work with dense linear operators provided as 2D Warp arrays. Custom operators can be implemented using the :class:`LinearOperator` interface.

.. automodule:: warp.optim.linear
    :members: