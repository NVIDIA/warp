Tiles
=====

.. currentmodule:: warp

Block-based programming models such as those in OpenAI Triton have proved to be effective ways of expressing high-performance kernels that can leverage cooperative operations on modern GPUs.
With Warp 1.5.0 [1]_, developers now have access to new tile-based programming primitives in Warp kernels.
Leveraging cuBLASDx and cuFFTDx, these new tools provide developers with efficient matrix multiplication and Fourier transforms for accelerated simulation and scientific computing. 

Requirements
------------

Tile-based operations are fully supported on versions of Warp built with CUDA Toolkit 12.6.3 or newer due to the use
of the MathDx library to back linear-algebra tile operations
like :func:`wp.tile_cholesky <warp._src.lang.tile_cholesky>`,
:func:`wp.tile_fft <warp._src.lang.tile_fft>`, and :func:`wp.tile_matmul <warp._src.lang.tile_matmul>`.
See `Building with MathDx`_ for more details when building the Warp locally with support for
these linear-algebra tile operations.

Execution Model
---------------

Warp's execution model allows users to specify a grid of logical threads with up to 4 dimensions for kernel execution at launch time.
With the introduction of tile primitives, users can now specify the *block size* for kernel launches,
which partitions the thread grid into smaller sets of threads that are executed on a single compute unit.

Inside kernels, tile operations are executed cooperatively across each block of threads, allowing them to take advantage of efficient memory access, local memory, and dedicated hardware units like `Tensor Cores <https://www.nvidia.com/en-us/data-center/tensor-cores/>`__.

In the following example, we launch a grid of threads where each block is responsible for loading a row of data from a 2D array and computing its sum:

.. testcode::
    :skipif: wp.get_cuda_device_count() == 0
    
    TILE_SIZE = wp.constant(256)
    TILE_THREADS = 64

    @wp.kernel
    def compute(a: wp.array2d[float], b: wp.array2d[float]):

        # obtain our block index
        i = wp.tid()

        # load a row from global memory
        t = wp.tile_load(a[i], TILE_SIZE)

        # cooperatively compute the sum of the tile elements; s is a single element tile
        s = wp.tile_sum(t)

        # store s in global memory
        wp.tile_store(b[i], s)

    N = 10

    a_np = np.arange(N).reshape(-1, 1) * np.ones((1, 256), dtype=float)
    a = wp.array(a_np, dtype=float)
    b = wp.zeros((N,1), dtype=float)

    wp.launch_tiled(compute, dim=[a.shape[0]], inputs=[a, b], block_dim=TILE_THREADS)

    print(f"b = {b[:,0]}")

.. testoutput::

    b = [   0.  256.  512.  768. 1024. 1280. 1536. 1792. 2048. 2304.]
    
Here, we have used the new :func:`warp.launch_tiled` function which assigns ``TILE_THREADS`` threads to each of the elements in the launch grid. Each block of ``TILE_THREADS`` threads then loads an entire row of 256 values from the global memory array and computes its sum (cooperatively).


Tile Properties
---------------

In Warp, tile objects are arrays of data where the tile elements may be scalars, vectors, matrices, or user-defined structures. Tiles can have up to four dimensions. We can load 2D tiles directly from 2D global memory arrays as follows:

.. code:: python
    
    TILE_M = wp.constant(16)
    TILE_N = wp.constant(16)    
    TILE_THREADS = 64

    @wp.kernel
    def compute(a: array2d(dtype=float)):
        
        # obtain our 2d block index
        i, j = wp.tid()

        # load a 2d tile from global memory
        t = wp.tile_load(array, shape=(TILE_M, TILE_N), offset=(i*TILE_M, j*TILE_N))
        s = wp.tile_sum(t)
        ...

    wp.launch_tiled(compute, dim=[a.shape[0]/TILE_M, a.shape[1]/TILE_N], inputs=[a], block_dim=TILE_THREADS)
    
Here, we divide the array ``a`` into 2D tiles of shape 16 x 16.
Each block cooperatively loads a tile from the input array and computes its sum.

Tile Storage
------------

When tiles are created, they are placed in either *register* or *shared* memory.
In general, Warp tries to determine the best storage location for tiles.
By default, tiles are allocated in register storage, but some operations such as matrix multiplication may migrate data from register to shared as necessary.

Register Tiles
^^^^^^^^^^^^^^

Values in register tiles are stored across the entire block.
For example, if the block dimension at launch is set to 64, a register tile with ``shape=(1, 256)`` will result in each thread storing 4 elements.
Register-based storage is the fastest storage on most hardware, but an individual thread cannot randomly access data that is assigned to another thread efficiently 
because the tile storage is spread across the threads in the block.
For this reason, operations on tiles tend to be expressed as higher-level maps, reductions, and reshaping operations that may transfer values through shared memory.

Shared Memory Tiles
^^^^^^^^^^^^^^^^^^^

Some operations like matrix multiplication require access to an entire tile of values.
In this case, the tile data may be stored in shared memory, which allows efficient random access.
Warp will automatically migrate tiles to shared memory as necessary for specific operations, some of which
require thread synchronization. For instance, when writing to a tile element, (e.g. ``tile[row, col] = val``),
Warp does not a priori know if the current thread is assigned to ``(row, col)``, so the tile is stored in shared
memory to allow random accessing. A thread synchronization must also follow, to prevent downstream race conditions.

Note that shared memory is a limited resource, and so the tile size must be set appropriately to avoid exceeding the hardware limitations.
Otherwise, kernel compilation may fail.

Note that shared memory tile allocations are guaranteed to be 16-byte aligned.

Shared Tile Load Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When loading tiles into shared memory, Warp uses a three-tier cascade to maximize memory bandwidth:

1. **Vectorized float4 (2D+ tiles)** — Reinterprets memory as 128-bit ``float4`` values for wide loads.
   Requires the last dimension to be float4-aligned (``last_dim * sizeof(T) % 16 == 0``), a contiguous
   source array, 16-byte aligned base address, tile in-bounds, and float4-aligned outer-dimension strides.
   Use ``aligned=True`` on :func:`tile_load <warp._src.lang.tile_load>` to skip runtime alignment checks
   when you can guarantee these conditions.

2. **Coalesced byte-copy (large element types, ``sizeof(T) > 16``)** — For types like ``mat33``, ``mat44``,
   and ``mat66``, copies raw bytes as ``float*`` with thread-striped access for perfect memory coalescing.
   This avoids the scalar path's per-element strided access pattern where each element load strides across
   multiple cache lines. Requires a contiguous source array, the tile must fit within array bounds, and the
   tile must span the full inner dimensions.

3. **Scalar fallback (universal)** — Per-element loads with bounds-checked zero-padding. Handles
   non-contiguous arrays, partial tiles, transposed layouts, and CPU execution. 1D tiles do not use the
   vectorized float4 path, but may use the coalesced byte-copy path for large element types.

The same cascade applies to :func:`tile_store <warp._src.lang.tile_store>`.
See :ref:`vectorized_tile_loads` for detailed conditions and optimization tips.

Example: General Matrix Multiply (GEMM)
---------------------------------------

.. code:: python

    import numpy as np
    import warp as wp

    # tile size
    TILE_M = wp.constant(8)
    TILE_N = wp.constant(4)
    TILE_K = wp.constant(8)

    # num threads per-tile
    TILE_THREADS = 64

    @wp.kernel
    def tile_gemm(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
        
        # output tile index
        i, j = wp.tid()

        sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

        M = A.shape[0]
        N = B.shape[1]
        K = A.shape[1]

        count = int(K / TILE_K)

        for k in range(0, count):
            a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i*TILE_M, k*TILE_K))
            b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k*TILE_K, j*TILE_N))

            # sum += a*b
            wp.tile_matmul(a, b, sum)

        wp.tile_store(C, sum, offset=(i*TILE_M, j*TILE_N))



    if __name__ == "__main__":

        # generate some tile aligned matrix dimensions
        M = TILE_M * 7
        K = TILE_K * 6
        N = TILE_N * 5

        rng = np.random.default_rng(42)
        A = rng.random((M, K), dtype=np.float32)
        B = rng.random((K, N), dtype=np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        A_wp = wp.array(A)
        B_wp = wp.array(B)
        C_wp = wp.array(C)

        with wp.Tape() as tape:
            wp.launch_tiled(
                tile_gemm,
                dim=(int(M / TILE_M), int(N / TILE_N)),
                inputs=[A_wp, B_wp, C_wp],
                block_dim=TILE_THREADS)

        np.testing.assert_allclose(C_wp.numpy(), A @ B, rtol=1e-3)

        print("Example matrix multiplication passed")


Tile Operations
---------------


Construction
^^^^^^^^^^^^

* :func:`tile_zeros <warp._src.lang.tile_zeros>`
* :func:`tile_ones <warp._src.lang.tile_ones>`
* :func:`tile_full <warp._src.lang.tile_full>`
* :func:`tile_from_thread <warp._src.lang.tile_from_thread>`
* :func:`tile_randi <warp._src.lang.tile_randi>`
* :func:`tile_randf <warp._src.lang.tile_randf>`
* :func:`tile_arange <warp._src.lang.tile_arange>`
* :func:`tile <warp._src.lang.tile>`
* :func:`untile <warp._src.lang.untile>`
* :func:`tile_view <warp._src.lang.tile_view>`
* :func:`tile_broadcast <warp._src.lang.tile_broadcast>`
* :func:`tile_reshape <warp._src.lang.tile_reshape>`
* :func:`tile_squeeze <warp._src.lang.tile_squeeze>`
* :func:`tile_astype <warp._src.lang.tile_astype>`

Load/Store
^^^^^^^^^^

* :func:`tile_load <warp._src.lang.tile_load>`
* :func:`tile_load_indexed <warp._src.lang.tile_load_indexed>`
* :func:`tile_store <warp._src.lang.tile_store>`
* :func:`tile_store_indexed <warp._src.lang.tile_store_indexed>`
* :func:`tile_atomic_add <warp._src.lang.tile_atomic_add>`
* :func:`tile_atomic_add_indexed <warp._src.lang.tile_atomic_add_indexed>`
* :func:`tile_assign <warp._src.lang.tile_assign>`
* :func:`tile_extract <warp._src.lang.tile_extract>`
* :func:`tile_scatter_add <warp._src.lang.tile_scatter_add>`
* :func:`tile_scatter_masked <warp._src.lang.tile_scatter_masked>`

Maps/Reductions
^^^^^^^^^^^^^^^

* :func:`tile_map <warp._src.lang.tile_map>`
* :func:`tile_reduce <warp._src.lang.tile_reduce>`
* :func:`tile_sum <warp._src.lang.tile_sum>`
* :func:`tile_min <warp._src.lang.tile_min>`
* :func:`tile_max <warp._src.lang.tile_max>`
* :func:`tile_argmin <warp._src.lang.tile_argmin>`
* :func:`tile_argmax <warp._src.lang.tile_argmax>`
* :func:`tile_sort <warp._src.lang.tile_sort>`
* :func:`tile_scan_inclusive <warp._src.lang.tile_scan_inclusive>`
* :func:`tile_scan_exclusive <warp._src.lang.tile_scan_exclusive>`
* :func:`tile_scan_max_inclusive <warp._src.lang.tile_scan_max_inclusive>`
* :func:`tile_scan_min_inclusive <warp._src.lang.tile_scan_min_inclusive>`

Arithmetic
^^^^^^^^^^

Tiles support standard Python arithmetic operators for element-wise operations.
These operations are cooperative and execute across all threads in the block.

**Addition and Subtraction**

The ``+`` and ``-`` operators perform element-wise addition and subtraction between two tiles
of the same shape and dtype:

.. code:: python

    @wp.kernel
    def add_sub_example(arr_a: wp.array[float], arr_b: wp.array[float]):
        a = wp.tile_load(arr_a, shape=TILE_SIZE)
        b = wp.tile_load(arr_b, shape=TILE_SIZE)

        c = a + b    # element-wise addition
        d = a - b    # element-wise subtraction
        e = -a       # element-wise negation

Tiles also support the ``+=`` and ``-=`` in-place operators.

**Multiplication** (``*``)

The ``*`` operator supports three forms:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Expression
     - Operand Types
     - Result
   * - ``tile * tile``
     - Both tiles have matching shape; at least one must have scalar dtype
     - Element-wise multiplication
   * - ``tile * constant``
     - Tile and a scalar, vector, or matrix constant
     - Multiply each tile element by the constant
   * - ``constant * tile``
     - A scalar, vector, or matrix constant and a tile
     - Multiply each tile element by the constant

When one operand is a tile and the other is a constant, the constant is broadcast to all elements.
At least one of the tile's element type or the constant type must be a scalar. The underlying
scalar types must match. For example:

.. code:: python

    @wp.kernel
    def mul_example(arr: wp.array[float]):
        a = wp.tile_load(arr, shape=TILE_SIZE)     # a tile of floats

        # tile * tile (element-wise)
        b = a * a

        # tile * scalar
        c = a * 2.0

        # scalar * tile
        d = 2.0 * a

        # float tile * vec3f constant -> vec3f tile
        e = a * wp.vec3(1.0, 2.0, 3.0)

        # vec3f constant * float tile -> vec3f tile
        f = wp.vec3(1.0, 2.0, 3.0) * a

**Division** (``/``)

The ``/`` operator supports three forms with the same type rules as multiplication:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Expression
     - Operand Types
     - Result
   * - ``tile / tile``
     - Both tiles have matching shape; at least one must have scalar dtype
     - Element-wise division
   * - ``tile / constant``
     - Tile and a scalar, vector, or matrix constant
     - Divide each tile element by the constant
   * - ``constant / tile``
     - A scalar, vector, or matrix constant and a tile
     - Divide the constant by each tile element

As with multiplication, at least one of the tile's element type or the constant type must be
a scalar, and the underlying scalar types must match. For example:

.. code:: python

    @wp.kernel
    def div_example(arr: wp.array[float], vec_arr: wp.array[wp.vec3]):
        a = wp.tile_load(arr, shape=TILE_SIZE)     # a tile of floats

        # tile / tile (element-wise)
        b = a / a

        # tile / scalar
        c = a / 2.0

        # scalar / tile (divides constant by each element)
        d = 1.0 / a

        # vec3f tile / scalar
        v = wp.tile_load(vec_arr, shape=TILE_SIZE)  # a tile of vec3f
        e = v / 2.0

        # float tile / vec3f constant -> vec3f tile
        f = a / wp.vec3(1.0, 2.0, 4.0)

        # vec3f constant / float tile -> vec3f tile
        g = wp.vec3(1.0, 2.0, 4.0) / a

**Type Promotion Rules**

The following table summarizes the result type for ``*`` and ``/`` operations between
tiles and constants:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Tile Element Type
     - Constant Type
     - Result Type
   * - ``float``
     - ``float``
     - ``float``
   * - ``float``
     - ``vec3f``
     - ``vec3f``
   * - ``float``
     - ``mat33f``
     - ``mat33f``
   * - ``vec3f``
     - ``float``
     - ``vec3f``
   * - ``mat33f``
     - ``float``
     - ``mat33f``

Combinations where both the tile element type and the constant type are non-scalar
(e.g., ``tile<vec3f> * vec3f``) are not supported. Use :func:`tile_map <warp._src.lang.tile_map>`
with :func:`wp.cw_mul <warp._src.lang.cw_mul>` or :func:`wp.cw_div <warp._src.lang.cw_div>` for
component-wise vector/matrix operations.

All arithmetic operators support automatic differentiation.

Linear Algebra
^^^^^^^^^^^^^^

* :func:`tile_matmul <warp._src.lang.tile_matmul>`
* :func:`tile_transpose <warp._src.lang.tile_transpose>`
* :func:`tile_fft <warp._src.lang.tile_fft>`
* :func:`tile_ifft <warp._src.lang.tile_ifft>`
* :func:`tile_cholesky <warp._src.lang.tile_cholesky>`
* :func:`tile_cholesky_inplace <warp._src.lang.tile_cholesky_inplace>`
* :func:`tile_cholesky_solve <warp._src.lang.tile_cholesky_solve>`
* :func:`tile_cholesky_solve_inplace <warp._src.lang.tile_cholesky_solve_inplace>`
* :func:`tile_lower_solve <warp._src.lang.tile_lower_solve>`
* :func:`tile_lower_solve_inplace <warp._src.lang.tile_lower_solve_inplace>`
* :func:`tile_upper_solve <warp._src.lang.tile_upper_solve>`
* :func:`tile_upper_solve_inplace <warp._src.lang.tile_upper_solve_inplace>`
* :func:`tile_diag_add <warp._src.lang.tile_diag_add>`

Spatial Queries
^^^^^^^^^^^^^^^

* :func:`tile_bvh_query_aabb <warp._src.lang.tile_bvh_query_aabb>`
* :func:`tile_bvh_query_ray <warp._src.lang.tile_bvh_query_ray>`
* :func:`tile_bvh_query_next <warp._src.lang.tile_bvh_query_next>`
* :func:`tile_mesh_query_aabb <warp._src.lang.tile_mesh_query_aabb>`
* :func:`tile_mesh_query_aabb_next <warp._src.lang.tile_mesh_query_aabb_next>`
* :func:`tile_query_valid <warp._src.lang.tile_query_valid>`

Stack
^^^^^

* :func:`tile_stack <warp._src.lang.tile_stack>`
* :func:`tile_stack_push <warp._src.lang.tile_stack_push>`
* :func:`tile_stack_pop <warp._src.lang.tile_stack_pop>`
* :func:`tile_stack_clear <warp._src.lang.tile_stack_clear>`
* :func:`tile_stack_count <warp._src.lang.tile_stack_count>`

Tile Stack
----------

Tile stacks provide a cooperative, block-scoped stack data structure in shared memory.
They enable patterns such as stream compaction and dynamic load balancing where threads
within a block need to collectively build and consume a shared work-list.

Most tile stack operations are **cooperative** — every thread in the block must call them,
even if a particular thread has no data to contribute, because they contain internal
synchronization barriers. The exception is :func:`wp.tile_stack_count
<warp._src.lang.tile_stack_count>`, which contains no barrier and may be called by a
single thread or from within a divergent branch.

Creating a Stack
^^^^^^^^^^^^^^^^

Use :func:`wp.tile_stack() <warp._src.lang.tile_stack>` to allocate a stack in shared memory.
The ``capacity`` must be a compile-time constant and ``dtype`` specifies the element type:

.. code:: python

    CAPACITY = wp.constant(256)

    @wp.kernel
    def my_kernel(data: wp.array(dtype=float)):
        i, j = wp.tid()
        s = wp.tile_stack(capacity=CAPACITY, dtype=float)
        ...

Push and Pop
^^^^^^^^^^^^

:func:`wp.tile_stack_push() <warp._src.lang.tile_stack_push>` conditionally pushes a value
onto the stack. Each thread provides a value and a boolean ``has_value`` flag — only threads
with ``has_value=True`` write to the stack. The function returns the slot index where the
value was written, or ``-1`` if the thread did not push (either because ``has_value`` was
``False`` or because the stack overflowed):

.. code:: python

    idx = wp.tile_stack_push(s, value, keep)

:func:`wp.tile_stack_pop() <warp._src.lang.tile_stack_pop>` removes values from the stack.
Each calling thread races for a slot. It returns a tuple ``(value, slot)`` where ``slot``
identifies which stack slot was popped (in ``[0, capacity-1]`` when non-negative), or
``-1`` if the stack was empty — consistent with
:func:`wp.tile_stack_push() <warp._src.lang.tile_stack_push>` which also returns ``-1`` on failure:

.. code:: python

    value, slot = wp.tile_stack_pop(s)
    if slot != -1:
        out[slot] = value

Clear and Count
^^^^^^^^^^^^^^^

:func:`wp.tile_stack_clear() <warp._src.lang.tile_stack_clear>` resets the stack count to zero,
allowing it to be reused within the same kernel invocation:

.. code:: python

    wp.tile_stack_clear(s)

:func:`wp.tile_stack_count() <warp._src.lang.tile_stack_count>` returns the current number of
elements. Unlike the other operations it is **not** cooperative — it contains no
synchronization barrier and may be called by a single thread or from within a divergent
branch. It is safe to call after any push, pop, or clear, all of which end with a barrier
that makes the count visible:

.. code:: python

    n = wp.tile_stack_count(s)

.. code:: python

    # may be called by a single thread or from within a branch
    if j == 0:
        n = wp.tile_stack_count(s)

Example: Stream Compaction
^^^^^^^^^^^^^^^^^^^^^^^^^^

A common use case is filtering elements that satisfy a condition. Because
``tile_stack`` is a within-block primitive, the slot returned by
:func:`wp.tile_stack_pop() <warp._src.lang.tile_stack_pop>` is compact only
within a single tile — for a single-tile launch this maps directly to a compact
output array:

.. code:: python

    BLOCK_DIM = 256
    CAPACITY = wp.constant(BLOCK_DIM)  # at most one output per thread

    @wp.kernel
    def compact_kernel(data: wp.array(dtype=float), out: wp.array(dtype=float)):
        _i, j = wp.tid()

        val = data[j]
        keep = val > 0.5

        # Cooperative push — only threads with keep=True write to the stack
        s = wp.tile_stack(capacity=CAPACITY, dtype=float)
        wp.tile_stack_push(s, val, keep)

        # Each thread races for a slot. The returned slot is a unique compact
        # index in [0, capacity-1] — use it directly as the output index.
        result, slot = wp.tile_stack_pop(s)
        if slot != -1:
            out[slot] = result

    n = BLOCK_DIM
    data = wp.array(..., dtype=float)
    out = wp.zeros(n, dtype=float)  # sized to CAPACITY: safe for any filter result
    wp.launch_tiled(compact_kernel, dim=[1], inputs=[data, out], block_dim=BLOCK_DIM)

.. note::

    In a multi-tile launch, each tile's pop slots are compact only within that
    tile and would overlap across tiles. Multi-tile compaction requires additional
    coordination, such as a 2D output array indexed by ``[tile, slot]`` or a
    global atomic counter to claim output indices.

.. note::

    Tile stacks are allocated in shared memory, which is a limited resource.
    The capacity should be chosen to avoid exceeding hardware limitations.
    The stack supports any Warp data type including scalars, vectors, and matrices.

.. note::

    Tile stack operations are not differentiable and cannot be used in the
    backward pass of a differentiable kernel.

Tiles and SIMT Code
-------------------

Traditionally, Warp kernels are primarily written in the SIMT programming model, where each thread's execution happens independently. 
Tiles, on the other hand, allow threads to work **cooperatively** to perform operations.
Warp exposes the :func:`warp.tile <warp._src.lang.tile>`, and :func:`warp.untile <warp._src.lang.untile>` methods to convert data between per-thread value types and
the equivalent tile representation. For example:

.. code:: python
    
    TILE_THREADS = 64

    @wp.kernel
    def compute():
        i = wp.tid()

        # perform some per-thread computation
        x = i*2.0 + wp.sin(float(i))

        # tile the value x across the block
        # returns a tile with shape=(1, TILE_THREADS)
        t = wp.tile(x)
        ...

    # launch as regular SIMT kernel
    wp.launch(compute, dim=[N], inputs=[], block_dim=TILE_THREADS)

In this example, we have launched a regular SIMT grid with ``N`` logical threads using :func:`wp.launch() <warp.launch>`.
The kernel performs some per-thread computations and then converts the scalar ``x`` value into a tile object using :func:`warp.tile <warp._src.lang.tile>`.
This function takes a single value as input and returns a tile with the same dimensions as the number of threads in the block (as set by the ``block_dim`` argument in :func:`wp.launch() <warp.launch>`),
which implies that each thread in the block is assigned to a particular tile element.
From here, the tile can be used in other regular cooperative operations such as reductions, GEMMs, etc.

Similarly, we can `untile` tile objects back to their per-thread scalar equivalent values.

.. Note:: All threads in a block must execute tile operations, but code surrounding tile operations may contain arbitrary conditional logic.

Extra consideration is needed when using tiles in SIMT kernels that are meant to run on both the GPU and the CPU.
On the CPU, ``block_dim`` is set to 1, which can change the behavior of kernels using :func:`wp.tile() <warp._src.lang.tile>`. Consider the following example:

.. testcode::
    :skipif: wp.get_device() == "cpu" or wp.get_cuda_device_count() == 0

    import warp as wp

    TILE_DIM = 4

    @wp.kernel
    def tile_reduce_blockwise_simt_kernel(output: wp.array[int]):
        i = wp.tid()

        t = wp.tile(i)
        s = wp.tile_sum(t)

        wp.tile_store(output, s, offset=i)

    N = TILE_DIM * 3

    output = wp.zeros(shape=N, dtype=int)

    wp.launch(tile_reduce_blockwise_simt_kernel, dim=N, outputs=[output], block_dim=TILE_DIM)

    print(output.numpy())

.. testoutput::

    [ 6  0  0  0 22  0  0  0 38  0  0  0]

Here, we launch ``N=12`` logical threads. The tile size is 4, so there are three blocks in total that are created with :func:`wp.tile() <warp._src.lang.tile>`.
The tile reduction operation stores the block's sum in the first thread of the block, so we see 6, 22, and 38 stored at indices 0, 4, and 8.
If we instead launch this kernel on the CPU, we get the following output:

.. code-block:: text

    [ 0  1  2  3  4  5  6  7  8  9 10 11]

When launching on the CPU, ``block_dim`` is set to 1, so the tile generated with :func:`wp.tile() <warp._src.lang.tile>` has a size of 1, and the reduction of each tile simply returns the value of the tile.
So if you are designing a kernel that is meant to get the same result running on the GPU or the CPU, it should be designed to be independent of the value of ``block_dim``.
For instance, if we want a full array reduction that works consistently across devices, we can use :func:`wp.tile_atomic_add() <warp._src.lang.tile_atomic_add>` to accumulate results from all blocks:

.. testcode::

    import warp as wp

    TILE_DIM = 4

    @wp.kernel
    def tile_reduce_simt_kernel(output: wp.array[int]):
        i = wp.tid()

        t = wp.tile(i)
        s = wp.tile_sum(t)

        # update global sum
        wp.tile_atomic_add(output, s)

    N = TILE_DIM * 3

    output = wp.zeros(shape=1, dtype=int)

    wp.launch(tile_reduce_simt_kernel, dim=N, outputs=[output], block_dim=TILE_DIM)

    print(output.numpy())

.. testoutput::

    [66]


Type Preservation
^^^^^^^^^^^^^^^^^

:func:`warp.tile <warp._src.lang.tile>` includes the optional parameter ``preserve_type``, which is ``False`` by default.
When ``preserve_type`` is ``False``, this function expands non-scalar inputs into a multi-dimensional tile.
Vectors are expanded into a 2D tile of scalar values with shape ``(length(vector), block_dim)``,
while matrices are expanded into a 3D tile of scalar values with shape ``(rows, cols, block_dim)``.

When ``preserve_type`` is ``True``, this function outputs a 1D tile of length ``block_dim`` with the same
data type as the input value. So if you tile a vector across the block with ``preserve_type=True``, a 1D
tile of vectors will be returned. This is useful for collective operations that operate on the entire
vector or matrix rather than their individual components, as in the following example demonstrating
a matrix tile reduction:

.. testcode::
    :skipif: wp.get_device() == "cpu" or wp.get_cuda_device_count() == 0

    import warp as wp

    TILE_DIM = 32

    @wp.kernel
    def matrix_reduction_kernel(y: wp.array[wp.mat33]):
        i = wp.tid()
        I = wp.identity(3, dtype=wp.float32)
        m = wp.float32(i) * I

        t = wp.tile(m, preserve_type=True)
        sum = wp.tile_reduce(wp.add, t)

        wp.tile_store(y, sum)

    y = wp.zeros(shape=1, dtype=wp.mat33)

    wp.launch(matrix_reduction_kernel, dim=TILE_DIM, inputs=[], outputs=[y], block_dim=TILE_DIM)

    print(y.numpy()[0])

.. testoutput::

    [[496.   0.   0.]
     [  0. 496.   0.]
     [  0.   0. 496.]]

Example: Using tiles to accelerate array-wide reductions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prior to the addition of tile support in Warp, array-wide reductions were commonly performed in a single kernel
using a built-in atomic function like :func:`wp.atomic_add() <warp._src.lang.atomic_add>`.
This could be very inefficient when compared to optimized mechanisms like
`cub::BlockReduce <https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockReduce.html>`__.
Consider the following sum-of-squares reduction on an array:

.. code-block:: python

    import  numpy as np

    import warp as wp

    @wp.kernel
    def reduce_array_simt(
        a: wp.array2d[wp.float64],
        result: wp.array[wp.float64],
    ):
        i, j = wp.tid()

        local_val = a[i, j]*a[i, j]

        wp.atomic_add(result, 0, local_val)

    rng = np.random.default_rng(42)

    data = wp.array(rng.random((4096, 4096), dtype=np.float64), dtype=wp.float64)
    result = wp.zeros((1,), dtype=wp.float64)

    wp.launch(reduce_array_simt, (4096, 4096), inputs=[data], outputs=[result])

The above kernel in Warp 1.6.1 runs in about 27.5 ms on an RTX 3090 GPU.
We can use tiles to accelerate this reduction by first creating a tile from the scalar ``local_val``
and then using :func:`wp.tile_sum() <warp._src.lang.tile_sum>` to cooperatively compute
the tile sum using shared memory. We can then accumulate the result of the reduced
tile into global memory using :func:`wp.tile_atomic_add() <warp._src.lang.tile_atomic_add>`:

.. code-block:: python

    import  numpy as np

    import warp as wp

    @wp.kernel
    def reduce_array_tile(
        a: wp.array2d[wp.float64],
        result: wp.array[wp.float64],
    ):
        i, j = wp.tid()

        local_val = a[i, j]*a[i, j]

        t = wp.tile(local_val)
        s = wp.tile_sum(t)

        wp.tile_atomic_add(result, s)

    rng = np.random.default_rng(42)

    data = wp.array(rng.random((4096, 4096), dtype=np.float64), dtype=wp.float64)
    result = wp.zeros((1,), dtype=wp.float64)

    wp.launch(reduce_array_tile, (4096, 4096), inputs=[data], outputs=[result])

The reduction kernel using tiles runs in about 0.528 ms, a 52x improvement over the original kernel.

Further speed improvements could be obtained by experimenting with different block sizes.
If we reduce the block size from the default of 256 threads to 128 threads
in this example by adding ``block_dim=128`` to the :func:`wp.launch() <warp.launch>`,
the kernel only takes about 0.436 ms to complete, while the pure SIMT kernel is
relatively unaffected.

Automatic Differentiation
-------------------------

Warp can automatically generate the backward version of tile-based programs.
In general, tile programs must obey the same rules for auto-diff as regular Warp programs.
In-place addition and subtraction (``+=``, ``-=``) on tiles are differentiable;
in-place multiplication and division are not supported for tiles.
Please see the :ref:`differentiability` section for more details.

Tiles in ``@wp.func`` Functions
-------------------------------

Tile parameters in :func:`@wp.func <warp.func>` functions are **passed by reference**. This is
a departure from other Warp types (scalars, vectors, matrices, etc.), which are passed by value.
The difference is observable when a function modifies a tile in place:

.. code:: python

    @wp.func
    def add_bias(t: wp.tile[float, TILE_M, TILE_N]):
        t += wp.tile_ones(dtype=float, shape=(TILE_M, TILE_N), storage="register") * 5.0

    @wp.kernel(enable_backward=False)
    def compute(input: wp.array2d[float], out: wp.array2d[float]):
        i = wp.tid()
        t = wp.tile_load(input, shape=(TILE_M, TILE_N), offset=(0, 0), storage="shared")
        add_bias(t)          # modifies t in place — caller sees the change
        wp.tile_store(out, t) # t now contains input + 5.0

This behavior applies to **both** shared-memory and register tiles, and matches Python's
semantics for mutable objects. The reasons for pass-by-reference are:

- Shared-memory tiles are handles to a fixed region of shared memory and cannot be copied.
  Passing by reference lets functions operate on shared tiles directly.
- Register tiles use the same calling convention for consistency, so that
  ``@wp.func`` code works identically regardless of the caller's storage choice.

.. note::

   Simple rebinding (``alias = t``) inside a ``@wp.func`` creates a new C++ variable.
   For register tiles this is a full value copy; for shared tiles the new variable is a
   non-owning handle to the same shared memory, so element-level writes through either
   variable affect the same data. In-place assignments (``+=``, ``-=``, etc.) on the
   original parameter mutate the tile through the reference regardless of storage type.

.. _vectorized_tile_loads:

Vectorized Tile Loads
---------------------

When loading shared-memory tiles, Warp automatically attempts a **vectorized path**
that uses 128-bit (``float4``) memory transactions instead of per-element loads. This
gives up to 1.3× higher bandwidth because each thread issues fewer, wider loads.

The vectorized path activates when **all** of the following hold:

1. **Last-dimension alignment:** ``tile_last_dim × sizeof(element) % 16 == 0``.
   For ``float32`` (4 bytes), the last dimension must be divisible by 4.
   For ``vec3`` (12 bytes), it must be divisible by 4 (since 4 × 12 = 48, divisible by 16).
   For ``float64`` (8 bytes), it must be divisible by 2.

2. **Contiguous array:** the source array is densely packed (no gaps between rows).

3. **16-byte aligned base address:** the start of the tile's data in global memory is
   aligned to a 16-byte boundary. Warp GPU arrays are 256-byte aligned (CPU arrays are
   64-byte aligned); either satisfies the 16-byte requirement for offset-zero tiles.
   Offset tiles are aligned when the linear byte offset
   ``sum(offset[d] * strides[d])`` is a multiple of 16.

4. **Tile fits within array bounds:** the tile does not extend past the array dimensions.

5. **2D or higher tile:** 1D tiles do not use the vectorized float4 path, but may use
   the coalesced byte-copy path for large element types (``sizeof(T) > 16``).

6. **Outer-dimension strides are float4-aligned:** all strides except the last dimension
   must be multiples of 16 bytes. A contiguous ``float32`` array with ``shape[1]=35``
   has ``strides[0]=140`` bytes, and ``140 % 16 != 0``, so vectorization is rejected
   even if the tile's last dimension is aligned.

When any condition fails, Warp tries a **coalesced byte-copy path** for large element
types (``sizeof(T) > 16``) such as ``mat33``, ``mat44``, and ``mat66``. This path copies
raw bytes as ``float*`` with thread-striped access for perfect memory coalescing, and works
for both 1D and N-D tiles. If that path is also ineligible, Warp falls back to a **scalar
load path** that handles arbitrary alignment, strides, and partial (out-of-bounds) tiles
with zero-padding.

**Padding for vectorized loads:** if your data dimensions are not naturally aligned, pad
them to hit the vectorized path. For example, with ``float32`` tiles of width 30 (not
divisible by 4), pad to 32:

.. code:: python

    # Pad array width from 30 → 32 for vectorized float4 loads
    data_np = np.pad(data_np, ((0, 0), (0, 2)), mode='constant')
    data = wp.array(data_np, dtype=float, device=device)

    # Now tile_load with width=32 hits the fast vectorized path
    t = wp.tile_load(data, shape=(8, 32), offset=(i * 8, 0), storage="shared")

**The ``aligned`` parameter:** when you know all conditions above are met, pass
``aligned=True`` to skip the runtime eligibility checks:

.. code:: python

    # Caller guarantees: contiguous array, 16-byte aligned, tile fits in bounds
    t = wp.tile_load(data, shape=(8, 32), offset=(i * 8, 0), storage="shared", aligned=True)
    wp.tile_store(out, t, offset=(i * 8, 0), aligned=True)

This eliminates a small amount of per-load overhead from the runtime checks. The
``aligned`` parameter is only meaningful for shared-memory tiles (register tiles do not
use the vectorized path).

.. warning::

   Passing ``aligned=True`` when the conditions are not met results in undefined behavior
   (incorrect results or GPU faults). Use it only when you can guarantee alignment. The
   address alignment check is always active (even in release builds), but bounds and
   contiguity checks are debug-only.


.. _software_pipelining:

Software Pipelining with Register Tiles
----------------------------------------

When a kernel iterates over many tiles, the default pattern — load, compute, store,
repeat — leaves the GPU's memory pipeline idle during compute and the ALU idle during
loads.

**Sequential (baseline):** each iteration loads a tile, processes it, stores the result,
then moves on. The next load cannot begin until the current store completes:

.. testcode::
    :skipif: wp.get_cuda_device_count() == 0

    import time
    import warp as wp
    import numpy as np

    @wp.func
    def activation(x: float):
        return wp.sin(x * 1.1 + 0.1)

    @wp.func
    def normalize(x: float):
        return wp.exp(-wp.abs(x) * 0.5)

    @wp.func
    def transform(x: float):
        return wp.sqrt(wp.abs(x) + 1.0)

    @wp.func
    def squash(x: float):
        return wp.tanh(x * 0.7)

    TILE_N = wp.constant(256)
    N_ROWS = wp.constant(512)

    @wp.kernel
    def sequential(
        inp: wp.array2d(dtype=float),
        out: wp.array2d(dtype=float),
    ):
        for i in range(N_ROWS):
            a = wp.tile_load(inp, shape=(1, TILE_N), offset=(i, 0), storage="register")
            a = wp.tile_map(activation, a)
            a = wp.tile_map(normalize, a)
            a = wp.tile_map(transform, a)
            a = wp.tile_map(squash, a)
            wp.tile_store(out, a, offset=(i, 0))

    @wp.kernel
    def pipelined(
        inp: wp.array2d(dtype=float),
        out: wp.array2d(dtype=float),
    ):
        # Load first tile
        a = wp.tile_load(inp, shape=(1, TILE_N), offset=(0, 0), storage="register")

        for i in range(1, N_ROWS):
            # Issue next load — GPU fetches in background during compute below
            b = wp.tile_load(inp, shape=(1, TILE_N), offset=(i, 0), storage="register")

            # Heavy compute overlaps with the memory fetch for b
            a = wp.tile_map(activation, a)
            a = wp.tile_map(normalize, a)
            a = wp.tile_map(transform, a)
            a = wp.tile_map(squash, a)
            wp.tile_store(out, a, offset=(i - 1, 0))

            a = b  # by now b's data has arrived

        # Epilogue
        a = wp.tile_map(activation, a)
        a = wp.tile_map(normalize, a)
        a = wp.tile_map(transform, a)
        a = wp.tile_map(squash, a)
        wp.tile_store(out, a, offset=(N_ROWS - 1, 0))

    rng = np.random.default_rng(42)
    device = wp.get_cuda_device()
    inp = wp.array(rng.random((N_ROWS, TILE_N), dtype=np.float32), dtype=float, device=device)
    out_seq = wp.zeros_like(inp)
    out_pipe = wp.zeros_like(inp)

    # Verify correctness
    wp.launch_tiled(sequential, dim=[1], inputs=[inp, out_seq], block_dim=TILE_N, device=device)
    wp.launch_tiled(pipelined, dim=[1], inputs=[inp, out_pipe], block_dim=TILE_N, device=device)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out_seq.numpy(), out_pipe.numpy(), rtol=1e-5)

    # Benchmark
    for _ in range(10):
        wp.launch_tiled(sequential, dim=[1], inputs=[inp, out_seq], block_dim=TILE_N, device=device)
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(100):
        wp.launch_tiled(sequential, dim=[1], inputs=[inp, out_seq], block_dim=TILE_N, device=device)
    wp.synchronize_device(device)
    seq_ms = (time.perf_counter() - t0) / 100 * 1000

    for _ in range(10):
        wp.launch_tiled(pipelined, dim=[1], inputs=[inp, out_pipe], block_dim=TILE_N, device=device)
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(100):
        wp.launch_tiled(pipelined, dim=[1], inputs=[inp, out_pipe], block_dim=TILE_N, device=device)
    wp.synchronize_device(device)
    pipe_ms = (time.perf_counter() - t0) / 100 * 1000

    speedup = seq_ms / pipe_ms
    print(f"Sequential: {seq_ms:.3f} ms")
    print(f"Pipelined:  {pipe_ms:.3f} ms")
    print(f"Speedup:    {speedup:.1f}x")

.. testoutput::
    :options: +ELLIPSIS

    Sequential: ... ms
    Pipelined:  ... ms
    Speedup:    ...x

The key insight: ``b = wp.tile_load(...)`` issues memory load instructions that enter the
GPU's memory pipeline immediately, but the result registers are not read until ``a = b``
on the next iteration. The GPU's instruction scheduler fills the gap with the ALU-heavy
``tile_map`` calls, effectively hiding the memory latency for free.

No special parameters or API calls are needed — the hardware pipeline handles the overlap
automatically. The benefit scales with the amount of compute per tile: with four
transcendental ``tile_map`` calls per iteration (as above), expect **1.2–1.6× speedup**
depending on the GPU.

.. note::

   This pattern uses ``storage="register"`` (the default). Shared-memory tiles require
   thread-block synchronization after each load, which prevents the same overlap. Use
   register tiles for pipelined workloads.


.. _mathdx:

``Failed to compile LTO`` Error Message
---------------------------------------

Some tile operations invoke MathDx APIs to generate and compile link-time objects (LTOs) at runtime.
If the compilation fails, you may see an error message that mentions ``Failed to compile LTO``.
A common cause of this error is that the tile sizes involved are too large for the current device, as shared memory
is a limited resource.

To get more information about the error, you can set the ``LIBMATHDX_LOG_LEVEL`` environment
variable to 5 and rerun the program. Batching the problem into smaller tiles may be required to work around the shared
memory limitations.
In the case of FFT operations, using more threads per block may help (see the
`cuFFTDx requirements <https://docs.nvidia.com/cuda/cufftdx/requirements_func.html>`__ for more details).

Building with MathDx
--------------------

Most tile operations described in `Linear Algebra`_ require Warp to be built with the MathDx library.
Starting with Warp 1.5.0, PyPI distributions will come with out-of-the-box support for tile operations
leveraging MathDx APIs.

When building Warp locally using ``build_lib.py``, the script will attempt to automatically download ``libmathdx``
from the `cuBLASDx Downloads Page <https://developer.nvidia.com/cublasdx-downloads>`__.
A path to an existing ``libmathdx`` installation can also be specified using the ``--libmathdx-path`` option
when running ``build_lib.py`` or by defining the path in the ``LIBMATHDX_HOME`` environment variable.

Please note that CUDA Toolkit 12.6.3 or higher is required for full MathDx support when building Warp from source.
Warp + MathDx will compile with earlier CUDA 12 versions, but matrix multiplication, triangular solves, Cholesky factorization,
and the Cholesky solver will fail at runtime.

.. [1] `Technical Blog: Introducing Tile-Based Programming in Warp 1.5.0 <https://developer.nvidia.com/blog/introducing-tile-based-programming-in-warp-1-5-0/>`_
