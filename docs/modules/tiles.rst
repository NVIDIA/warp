Tiles
=====

.. currentmodule:: warp

.. warning:: Tile-based operations in Warp are under preview, APIs are subject to change.

Block-based programming models such as those in OpenAI Triton have proved to be effective ways of expressing high-performance kernels that can leverage cooperative operations on modern GPUs.
With Warp 1.5.0 [1]_, developers now have access to new tile-based programming primitives in Warp kernels.
Leveraging cuBLASDx and cuFFTDx, these new tools provide developers with efficient matrix multiplication and Fourier transforms for accelerated simulation and scientific computing. 

Requirements
------------

Tile-based operations are fully supported on versions of Warp built against the CUDA 12 runtime.
Most linear-algebra tile operations like :func:`tile_cholesky`, :func:`tile_fft`, and :func:`tile_matmul`
require Warp to be built with the MathDx library, which is not supported on CUDA 11.
See `Building with MathDx`_ for more details when building the Warp locally with support for
linear-algebra tile operations.

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
    def compute(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float)):

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
    def tile_gemm(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
        
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

        assert(np.allclose(C_wp.numpy(), A@B))

        print("Example matrix multiplication passed")


Tile Operations
---------------


Construction
^^^^^^^^^^^^

* :func:`tile_zeros`
* :func:`tile_ones`
* :func:`tile_arange`
* :func:`tile`
* :func:`untile`
* :func:`tile_view`
* :func:`tile_broadcast`
* :func:`tile_reshape`
* :func:`tile_squeeze`
* :func:`tile_astype`

Load/Store
^^^^^^^^^^

* :func:`tile_load`
* :func:`tile_store`
* :func:`tile_atomic_add`

Maps/Reductions
^^^^^^^^^^^^^^^

* :func:`tile_map`
* :func:`tile_reduce`
* :func:`tile_sum`
* :func:`tile_min`
* :func:`tile_max`

Linear Algebra
^^^^^^^^^^^^^^

* :func:`tile_matmul`
* :func:`tile_transpose`
* :func:`tile_fft`
* :func:`tile_ifft`
* :func:`tile_cholesky`
* :func:`tile_cholesky_solve`
* :func:`tile_diag_add`

Tiles and SIMT Code
-------------------

Traditionally, Warp kernels are primarily written in the SIMT programming model, where each thread's execution happens independently. 
Tiles, on the other hand, allow threads to work **cooperatively** to perform operations.
Warp exposes the :func:`warp.tile`, and :func:`warp.untile` methods to convert data between per-thread value types and
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

In this example, we have launched a regular SIMT grid with ``N`` logical threads using ``wp.launch()``.
The kernel performs some per-thread computations and then converts the scalar ``x`` value into a tile object using :func:`warp.tile`.
This function takes a single value as input and returns a tile with the same dimensions as the number of threads in the block.
From here, the tile can be used in other regular cooperative operations such as reductions, GEMMs, etc.

Similarly, we can `untile` tile objects back to their per-thread scalar equivalent values.

.. Note:: All threads in a block must execute tile operations, but code surrounding tile operations may contain arbitrary conditional logic.

Type Preservation
^^^^^^^^^^^^^^^^^

:func:`warp.tile` includes the optional parameter ``preserve_type``, which is ``False`` by default.
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
    def matrix_reduction_kernel(y: wp.array(dtype=wp.mat33)):
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
using a built-in atomic function like :func:`wp.atomic_add() <warp.atomic_add>`.
This could be very inefficient when compared to optimized mechanisms like
`cub::BlockReduce <https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockReduce.html>`__.
Consider the following sum-of-squares reduction on an array:

.. code-block:: python

    import  numpy as np

    import warp as wp

    @wp.kernel
    def reduce_array_simt(
        a: wp.array2d(dtype=wp.float64),
        result: wp.array(dtype=wp.float64),
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
and then using :func:`wp.tile_sum() <warp.tile_sum>` to cooperatively compute
the tile sum using shared memory. We can then accumulate the result of the reduced
tile into global memory using :func:`wp.tile_atomic_add() <warp.tile_atomic_add>`:

.. code-block:: python

    import  numpy as np

    import warp as wp

    @wp.kernel
    def reduce_array_tile(
        a: wp.array2d(dtype=wp.float64),
        result: wp.array(dtype=wp.float64),
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
In general, tile programs must obey the same rules for auto-diff as regular Warp programs, e.g. avoiding in-place operations, etc.
Please see the :ref:`differentiability` section for more details.

.. _mathdx:

Building with MathDx
--------------------

Most tile operations described in `Linear Algebra`_ require Warp to be built with the MathDx library.
Starting with Warp 1.5.0, PyPI distributions will come with out-of-the-box support for tile operations
leveraging MathDx APIs.

When building Warp locally using ``build_lib.py``, the script will attempt to automatically download ``libmathdx``
from the `cuBLASDx Downloads Page <https://developer.nvidia.com/cublasdx-downloads>`__.
A path to an existing ``libmathdx`` installation can also be specified using the ``--libmathdx_path`` option
when running ``build_lib.py`` or by defining the path in the ``LIBMATHDX_HOME`` environment variable.

Please note that CUDA Toolkit 12.6.3 or higher is required for full MathDx support when building Warp from source.
Warp + MathDx will compile with earlier CUDA 12 versions, but matrix multiplication, triangular solves, Cholesky factorization,
and the Cholesky solver will fail at runtime.

.. [1] `Technical Blog: Introducing Tile-Based Programming in Warp 1.5.0 <https://developer.nvidia.com/blog/introducing-tile-based-programming-in-warp-1-5-0/>`_
