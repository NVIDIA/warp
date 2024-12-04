Tiles
=====

.. currentmodule:: warp

.. warning:: Tile-based operations in Warp are under preview, APIs are subject to change.

Block-based programming models such as those in OpenAI Triton have proved to be effective ways of expressing high-performance kernels that can leverage cooperative operations on modern GPUs. With Warp 1.5.0 developers now have access to new tile-based programming primitives in Warp kernels. Leveraging cuBLASDx and cuFFTDx, these new tools provide developers with efficient matrix multiplication and Fourier transforms for accelerated simulation and scientific computing. 

Requirements
------------

Tile-based operations are currently only supported on versions of Warp built against the CUDA 12 runtime.
See `Building with MathDx`_ for more details when building the Warp locally with support for tile operations.

Execution Model
---------------

Warp's execution model allows users to specify a grid of logical threads with up to 4 dimensions for kernel execution at launch time. With the introduction of tile primitives, users can now specify the *block size* for kernel launches, which partitions the thread grid into smaller sets of threads that are executed on a single compute unit.

Inside kernels, tile operations are executed cooperatively across each block of threads, allowing them to take advantage of efficient memory access, local memory, and dedicated hardware units like `Tensor Cores <https://www.nvidia.com/en-us/data-center/tensor-cores/>`__.

In the following example, we launch a grid of threads where each block is responsible for loading a row of data from a 2D array and computing its sum:

.. code:: python
    
    TILE_SIZE = wp.constant(256)
    TILE_THREADS = 64

    @wp.kernel
    def compute(a: array2d(dtype=float))
        
        # obtain our block index
        i = wp.tid()

        # load a row from global memory
        t = wp.tile_load(array[i], i, TILE_SIZE)
        s = wp.tile_sum(t)
        ...

    wp.launch_tiled(compute, dim=[a.shape[0]], inputs=[a], block_dim=TILE_THREADS)
    
Here, we have used the new :func:`warp.launch_tiled` function which assigns ``TILE_THREADS`` threads to each of the elements in the launch grid. Each block of ``TILE_THREADS`` threads then loads an entire row of 256 values from the global memory array and computes its sum (cooperatively).


Tile Properties
---------------

In Warp, tile objects are 2D arrays of data where the tile elements may be scalars, vectors, matrices, or user-defined structures. We can load 2D tiles directly from 2D global memory arrays as follows:

.. code:: python
    
    TILE_M = wp.constant(16)
    TILE_N = wp.constant(16)    
    TILE_THREADS = 64

    @wp.kernel
    def compute(a: array2d(dtype=float))
        
        # obtain our 2d block index
        i, j = wp.tid()

        # load a 2d tile from global memory
        t = wp.tile_load(array, i, j, m=TILE_M, n=TILE_N)
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
Warp will automatically migrate tiles to shared memory as necessary for specific operations.
Shared memory is a limited resource, and so the tile size must be set appropriately to avoid exceeding the hardware limitations.
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

        sum = wp.tile_zeros(m=TILE_M, n=TILE_N, dtype=wp.float32)

        M = A.shape[0]
        N = B.shape[1]
        K = A.shape[1]

        count = int(K / TILE_K)

        for k in range(0, count):
            a = wp.tile_load(A, i, k, m=TILE_M, n=TILE_K)
            b = wp.tile_load(B, k, j, m=TILE_K, n=TILE_N)

            # sum += a*b
            wp.tile_matmul(a, b, sum)

        wp.tile_store(C, i, j, sum)



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

Tiles and SIMT Code
-------------------

Traditionally, Warp kernels are primarily written in the SIMT programming model, where each thread's execution happens independently. Tiles, on the other hand, allow threads to work **cooperatively** to perform operations. Warp exposes the :func:`warp.tile`, and :func:`warp.untile` methods to convert data between per-thread value types and the equivalent tile representation. For example:

.. code:: python
    
    TILE_THREADS = 64

    @wp.kernel
    def compute()
        i = wp.tid()

        # perform some per-thread computation
        x = i*2.0 + wp.sin(float(i))

        # tile the value x across the block
        # returns a tile with shape=(1, TILE_THREADS)
        t = wp.tile(x)
        ...

    # launch as regular SIMT kernel
    wp.launch(compute, dim=[N], inputs=[], block_dim=TILE_THREADS)

In this example, we have launched a regular SIMT grid with ``N`` logical threads using ``wp.launch()``. The kernel performs some per-thread computations and then converts the scalar ``x`` value into a tile object using :func:`warp.tile`. This function takes a single value as input and returns a tile with the same dimensions as the number of threads in the block. From here, the tile can be used in other regular cooperative operations such as reductions, GEMMs, etc.

Similarly, we can `untile` tile objects back to their per-thread scalar equivalent values.

.. Note:: All threads in a block must execute tile operations, but code surrounding tile operations may contain arbitrary conditional logic.

Automatic Differentiation
-------------------------

Warp can automatically generate the backward version of tile-based programs.
In general, tile programs must obey the same rules for auto-diff as regular Warp programs, e.g. avoiding in-place operations, etc.
Please see the :ref:`differentiability` section for more details.

Building with MathDx
--------------------

The tile operations described in `Linear Algebra`_ require Warp to be built with the MathDx library.
Starting with Warp 1.5.0, PyPI distributions will come with out-of-the-box support for tile operations
leveraging MathDx APIs.

When building Warp locally using ``build_lib.py``, the script will attempt to automatically download ``libmathdx``
from the `cuBLASDx Downloads Page <https://developer.nvidia.com/cublasdx-downloads>`__.
A path to an existing ``libmathdx`` installation can also be specified using the ``--libmathdx_path`` option
when running ``build_lib.py`` or by defining the path in the ``LIBMATHDX_HOME`` environment variable.
