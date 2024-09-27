Tiles (Preview)
===============

Block-based programming models such as those in OpenAI Triton have proved to be effective ways of expressing high performance kernels that can leverage cooperative operations on modern GPUs.

Warp 1.4.0 introduces tile extensions that expose a block-based programming to Warp kernels. 

Execution Model
---------------

Warp's execution model allows users to specify an up to 4-dimensional grid of logical threads for kernel execution at launch time. With the introduction of tiles, users can also specify a block size, which partitions the grid into smaller sets of threads that are executed on a single compute unit.

Inside kernels, tile operations are executed cooperatively across each block of threads, allowing them to take advantage of efficient memory access, local memory, and dedicated hardware units like TensorCores.

As an example, consider the following kernel:

.. code:: python
    
    TILE_SIZE = wp.constant(256)
    TILE_THREADS = 64

    @wp.kernel
    def compute(a: array(dtype=float))
        i = wp.tid()/TILE_SIZE

        t = wp.tile_load(array, x=i, n=TILE_SIZE)
        ...

    wp.launch(compute, dim=[len(a)], inputs=[a], block_dim=TILE_THREADS)
    
Here, we load a 1D tile of 256 values from a global memory array ``a``, where the load operation is performed cooperatively by all 64 threads in the block, as specified by the ``block_dim`` argument to :func:`warp.launch`. In this case each thread is responsible for loading 4 values from global memory, which may then be stored in registers, or shared memory across the block.

Tile Properties
---------------

In Warp, tile objects are 2D arrays of data where the tile elements may be scalars, vectors, matrices, or user defined structures.

In a more complex example, we launch a grid of threads where each block is responsible for loading a row of data from a 2D array and computing its sum:

.. code:: python
    
    TILE_SIZE = wp.constant(256)
    TILE_THREADS = 64

    @wp.kernel
    def compute(a: array2d(dtype=float))
        i, _= wp.tid()

        # load a row from global memory
        t = wp.tile_load(array, i, TILE_SIZE)
        s = wp.sum(t)
        ...

    wp.launch(compute, dim=[a.shape[0], TILE_THREADS], inputs=[a], block_dim=TILE_THREADS)
    
Here, we launch a 2D grid of threads where the trailing dimension is equal to the block size. This ensures we have an entire block of threads dedicated to each row. Each block then loads an entire row of 256 values from the global memory array and computes its sum.

To streamline this common pattern Warp provides a helper ``wp.tiled_launch()`` which takes care of adding the trailing tile dimension to the thread grid, for example, to assign a block of 64 threads to load and sum a 2D array of values we can do the following:

.. code:: python
    
    TILE_M = wp.constant(16)
    TILE_N = wp.constant(16)    
    TILE_THREADS = 64

    @wp.kernel
    def compute(a: array2d(dtype=float))
        i, j = wp.tid()

        # load a row from global memory
        t = wp.tile_load(array, i, j, TILE_M, TILE_N)
        s = wp.sum(t)
        ...

    wp.launch_tiled(compute, dim=[a.shape[0]/TILE_M, a.shape[1]/TILE_N], inputs=[a], block_dim=TILE_THREADS)
    
In this example, we use :func:`warp.launch_tiled` to automatically insert the trailing dimension, and assign ``TILE_THREADS`` to each 2D tile of the array. Each tile consists of ``16*16=256`` values, which are loaded cooperatively by the 64 threads in each block.

Tile Storage
------------

When tiles are created they are placed in either `register` or `shared` memory. In general Warp tries to determine the best storage for each, the default is generally for register storage, although some operations such as matrix multiplies may migrate data from register to shared as necessary.

Register Tiles
++++++++++++++

Values in register tiles are stored across the entire block, for example, if the block dimension at launch is set to 64, a register tile with ``shape=(1, 256)`` will result in each thread storing 4 elements. Reigster based storage is the fastest storage on most hardware, however, because the tile storage is spread across the threads in the block, an individual thread cannot randomly access data that is assigned to another thread efficiently. For this reason operations on tiles tend to expressed as higher level maps, reductions, and reshaping operations that may transfer values through shared memory.

Shared Memory Tiles
+++++++++++++++++++

Some operations like matrix multiplication, require access to an entire tile of values. In this case the tile data may stored in shared memory, which allows efficient random access. Warp will automatically migrate tiles to shared memory as necessary for specific operations. Shared memory is a limited resource, and so tile size must be set appropriately to avoid exceeding the hardware limitations, otherwise kernel compilation may fail.

Tile Operations
---------------

Creation
++++++++

* :func:`warp.tile_zeros`
* :func:`warp.tile_ones`
* :func:`warp.tile_arange`

Conversion
++++++++++

* :func:`warp.tile`
* :func:`warp.untile`


Load/Store
++++++++++

* :func:`warp.tile_load`
* :func:`warp.tile_store`
* :func:`warp.tile_atomic_add`

Maps/Reductions
+++++++++++++++

* :func:`warp.tile_map`
* :func:`warp.tile_sum`

Linear Algebra
++++++++++++++

* :func:`warp.tile_matmul`
* :func:`warp.tile_fft`
* :func:`warp.tile_ifft`

Tiles and SIMT Code
+++++++++++++++++++

Warp kernels are primarily written in the SIMT programming model in mind, where each thread's execution happens completely independently. Tiles on the other hand allow threads to work cooperatively to perform operations.

Warp aims to give users a way to seamlessly integrate tile operations with existing SIMT code. To this end, we expose two operations, :func:`warp.tile`, and :func:`warp.untile` which can be used as follows:

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

In this example we perform some per-thread computations, and then convert the scalar ``x`` value into a tile object using the  :func:`warp.tile` function. This function takes a single value as input, and returns a tile with the same dimensions as the number of threads in the block. From here, the tile can used in other regular cooperative operations such as reductions, GEMMs, etc.

Similarly, we can `untile` tile objects back to their per-thread scalar equivalent values.






