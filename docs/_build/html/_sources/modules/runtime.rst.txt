Runtime Reference
=================

.. currentmodule:: warp

.. toctree::
   :maxdepth: 2

Initialization
--------------

Before use Warp should be explicitly initialized with the ``wp.init()`` method::

   import warp as wp

   wp.init()

Users can query the supported compute devices using the ``wp.get_devices()`` method::

   print(wp.get_devices())

   >> ['cpu', 'cuda:0']

These device strings can then be used to allocate memory and launch kernels as described below.  More information about working with devices is available in :ref:`devices`.

Kernels
-------

In Warp, kernels are defined as Python functions, decorated with the ``@warp.kernel`` decorator. Kernels have a 1:1 correspondence with CUDA kernels, and may be launched with any number of parallel execution threads: ::

    @wp.kernel
    def simple_kernel(a: wp.array(dtype=wp.vec3),
                      b: wp.array(dtype=wp.vec3),
                      c: wp.array(dtype=float)):

        # get thread index
        tid = wp.tid()

        # load two vec3s
        x = a[tid]
        y = b[tid]

        # compute the dot product between vectors
        r = wp.dot(x, y)

        # write result back to memory
        c[tid] = r

Kernels are launched with the ``warp.launch()`` function on a specific device (CPU/GPU). The following example shows how to launch the kernel above::

   wp.launch(simple_kernel, dim=1024, inputs=[a, b, c], device="cuda")
 
Kernels may be launched with multi-dimensional grid bounds. In this case threads are not assigned a single index, but a coordinate in of an n-dimensional grid, e.g.::

   wp.launch(complex_kernel, dim=(128, 128, 3), ...)

Launches a 3d grid of threads with dimension 128x128x3. To retrieve the 3d index for each thread use the following syntax::

   i,j,k = wp.tid()

.. note:: 
   Currently kernels launched on ``cpu`` devices will be executed in serial. Kernels launched on ``cuda`` devices will be launched in parallel with a fixed block-size.

.. note:: 
   Note that all the kernel inputs must live on the target device, or a runtime exception will be raised.

.. autofunction:: launch

Arrays
------

Arrays are the fundamental memory abstraction in Warp; they are created through the following global constructors: ::

    wp.empty(shape=1024, dtype=wp.vec3, device="cpu")
    wp.zeros(shape=1024, dtype=float, device="cuda")
    

Arrays can also be constructed directly from ``numpy`` ndarrays as follows: ::

   r = np.random.rand(1024)

   # copy to Warp owned array
   a = wp.array(r, dtype=float, device="cpu")

   # return a Warp array wrapper around the numpy data (zero-copy)
   a = wp.array(r, dtype=float, copy=False, device="cpu")

   # return a Warp copy of the array data on the GPU
   a = wp.array(r, dtype=float, device="cuda")

Note that for multi-dimensional data the ``dtype`` parameter must be specified explicitly, e.g.: ::

   r = np.random.rand((1024, 3))

   # initialize as an array of vec3 objects
   a = wp.array(r, dtype=wp.vec3, device="cuda")

If the shapes are incompatible an error will be raised. 


Arrays can be moved between devices using the ``array.to()`` method: ::

   host_array = wp.array(a, dtype=float, device="cpu")
   
   # allocate and copy to GPU
   device_array = host_array.to("cuda")

Additionally, arrays can be copied directly between memory spaces: ::

   src_array = wp.array(a, dtype=float, device="cpu")
   dest_array = wp.empty_like(host_array)

   # copy from source CPU buffer to GPU
   wp.copy(dest_array, src_array)

Multi-dimensional arrays
########################

Multi-dimensional arrays can be constructed by passing a tuple of sizes for each dimension, e.g.: the following constructs a 2d array of size 1024x16::

    wp.zeros(shape=(1024, 16), dtype=float, device="cuda")

When passing multi-dimensional arrays to kernels users must specify the expected array dimension inside the kernel signature,
e.g. to pass a 2d array to a kernel the number of dims is specified using the ``ndim=2`` parameter::

   @wp.kernel
   def test(input: wp.array(dtype=float, ndim=2)):

Type-hint helpers are provided for common array sizes, e.g.: ``array2d()``, ``array3d()``, which are equivalent to calling ``array(..., ndim=2)```, etc. To index a multi-dimensional array use a the following kernel syntax::

   # returns a float from the 2d array
   value = input[i,j]

To create an array slice use the following syntax, where the number of indices is less than the array dimensions::

   # returns an 1d array slice representing a row of the 2d array
   row = input[i]

Slice operators can be concatenated, e.g.: via. ``s = array[i][j][k]``. Slices can be passed to ``wp.func`` user functions provided
the function also declares the expected array dimension. Currently only single-index slicing is supported.

.. note:: 
   Currently Warp limits arrays to 4 dimensions maximum. This is in addition to the contained datatype, which may be 1-2 dimensional for vector and matrix types such as ``vec3``, and ``mat33``.


The following construction methods are provided for allocating zero-initialized and empty (non-initialized) arrays:

.. autofunction:: zeros
.. autofunction:: zeros_like
.. autofunction:: empty
.. autofunction:: empty_like

.. autoclass:: array

User Functions
--------------

Users can write their own functions using the ``wp.func`` decorator, for example::

   @wp.func
   def square(x: float):
      return x*x

User functions can be called freely from within kernels inside the same module and accept arrays as inputs. 

Data Types
----------

Scalar Types
############

The following scalar storage types are supported for array structures:

+---------+------------------------+
| int8    | signed byte            |
+---------+------------------------+
| uint8   | unsigned byte          |
+---------+------------------------+
| int16   | signed short           |
+---------+------------------------+
| uint16  | unsigned short         |
+---------+------------------------+
| int32   | signed integer         |
+---------+------------------------+
| uint32  | unsigned integer       |
+---------+------------------------+
| int64   | signed long integer    |
+---------+------------------------+
| uint64  | unsigned long integer  |
+---------+------------------------+
| float32 | single precision float |
+---------+------------------------+
| float64 | double precision float |
+---------+------------------------+

Warp supports ``float`` and ``int`` as aliases for ``wp.float32`` and ``wp.int32`` respectively.

.. note:: 
   Currently Warp treats ``int32`` and ``float32`` as the two basic scalar *compute types*, and all other types as *storage types*. Storage types can be loaded and stored to arrays, but not participate in arithmetic operations directly. Users should cast storage types to a compute type (``int`` or ``float``) to perform computations.


Vector Types
############

Warp provides built-in math and geometry types for common simulation and graphics problems. A full reference for operators and functions for these types is available in the :any:`functions`.


+-----------------+---------------------------------------------------------------------------------------------------------------------------------+
| vec2            | 2d vector of floats                                                                                                             |
+-----------------+---------------------------------------------------------------------------------------------------------------------------------+
| vec3            | 3d vector of floats                                                                                                             |
+-----------------+---------------------------------------------------------------------------------------------------------------------------------+
| vec4            | 4d vector of floats                                                                                                             |
+-----------------+---------------------------------------------------------------------------------------------------------------------------------+
| mat22           | 2x2 matrix of floats                                                                                                            |
+-----------------+---------------------------------------------------------------------------------------------------------------------------------+
| mat33           | 3x3 matrix of floats                                                                                                            |
+-----------------+---------------------------------------------------------------------------------------------------------------------------------+
| mat44           | 4x4 matrix of floats                                                                                                            |
+-----------------+---------------------------------------------------------------------------------------------------------------------------------+
| quat            | Quaternion in form i,j,k,w where w is the real part                                                                             |
+-----------------+---------------------------------------------------------------------------------------------------------------------------------+
| transform       | 7d vector of floats representing a spatial rigid body transformation in format (p, q) where p is a vec3, and q is a quaternion  |
+-----------------+---------------------------------------------------------------------------------------------------------------------------------+
| spatial_vector  | 6d vector of floats, see wp.spatial_top(), wp.spatial_bottom(), useful for representing rigid body twists                       |
+-----------------+---------------------------------------------------------------------------------------------------------------------------------+
| spatial_matrix  | 6x6 matrix of floats used to represent spatial inertia matrices                                                                 |
+-----------------+---------------------------------------------------------------------------------------------------------------------------------+

Type Conversions
################

Warp is particularly strict regarding type conversions and does not perform *any* implicit conversion between numeric types. The user is responsible for ensuring types for most arithmetric operators match, e.g.: ``x = float(0.0) + int(4)`` will result in an error. This can be surprising for users that are accustomed to C-type conversions but avoids a class of common bugs that result from implicit conversions.

In addition, users should always cast storage types to a compute type (``int`` or ``float``) before computation. Compute types can be converted back to storage types through explicit casting, e.g.: ``byte_array[index] = wp.uint8(i)``.

.. note:: Warp does not currently perform implicit type conversions between numeric types. Users should explicitly cast variables to compatible types using ``int()`` or ``float()`` constructors.

Constants
---------

In general, Warp kernels cannot access variables in the global Python interpreter state. One exception to this is for compile-time constants, which may be declared globally (or as class attributes) and folded into the kernel definition.

Constants are defined using the ``warp.constant`` type. An example is shown below::

   TYPE_SPHERE = wp.constant(0)
   TYPE_CUBE = wp.constant(1)
   TYPE_CAPSULE = wp.constant(2)

   @wp.kernel
   def collide(geometry: wp.array(dtype=int)):

      t = geometry[wp.tid()]

      if (t == TYPE_SPHERE):
         print("sphere")
      if (t == TYPE_CUBE):
         print("cube")
      if (t == TYPE_CAPSULE):
         print("capsule")


.. autoclass:: constant


Operators
----------

Boolean Operators
#################

+--------------+--------------------------------------+
|   a and b    | True if a and b are True             |
+--------------+--------------------------------------+
|   a or b     | True if a or b is True               |
+--------------+--------------------------------------+
|   not a      | True if a is False, otherwise False  |
+--------------+--------------------------------------+

.. note:: 

   Expressions such as ``if (a and b):`` currently do not perform short-circuit evaluation. In this case ``b`` will also be evaluated even when ``a`` is ``False``. Users should take care to ensure that secondary conditions are safe to evaluate (e.g.: do not index out of bounds) in all cases.


Comparison Operators
####################

+----------+---------------------------------------+
| a > b    | True if a strictly greater than b     |
+----------+---------------------------------------+
| a < b    | True if a strictly less than b        |
+----------+---------------------------------------+
| a >= b   | True if a greater than or equal to b  |
+----------+---------------------------------------+
| a <= b   | True if a less than or equal to b     |
+----------+---------------------------------------+
| a == b   | True if a equals b                    |
+----------+---------------------------------------+
| a != b   | True if a not equal to b              |
+----------+---------------------------------------+

Arithmetic Operators
####################

+-----------+--------------------------+
|  a + b    | Addition                 |
+-----------+--------------------------+
|  a - b    | Subtraction              |
+-----------+--------------------------+
|  a * b    | Multiplication           |
+-----------+--------------------------+
|  a / b    | Floating point division  |
+-----------+--------------------------+
|  a // b   | Floored division         |
+-----------+--------------------------+
|  a ** b   | Exponentiation           |
+-----------+--------------------------+
|  a % b    | Modulus                  |
+-----------+--------------------------+

.. note::
   Arguments types to operators should match since implicit conversions are not performed, users should use the type constructors ``float()``, ``int()`` to cast variables to the correct type. Also note that the multiplication expression ``a * b`` is used to represent scalar multiplication and matrix multiplication. Currently the ``@`` operator is not supported in this version.

Meshes
------

Warp provides a ``warp.Mesh`` class to manage triangle mesh data. To create a mesh users provide a points, indices and optionally a velocity array::

   mesh = wp.Mesh(points, indices, velocities)

.. note::
   Mesh objects maintain references to their input geometry buffers. All buffers should live on the same device.

Meshes can be passed to kernels using their ``id`` attribute which uniquely identifies the mesh by a unique ``uint64`` value. Once inside a kernel you can perform geometric queries against the mesh such as ray-casts or closest point lookups::

   @wp.kernel
   def raycast(mesh: wp.uint64,
               ray_origin: wp.array(dtype=wp.vec3),
               ray_dir: wp.array(dtype=wp.vec3),
               ray_hit: wp.array(dtype=wp.vec3)):

      tid = wp.tid()

      t = float(0.0)                   # hit distance along ray
      u = float(0.0)                   # hit face barycentric u
      v = float(0.0)                   # hit face barycentric v
      sign = float(0.0)                # hit face sign
      n = wp.vec3()       # hit face normal
      f = int(0)                       # hit face index

      color = wp.vec3()

      # ray cast against the mesh
      if wp.mesh_query_ray(mesh, ray_origin[tid], ray_dir[tid], 1.e+6, t, u, v, sign, n, f):

         # if we got a hit then set color to the face normal
         color = n*0.5 + wp.vec3(0.5, 0.5, 0.5)

      ray_hit[tid] = color


Users may update mesh vertex positions at runtime simply by modifying the points buffer. After modifying point locations users should call ``Mesh.refit()`` to rebuild the bounding volume hierarchy (BVH) structure and ensure that queries work correctly.

.. note::
   Updating Mesh topology (indices) at runtime is not currently supported, users should instead re-create a new Mesh object.

.. autoclass:: Mesh
   :members:

Volumes
-------

Sparse volumes are incredibly useful for representing grid data over large domains, such as signed distance fields (SDFs) for complex objects, or velocities for large-scale fluid flow. Warp supports reading sparse volumetric grids stored using the `NanoVDB <https://developer.nvidia.com/nanovdb>`_ standard. Users can access voxels directly, or use built-in closest point or trilinear interpolation to sample grid data from world or local-space.


Volume object can be created directly from Warp arrays containing a NanoVDB grid or from the contents of a standard ``.nvdb`` file, using the ``load_from_nvdb`` method.


Below we give an example of creating a Volume object from an existing NanoVDB file::

   # open NanoVDB file on disk
   file = open("mygrid.nvdb", "rb")

   # create Volume object
   volume = wp.Volume.load_from_nvdb(file, device="cpu")

.. note::
   Files written by the NanoVDB library, commonly marked by the ``.nvdb`` extension, can contain multiple grids with various compression methods, but a ``Volume`` object represents a single NanoVDB grid therefore only files with a single grid are supported. NanoVDB's uncompressed and zip compressed file formats are supported.

To sample the volume inside a kernel we pass a reference to it by id, and use the built-in sampling modes::

   @wp.kernel
   def sample_grid(volume: wp.uint64,
                   points: wp.array(dtype=wp.vec3),
                   samples: wp.array(dtype=float)):

      tid = wp.tid()

      # load sample point in world-space
      p = points[tid]

      # transform position to the volume's local-space
      q = wp.volume_world_to_index(volume, p)
      # sample volume with trilinear interpolation     
      f = wp.volume_sample_f(volume, q, wp.Volume.LINEAR)

      # write result
      samples[tid] = f



.. note:: Warp does not currently support modifying sparse-volumes at runtime. We expect to address this in a future update. Users should create volumes using standard VDB tools such as OpenVDB, Blender, Houdini, etc.

.. autoclass:: Volume
   :members:

.. seealso:: `Reference <functions.html#volumes>`__ for the volume functions available in kernels.

Hash Grids
----------

Many particle-based simulation methods such as the Discrete Element Method (DEM), or Smoothed Particle Hydrodynamics (SPH), involve iterating over spatial neighbors to compute force interactions. Hash grids are a well-established data structure to accelerate these nearest neighbor queries, and particularly well-suited to the GPU.

To support spatial neighbor queries Warp provides a ``HashGrid`` object that may be created as follows:: 

   grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device="cuda")

   grid.build(points=p, radius=r)

Where ``p`` is an array of ``warp.vec3`` point positions, and ``r`` is the radius to use when building the grid. Neighbors can then be iterated over inside the kernel code as follows::

   @wp.kernel
   def sum(grid : wp.uint64,
         points: wp.array(dtype=wp.vec3),
         output: wp.array(dtype=wp.vec3),
         radius: float):

      tid = wp.tid()

      # query point
      p = points[tid]

      # create grid query around point
      query = wp.hash_grid_query(grid, p, radius)
      index = int(0)

      sum = wp.vec3()

      while(wp.hash_grid_query_next(query, index)):
            
         neighbor = points[index]
         
         # compute distance to neighbor point
         dist = wp.length(p-neighbor)
         if (dist <= radius):
               sum += neighbor

      output[tid] = sum

.. note:: The HashGrid query will give back all points in *cells* that fall inside the query radius. When there are hash conflicts it means that some points outside of query radius will be returned, and users should check the distance themselves inside their kernels. The reason the query doesn't do the check itself for each returned point is because it's common for kernels to compute the distance themselves, so it would redundant to check/compute the distance twice.


.. autoclass:: HashGrid
   :members:

Differentiability
-----------------

By default Warp generates a foward and backward (adjoint) version of each kernel definition. Buffers that participate in the chain of computation should be created with ``requires_grad=True``, for example::

   a = wp.zeros(1024, dtype=wp.vec3, device="cuda", requires_grad=True)

The ``warp.Tape`` class can then be used to record kernel launches, and replay them to compute the gradient of a scalar loss function with respect to the kernel inputs::

   tape = wp.Tape()

   # forward pass
   with tape:
      wp.launch(kernel=compute1, inputs=[a, b], device="cuda")
      wp.launch(kernel=compute2, inputs=[c, d], device="cuda")
      wp.launch(kernel=loss, inputs=[d, l], device="cuda")

   # reverse pass
   tape.backward(l)

After the backward pass has completed the gradients with respect to the inputs are available via a mapping in the Tape object: ::

   # gradient of loss with respect to input a
   print(tape.gradients[a])

Note that gradients are accumulated on the participating buffers, so if you wish to re-use the same buffers for multiple backward passes you should first zero the gradients::

   tape.zero()

.. note:: 

   Warp uses a source-code transformation approach to auto-differentiation. In this approach the backwards pass must keep a record of intermediate values computed during the foward pass. This imposes some restrictions on what kernels can do and still be differentiable:

   * Dynamic loops should not mutate any previously declared local variable. This means the loop must be side-effect free. A simple way to ensure this is to move the loop body into a separate function. Static loops that are unrolled at compile time do not have this restriction and can perform any computation.
         
   * Kernels should not overwrite any previously used array values except to perform simple linear add/subtract operations (e.g.: via ``wp.atomic_add()``)

Jacobians
#########

To compute the Jacobian matrix :math:`J\in\mathbb{R}^{m\times n}` of a multi-valued function :math:`f: \mathbb{R}^n \to \mathbb{R}^m`, we can evaluate an entire row of the Jacobian in parallel by finding the Jacobian-vector product :math:`J^\top \mathbf{e}`. The vector :math:`\mathbf{e}\in\mathbb{R}^m` selects the indices in the output buffer to differentiate with respect to.
In Warp, instead of passing a scalar loss buffer to the ``tape.backward()`` method, we pass a dictionary ``grads`` mapping from the function output array to the selection vector :math:`\mathbf{e}` having the same type::

   # compute the Jacobian for a function of single output
   jacobian = np.empty((ouput_dim, input_dim), dtype=np.float32)
   tape = wp.Tape()
   with tape:
      output_buffer = launch_kernels_to_be_differentiated(input_buffer)
   for output_index in range(output_dim):
      # select which row of the Jacobian we want to compute
      select_index = np.zeros(output_dim)
      select_index[output_index] = 1.0
      e = wp.array(select_index, dtype=wp.float32)
      # pass input gradients to the output buffer to apply selection
      tape.backward(grads={output_buffer: e})
      q_grad_i = tape.gradients[input_buffer]
      jacobian[output_index, :] = q_grad_i.numpy()
      tape.zero()

When we run simulations independently in parallel, the Jacobian corresponding to the entire system dynamics is a block-diagonal matrix. In this case, we can compute the Jacobian in parallel for all environments by choosing a selection vector that has the output indices active for all environment copies. For example, to get the first rows of the Jacobians of all environments, :math:`\mathbf{e}=[\begin{smallmatrix}1 & 0 & 0 & \dots & 1 & 0 & 0 & \dots\end{smallmatrix}]^\top`, to compute the second rows, :math:`\mathbf{e}=[\begin{smallmatrix}0 & 1 & 0 & \dots & 0 & 1 & 0 & \dots\end{smallmatrix}]^\top`, etc.::

   # compute the Jacobian for a function over multiple environments in parallel
   jacobians = np.empty((num_envs, ouput_dim, input_dim), dtype=np.float32)
   tape = wp.Tape()
   with tape:
      output_buffer = launch_kernels_to_be_differentiated(input_buffer)
   for output_index in range(output_dim):
      # select which row of the Jacobian we want to compute
      select_index = np.zeros(output_dim)
      select_index[output_index] = 1.0
      # assemble selection vector for all environments (can be precomputed)
      e = wp.array(np.tile(select_index, num_envs), dtype=wp.float32)
      tape.backward(grads={output_buffer: e})
      q_grad_i = tape.gradients[input_buffer]
      jacobians[:, output_index, :] = q_grad_i.numpy().reshape(num_envs, input_dim)
      tape.zero()

.. autoclass:: Tape
   :members:

Graphs
-----------

Launching kernels from Python introduces significant additional overhead compared to C++ or native programs. To address this, Warp exposes the concept of `CUDA graphs <https://developer.nvidia.com/blog/cuda-graphs/>`_ to allow recording large batches of kernels and replaying them with very little CPU overhead.

To record a series of kernel launches use the ``warp.capture_begin()`` and ``warp.capture_end()`` API as follows: ::

   # begin capture
   wp.capture_begin()

   # record launches
   for i in range(100):
      wp.launch(kernel=compute1, inputs=[a, b], device="cuda")

   # end capture and return a graph object
   graph = wp.capture_end()


Once a graph has been constructed it can be executed: ::

   wp.capture_launch(graph)

Note that only launch calls are recorded in the graph, any Python executed outside of the kernel code will not be recorded. Typically it only makes sense to use CUDA graphs when the graph will be reused / launched multiple times.

.. autofunction:: capture_begin
.. autofunction:: capture_end
.. autofunction:: capture_launch

Interopability
-----------------

Warp can interop with other Python-based frameworks such as NumPy through standard interface protocols.

NumPy
#####

Warp arrays may be converted to a NumPy array through the ``warp.array.numpy()`` method. When the Warp array lives on the ``cpu`` device this will return a zero-copy view onto the underlying Warp allocation. If the array lives on a ``cuda`` device then it will first be copied back to a temporary buffer and copied to NumPy.

Warp CPU arrays also implement  the ``__array_interface__`` protocol and so can be used to construct NumPy arrays directly::

   w = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu")
   a = np.array(w)
   print(a)   
   > [1. 2. 3.]


PyTorch
#######

Warp provides helper functions to convert arrays to/from PyTorch. Please see the ``warp.torch`` module for more details. Example usage is shown below::

   import warp.torch

   w = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu")

   # convert to Torch tensor
   t = warp.to_torch(w)

   # convert from Torch tensor
   w = warp.from_torch(t)


CuPy/Numba
##########

Warp GPU arrays support the ``__cuda_array_interface__`` protocol for sharing data with other Python GPU frameworks. Currently this is one-directional, so that Warp arrays can be used as input to any framework that also supports the ``__cuda_array_interface__`` protocol, but not the other way around.

JAX
###

Interop with JAX arrays is not currently well supported, although it is possible to first convert arrays to a Torch tensor and then to JAX via. the dlpack mechanism.


Debugging
---------

Printing Values
#################

Often one of the best debugging methods is to simply print values from kernels. Warp supports printing all built-in types using the ``print()`` function, e.g.::

   v = wp.vec3(1.0, 2.0, 3.0)

   print(v)   

In addition, formatted C-style printing is available through the ``printf()`` function, e.g.::

   x = 1.0
   i = 2

   wp.printf("A float value %f, an int value: %d", x, i)

.. note:: Formatted printing is only available for scalar types (e.g.: ``int`` and ``float``) not vector types.

Printing Launches
#################

For complex applications it can be difficult to understand the order-of-operations that lead to a bug. To help diagnose these issues Warp supports a simple option to print out all launches and arguments to the console::

   wp.config.print_launches = True


Step-Through Debugging
######################

It is possible to attach IDE debuggers such as Visual Studio to Warp processes to step through generated kernel code. Users should first compile the kernels in debug mode by setting::
   
   wp.config.mode = "debug"
   
This setting ensures that line numbers, and debug symbols are generated correctly. After launching the Python process, the debugger should be attached, and a breakpoint inserted into the generated code (exported in the ``warp/gen`` folder).

.. note:: Generated kernel code is not a 1:1 correspondence with the original Python code, but individual operations can still be replayed and variables inspected.

Bounds Checking
###############

Warp will perform bounds checking in debug build configurations to ensure that all array accesses lie within the defined shape.

CUDA Verification
#################

It is possible to generate out-of-bounds memory access violations through poorly formed kernel code or inputs. In this case the CUDA runtime will detect the violation and put the CUDA context into an error state. Subsequent kernel launches may silently fail which can lead to hard to diagnose issues.

If a CUDA error is suspected a simple verification method is to enable::

   wp.config.verify_cuda = True

This setting will check the CUDA context after every operation to ensure that it is still valid. If an error is encountered it will raise an exception that often helps to narrow down the problematic kernel.

.. note:: Verifying CUDA state at each launch requires synchronizing CPU and GPU which has a significant overhead. Users should ensure this setting is only used during debugging.


.. _devices:

Devices
-------

Warp assigns unique string aliases to all supported compute devices in the system.  There is currently a single CPU device exposed as ``"cpu"``.  Each CUDA-capable GPU gets an alias of the form ``"cuda:i"``, where ``i`` is the CUDA device ordinal.  This convention should be familiar to users of other popular frameworks like PyTorch.

It is possible to explicitly target a specific device with each Warp API call using the ``device`` argument::

   a = wp.zeros(n, device="cpu")
   wp.launch(kernel, dim=a.size, inputs=[a], device="cpu")

   b = wp.zeros(n, device="cuda:0")
   wp.launch(kernel, dim=b.size, inputs=[b], device="cuda:0")

   c = wp.zeros(n, device="cuda:1")
   wp.launch(kernel, dim=c.size, inputs=[c], device="cuda:1")

.. note::

   A Warp CUDA device (``"cuda:i"``) corresponds to the primary CUDA context of device ``i``.  This is compatible with frameworks like PyTorch and other software that uses the CUDA Runtime API.  It makes interoperability easy, because GPU resources like memory can be shared with Warp.

Default Device
##############

To simplify writing code, Warp has the concept of **default device**.  When the ``device`` argument is omitted from a Warp API call, the default device will be used.

During Warp initialization, the default device is set to be ``"cuda:0"`` if CUDA is available.  Otherwise, the default device is ``"cpu"``.

The function ``wp.set_device()`` can be used to change the default device::

   wp.set_device("cpu")
   a = wp.zeros(n)
   wp.launch(kernel, dim=a.size, inputs=[a])
   
   wp.set_device("cuda:0")
   b = wp.zeros(n)
   wp.launch(kernel, dim=b.size, inputs=[b])
   
   wp.set_device("cuda:1")
   c = wp.zeros(n)
   wp.launch(kernel, dim=c.size, inputs=[c])

.. note::

   For CUDA devices, ``wp.set_device()`` does two things: it sets the Warp default device and it makes the device's CUDA context current.  This helps to minimize the number of CUDA context switches in blocks of code targeting a single device.

For PyTorch users, this function is similar to ``torch.cuda.set_device()``.  It is still possible to specify a different device in individual API calls, like in this snippet::

   # set default device
   wp.set_device("cuda:0")
   
   # use default device
   a = wp.zeros(n)
   
   # use explicit devices
   b = wp.empty(n, device="cpu")
   c = wp.empty(n, device="cuda:1")
   
   # use default device
   wp.launch(kernel, dim=a.size, inputs=[a])
   
   wp.copy(b, a)
   wp.copy(c, a)

Scoped Devices
##############

Another way to manage the default device is using ``wp.ScopedDevice`` objects.  They can be arbitrarily nested and restore the previous default device on exit::

   with wp.ScopedDevice("cpu"):
      # alloc and launch on "cpu"
      a = wp.zeros(n)
      wp.launch(kernel, dim=a.size, inputs=[a])
 
   with wp.ScopedDevice("cuda:0"):
      # alloc on "cuda:0"
      b = wp.zeros(n)
   
      with wp.ScopedDevice("cuda:1"):
         # alloc and launch on "cuda:1"
         c = wp.zeros(n)
         wp.launch(kernel, dim=c.size, inputs=[c])
   
      # launch on "cuda:0"
      wp.launch(kernel, dim=b.size, inputs=[b])

.. note::

   For CUDA devices, ``wp.ScopedDevice`` makes the device's CUDA context current and restores the previous CUDA context on exit.  This is handy when running Warp scripts as part of a bigger pipeline, because it avoids any side effects of changing the CUDA context in the enclosed code.

Current CUDA Device
###################

Warp uses the device alias ``"cuda"`` to target the current CUDA device.  This allows external code to manage the CUDA device on which to execute Warp scripts.  It is analogous to the PyTorch ``"cuda"`` device, which should be familiar to Torch users and simplify interoperation.

In this snippet, we use PyTorch to manage the current CUDA device and invoke a Warp kernel on that device::

   def example_function():
      # create a Torch tensor on the current CUDA device
      t = torch.arange(10, dtype=torch.float32, device="cuda")

      a = wp.from_torch(t)

      # launch a Warp kernel on the current CUDA device
      wp.launch(kernel, dim=a.size, inputs=[a], device="cuda")

   # use Torch to set the current CUDA device and run example_function() on that device
   torch.cuda.set_device(0)
   example_function()

   # use Torch to change the current CUDA device and re-run example_function() on that device
   torch.cuda.set_device(1)
   example_function()

.. note::

   Using the device alias ``"cuda"`` can be problematic if the code runs in an environment where another part of the code can unpredictably change the CUDA context.  Using an explicit CUDA device like ``"cuda:i"`` is recommended to avoid such issues.

Device Synchronization
######################

CUDA kernel launches and memory operations can execute asynchronously.  This allows for overlapping compute and memory operations on different devices.  Warp allows synchronizing the host with outstanding asynchronous operations on a specific device::

   wp.synchronize_device("cuda:1")

The ``wp.synchronize_device()`` function offers more fine-grained synchronization than ``wp.synchronize()``, as the latter waits for *all* devices to complete their work.

Custom CUDA Contexts (Advanced)
###############################

Warp is designed to work with arbitrary CUDA contexts so it can easily integrate into different workflows.

Applications built on the CUDA Runtime API target the *primary context* of each device.  The Runtime API hides CUDA context management under the hood.  In Warp, device ``"cuda:i"`` represents the primary context of device ``i``, which aligns with the CUDA Runtime API.

Applications built on the CUDA Driver API work with CUDA contexts directly and can create custom CUDA contexts on any device.  Custom CUDA contexts can be created with specific affinity or interop features that benefit the application.  Warp can work with these CUDA contexts as well.

The special device alias ``"cuda"`` can be used to target the current CUDA context, whether this is a primary or custom context.

In addition, Warp allows registering new device aliases for custom CUDA contexts, so that they can be explicitly targeted by name.  If the ``CUcontext`` pointer is available, it can be used to create a new device alias like this::

   wp.map_cuda_device("foo", ctypes.c_void_p(context_ptr))

Alternatively, if the custom CUDA context was made current by the application, the pointer can be omitted::

   wp.map_cuda_device("foo")

In either case, mapping the custom CUDA context allows us to target the context directly using the assigned alias::

   with wp.ScopedDevice("foo"):
      a = wp.zeros(n)
      wp.launch(kernel, dim=a.size, inputs=[a])
