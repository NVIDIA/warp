Runtime Reference
=================

.. currentmodule:: warp

.. toctree::
   :maxdepth: 2

Initialization
--------------

Before initialization it is possible to set runtime options through the ``warp.config`` module, e.g.: ::

   import warp as wp

   wp.config.mode = "debug"
   wp.config.verify_cuda = True

   wp.init()

This example sets the kernel build mode to debug, and enables CUDA verification that will check the CUDA context for any errors after each kernel launch, which can be useful for debugging.

.. autofunction:: init

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

Kernels are launched with the ``warp.launch()`` function on a specific device (CPU/GPU). Note that all the kernel inputs must live on the target device, or a runtime exception will be raised.

.. autofunction:: launch

Arrays
------

Arrays are the fundamental memory abstraction in Warp; they are created through the following global constructors: ::

    wp.empty(n=1024, dtype=wp.vec3, device="cpu")
    wp.zeros(n=1024, dtype=float, device="cuda")
    

Arrays can also be constructed directly from ``numpy`` ndarrays as follows: ::

   r = np.random.rand(1024)

   # copy to Warp owned array
   a = wp.array(r, dtype=float, device="cpu")

   # return a Warp array wrapper around the numpy data (zero-copy)
   a = wp.array(r, dtype=float, copy=False, device="cpu")

   # return a Warp copy of the array data on the GPU
   a = wp.array(r, dtype=float, device="cuda")

Note that for multi-dimensional data the datatype, ``dtype`` parameter, must be specified explicitly, e.g.: ::

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

.. autofunction:: zeros
.. autofunction:: zeros_like
.. autofunction:: empty
.. autofunction:: empty_like

.. autoclass:: array

Data Types
----------

Warp provides built-in math and geometry types for common simulation and graphics problems. A full reference for operators and functions for these types is available in the :any:`functions`.

.. autoclass:: vec2
.. autoclass:: vec3
.. autoclass:: vec4
.. autoclass:: quat
.. autoclass:: mat22
.. autoclass:: mat33
.. autoclass:: mat44
.. autoclass:: transform
.. autoclass:: spatial_vector
.. autoclass:: spatial_matrix

Meshes
------

Warp provides a ``wp.Mesh`` class to manage triangle mesh data. To create a mesh users provide a points, indices and optionally a velocity array::

   mesh = wp.Mesh(points, velocities, indices)

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
      n = wp.vec3(0.0, 0.0, 0.0)       # hit face normal
      f = int(0)                       # hit face index

      color = wp.vec3(0.0, 0.0, 0.0)

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

Warp supports reading sparse volumetric grids stored using the `NanoVDB <https://developer.nvidia.com/nanovdb>`_ standard. Users can access voxels directly, or use built-in closest point or trilinear interpolation to sample grid data from world or local-space.

Below we give an example of creating a Volume object from an existing NanoVDB file::

   # load NanoVDB bytes from disk
   file = np.fromfile("mygrid.nvdbraw", dtype=np.byte)

   # create Volume object
   volume = wp.Volume(wp.array(file, device="cpu"))

To sample the volume inside a kernel we pass a reference to it by id, and use the built-in sampling modes::

   @wp.kernel
   def sample_grid(volume: wp.uint64,
                   points: wp.array(dtype=wp.vec3),
                   samples: wp.array(dtype=float)):

      tid = wp.tid()

      # load sample point in world-space
      p = points[tid]

      # sample grid with trilinear interpolation     
      f = wp.volume_sample_world(volume, p, wp.Volume.LINEAR)

      # write result
      samples[tid] = f



.. note:: Warp does not currently support modifying sparse-volumes at runtime. We expect to address this in a future update. Users should create volumes using standard VDB tools such as OpenVDB, Blender, Houdini, etc.

.. autoclass:: Volume
   :members:

Hash Grids
----------

Warp provides a hash-grid data structure for performing fast spatial nearest neighbor queries. Hash grids are created as follows:



.. autoclass:: HashGrid
   :members:

Differentiability
-----------------

By default Warp generates a foward and backward (adjoint) version of each kernel definition. Buffers that participate in the chain of computation should be created with ``requires_grad=True``, for example::

   a = wp.zeros(1024, dtype=wp.vec3, device="cuda", requires_grad=True)

The ``wp.Tape`` class can then be used to record kernel launches, and replay them to compute the gradient of a scalar loss function with respect to the kernel inputs::

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

.. autoclass:: Tape
   :members:

CUDA Graphs
-----------

Launching kernels from Python introduces significant additional overhead compared to C++ or native programs. To address this, Warp exposes the concept of `CUDA graphs <https://developer.nvidia.com/blog/cuda-graphs/>`_ to allow recording large batches of kernels and replaying them with very little CPU overhead.

To record a series of kernel launches use the ``wp.capture_begin()`` and ``wp.capture_end()`` API as follows: ::

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

