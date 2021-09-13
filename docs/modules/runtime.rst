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

This example sets the kernel build mode to debug, and ensure verify the CUDA context for any errors after each kernel launch, which can be useful for debugging.

.. autofunction:: init

Kernels
-------

In Warp, kernels are defined as Python functions, decorated with the ``@warp.kernel`` decorator. Kernels have a 1:1 correspondence with CUDA kernels, and may be launched with any number of parallel execution threads. ::

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

Arrays are the fundamental memory abstraction in Warp, they are created through the following global constructors: ::

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

CUDA Graphs
-----------

Launching kernels from Python introduces significant additional overhead compared to C++ or native programs. To address this Warp exposes the concept of `CUDA graphs <https://developer.nvidia.com/blog/cuda-graphs/>`_ to allow recording large batches of kernels and replaying them with very little CPU overhead.

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

Data Types
----------

.. autoclass:: vec3
.. autoclass:: vec4
.. autoclass:: quat
.. autoclass:: mat22
.. autoclass:: mat33
.. autoclass:: mat44
.. autoclass:: spatial_transform
.. autoclass:: spatial_vector
.. autoclass:: spatial_matrix


Differentiability
-----------------

By default Warp generates a foward and backward (adjoint) version of each kernel definition. The ``wp.Tape`` class can be used to record kernel launches, and replay them to compute the gradient of a scalar loss function with respect to the kernel inputs. ::

   tape = wp.Tape()

   # forward pass
   with tape:
      wp.launch(kernel=compute1, inputs=[a, b], device="cuda")
      wp.launch(kernel=compute2, inputs=[c, d], device="cuda")
      wp.launch(kernel=loss, inputs=[d, l], device="cuda")

   # reverse pass
   tape.backward()

After the backward pass has completed the gradients w.r.t. inputs are available via. a mapping in the Tape object: ::

   # gradient of loss with respect to input a
   print(tape.adjoints[a])


