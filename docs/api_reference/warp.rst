warp
====

.. automodule:: warp
   :no-members:

.. currentmodule:: warp

Submodules
----------

These modules are automatically available when you ``import warp``.

- :mod:`warp.config`
- :mod:`warp.types`
- :mod:`warp.utils`

Additional Submodules
---------------------

These modules must be explicitly imported (e.g., ``import warp.autograd``).

- :mod:`warp.autograd`
- :mod:`warp.fem`
- :mod:`warp.jax_experimental`
- :mod:`warp.optim`
- :mod:`warp.render`
- :mod:`warp.sparse`

Type Annotations
----------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   DeviceLike
   Float
   Int
   Scalar

Data Types
----------

Scalars
^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: _generated

   bool
   float16
   float32
   float64
   int8
   int16
   int32
   int64
   uint8
   uint16
   uint32
   uint64

Vectors
^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: _generated

   vec2
   vec3
   vec4
   vec2b
   vec2d
   vec2f
   vec2h
   vec2i
   vec2l
   vec2s
   vec2ub
   vec2ui
   vec2ul
   vec2us
   vec3b
   vec3d
   vec3f
   vec3h
   vec3i
   vec3l
   vec3s
   vec3ub
   vec3ui
   vec3ul
   vec3us
   vec4b
   vec4d
   vec4f
   vec4h
   vec4i
   vec4l
   vec4s
   vec4ub
   vec4ui
   vec4ul
   vec4us

Matrices
^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: _generated

   mat22
   mat33
   mat44
   mat22d
   mat22f
   mat22h
   mat33d
   mat33f
   mat33h
   mat44d
   mat44f
   mat44h
   matrix_from_cols
   matrix_from_rows

Quaternions
^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: _generated

   quat
   quatd
   quatf
   quath
   quat_between_vectors

Transformations
^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: _generated

   transform
   transformd
   transformf
   transformh
   transform_expand

Spatial Vectors and Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: _generated

   spatial_matrix
   spatial_matrixd
   spatial_matrixf
   spatial_matrixh
   spatial_vector
   spatial_vectord
   spatial_vectorf
   spatial_vectorh

Arrays
------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   array
   fixedarray
   tile
   array1d
   array2d
   array3d
   array4d
   clone
   copy
   empty
   empty_like
   from_ptr
   full
   full_like
   ones
   ones_like
   zeros
   zeros_like

Indexed Arrays
^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: _generated

   indexedarray
   indexedarray1d
   indexedarray2d
   indexedarray3d
   indexedarray4d

Spatial Acceleration
--------------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   Bvh
   BvhQuery
   BvhQueryTiled
   HashGrid
   HashGridQuery
   Mesh
   MeshQueryAABB
   MeshQueryAABBTiled
   MeshQueryPoint
   MeshQueryRay
   Texture
   Texture2D
   Texture3D
   TextureAddressMode
   TextureFilterMode
   Volume

Runtime
-------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   clear_kernel_cache
   clear_lto_cache
   init
   is_cpu_available
   is_cuda_available

Kernel Programming
------------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   WarpCodegenAttributeError
   WarpCodegenError
   WarpCodegenIndexError
   WarpCodegenKeyError
   WarpCodegenTypeError
   WarpCodegenValueError
   func
   func_grad
   func_native
   func_replay
   grad
   kernel
   map
   overload
   static
   struct

Kernel Execution
----------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   Function
   Kernel
   Launch
   Module
   launch
   launch_tiled
   synchronize

Automatic Differentiation
-------------------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   Tape

Device Management
-----------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   Device
   ScopedDevice
   get_cuda_device
   get_cuda_device_count
   get_cuda_devices
   get_cuda_driver_version
   get_cuda_supported_archs
   get_cuda_toolkit_version
   get_device
   get_devices
   get_preferred_device
   is_device_available
   map_cuda_device
   set_device
   synchronize_device
   unmap_cuda_device

Module Management
-----------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   compile_aot_module
   force_load
   get_module
   get_module_options
   load_aot_module
   load_module
   set_module_options

CUDA Stream Management
----------------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   ScopedStream
   Stream
   get_stream
   set_stream
   synchronize_stream
   wait_stream

CUDA Event Management
---------------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   Event
   get_event_elapsed_time
   record_event
   synchronize_event
   wait_event

CUDA Memory Management
----------------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   ScopedMempool
   ScopedMempoolAccess
   ScopedPeerAccess
   get_mempool_release_threshold
   get_mempool_used_mem_current
   get_mempool_used_mem_high
   is_mempool_access_enabled
   is_mempool_access_supported
   is_mempool_enabled
   is_mempool_supported
   is_peer_access_enabled
   is_peer_access_supported
   set_mempool_access_enabled
   set_mempool_enabled
   set_mempool_release_threshold
   set_peer_access_enabled

CUDA Graph Management
---------------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   ScopedCapture
   capture_begin
   capture_debug_dot_print
   capture_end
   capture_if
   capture_launch
   capture_while
   is_conditional_graph_supported

CUDA Interprocess Communication
-------------------------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   event_from_ipc_handle
   from_ipc_handle

Profiling
---------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   ScopedTimer
   TimingResult
   timing_begin
   timing_end
   timing_print

Timing Flags
^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: _generated

   TIMING_ALL
   TIMING_GRAPH
   TIMING_KERNEL
   TIMING_KERNEL_BUILTIN
   TIMING_MEMCPY
   TIMING_MEMSET

NumPy Interop
-------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   dtype_from_numpy
   dtype_to_numpy
   from_numpy

DLPack Interop
--------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   from_dlpack
   to_dlpack

JAX Interop
-----------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   device_from_jax
   device_to_jax
   dtype_from_jax
   dtype_to_jax
   from_jax
   to_jax

PyTorch Interop
---------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   device_from_torch
   device_to_torch
   dtype_from_torch
   dtype_to_torch
   from_torch
   stream_from_torch
   stream_to_torch
   to_torch

Omniverse Runtime Fabric Interop
--------------------------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   fabricarray
   indexedfabricarray
   fabricarrayarray
   indexedfabricarrayarray

Paddle Interop
--------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   device_from_paddle
   device_to_paddle
   dtype_from_paddle
   dtype_to_paddle
   from_paddle
   stream_from_paddle
   to_paddle

Constants
---------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   constant
   E
   HALF_PI
   INF
   LN2
   LN10
   LOG10E
   LOG2E
   NAN
   PHI
   PI
   TAU
   e
   half_pi
   inf
   ln2
   ln10
   log10e
   log2e
   nan
   phi
   pi
   tau

Misc
----

.. autosummary::
   :nosignatures:
   :toctree: _generated

   MarchingCubes
   RegisteredGLBuffer
