Rendering
=========

.. currentmodule:: warp.render

The :mod:`warp.render` module provides a set of renderers that can be used for visualizing scenes involving shapes of various types.

Standalone Renderers
--------------------

The :class:`OpenGLRenderer <warp.render.OpenGLRenderer>` provides an interactive renderer to play back animations in real time and is mostly intended for debugging,
whereas more sophisticated rendering can be achieved with the help of the :class:`UsdRenderer <warp.render.UsdRenderer>`, which allows exporting the scene to
a USD file that can then be rendered in an external 3D application or renderer of your choice.



CUDA Graphics Interface
-----------------------

Warp provides a CUDA graphics interface that allows you to access OpenGL buffers from CUDA kernels.
This is useful for manipulating OpenGL array buffers without having to copy them back and forth between the CPU and GPU.

See the `CUDA documentation on OpenGL Interoperability <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GL.html>`_ for more information.

The :class:`wp.RegisteredGLBuffer <warp.RegisteredGLBuffer>` class wraps an OpenGL buffer and registers it with CUDA, allowing it to be
mapped as a Warp array for use in kernels without copying data between the CPU and GPU.

