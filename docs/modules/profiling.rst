Profiling
=========

``wp.ScopedTimer`` objects can be used to gain some basic insight into the performance of Warp applications::

   with wp.ScopedTimer("grid build"):
      self.grid.build(self.x, self.point_radius)

This results in a printout at runtime to the standard output stream like::

   grid build took 0.06 ms

The ``wp.ScopedTimer`` object does not synchronize (e.g. by calling ``wp.synchronize()``)
upon exiting the ``with`` statement, so this can lead to misleading numbers if the body
of the ``with`` statement launches device kernels.

When a ``wp.ScopedTimer`` object is passed ``use_nvtx=True`` as an argument, the timing functionality is replaced by calls
to ``nvtx.start_range()`` and ``nvtx.end_range()``::

   with wp.ScopedTimer("grid build", use_nvtx=True, color="cyan"):
      self.grid.build(self.x, self.point_radius)

These range annotations can then be collected by a tool like `NVIDIA Nsight Systems <https://developer.nvidia.com/nsight-systems>`_
for visualization on a timeline, e.g.::

   nsys profile python warp_application.py

This code snippet also demonstrates the use of the ``color`` argument to specify a color
for the range, which may be a number representing the ARGB value or a recognized string
(refer to `colors.py <https://github.com/NVIDIA/NVTX/blob/release-v3/python/nvtx/colors.py>`_ for
additional color examples).
The `nvtx module <https://github.com/NVIDIA/NVTX>`_ must be
installed in the Python environment for this capability to work.
An equivalent way to create an NVTX range without using ``wp.ScopedTimer`` is::

   import nvtx

   with nvtx.annotate("grid build", color="cyan"):
      self.grid.build(self.x, self.point_radius)

This form may be more convenient if the user does not need to frequently switch
between timer and NVTX capabilities of ``wp.ScopedTimer``. 

.. autoclass:: warp.ScopedTimer
