Programming Model
=================

Warp programs span two execution environments. Python code configures work,
allocates resources, and launches operations from the host, while kernels and
user functions run as statically typed native code on a CPU or CUDA device.
Understanding that boundary also explains when Warp compiles code, why values
sometimes become compile-time constants, and when specialized programming
features are useful.

Guides in this section
----------------------

* :doc:`Generics <programming_model/generics>`: Write kernels and functions
  that specialize for multiple concrete data types.
* :doc:`Tiles <programming_model/tiles>`: Use cooperative thread-block
  programming for structured computation and data reuse.
* :doc:`Code Generation <programming_model/code_generation>`: Understand
  compilation, module hashing, caching, generated code, and ahead-of-time
  workflows.
* :doc:`C++ and CUDA Workflows <programming_model/cpp_cuda_workflows>`:
  Integrate native code, inline PTX, generated source, and standalone graph
  replay.

.. _python-scope-vs-kernel-scope-api:

Python Scope and Kernel Scope
-----------------------------

Some of the Warp API can only be called from the Python scope (i.e. outside of Warp user functions and kernels),
while others can only be called from the kernel scope.

The Python-scope API is documented in the :doc:`API Reference </api_reference/warp>`,
while the kernel-scope API is documented under :doc:`Built-Ins </language_reference/builtins>`.
Many kernel-scope functions can also be called from Python.

Not all of the Python language is supported inside the kernel scope. Some features haven't been implemented yet, while
other features do not map well to the GPU from a performance perspective.

See the :doc:`Limitations <limitations>` documentation for more details.

.. _Compilation Model:

Compilation Model
-----------------

Warp uses a Python->C++/CUDA compilation model that generates kernel code from Python function definitions.
All kernels belonging to a Python module are runtime compiled into dynamic libraries and PTX.
The result is then cached between application restarts for fast startup times.

Note that compilation is triggered on the first kernel launch for that module.
Any kernels registered in the module with :func:`@wp.kernel <warp.kernel>` will be included in the shared library.

.. image:: ../img/compiler_pipeline.svg

For more information, see the :doc:`programming_model/code_generation` section.

Execution Models
----------------

An ordinary Warp kernel maps each point in its launch grid to one logical
thread. The thread obtains its index with :func:`wp.tid() <warp.tid>` and works
independently unless the algorithm uses explicit synchronization or atomic
operations. This SIMT model is the default for element-wise and irregular
parallel work.

:doc:`programming_model/tiles` provides a cooperative alternative for block-level
algorithms. Tile operations let threads jointly load, transform, and store
multi-dimensional regions of data, which is useful for dense linear algebra and
other workloads with structured data reuse.

.. toctree::
   :hidden:
   :titlesonly:

   programming_model/generics
   programming_model/tiles
   programming_model/code_generation
   programming_model/cpp_cuda_workflows
