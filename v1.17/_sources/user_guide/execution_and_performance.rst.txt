Execution and Performance
=========================

Warp performance depends on more than device kernel time. Python submits work,
Warp may compile or load native modules, arrays may allocate or move memory, and
CUDA devices usually execute their queued operations asynchronously. Separate
those costs before deciding what to optimize.

Guides in this section
----------------------

* :doc:`Concurrency <execution_and_performance/concurrency>`: Streams,
  events, synchronization, graph capture, and multi-device execution.
* :doc:`Memory Allocation and Access <execution_and_performance/memory_management>`:
  Allocation strategies, memory pools, residency, and cross-device access.
* :doc:`Profiling <execution_and_performance/profiling>`: Measuring host and
  device work with Warp timers, NVTX, and NVIDIA profiling tools.
* :doc:`Deterministic Execution <execution_and_performance/deterministic_execution>`:
  Reproducibility guarantees, limitations, and performance tradeoffs.

.. toctree::
   :hidden:
   :titlesonly:

   execution_and_performance/concurrency
   execution_and_performance/memory_management
   execution_and_performance/profiling
   execution_and_performance/deterministic_execution
