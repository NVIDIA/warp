.. _faq:

FAQ
===

Use this page for quick answers to common Warp questions. Each answer links to
the maintained details. The :doc:`publications` page collects examples of Warp
in research and production projects.

About Warp
----------

What is Warp, and how does it fit into an existing application?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Warp is a Python framework for writing kernels that run on CPUs and NVIDIA
GPUs. Authors express parallel work through logical thread indices, array
operations, and kernel launches. Warp maps that work onto the target device and
handles many lower-level execution details. Warp lowers typed Python kernel
code to generated C++ or CUDA C++ source. It then uses LLVM/Clang for CPU code
or the CUDA runtime compiler (NVRTC) for CUDA code. The :doc:`basics` and
:doc:`../deep_dive/codegen` guides explain the programming and compilation
models.

Warp generates reverse-mode automatic differentiation code for supported
kernels and functions. Its spatial API includes GPU-accelerated BVHs, hash
grids, triangle meshes, and sparse volumes, together with the query primitives
needed to use them. Warp also includes sparse linear algebra and a finite
element toolkit for building simulation and partial differential equation
(PDE) solvers. See :doc:`differentiability`, the
:doc:`Warp API reference <../api_reference/warp>`, the
:doc:`sparse API <../api_reference/warp_sparse>`, and the :doc:`finite element
method (FEM) toolkit <../domain_modules/fem>`.

Warp is designed to interoperate with existing Python libraries and frameworks.
An application can use it for a single kernel, a performance-critical
subsystem, or most of its computation. Array interfaces, DLPack, and dedicated
converters enable data sharing without a copy when the underlying protocol
allows it, so existing NumPy, PyTorch, JAX, and other framework code can remain
in place. :doc:`interoperability` documents the available paths.

What are some projects built with Warp?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The projects below use Warp for kernels, interoperability, or as the basis of a
larger application:

* `Newton <https://github.com/newton-physics/newton>`__ is a GPU-accelerated
  physics engine for robotics and simulation research. Its higher-level
  models, states, solvers, importers, and viewers build on Warp's arrays,
  runtime, and GPU programming model.
* `waxMorph
  <https://github.com/Computational-Morphogenomics-Group/waxmorph>`__ is a
  differentiable cell-based framework for three-dimensional morphogenesis.
  Its forward-simulation kernels and primary rendering paths use Warp, while
  learned emulation workflows integrate with PyTorch and JAX.
* `NVIDIA ALCHEMI Toolkit-Ops
  <https://github.com/NVIDIA/nvalchemi-toolkit-ops>`__ implements batched
  operations for computational chemistry and atomistic simulation. Its Warp
  kernels cover neighbor lists, molecular-dynamics integrators, dispersion
  corrections, and electrostatics, with optional PyTorch and JAX bindings.
* `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`__ is a
  GPU-accelerated implementation of MuJoCo for NVIDIA hardware that uses Warp
  kernels for high-throughput parallel robotics simulation. Google DeepMind
  and NVIDIA maintain it as part of Newton.
* `XLB <https://github.com/Autodesk/XLB>`__ is a differentiable
  lattice-Boltzmann library for fluid simulation and physics-based machine
  learning. Warp is one of its main compute backends and implements fluid
  operators as Python-authored GPU kernels.
* `NCLaw <https://github.com/PingchuanMa/NCLaw>`__ is the International
  Conference on Machine Learning (ICML) 2023 research implementation of neural
  constitutive laws learned from observed motion. It couples PyTorch training
  to a differentiable material-point simulation built with an older vendored
  version of Warp, so it is best treated as a research artifact rather than a
  current starter project.

See :doc:`publications` for a broader collection.

How does Warp compare with other Python GPU programming approaches?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Warp is built for authoring typed kernels and can run alongside other
frameworks. The main differences are in the level of programming each exposes.

PyTorch and JAX center on tensor operations and program transformations. With
Warp, authors express parallel work as kernels over logical threads and can
control memory access and kernel decomposition when needed. Warp maps that work
onto the target device. This flexibility suits irregular and sparse algorithms,
simulations, and geometry processing.

`CuPy <https://cupy.dev/>`__ provides NumPy- and SciPy-compatible arrays on
NVIDIA and AMD GPUs. Many operations call optimized GPU libraries, and CuPy
also has custom-kernel and fusion APIs. Warp's API is organized around
authored kernels and includes automatic differentiation and spatial APIs.

`Triton <https://triton-lang.org/>`__ is a language and compiler for custom GPU
kernels. Its blocked programming model targets dense machine-learning tensor
computations. Warp primarily exposes explicit single-instruction,
multiple-thread (SIMT) execution, also supports cooperative tile operations,
and can run kernels on CPUs as well as NVIDIA GPUs.

An application can also combine Warp with these frameworks, libraries, and
languages. On CUDA, Warp arrays interoperate with PyTorch, JAX, CuPy, and other
array libraries through framework adapters and standard exchange protocols.
See :doc:`interoperability` for the supported paths.

How does Warp's tile programming model differ from cuTile Python?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Warp tiles and cuTile Python use different compiler models. In Warp, tiles are
cooperative abstractions within ordinary `SIMT kernels
<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html>`__.
Warp generates CUDA C++ and passes it to NVRTC, which produces PTX or a compiled
CUDA binary (cubin). cuTile Python instead targets the CUDA Tile intermediate
representation (Tile IR), a tile-native virtual instruction set. Its compiler
maps tile programs onto GPU threads, memory, Tensor Cores, and other hardware.

The Python authoring experience looks similar because both models use decorated
Python kernels, multidimensional array and tile values, and explicit load and
store operations. Warp tiles still run as part of SIMT kernels. cuTile leaves
the mapping of individual threads to the Tile IR compiler.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Dimension
     - Warp tiles
     - cuTile Python
   * - Machine model
     - Explicit SIMT threads and blocks with cooperative tile operations
     - First-class tile blocks and multidimensional tile values
   * - Hardware mapping
     - Generated CUDA C++ follows SIMT execution; selected operations call
       device libraries that can use Tensor Cores
     - Tile IR compiler mapping to threads, memory, Tensor Cores, and the
       Tensor Memory Accelerator (TMA) for supported access patterns
   * - Control
     - Users choose ``block_dim`` and may mix SIMT and tile code
     - Individual hardware threads are hidden and mapped by the compiler
   * - Library scope
     - CPU and CUDA kernel runtime with built-in reverse-mode automatic
       differentiation, BVHs, triangle meshes, hash grids, sparse volumes, and
       spatial query primitives
     - GPU-focused tile language and compiler for authoring custom kernels

Both models can use Tensor Cores, but their access to other hardware features
differs. The CUDA Tile compiler can lower structured tile-space loads to the
`Tensor Memory Accelerator (TMA)
<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html>`__
on supported GPUs. Warp's tile load and store implementation does not currently
use TMA. This is a difference between the current implementations, not an
inherent limitation of CUDA C++ or PTX.

Warp's tile backend does not currently use CUDA Tile IR or CUDA Tile C++. See
Warp's :doc:`tiles` guide, the `Tile IR documentation
<https://docs.nvidia.com/cuda/tile-ir/latest/>`__, and the `CUDA Tile C++ guide
<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html>`__.

Is Warp a physics engine, and how does Newton relate to it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Warp itself is not a turnkey physics engine. It provides a programming model,
geometry and numerical primitives, automatic differentiation, and examples for
building simulators or accelerating parts of a larger physics system.

`Newton <https://github.com/newton-physics/newton>`__ is a separate
GPU-accelerated physics simulation engine built on Warp. Warp supplies Newton's
kernel compiler, runtime, arrays, geometry operations, and automatic
differentiation. Newton adds models, states, solvers, importers, viewers, and
physics workflows. It extends and generalizes the former ``warp.sim`` module,
with MuJoCo Warp as its primary backend.

Applications that need a ready-made simulation stack can start with Newton.
Applications that need custom kernels or lower-level computation can use Warp
directly, including alongside Newton. See Newton's `migration guide
<https://newton-physics.github.io/newton/migration.html>`__ and Warp's
:doc:`publications` page for examples.

Installation and Compatibility
------------------------------

Which operating systems, Python versions, and GPUs does Warp support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Warp requires Python 3.10 or newer and supports Windows and Linux on x86-64,
Linux on ARM64, and Apple Silicon macOS. CUDA acceleration needs a supported
NVIDIA GPU and driver; macOS uses the CPU backend.

Python, operating-system, GPU-architecture, and driver requirements may change
between releases. Check :doc:`compatibility` for the current requirements.

Does Warp support NVIDIA Grace and Grace Blackwell systems such as DGX Spark and GB200?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, yes. Warp publishes Linux AArch64 packages and supports CPU and CUDA
execution on Linux AArch64 systems. That includes Grace, Grace Hopper, and Grace
Blackwell platforms when the system meets the current driver and runtime
requirements. Run :func:`wp.print_diagnostics() <warp.print_diagnostics>` to
see the installed Warp build and the devices it detects.

On DGX Spark, use ``warp-lang[examples]`` when the examples or Universal Scene
Description (USD) rendering are needed. It installs ``usd-exchange`` because
``usd-core`` does not publish Linux AArch64 wheels. The core ``warp-lang``
package does not require either one. See :doc:`installation` for the current
package choices.

CPU/GPU coherence does not make ordinary Warp CPU and CUDA arrays
interchangeable. Standard Warp CUDA arrays are not managed-memory allocations.
Warp supports opt-in CUDA managed-memory arrays through
:class:`wp.CudaManagedAllocator <warp.CudaManagedAllocator>`. On systems where
CUDA reports compatible access, CPU and GPU code can address these arrays
subject to CUDA managed-memory synchronization rules. See :ref:`Managed Memory
Allocator <managed_memory_allocation_options>` for the allocation options and
usage example.

For the underlying memory model on Grace Hopper and Grace Blackwell, see the
CUDA Programming Guide's `Unified and System Memory
<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html>`__
chapter.

Query the device capability properties and use :func:`wp.can_access()
<warp.can_access>` before relying on cross-device access. On multi-GPU GB200
systems, Warp exposes each GPU as a separate CUDA device. Applications still
choose devices and coordinate data movement explicitly. Performance depends on
the kernel and its memory-access pattern, so profile the target system. See
:doc:`../deep_dive/memory_access`, :doc:`devices`, and
:doc:`../deep_dive/profiling`.

Do I need to install the CUDA Toolkit?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pre-built Warp package does not require a system CUDA Toolkit. CUDA-enabled
packages include the components that Warp needs, but the system still needs a
compatible NVIDIA driver.

Building Warp with CUDA support from source does require a CUDA Toolkit. If a
build uses shared CUDA libraries, those libraries must also be available at
runtime. The current driver and build requirements are in
:doc:`installation`.

Which Warp package or build should I install?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most users should install the stable ``warp-lang`` package from PyPI. The
`warp-lang packages on conda-forge <https://anaconda.org/conda-forge/warp-lang>`__
provide managed CPU and CUDA variants. Nightly packages contain unreleased
changes, GitHub Releases provide wheels for alternate CUDA runtimes, and source
builds support custom toolchains or build options.

Package variants and commands change over time. Follow :doc:`installation`
instead of copying a version-specific command from an old issue or message.

Programming with Warp
---------------------

What Python features can I use inside Warp kernels and functions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Host-side Warp code is ordinary Python. Code inside
:func:`@wp.kernel <warp.kernel>` and :func:`@wp.func <warp.func>`, however, runs
without a Python interpreter and is limited to Warp's documented language
subset, which includes typed scalar and aggregate values, control flow,
fixed-size local values, and Warp's device-callable built-ins.

Compiled code cannot use dynamic containers, list comprehensions, lambdas,
exceptions, recursion, ``eval()``, or arbitrary Python and standard-library
calls. Check :doc:`limitations` and the
:doc:`../language_reference/builtins` reference for a specific feature.

Why is the first kernel launch slow, and when does Warp recompile?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first launch can be slow when Warp must compile the kernel's module for the
selected device. Warp generates C++ or CUDA source, compiles it, and saves the
result in its kernel cache. Later processes can load a matching cached module
much faster.

Warp caches kernels by module. The generated module content and target form the
cache key, so changes to kernel code, referenced compile-time values, overloads,
module options, or the target can trigger another compilation.

Module-load messages distinguish a compiled module from one loaded out of the
cache. Initialization reports the cache location and lets applications
configure it for deployment or process isolation. Warm up the required modules
before measuring execution, and measure cold-start compilation separately if
startup latency matters.

In containers and other ephemeral environments, put the cache on storage that
survives the process. Set ``WARP_CACHE_PATH`` to a writable mounted directory,
or set :attr:`warp.config.kernel_cache_dir` before calling
:func:`wp.init() <warp.init>`. For a fixed deployment, warm the required modules
while building the image or follow :ref:`Ahead-of-Time Compilation Workflows
<ahead_of_time_compilation_workflows>`.

If several workers may compile at once, populate a shared cache before starting
them or give each worker a separate writable cache directory. Multiple
processes should not populate the same cache concurrently.
:doc:`../deep_dive/codegen` describes module hashing and compilation in detail.

Why can't I assign individual Warp array elements from Python?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Warp does not expose element assignment from Python because a single write to
GPU memory would require hidden synchronization and data transfer. Bulk
operations make those costs explicit and preserve asynchronous execution.

Create arrays from Python, NumPy, or another supported framework. For bulk
updates, use
:meth:`array.fill_() <warp.array.fill_>`,
:meth:`array.assign() <warp.array.assign>`, or
:func:`wp.copy() <warp.copy>`. Launch a kernel when writes need individual
indices. The :ref:`Arrays` section of :doc:`runtime` covers these operations.

How do runtime values, constants, generics, and specialized kernels affect compilation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass frequently changing scalar values as kernel arguments. Ordinary runtime
arguments do not need a new specialization. Supported Python values referenced
from kernel scope, however, are folded into generated code. Concrete generic
types and generated kernels can also produce new compiled module content.

Rebinding a captured Python value after its module has loaded does not update
the compiled code. Keep compile-time choices stable, explicitly instantiate the
generic overloads needed for deployment, and cache runtime-generated kernel
objects in application code. See
:ref:`External References and Constants <external_references>` and
:doc:`generics`.

Devices, Memory, and Execution
------------------------------

How does CPU execution differ from CUDA execution?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPU kernel launches currently run serially and synchronously, whereas CUDA
launches run many threads in parallel and are generally asynchronous with
respect to Python. Both devices use the same kernel language, but they have
different performance and concurrency characteristics.

Tile kernels need additional care on the CPU backend because its effective
``block_dim`` is one. Consult :ref:`CPU Tile Semantics <cpu_tile_semantics>`
when the same tile kernel must run on both backends. Other device differences
are covered in :doc:`devices`.

When do I need to synchronize explicitly?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If an application uses the CPU backend, or one GPU on one CUDA stream, it
generally does not need explicit synchronization. CPU launches are synchronous,
CUDA operations on the same stream are ordered, and convenience readbacks such
as :meth:`array.numpy() <warp.array.numpy>` perform the required wait.

Explicit ordering matters when work crosses streams, devices, or frameworks
that have not already established a dependency. Prefer device-side ordering
with :func:`wp.wait_stream() <warp.wait_stream>` or
:func:`wp.record_event() <warp.record_event>` and
:func:`wp.wait_event() <warp.wait_event>`. These operations do not block Python.

When Python itself must wait, choose the narrowest host-blocking call:
:func:`wp.synchronize_event() <warp.synchronize_event>`,
:func:`wp.synchronize_stream() <warp.synchronize_stream>`, or
:func:`wp.synchronize_device() <warp.synchronize_device>`. Use
:func:`wp.synchronize() <warp.synchronize>` only when every device must finish.
One easy-to-miss case is an asynchronous :func:`wp.copy() <warp.copy>` into a
pinned CPU array. Wait for the relevant event, stream, or device before reading
the destination. See :ref:`Synchronization Guidance
<synchronization_guidance>`.

How do I safely use Warp with external or non-blocking CUDA streams?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An external stream is not necessarily non-blocking. Streams created by Warp are
blocking. PyTorch's CUDA default stream is also blocking, while non-default
streams created with ``torch.cuda.Stream()`` are non-blocking. Converting or
wrapping a stream preserves this behavior; check
:attr:`wp.Stream.is_blocking <warp.Stream.is_blocking>` when in doubt.

Every Warp resource used on a non-blocking stream must remain alive until work
on that stream has completed.

Before releasing temporary arrays, meshes, hash grids, or volumes, synchronize
the non-blocking stream or make a Warp-created blocking stream wait for it. A
stream wait keeps Python asynchronous. See
:ref:`Blocking and Non-Blocking Streams <nonblocking_streams>` and the CUDA
Programming Guide's `Asynchronous Execution
<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html>`__
chapter.

When are arrays copied, and when is memory shared?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constructing an array copies data unless the chosen API explicitly creates a
view. CPU arrays can share memory with NumPy. Framework converters and array
interfaces can share compatible CPU or CUDA allocations. DLPack can exchange
zero-copy views with standardized stream ordering. Calling
:meth:`array.numpy() <warp.array.numpy>` on a CUDA array copies its data to host
memory.

The original producer still owns an allocation shared through a zero-copy
wrapper. The producer and consumer must follow the lifetime rules of the
selected protocol. :doc:`interoperability` contains the conversion matrix and
lifetime guidance.

How do I use multiple GPUs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-GPU Warp programs usually follow one of these execution models:

* One Python process can control several GPUs on the same host. Use explicit
  ``cuda:i`` device aliases to allocate arrays and launch kernels. Work on
  separate devices may run concurrently. Check peer or memory-pool access
  before relying on direct cross-GPU access, and order transfers with events or
  stream waits. See :doc:`devices` and the CUDA Programming Guide's
  `Programming Systems with Multiple GPUs
  <https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html>`__
  chapter.
* A process-per-GPU program uses an external launcher. Python multiprocessing
  can start workers on one host, while a cluster launcher can start them across
  nodes. In each worker, select the process-local GPU before allocating Warp
  arrays, loading modules, or capturing graphs. If another framework selects
  the GPU before :func:`wp.init() <warp.init>`, Warp adopts that current CUDA
  context as its default. Otherwise, set Warp's default explicitly with
  :func:`wp.set_device() <warp.set_device>` before beginning device-dependent
  work. When ``CUDA_VISIBLE_DEVICES`` exposes one GPU per worker, that GPU
  normally appears as ``cuda:0`` inside the process. The CUDA Programming
  Guide's `CUDA Environment Variables
  <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html>`__
  chapter explains device visibility and enumeration. For new GPU-native
  distributed applications, prefer `NCCL4Py
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py.html>`__
  for collective and point-to-point communication, or `NVSHMEM4Py
  <https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/index.html>`__
  for symmetric-memory and one-sided communication. NCCL4Py accepts Warp arrays
  through DLPack or the CUDA Array Interface. NVSHMEM4Py allocations can be
  wrapped by Warp through DLPack.
* The Message Passing Interface (MPI) remains useful when the application
  already uses it. GPU buffers passed through `mpi4py
  <https://mpi4py.readthedocs.io/en/stable/install.html#building-from-sources>`__
  require a CUDA-aware MPI implementation. Installing mpi4py from PyPI or
  conda-forge does not by itself provide a CUDA-aware MPI stack. One way to
  produce a CUDA-aware mpi4py installation is to download `NVIDIA HPC-X
  <https://developer.nvidia.com/networking/hpc-x>`__ and build mpi4py from
  source against its ``mpicc``. Call :func:`wp.synchronize_device()
  <warp.synchronize_device>` before passing a GPU buffer to MPI because mpi4py
  cannot synchronize GPU work for you. See the `distributed Jacobi example
  <https://github.com/NVIDIA/warp/blob/main/warp/examples/distributed/example_jacobi_mpi.py>`__.
* In a `PyTorch Distributed
  <https://docs.pytorch.org/docs/stable/distributed.html>`__ or JAX application,
  let the framework manage processes, devices, and communication. Warp runs the
  local computation on each process's tensors or shards through
  :doc:`interoperability_pytorch` or :ref:`JAX shard_map <jax-shard-map>`.

Multi-process deployments should normally assign one GPU to each process. Keep
communication on the same CUDA stream as the Warp work when the API supports
it. Otherwise, synchronize explicitly before sharing a buffer. The next
question covers Warp initialization and kernel-cache setup for child processes.

How should I initialize and configure Warp in multiprocessing applications?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each child process should initialize Warp before allocating Warp objects or
launching kernels. Do not use a CUDA context inherited through ``fork``. Use
``spawn`` or ``forkserver`` instead, or make sure the parent has not initialized
CUDA before it forks.

Processes that compile concurrently need separate kernel-cache directories to
avoid cache races. Set :attr:`warp.config.kernel_cache_dir` before calling
:func:`wp.init() <warp.init>`, or assign a distinct ``WARP_CACHE_PATH`` to each
process. Current multiprocessing restrictions are listed in
:doc:`limitations`.

Differentiation and Interoperability
------------------------------------

How does automatic differentiation work, and what state must I preserve?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Warp generates the backward, or adjoint, code for differentiable kernels and
functions. Mark participating arrays with ``requires_grad=True``, record
launches in a :class:`wp.Tape <warp.Tape>`, and call
:meth:`Tape.backward() <warp.Tape.backward>` with a scalar loss or explicit
output gradients.

A tape records launches and references to their arrays. It does not save a new
version of an array each time the array is written. The backward pass may need
values that were present during an earlier forward launch, so the application
must preserve those values. PyTorch and JAX commonly create new tensor values
for operation outputs; Warp leaves output allocation and reuse to the
application. An iterative simulation may therefore need separate state arrays
for multiple time steps.

Keeping every state makes memory use grow with the number of simulation steps.
Gradient checkpointing trades computation for storage: it saves periodic
states and replays the work between them during the backward pass. Applications
currently implement this themselves. The `fluid checkpointing example
<https://github.com/NVIDIA/warp/blob/main/warp/examples/optim/example_fluid_checkpoint.py>`__
shows one implementation. See :doc:`differentiability` for the full overwrite
and replay rules.

Why can array overwrites produce unexpected gradients?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A tape tracks array objects, not historical versions of their contents. If the
backward pass needs an earlier value, overwriting that array destroys the value
that the adjoint calculation expects. Across kernel launches, Warp propagates
gradients through the final write to an array element. Retaining gradients on
values written more than once can count them twice.

Preserve required values in distinct state arrays or use differentiable
:func:`wp.copy() <warp.copy>`, :func:`wp.clone() <warp.clone>`, or
:meth:`array.assign() <warp.array.assign>` operations. During debugging, enable
:attr:`warp.config.verify_autograd_array_access` to find problematic overwrite
patterns. Supported in-place accumulation has separate rules. See
:ref:`Array Overwrite Tracking <array_overwrite_tracking>` and
:ref:`In-Place Math <in_place_math>`.

Can I define custom gradients for Warp functions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. Define the forward calculation with :func:`@wp.func <warp.func>`, then
register a replacement adjoint with
:func:`@wp.func_grad <warp.func_grad>`. Warp uses the custom gradient instead
of the derivative it would otherwise generate for that function.

Define a custom gradient to choose the behavior at a non-smooth point, avoid a
non-finite derivative, or supply a simpler analytic derivative. The custom
gradient function receives the original inputs and output adjoints, then
accumulates input gradients through ``wp.adjoint``.

For an expression built from existing functions, wrap the expression in a
user-defined Warp function and attach the custom gradient there. If the
backward pass must replay the forward function differently, use
:func:`@wp.func_replay <warp.func_replay>`. See
:ref:`Custom Gradient Functions <custom-gradient-functions>` for signatures
and examples.

How do gradients flow between Warp and PyTorch or JAX?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sharing an allocation does not connect two automatic-differentiation systems
by itself. Use Warp's framework-specific integration: wrap the Warp work in a
PyTorch custom autograd function or registered custom operator, or use
:func:`jax_kernel() <warp.jax_kernel>` and its custom vector-Jacobian product
(VJP) support for JAX.

The wrapper must define how output gradients launch Warp's adjoint code and
how the resulting input gradients return to the host framework. Integration
details are in :doc:`interoperability_pytorch` and
:doc:`interoperability_jax`.

Should I use direct framework converters, array interfaces, or DLPack?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a framework-specific converter when Warp provides one and the application
needs gradient buffers or framework-specific dtype handling. An
array-interface object can be passed directly when a compatible allocation is
needed only as a kernel argument. DLPack supports framework-neutral zero-copy
exchange with standardized stream synchronization.

Neither array interfaces nor DLPack carry automatic-differentiation metadata.
The producer must keep shared memory alive as required by the selected
protocol. The quick-reference table in :doc:`interoperability` compares these
options.

Performance, Debugging, and Deployment
--------------------------------------

Why can Warp results differ between runs or devices, and how can I make them deterministic?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Small floating-point differences can come from changes in operation order or
the math implementation used by a backend. Concurrent floating-point atomics
are another common source: CUDA does not guarantee the order in which threads
update the same value, and different orders can round differently. Larger or
erratic differences may point to a race, invalid memory access, or uninitialized
data.

The CUDA Programming Guide's `Floating-Point Computation
<https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html>`__
chapter explains how evaluation order and implementation differences affect
results.

Warp does not constrain atomic ordering by default. For supported atomic
patterns, :attr:`wp.DeterministicMode.RUN_TO_RUN
<warp.DeterministicMode.RUN_TO_RUN>` produces bit-exact repeated results on the
same GPU architecture. :attr:`wp.DeterministicMode.GPU_TO_GPU
<warp.DeterministicMode.GPU_TO_GPU>` uses a stronger path intended to match
across GPU architectures. These modes can take more time and temporary memory,
and they must be selected before the module is compiled.
:doc:`deterministic_execution` covers supported operations, configuration
scopes, and current limitations.

How should I benchmark asynchronous Warp operations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Warm up compilation before benchmarking, and measure GPU completion instead
of Python dispatch alone. Because CUDA launches are asynchronous, a host timer
around :func:`wp.launch() <warp.launch>` measures scheduling unless the timed
region synchronizes or uses CUDA events.

For a simple end-to-end measurement, use
:class:`wp.ScopedTimer <warp.ScopedTimer>` with ``synchronize=True``. CUDA
events provide targeted stream timing, while a profiler is better suited to
concurrent workloads. The CUDA Programming Guide's `Asynchronous Execution
<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html>`__
chapter explains stream and event timing. :doc:`../deep_dive/profiling`
describes each method.

How do I find which kernel caused an illegal memory access or a non-finite value?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA launches are asynchronous, so an error may surface during a later
operation. Call :func:`wp.synchronize_device() <warp.synchronize_device>` after
a suspected launch, or set :attr:`warp.config.verify_cuda` to ``True`` to check
the CUDA context after every launch. The latter adds synchronization overhead
and cannot be used during CUDA graph capture.

Set :attr:`warp.config.verify_fp` to ``True`` to locate non-finite values such
as NaN or infinity. Debug mode adds array bounds checks, assertions, and device
line information. Set :attr:`warp.config.print_launches` to ``True``, or set
:attr:`warp.config.log_level` to :data:`wp.LOG_DEBUG <warp.LOG_DEBUG>`, to
identify the launch that failed. If those checks do not isolate the problem,
run the application with NVIDIA Compute Sanitizer. After an illegal access,
rerun in a fresh process because later launches in the affected CUDA context
may fail and hide the original cause. See :doc:`debugging` for details and
additional tools.

How can I profile kernels and inspect generated code?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`wp.ScopedTimer <warp.ScopedTimer>` to summarize Warp activity and
NVIDIA Nsight Systems or Nsight Compute for device-level profiling. NVIDIA
Tools Extension (NVTX) ranges and Warp's CUDA profiler controls can restrict a
capture to the region of interest.

Set :attr:`warp.config.log_level` to :data:`wp.LOG_DEBUG <warp.LOG_DEBUG>` to
log detailed module and codegen messages. Warp also writes generated C++ and
CUDA source into its kernel cache for direct inspection. See
:doc:`../deep_dive/profiling` and :doc:`../deep_dive/codegen`.

When should I use CUDA graph capture, and what is currently supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`CUDA graph capture
<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>`__
is useful when the same stable operation sequence runs many times and Python or
driver launch overhead is significant. Capture has a setup cost, records
supported device operations rather than arbitrary Python, and is most useful
after modules and external libraries have been warmed up.

Prefer to allocate long-lived inputs, outputs, and reusable scratch storage
before capture. Keep every array and spatial object referenced by the captured
work alive for as long as the graph may be replayed.

Warp can record CUDA allocations made during capture when CUDA memory pools are
supported and enabled. An array allocated this way is not usable until the
graph has been launched. Managed-memory allocations and operations that need
unsupported temporary or staging allocations must still be prepared before
capture. See :doc:`../deep_dive/allocators` for the allocation rules.

Requirements for capture-safe operations and conditional, multi-stream, CPU,
and serialized graph behavior evolve with CUDA and Warp. Check the
:ref:`Graphs <graphs>` section of :doc:`runtime` for current support.

Can I precompile and package kernels for deployment?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. Warp can compile a module ahead of time with
:func:`wp.compile_aot_module() <warp.compile_aot_module>` and load the resulting
artifacts with :func:`wp.load_aot_module() <warp.load_aot_module>`. Deployment
artifacts must cover the required target architectures, module options,
generic overloads, and other specializations that the application will use.

Any runtime-generated kernel or omitted specialization still needs
compilation. See :ref:`Ahead-of-Time Compilation Workflows
<ahead_of_time_compilation_workflows>` in :doc:`../deep_dive/codegen`.

Can Warp use existing C++ or CUDA code?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. Applications can insert small C++ or CUDA operations into generated Warp
modules with :func:`@wp.func_native <warp.func_native>`. Larger library
integrations require extending Warp's native bindings and build. See
:doc:`cpp_cuda_workflows` for both approaches.

Can Warp computations run from C or C++ without Python?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. During an ahead-of-time build, Warp can write generated CUDA C++ source,
metadata, PTX, and CUBIN files to disk. A native CUDA C++ application can load
the generated CUBIN through the `CUDA Driver API
<https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html>`__
or include the generated ``.cu`` source in its own translation unit. The
``01_source_include`` example in :doc:`cpp_cuda_workflows` launches the
generated forward and backward kernels directly, making this workflow useful
for both ordinary kernel execution and reverse-mode differentiation.

For applications that need to replay a larger sequence of Warp operations,
API Capture can serialize a captured sequence and replay it from a standalone
C++ program without a Python runtime.

The native workflows expose fewer operations than the Python runtime and work
at a lower level. API Capture can record only its documented set of operations.
The C++ examples in :doc:`cpp_cuda_workflows` show each workflow.

Community and Support
---------------------

Submit bug reports, feature requests, and technical questions through
`GitHub Issues <https://github.com/NVIDIA/warp/issues>`__. Before opening an
issue, run:

.. code-block:: console

    $ python -c "import warp as wp; wp.print_diagnostics()"

Include the output along with the smallest reproducible example, the observed
error, and the expected behavior. For inquiries that do not belong in a public
issue, email ``warp-python@nvidia.com``.

How can I share work built with Warp?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Share projects, demos, research, and other examples in the Warp GitHub
Discussions `Show and tell
<https://github.com/NVIDIA/warp/discussions/categories/show-and-tell>`__
section. For a public GitHub repository, add ``nvidia-warp`` to its repository
topics so that it appears on the `nvidia-warp topic page
<https://github.com/topics/nvidia-warp>`__ and is easier for other Warp users
to find. Include a short explanation of how the project uses Warp, along with
images, videos, or results when useful.
