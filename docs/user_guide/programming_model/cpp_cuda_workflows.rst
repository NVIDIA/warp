C++ and CUDA Workflows
======================

.. currentmodule:: warp

Warp is primarily authored from Python, but several workflows expose generated
C++/CUDA code or replay captured Warp work from a native host application. This
page collects the public entry points for those non-Python integration paths and
links to the detailed workflow documentation and examples.

Use this page when you need to:

- insert native C++/CUDA snippets into generated Warp kernels.
- ahead-of-time compile Warp kernels into source, PTX, or CUBIN files.
- load generated Warp binaries or source from a CUDA C++ application.
- serialize captured Warp work and replay it from C++ without a Python runtime.

.. _native_functions:

Native Snippets in Warp Kernels
-------------------------------

Use :func:`@wp.func_native <warp.func_native>` to insert native C++/CUDA code
into generated Warp modules. Native functions are useful when Warp does not
provide a built-in operation, CUDA intrinsic, synchronization pattern, or
low-level expression that your kernel needs.

Pure C++ snippets, meaning snippets without CUDA-only constructs, can be used
by CPU kernels. The same snippet can also be used by CUDA kernels if the code is
valid device code. CUDA-specific constructs, such as ``__shared__`` memory,
``__syncthreads()``, and CUDA atomics, require CUDA kernels.

The decorator takes native source code as a string. The decorated Python
function is a typed stub: its arguments define the names and types available to
the snippet, and its body should be ``...`` because Warp replaces the body with
the native snippet during code generation.

The thread index should be computed by the caller and passed explicitly. Native
snippets are inserted into generated C++/CUDA, so they cannot call
:func:`wp.tid() <warp.tid>` directly.

.. code-block:: python

    import numpy as np
    import warp as wp

    snippet = "out[tid] = x[tid] + 1.0f;"


    @wp.func_native(snippet)
    def increment(x: wp.array[wp.float32], out: wp.array[wp.float32], tid: int):
        ...


    @wp.kernel
    def increment_kernel(x: wp.array[wp.float32], out: wp.array[wp.float32]):
        tid = wp.tid()
        increment(x, out, tid)


    device = "cpu"
    x = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device=device)
    out = wp.zeros_like(x)
    wp.launch(increment_kernel, dim=x.shape, inputs=[x], outputs=[out], device=device)

CUDA Shared Memory
~~~~~~~~~~~~~~~~~~

Native snippets can use CUDA features that Warp does not expose directly. The
following example performs a reduction within a single 128-thread block using
shared memory. It assumes the launch uses exactly one block. Generalizing this
pattern to multiple blocks requires using a per-block thread index and storing
one result per block.

.. code-block:: python

    import numpy as np
    import warp as wp

    snippet = """
        __shared__ int sum[128];

        sum[tid] = arr[tid];
        __syncthreads();

        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sum[tid] += sum[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out[0] = sum[0];
        }
        """


    @wp.func_native(snippet)
    def reduce(arr: wp.array[int], out: wp.array[int], tid: int):
        ...


    @wp.kernel
    def reduce_kernel(arr: wp.array[int], out: wp.array[int]):
        tid = wp.tid()
        reduce(arr, out, tid)


    arr = wp.array(np.arange(128, dtype=np.int32), dtype=wp.int32, device="cuda")
    out = wp.zeros(1, dtype=wp.int32, device="cuda")
    wp.launch(reduce_kernel, dim=128, inputs=[arr], outputs=[out], block_dim=128, device="cuda")

.. _thread-block-clusters:

Thread Block Clusters and Distributed Shared Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`CUDA Thread Block Clusters
<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters>`_
group adjacent CTAs into a cluster whose blocks the hardware guarantees to
co-schedule on a single GPU Processing Cluster (GPC) — a hardware group of SMs
linked by an on-chip interconnect. That SM-to-SM path unlocks *distributed shared
memory*: each block can address the shared memory of every other block in its
cluster. Clusters require compute capability 9.0 (Hopper) or higher.

Set the cluster size with the :ref:`cluster_dim <kernel-cluster-dim>` kernel
argument. Warp emits the kernel's ``__cluster_dims__`` attribute from that value,
but the cluster machinery itself — distributed shared memory, cluster barriers,
and cluster rank queries — is reachable only from native CUDA code, so a
``@wp.func_native`` snippet is how you use it.

The example below reduces an array with a single cluster of ``CLUSTER_DIM``
blocks. Each block sums its own slice into shared memory; after a cluster
barrier, block 0 reaches into every peer block's shared memory through the
``__cluster_map_shared_rank`` device builtin to form the cluster-wide total. A
second cluster barrier keeps every block alive until block 0 has finished
reading, because a block must not exit while its shared memory is still being
accessed.

.. testcode::
    :skipif: wp.get_cuda_device_count() == 0 or wp.get_device("cuda:0").arch < 90

    import numpy as np
    import warp as wp

    CLUSTER_DIM = 4
    BLOCK_DIM = 32

    snippet = r"""
        const unsigned int lane = threadIdx.x;

        // Each block reduces its slice into a single shared-memory accumulator.
        __shared__ int s_partial;
        if (lane == 0) s_partial = 0;
        __syncthreads();
        atomicAdd(&s_partial, value);
        __syncthreads();

        // Cluster barrier: every block's s_partial is finalized and visible
        // across the cluster's distributed shared memory.
        asm volatile("barrier.cluster.arrive;" ::: "memory");
        asm volatile("barrier.cluster.wait;" ::: "memory");

        unsigned int rank, num_blocks;
        asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
        asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(num_blocks));

        // Block 0 gathers every peer's partial sum via distributed shared memory.
        if (rank == 0 && lane == 0) {
            int total = 0;
            for (unsigned int r = 0; r < num_blocks; ++r) {
                int *remote = (int *)__cluster_map_shared_rank(&s_partial, r);
                total += *remote;
            }
            out[0] = total;
        }

        // Second barrier: no block exits (freeing its shared memory) until
        // block 0 has finished reading every peer's accumulator.
        asm volatile("barrier.cluster.arrive;" ::: "memory");
        asm volatile("barrier.cluster.wait;" ::: "memory");
        """


    @wp.func_native(snippet)
    def cluster_reduce(value: int, out: wp.array[int]):
        ...


    @wp.kernel(cluster_dim=CLUSTER_DIM, enable_backward=False)
    def cluster_reduce_kernel(values: wp.array[int], out: wp.array[int]):
        tid = wp.tid()
        cluster_reduce(values[tid], out)


    # One cluster of CLUSTER_DIM blocks: launch exactly CLUSTER_DIM * BLOCK_DIM threads.
    n = CLUSTER_DIM * BLOCK_DIM
    values = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device="cuda")
    out = wp.zeros(1, dtype=wp.int32, device="cuda")
    wp.launch(
        cluster_reduce_kernel,
        dim=n,
        inputs=[values],
        outputs=[out],
        block_dim=BLOCK_DIM,
        device="cuda",
    )

    print(int(out.numpy()[0]))

.. testoutput::
    :skipif: wp.get_cuda_device_count() == 0 or wp.get_device("cuda:0").arch < 90

    8128

The ``"memory"`` clobber on each barrier stops the compiler from reordering
shared memory accesses across it. ``cluster_dim`` values 2–8 are portable across
all cluster-capable devices; values 9–16 are non-portable and depend on the GPU,
so query :func:`warp.get_cuda_max_cluster_dim` before using them.

Inline PTX
~~~~~~~~~~

Native snippets can also use `inline Parallel Thread Execution (PTX) assembly
<https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html>`_ inside CUDA
code. Inline PTX is useful when you need a GPU instruction that is not exposed
directly through Warp or CUDA C++.

The following example computes the sum of four byte-wise absolute differences
between two packed 8-bit values. The `PTX vabsdiff4 instruction
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd4-vsub4-vavrg4-vabsdiff4-vmin4-vmax4>`_
performs four byte-wise absolute differences and, with the ``.add`` modifier,
accumulates them into one 32-bit result.

.. testcode::
    :skipif: wp.get_cuda_device_count() == 0

    import numpy as np
    import warp as wp

    snippet = r"""
        unsigned int result;
        unsigned int zero = 0;
        asm("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"
            : "=r"(result)
            : "r"(a), "r"(b), "r"(zero));
        return result;
        """


    @wp.func_native(snippet)
    def absdiff4_sum_u8(a: wp.uint32, b: wp.uint32) -> wp.uint32:
        ...


    @wp.kernel
    def absdiff4_kernel(
        a: wp.array[wp.uint32],
        b: wp.array[wp.uint32],
        out: wp.array[wp.uint32],
    ):
        tid = wp.tid()
        out[tid] = absdiff4_sum_u8(a[tid], b[tid])


    def pack4(values):
        return np.uint32(values[0] | (values[1] << 8) | (values[2] << 16) | (values[3] << 24))


    a_host = np.array([pack4([10, 20, 30, 40]), pack4([0, 128, 255, 13])], dtype=np.uint32)
    b_host = np.array([pack4([13, 18, 41, 35]), pack4([255, 120, 0, 15])], dtype=np.uint32)

    a = wp.array(a_host, dtype=wp.uint32, device="cuda")
    b = wp.array(b_host, dtype=wp.uint32, device="cuda")
    out = wp.zeros_like(a)
    wp.launch(absdiff4_kernel, dim=a.shape, inputs=[a, b], outputs=[out], device="cuda")

    # [3 + 2 + 11 + 5, 255 + 8 + 255 + 2]
    print(out.numpy().tolist())
    np.testing.assert_array_equal(out.numpy(), np.array([21, 520], dtype=np.uint32))

.. testoutput::
    :skipif: wp.get_cuda_device_count() == 0

    [21, 520]

The ``"r"`` constraints bind the operands to 32-bit integer registers, which
matches the ``.u32`` instruction operands. The final PTX operand is an
accumulator and is supplied as a zero-initialized register in this example. If
the assembly reads or writes memory through pointers, add the appropriate
``"memory"`` clobber as described in NVIDIA's inline PTX documentation.

Returning Values
~~~~~~~~~~~~~~~~

A native snippet can return a value when the Python stub declares a return type.
Warp supports scalar, vector, matrix, quaternion, array, and fixed-array return
types. Struct return values are not supported.

.. code-block:: python

    snippet = """
        float sq = x * x;
        return sq;
        """


    @wp.func_native(snippet)
    def square(x: wp.float32) -> wp.float32:
        ...

Pass-by-reference Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Native functions can use :class:`wp.ref[T] <warp.ref>` parameters for scalar,
vector, matrix, quaternion, or struct values that should be mutated in place.
Inside the native snippet, a ``wp.ref[T]`` parameter is visible as a C++/CUDA
reference named after the Python parameter:

.. code-block:: python

    @wp.func_native("x = x + 5;")
    def add_five(x: wp.ref[wp.int32]):
        ...


    @wp.kernel(enable_backward=False)
    def add_five_kernel(values: wp.array[wp.int32]):
        i = wp.tid()
        add_five(values[i])

Call sites must pass an addressable expression, such as a local variable,
function parameter, array element, struct field, vector or matrix component, or
nested field rooted at an array element. If the native function is used in a
tape-recorded computation, provide an ``adj_snippet``; adjoint variables for
``wp.ref`` parameters use the same ``adj_`` prefix and are also snippet-visible
as references.

Use :func:`wp.address_of(expr) <warp.address_of>` when a native snippet needs a
raw pointer to a specific addressable expression instead of a ``wp.ref[T]``
parameter. Use ``array.ptr`` for the base pointer of an entire array, and use
``wp.address_of(expr)`` for local variables, array elements, components, or
nested fields such as ``wp.address_of(v.y)`` and
``wp.address_of(outers[i].inner.value)``:

.. code-block:: python

    @wp.func_native("*(float*)ptr += delta;")
    def add_to_ptr(ptr: wp.uint64, delta: wp.float32):
        ...


    @wp.kernel(enable_backward=False)
    def add_to_ptr_kernel(values: wp.array[wp.float32]):
        i = wp.tid()
        local = wp.float32(1.0)
        add_to_ptr(wp.address_of(local), wp.float32(2.0))
        add_to_ptr(wp.address_of(values[i]), local)

Raw pointer writes performed through native snippets are not automatically
differentiable by Warp. Keep those kernels forward-only or provide the
appropriate manual adjoint for the surrounding native operation.

Differentiable Native Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a native function participates in a tape-recorded computation, provide an
``adj_snippet`` that accumulates adjoints for the native operation. Adjoint
variables use the ``adj_`` prefix, and return-value adjoints are named
``adj_ret``.

.. code-block:: python

    snippet = "out[tid] = 2.0f * x[tid] + y[tid];"
    adj_snippet = """
        adj_x[tid] += 2.0f * adj_out[tid];
        adj_y[tid] += adj_out[tid];
        """


    @wp.func_native(snippet=snippet, adj_snippet=adj_snippet)
    def axpy(
        x: wp.array[wp.float32],
        y: wp.array[wp.float32],
        out: wp.array[wp.float32],
        tid: int,
    ):
        ...

During the backward pass, Warp runs a forward replay phase. By default, native
functions replay the original ``snippet``. If the forward snippet has side
effects that should not be repeated, such as mutating a counter with an atomic
operation, provide ``replay_snippet``. An empty string is a valid no-op replay
snippet.

.. code-block:: python

    snippet = """
        int next_index = atomicAdd(counter, 1);
        thread_values[tid] = next_index;
        """
    replay_snippet = ""


    @wp.func_native(snippet, replay_snippet=replay_snippet)
    def record_index(counter: wp.array[int], thread_values: wp.array[int], tid: int):
        ...

Native Function Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Native snippets are inserted into generated C++/CUDA and are not parsed as
  Warp code.
- The snippet can refer to variables named after the typed Python stub
  arguments.
- CUDA-specific snippets cannot run on CPU devices.
- Type hints must accurately describe the stub arguments and return type.
- Struct return values are unsupported.
- Users are responsible for native-code correctness, synchronization, memory
  safety, and portability.

External Build Inputs
---------------------

.. admonition:: Experimental

    The external-compilation APIs described in this section are experimental
    and may change without deprecation in future releases.

Extra build inputs for a module can be grouped in
:class:`wp.ModuleBuildOptions <warp.ModuleBuildOptions>`:

.. code:: python

    build_options = wp.ModuleBuildOptions(
        extra_cuda_include_dirs=[my_cuda_include_dir],
        extra_cpu_include_dirs=[my_cpu_include_dir],
        extra_cuda_preamble='#include "my_cuda_addon.h"',
        extra_cpu_preamble='#include "my_cpu_addon.h"',
        extra_build_dependencies=[my_cuda_addon_header, my_cpu_addon_header],
    )
    wp.set_module_options({"extra_build_options": build_options})

The preambles are inserted after Warp's own headers and before the generated
code, so an external header may use Warp macros such as ``CUDA_CALLABLE``, and
the generated kernels see everything the preamble declares. A preamble cannot
define macros that Warp's headers consume, because those are already included.

Warp hashes the configured include-directory paths, but not the contents of
headers resolved through them. List external headers and other build inputs in
``extra_build_dependencies`` so Warp also hashes their contents. Dependency
contents are re-read whenever the module hash is recomputed — after
``wp.set_module_options()`` or in a new process — not on every launch.

Changes to a :class:`wp.ModuleBuildOptions <warp.ModuleBuildOptions>` instance
take effect when it is passed to :func:`wp.set_module_options()
<warp.set_module_options>`. After modifying an instance, call
``wp.set_module_options()`` again so Warp invalidates the module's cached
compilation state.

If more than one addon contributes build inputs, combine them without
discarding earlier settings:

.. code:: python

    current = wp.get_module_options(kernel.module)["extra_build_options"]
    if current is None:
        current = wp.ModuleBuildOptions()
    wp.set_module_options(
        {"extra_build_options": current.merged(addon_build_options)},
        module=kernel.module,
    )

External Native Value Types
---------------------------

External C++ value types can be used directly as kernel annotations, built-in
arguments and return values, Warp struct fields, and array dtypes. Define the
host ABI with :class:`ctypes.Structure` and register the same class:

.. code:: python

    import ctypes
    import warp as wp

    class Color(ctypes.Structure):
        _fields_ = [
            ("r", ctypes.c_float),
            ("g", ctypes.c_float),
            ("b", ctypes.c_float),
        ]

    wp.build_experimental.add_native_type(
        Color,
        native_name="render::Color",
        fields={"r": wp.float32, "g": wp.float32, "b": wp.float32},
        initializer="aggregate",
    )

The exact C++ definition must be made visible using the module preamble and
include-directory options above. Warp emits compile-time checks for standard
layout, trivial copyability, size, and alignment. CPU and CUDA compilation
also check each exposed member's C++ type. CPU compilation checks every exposed
field offset, while CUDA compilation checks each field size. Setting
``initializer="aggregate"`` opts into ordered field construction such as
``Color(r, g, b)`` inside kernels. In this mode, ``fields`` must contain every
ctypes field in declaration order. Without it, only default construction is
available.

Pass ``fields=None`` (the default) for an opaque type. Opaque values can cross
kernel and registered built-in boundaries and can be stored in arrays, but
their members and captured constants are not exposed. Warp does not manage
resources referenced by opaque values; the external package remains
responsible for their ownership and lifetime.

Native value types do not automatically gain arithmetic or differentiation.
Register operations explicitly with :func:`warp.build_experimental.add_builtin`, and use
arrays without ``requires_grad``.

Complete Addon Example
----------------------

An addon should register its types and built-ins when imported, then expose a
small helper that applies its build inputs to the module containing the user's
kernels. For example, ``my_addon.py`` can contain:

.. code:: python

    import ctypes
    from pathlib import Path

    import warp as wp

    include_dir = Path(__file__).parent / "include"
    header = include_dir / "my_addon.h"


    class Color(ctypes.Structure):
        _fields_ = [
            ("r", ctypes.c_float),
            ("g", ctypes.c_float),
            ("b", ctypes.c_float),
        ]


    wp.build_experimental.add_native_type(
        Color,
        native_name="my_addon::Color",
        fields={"r": wp.float32, "g": wp.float32, "b": wp.float32},
        initializer="aggregate",
    )
    wp.build_experimental.add_builtin(
        "my_addon_scale_color",
        {"value": Color, "factor": wp.float32},
        Color,
        native_name="my_addon::scale_color",
    )

    _build_options = wp.ModuleBuildOptions(
        extra_cuda_include_dirs=[include_dir],
        extra_cpu_include_dirs=[include_dir],
        extra_cuda_preamble='#include "my_addon.h"',
        extra_cpu_preamble='#include "my_addon.h"',
        extra_build_dependencies=[header],
    )


    def configure_module(module):
        current = wp.get_module_options(module)["extra_build_options"]
        if current is None:
            current = wp.ModuleBuildOptions()
        wp.set_module_options(
            {"extra_build_options": current.merged(_build_options)},
            module=module,
        )

Users import the addon before compiling kernels that call its built-ins, and
configure each kernel module before its first launch or AOT compilation:

.. code:: python

    import warp as wp

    import my_addon


    @wp.kernel
    def scale_colors(colors: wp.array[my_addon.Color], factor: float):
        tid = wp.tid()
        colors[tid] = wp.my_addon_scale_color(colors[tid], factor)


    my_addon.configure_module(scale_colors.module)

    # A normal wp.launch() now JIT-compiles with the addon's build inputs.
    # For external runtimes, AOT compilation returns the generated artifact.
    artifacts = wp.compile_aot_module(
        scale_colors.module,
        device="cuda",
        use_ptx=True,
    )

Registered built-in names share Warp's global kernel namespace. Addons should
use a package-specific prefix unless they intentionally add an overload to an
existing Warp operation. Registrations are process-global and equivalent
registrations are idempotent, which makes normal module reloads safe.

Ahead-of-Time C++/CUDA Workflows
--------------------------------

Warp can compile kernels ahead of time and write the generated CUDA source,
metadata, PTX, or CUBIN files to disk. This is useful when a Python build step
authors and validates kernels, but a native CUDA C++ application owns runtime
execution.

The full AOT workflow is documented in
:ref:`ahead_of_time_compilation_workflows`. The C++ examples under
``warp/examples/cpp/`` show two deployment patterns:

- `00_cubin_launch <https://github.com/NVIDIA/warp/tree/main/warp/examples/cpp/00_cubin_launch>`_
  compiles a Warp kernel to a CUBIN, loads that module with the CUDA Driver API,
  and launches the generated kernel with ``cuLaunchKernel()``.
- `01_source_include <https://github.com/NVIDIA/warp/tree/main/warp/examples/cpp/01_source_include>`_
  includes the generated ``.cu`` source in a CUDA C++ translation unit and
  launches the generated forward and backward kernels directly.

Both examples use ``warp/native/aot.h`` for Warp's generated type definitions,
CUDA setup helpers, and error-checking macros. The generated code also depends
on the native type headers such as ``builtin.h`` that ship in ``warp/native/``.

API Capture Replay from C++
---------------------------

API Capture (APIC) can serialize a captured Warp graph to a ``.wrp`` file plus a
companion module directory. The saved graph can later be loaded from Python or
from a standalone C++ program that links against the Warp native library.

See :ref:`apic_save_load` for the C API surface, serialization format notes, and
current limitations. The C++ examples cover both device families:

- `02_apic_visualization <https://github.com/NVIDIA/warp/tree/main/warp/examples/cpp/02_apic_visualization>`_
  records and saves a CUDA workload in Python. The C++ loader reconstructs a
  CUDA graph, updates named inputs, and replays the frame with
  ``cudaGraphLaunch()``.
- `03_apic_visualization_cpu <https://github.com/NVIDIA/warp/tree/main/warp/examples/cpp/03_apic_visualization_cpu>`_
  captures and replays on the CPU device. The C++ viewer does not link against
  CUDA. It loads recorded CPU kernel objects and replays the graph with
  ``wp_apic_cpu_replay_graph()``.

Native Library Headers
----------------------

The C++ integration examples intentionally use a small native surface:

- ``warp/native/aot.h`` exposes utilities for generated AOT kernels and includes
  Warp's generated type support.
- ``warp/native/warp.h`` declares the core Warp C API exported by the native
  library.
- ``warp/native/apic.h`` declares the APIC graph loading and replay API used by
  ``.wrp`` graph consumers.

Other files in ``warp/native/`` implement Warp's runtime and kernel support
library. They are useful when inspecting generated code, but the examples above
are the recommended starting points for native host integration.

Related Topics
--------------

- :doc:`../basics` covers regular :func:`@wp.func <warp.func>` functions called
  from kernels.
- :doc:`../differentiability` covers custom gradients, custom replay functions,
  tapes, and native-function adjoints.
- :doc:`../runtime` covers runtime kernel creation, CUDA graph capture, and APIC
  save/load.
- :doc:`code_generation` covers generated C++/CUDA source and AOT
  compilation.
