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
  captures a CUDA graph in Python, loads it from C++, updates named inputs, and
  replays the frame with ``cudaGraphLaunch()``.
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

- :doc:`basics` covers regular :func:`@wp.func <warp.func>` functions called
  from kernels.
- :doc:`differentiability` covers custom gradients, custom replay functions,
  tapes, and native-function adjoints.
- :doc:`runtime` covers runtime kernel creation, CUDA graph capture, and APIC
  save/load.
- :doc:`../deep_dive/codegen` covers generated C++/CUDA source and AOT
  compilation.
