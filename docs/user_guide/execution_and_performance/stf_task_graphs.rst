Task Parallelism with CUDASTF
=============================

.. currentmodule:: warp.stf_experimental

Warp historically expresses concurrency by issuing kernels into a single
CUDA stream and, optionally, capturing them into a graph. That works well
when the work is naturally linear, but more complex pipelines often have
independent sub-steps that *should* run concurrently and only synchronize
on a few data dependencies. Coordinating that by hand -- creating extra
streams, signalling the right events, and threading them through every call
site -- is brittle and easy to get wrong.

The :mod:`warp.stf_experimental` module adds a task-parallel programming
model to Warp by integrating with CUDASTF, the experimental
``cuda.stf._experimental`` Python module developed as part of the
`CUDA C++ Core Libraries (CCCL) <https://github.com/NVIDIA/cccl>`_ and distributed
via the standalone ``cuda-stf`` package on PyPI. User code stays mostly
sequential: tasks declare which logical data (or tokens) they read and
write, and CUDASTF maps those dependencies onto a stream pool, inserting
the synchronization needed to honor them. The Warp helpers wrap CUDASTF
streams as :class:`warp.Stream` objects, alias CUDASTF array views as
zero-copy :class:`warp.array` views, and play nicely with Warp graph capture.

This module is optional. Install ``cuda-stf[cu12]`` or
``cuda-stf[cu13]`` separately, then check
:func:`warp.stf_experimental.is_available` before using STF features.

The rest of this guide walks through three integration layers, from the
simplest to the most coupled with CUDA graphs: pure task parallelism with no
graph capture, an STF-owned DAG recorded into a re-launchable CUDA graph,
and a stream-bound STF context running *inside* an existing CUDA graph
capture.

STF Without CUDA Graphs
-----------------------

The simplest way to use CUDASTF is as a pure asynchronous tasking layer.
Open a stream-bound :func:`warp.stf_experimental.context`, declare tokens for
the data each task reads or writes, run a sequence of
``ctx.task(...)`` blocks, and finalize the context. CUDASTF picks streams from
its internal pool and inserts the cross-stream events needed to honor the
declared dependencies -- no CUDA graph capture is involved.
The task body inherits the active Warp device from :func:`warp.get_device`;
wrap setup code in :class:`warp.ScopedDevice` when building tasks for a
non-default device.

.. code-block:: python

    import numpy as np

    import warp as wp
    from warp import stf_experimental as wp_stf

    with wp_stf.context(stream=wp.get_stream()) as ctx:
        tok = ctx.token()

        with ctx.task(tok.write()) as (stream,):
            simulate_robot(stream)
        with ctx.task(tok.read()) as (stream,):
            simulate_sand(stream)

This mode is the natural fit for one-shot or rarely-repeated workloads where
the per-frame recording cost would not amortize, and for code paths that
need data-dependent control flow that a single captured graph could not
express. The :func:`task` context manager and the token API are exactly the
same as in the re-launchable graph pattern below; switching to that pattern
is purely a matter of creating a :func:`warp.stf_experimental.task_graph`.

Tracking Warp Arrays As STF Dependencies
----------------------------------------

Tokens are useful for ordering-only dependencies, but many tasks already
communicate through :class:`warp.array` objects. ``ctx.dep(array)`` memoizes an
ordering token for an exact CUDA Warp array view. The returned token uses
CUDASTF's native ``read()``, ``write()``, and ``rw()`` dependency methods, but
it does not carry array payload:

.. code-block:: python

    with wp_stf.context(stream=wp.get_stream()) as ctx:
        a = wp.zeros(n, dtype=wp.float32, device="cuda:0")
        b = wp.zeros_like(a)

        with ctx.task(ctx.dep(a).write()) as (stream,):
            fill_a(stream, a)

        with ctx.task(ctx.dep(a).read(), ctx.dep(b).write()) as (stream,):
            copy_a_to_b(stream, a, b)

Tokens do not appear in the tuple yielded by ``ctx.task(...)``. A task whose
dependencies are all tokens yields ``(stream,)``; if a task mixes array deps
with explicit logical data, yielded array views are produced only for the
explicit logical data. Use ``ctx.logical_data(...)`` when the task body should
receive an STF-owned/staged array view:

.. code-block:: python

    host_x = np.ones(n, dtype=np.float32)

    with wp_stf.context(stream=wp.get_stream()) as ctx:
        l_x = ctx.logical_data(host_x)

        with ctx.task(l_x.rw()) as (stream, x_view):
            scale_in_place(stream, x_view)

Calling ``ctx.dep`` on an existing CUDASTF logical data object, including a
token returned by ``ctx.token()``, is a no-op. This lets helper code accept
either raw Warp arrays or already-constructed CUDASTF logical data:

.. code-block:: python

    def scale(ctx, x):
        with ctx.task(ctx.dep(x).rw()) as (stream,):
            scale_in_place(stream, x)

The mapping is local to the context. For Warp arrays, the cache key is the
exact array view: device, pointer, byte span, dtype, shape, and strides.
Registering the same view twice returns the same ordering token, while
differently shaped or strided aliases over the same allocation intentionally
produce distinct tokens. ``ctx.dep`` is not a general alias analysis; reuse
the same array view or use a shared token when differently shaped views must be
ordered against each other.

Because ``ctx.dep(array)`` creates tokens rather than payload-carrying logical
data, it is safe to call inline inside :func:`task_graph` recording blocks and
stackable ``graph_scope()`` / ``while_loop()`` frames:

.. code-block:: python

    graph = wp_stf.task_graph()
    ctx = graph.context

    with graph:
        with ctx.task(ctx.dep(a).read(), ctx.dep(b).write()) as (stream,):
            copy_a_to_b(stream, a, b)

STF-Owned Task Graphs
---------------------

When the same DAG is replayed many times -- for example, once per simulation
step -- :func:`warp.stf_experimental.task_graph` creates an STF-owned
recorder for a CUDA graph that can be re-launched with one
``cudaGraphLaunch`` per frame. Each Warp sub-step is recorded as a CUDASTF
task, and dependencies between tasks determine which sub-steps can run
concurrently.

.. code-block:: python

    from warp import stf_experimental as wp_stf

    graph = wp_stf.task_graph()
    ctx = graph.context
    token = ctx.token()

    with graph:
        with ctx.task(token.write()) as (stream,):
            simulate_robot(stream)
        with ctx.task(token.read()) as (stream,):
            simulate_sand(stream)

    graph.launch()
    graph.finalize()

The graph object owns the underlying CUDASTF context, exposed as
``graph.context`` for tokens, logical data, and tasks. Enter ``with graph:``
exactly once to record tasks. After the context manager exits successfully,
the graph is sealed: call ``launch()`` many times, then call ``finalize()``
when the graph is no longer needed. Re-recording, appending tasks after the
graph is sealed, or launching after ``reset()`` all raise explicit errors;
create a new :func:`task_graph` for a new graph structure.

The :func:`warp.stf_experimental.task` context manager wraps the stream handed
out by CUDASTF as a :class:`warp.Stream`, pushes it as Warp's active stream, and
tracks external capture bookkeeping when the task is already inside a CUDA graph
capture.

The stackable-frame API is not hidden by these helpers. Calling
:func:`warp.stf_experimental.context` without a stream returns a stackable
context whose full CUDASTF surface (``push()``, ``pop()``, ``graph_scope()``,
``while_loop()``, ...) remains available directly on the wrapper for fully
manual frame management; :func:`task_graph` only adds a guard-railed
record-once/launch-many lifecycle on top. Advanced users can also open
nested frames inside the single ``with graph:`` block -- spelled
``graph.context.raw`` below to make the departure from the managed lifecycle
explicit, although the wrapper forwards these calls as well. Each inner
``push()`` opens a new frame on the same context, the work recorded inside it
is folded into the surrounding frame on ``pop()``, and only the outer
``with graph:`` produces a launchable CUDA graph for the whole tree. This can
be useful when a re-usable sub-step should be folded into a larger DAG without
the outer code knowing how the sub-step is built.

.. code-block:: python

    from warp import stf_experimental as wp_stf

    def build_sub_dag(ctx, tok_a, tok_b):
        # Inner frame: two sibling tasks folded into the surrounding DAG.
        ctx.raw.push()
        with ctx.task(tok_a.write()) as (s,):
            sub_phase_a(s)
        with ctx.task(tok_b.write()) as (s,):
            sub_phase_b(s)
        ctx.raw.pop()

    graph = wp_stf.task_graph()
    ctx = graph.context
    tok_a = ctx.token()
    tok_b = ctx.token()
    tok_c = ctx.token()

    with graph:
        with ctx.task(tok_c.write()) as (s,):
            init_c(s)

        build_sub_dag(ctx, tok_a, tok_b)  # nested raw push/pop, no graph returned

        with ctx.task(tok_a.read(), tok_b.read(), tok_c.read()) as (s,):
            join(s)

    graph.launch()  # replays the full outer DAG, sub-DAG included

Inner ``pop()`` calls do not produce launchable graphs of their own; they
just close the current frame and let its tasks continue to participate in
the outer DAG with all their token dependencies intact. Because every
nested frame uses the *same* context and shares its token table,
dependencies between inner and outer tasks fall out automatically -- there
is no extra plumbing to thread tokens through nested calls.

STF Inside The Capture
----------------------

In the inner pattern, a captured Warp workload opens a stream-bound context on
the capture stream. The context expresses finer-grained fork-join work inside
one captured task, then finalizes before the surrounding capture ends.

.. code-block:: python

    from warp import stf_experimental as wp_stf

    with wp_stf.context(stream=stream) as ctx:
        tok_a = ctx.token()
        tok_b = ctx.token()

        with ctx.task(tok_a.write()) as (s,):
            phase_a(s)

        with ctx.task(tok_b.write()) as (s,):
            phase_b(s)

        with ctx.task(tok_a.read(), tok_b.read()) as (s,):
            join(s)

This is useful when a solver has a natural sequential call site but contains
independent sub-work that can run as sibling tasks. CUDASTF handles dependency
edges, stream selection, and synchronization back to the local context stream.

Forcing Eager Initialization
----------------------------

CUDASTF performs its one-time CUDA initialization the first time a
``stf.context`` opens. That lazy init can show up as a ``cudaFree(0)`` (or
similar) on the active stream, which interacts with CUDA graph capture:

* Captures opened with :attr:`warp.CaptureMode.RELAXED` (or with
  ``external=True`` after a CUDASTF context is already live) tolerate the
  lazy init.
* Captures opened with the default :attr:`warp.CaptureMode.THREAD_LOCAL`
  reject capture-unsafe runtime calls and will fail if the very first STF
  context is opened *inside* the captured region.

Either way, you can call :func:`warp.stf_experimental.warmup` once at startup
to force the init eagerly:

.. code-block:: python

    import warp as wp
    from warp import stf_experimental as wp_stf

    if wp_stf.is_available():
        wp_stf.warmup()  # idempotent; safe to call from library startup

This guarantees that the first ``stf.context`` inside a capture is a no-op
for CUDA initialization, so the failure mode -- if any -- surfaces outside
the capture rather than during recording.

Pinning Tasks To A Device
-------------------------

By default, :func:`warp.stf_experimental.task` lets CUDASTF use the current
execution place and wraps the resulting task stream on the active Warp device.
Pass ``exec_place`` when a task should run on a specific execution place. The
common Warp spelling is a :class:`warp.Device`; advanced users can pass a
single-device ``cuda.stf.exec_place`` directly.

.. code-block:: python

    import cuda.stf._experimental as stf
    import warp as wp
    from warp import stf_experimental as wp_stf

    with wp_stf.context() as ctx:
        tok = ctx.token()

        with ctx.task(tok.write(), exec_place=wp.get_device("cuda:1")) as (stream,):
            wp.launch(my_kernel, dim=n, inputs=[arr], stream=stream)

        place = stf.exec_place.green_ctx(view)
        with ctx.task(tok.read(), exec_place=place, symbol="green_ctx_task") as (stream,):
            wp.launch(other_kernel, dim=n, inputs=[arr], stream=stream)

The yielded :class:`warp.Stream` carries the device chosen for the execution
place, so :func:`warp.launch` can inherit the device from ``stream``.

When To Use STF
---------------

Use :mod:`warp.stf_experimental` when a workload already has meaningful task
boundaries and dependency edges, but manually managing streams and events would
make the code brittle. Prefer regular Warp graph capture when the work is
naturally linear or when a single stream already exposes enough parallelism
through kernels.

See ``warp/examples/interop/example_stf_task_graph.py`` for a complete example
that combines an outer STF-owned task graph with local task graphs inside its
tasks.
