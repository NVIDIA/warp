# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example CUDASTF Task-Safe Metadata Readback
#
# Demonstrates reading back a device-side counter (the essence of the
# BsrMatrix ``nnz`` bookkeeping, see NVIDIA/warp#1792) safely across CUDA
# graphs and multiple streams by expressing the readback as tasks.
#
# Hand-rolled versions of this pattern share one pinned buffer and one
# event per structure, which is unsound across streams and graph captures:
# concurrent readbacks race on the shared buffer, and a consumer's wait can
# resolve against the wrong producer. Expressed as STF tasks, the readback
# is a task writing an output and every consumer is ordered against it by
# the runtime:
#
# - Inside a captured scope (task graph), a slow update task and TWO
#   readback tasks of the same structure are declared. The runtime
#   serializes the readbacks against the update (read-after-write) and
#   against each other (write-after-write) -- the unordered-copy-nodes race
#   of the hand-rolled version is inexpressible.
# - The host-side consumption is itself a task: a host access declared
#   after the device accesses (``ctx.host_launch(lnnz.read(), fn=...)``).
#   The runtime owns the device-to-host staging and the ordering, and the
#   callback becomes a host node of the graph, re-executed on every launch
#   with the freshly staged value. There is no user-managed pinned buffer,
#   no event, and no explicit copy.
# - The sealed graph is then launched "in one stream" while the host
#   observation is ordered inside the replay itself:
#   ``graph.launch(stream=launch_stream)`` orders the launch stream behind
#   the STF-internal launch stream. No shared event identity anywhere.
###########################################################################

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

NROW = 4
NNZ_TARGET = 3
SPIN_ITERS = 20_000_000


@wp.kernel
def clear_offsets_kernel(offsets: wp.array[int]):
    offsets[wp.tid()] = 0


@wp.kernel
def spin_then_update_kernel(offsets: wp.array[int], mult: wp.array[float], nrow: int, val: int, iters: int):
    # Memory-fed multiplier and divergent arms defeat dead-code elimination:
    # this models a slow topology update whose completion a naive consumer
    # would race against.
    acc = mult[0]
    for _i in range(iters):
        acc = acc * mult[0] + 1.0e-7
    offsets[nrow] = wp.where(acc >= 0.0, val, val - 1)


@wp.kernel
def readback_kernel(offsets: wp.array[int], nrow: int, nnz_out: wp.array[int]):
    nnz_out[0] = offsets[nrow]


def build_task_graph(device, offsets: wp.array, spin_mult: wp.array, observed: list) -> wp_stf.task_graph:
    graph = wp_stf.task_graph()

    with graph:
        # Tasks may only be declared while the graph is recording, so the
        # context is scoped to the recording block.
        ctx = graph.context

        # The nnz counter is a logical data instance owned by the runtime,
        # so that a host access below can be tracked like any other access.
        lnnz = ctx.logical_data_empty((1,), dtype=np.int32, name="nnz")

        # Reset the structure so every replay is self-contained.
        with ctx.task(ctx.dep(offsets).write()) as (s,):
            wp.launch(clear_offsets_kernel, dim=NROW + 1, inputs=[offsets], stream=s)

        # Slow topology update.
        with ctx.task(ctx.dep(offsets).rw()) as (s,):
            wp.launch(
                spin_then_update_kernel,
                dim=1,
                inputs=[offsets, spin_mult, NROW, NNZ_TARGET, SPIN_ITERS],
                stream=s,
            )

        # Two independent consumers each request a readback of the same
        # structure into the same output -- the shape that races in the
        # hand-rolled single-buffer/single-event version. Declared as tasks,
        # both are ordered after the update, and against each other.
        with ctx.task(ctx.dep(offsets).read(), lnnz.write()) as (s, wnnz):
            wp.launch(readback_kernel, dim=1, inputs=[offsets, NROW, wnnz], stream=s)

        with ctx.task(ctx.dep(offsets).read(), lnnz.rw()) as (s, wnnz):
            wp.launch(readback_kernel, dim=1, inputs=[offsets, NROW, wnnz], stream=s)

        # Host access AFTER the device accesses: the nnz_sync() analog.
        # The runtime stages the value to the host and runs the callback as
        # a host node of the graph, once per launch, ordered after the
        # device readbacks.
        ctx.host_launch(lnnz.read(), fn=lambda nnz: observed.append(int(nnz[0])))

    return graph


def main(device):
    # Force CUDASTF lazy initialization outside any capture.
    wp_stf.warmup(device=device)

    offsets = wp.zeros(NROW + 1, dtype=wp.int32, device=device)
    spin_mult = wp.array(np.array([1.0000001], dtype=np.float32), device=device)

    observed = []
    graph = build_task_graph(device, offsets, spin_mult, observed)

    launch_stream = wp.Stream(device)

    try:
        for i in range(3):
            # Launch the sealed graph "in" launch_stream: the graph runs on
            # the STF-internal stream and launch_stream is ordered behind it.
            graph.launch(stream=launch_stream)

            # The host observation happened INSIDE the replay (host node),
            # ordered after the device readbacks; syncing the launch stream
            # is only needed here to inspect the result from this scope.
            wp.synchronize_stream(launch_stream)

            nnz = observed[-1]
            print(f"launch {i}: nnz = {nnz} (host observations: {len(observed)})")
            if nnz != NNZ_TARGET or len(observed) != i + 1:
                raise RuntimeError(f"stale readback: observed {observed}, expected {i + 1} x {NNZ_TARGET}")
    finally:
        graph.finalize()

    print("task-safe readback: all launches observed the post-update value")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda:0", help="Override the default Warp device.")
    parser.add_argument("--stage-path", type=str, default=None, help="Unused; accepted for test harness compatibility.")
    args = parser.parse_known_args()[0]

    wp.init()

    if not wp.is_cuda_available():
        raise RuntimeError("This example requires a CUDA device.")
    if not wp_stf.is_available():
        raise RuntimeError(
            "This example requires the cuda-stf package "
            "(pip install 'cuda-stf[cu13] @ git+https://github.com/NVIDIA/cccl.git@main#subdirectory=python/cuda_stf')."
        )

    device = wp.get_device(args.device)
    with wp.ScopedDevice(device):
        main(device)
