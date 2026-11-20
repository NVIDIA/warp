# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check Warp's JAX device lookup in one process of a two-process ``jax.distributed`` program.

``test_jax.py`` runs this script twice at the same time:
``python aux_test_jax_distributed_devices.py <coordinator address> <process ID> [--require-gpu]``.
Under ``jax.distributed``, ``jax.devices()`` and ``jax.Device.id`` cover both processes, while
Warp's CUDA ordinals are local to each process.
"""

import sys

import jax

import warp as wp


def main():
    """Join the ``jax.distributed`` program and check Warp's device lookup in this process."""
    jax.distributed.initialize(coordinator_address=sys.argv[1], num_processes=2, process_id=int(sys.argv[2]))
    rank = jax.process_index()

    if "--require-gpu" in sys.argv[3:]:
        # Without a local GPU, the GPU checks below would pass without checking anything.
        assert any(d.platform == "gpu" for d in jax.local_devices()), "JAX has no GPU in this process"

    # wp.device_to_jax() returns this process's CPU device, not process 0's.
    assert wp.device_to_jax("cpu").process_index == rank

    for jax_gpu in (d for d in jax.devices() if d.platform == "gpu"):
        if jax_gpu.process_index == rank:
            # This process's GPU maps to the Warp device with the same process-local ordinal, and back.
            warp_gpu = wp.device_from_jax(jax_gpu)
            assert warp_gpu.ordinal == jax_gpu.local_hardware_id, f"{jax_gpu} mapped to {warp_gpu}"
            assert wp.device_to_jax(warp_gpu) == jax_gpu
        else:
            # The other process's GPU is rejected instead of being mapped to one of this process's GPUs.
            try:
                wp.device_from_jax(jax_gpu)
            except RuntimeError:
                continue
            raise AssertionError(f"wp.device_from_jax() accepted {jax_gpu} of process {jax_gpu.process_index}")

    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
