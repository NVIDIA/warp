# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import warp as wp

THREAD_COUNT = 32
_THIS_MODULE = sys.modules[__name__]


@wp.kernel
def reserve_slots(counter: wp.array[wp.int32], slots: wp.array[wp.int32]):
    slot = wp.atomic_add(counter, 0, 1)
    slots[slot] = wp.tid()


def _set_mode(deterministic, max_records):
    wp.set_module_options(
        {"deterministic": deterministic, "deterministic_max_records": max_records},
        module=_THIS_MODULE,
    )


def main(device):
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    slots = wp.empty(2 * THREAD_COUNT, dtype=wp.int32, device=device)

    # Steps 4 and 5 revisit the variants loaded in steps 2 and 3; stale launch
    # metadata would run them through the wrong launch path (GH-1637).
    steps = (
        (wp.DeterministicMode.RUN_TO_RUN, 1, 256),
        (wp.DeterministicMode.NOT_GUARANTEED, 0, 256),
        (wp.DeterministicMode.RUN_TO_RUN, 1, 128),
        (wp.DeterministicMode.NOT_GUARANTEED, 0, 256),
        (wp.DeterministicMode.RUN_TO_RUN, 1, 128),
    )

    for i, (deterministic, max_records, block_dim) in enumerate(steps):
        counter.zero_()
        _set_mode(deterministic, max_records)
        wp.launch(reserve_slots, dim=THREAD_COUNT, inputs=[counter, slots], device=device, block_dim=block_dim)
        actual = int(counter.numpy()[0])
        if actual != THREAD_COUNT:
            raise AssertionError(
                f"step {i} ({deterministic.name}, block_dim={block_dim}) produced counter value "
                f"{actual}, expected {THREAD_COUNT}"
            )


if __name__ == "__main__":
    main(sys.argv[1])
