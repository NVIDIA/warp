# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise staged graph replay across a collective with asymmetric work on two GPUs.

Run with CUDA_SCALE_LAUNCH_QUEUES=0.25x to reach CUDA queue backpressure with fewer launches.
"""

from __future__ import annotations

import faulthandler
import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import warp as wp


@wp.kernel
def increment(values: wp.array1d[wp.float32], output: wp.array1d[wp.float32]) -> None:
    index = wp.tid()
    output[index] = values[index] + 1.0


def launch_increment(values: wp.array1d[wp.float32], output: wp.array1d[wp.float32]) -> None:
    wp.launch(increment, dim=values.size, inputs=[values], outputs=[output])


class StagedCollectiveTest(unittest.TestCase):
    def test_peer_can_enter_callback_during_replay(self) -> None:
        devices = jax.local_devices(backend="gpu")[:2]
        self.assertEqual(len(devices), 2)
        width = 1 << 20
        before_mean = 1 << 10
        after_mean = 1 << 11
        callback = wp.jax_callable(
            launch_increment,
            num_outputs=1,
            output_dims={"output": (width,)},
            graph_mode=wp.JaxCallableGraphMode.WARP_STAGED_EX,
        )

        def replay(_: jax.Array, values: jax.Array) -> jax.Array:
            return callback(values)[0]

        @functools.partial(jax.pmap, axis_name="gpu", devices=devices)
        def run(values: jax.Array) -> jax.Array:
            # One GPU can queue replays behind the mean while its peer still
            # needs the callback lock to finish the work preceding the mean.
            values = jax.lax.cond(
                jax.lax.axis_index("gpu") == 1,
                lambda x: jax.lax.fori_loop(0, before_mean, replay, x),
                lambda x: x,
                values,
            )
            return jax.lax.fori_loop(0, after_mean, replay, jax.lax.pmean(values, "gpu"))

        values = jnp.zeros((2, width), dtype=jnp.float32)
        for iteration in range(1, 17):
            faulthandler.dump_traceback_later(60, exit=True)
            try:
                values = run(values)
                values.block_until_ready()
            finally:
                faulthandler.cancel_dump_traceback_later()
            expected = iteration * (before_mean // 2 + after_mean)
            np.testing.assert_array_equal(np.asarray(values), expected)

        # Exercise concurrent timing writes with one replay per device.
        single_replay = jax.pmap(lambda x: callback(x)[0], devices=devices)
        single_replay(values).block_until_ready()
        wp.timing_begin(wp.TIMING_GRAPH | wp.TIMING_MEMCPY)
        try:
            single_replay(values).block_until_ready()
        finally:
            timings = wp.timing_end()
        for device in devices:
            recorded = [r.filter for r in timings if r.device == wp.device_from_jax(device)]
            self.assertCountEqual(recorded, [wp.TIMING_GRAPH, wp.TIMING_MEMCPY, wp.TIMING_MEMCPY])


if __name__ == "__main__":
    unittest.main()
