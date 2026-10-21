# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe integer texture sampling in a process that can safely hit a legacy CUDA trap."""

import numpy as np

import warp as wp


@wp.kernel(enable_backward=False)
def sample_texture(texture: wp.Texture2D, output: wp.array[wp.vec4f]):
    output[0] = wp.texture_sample(texture, wp.vec2f(0.5, 0.5), dtype=wp.vec4f)


def main():
    wp.config.quiet = True
    device = wp.get_device("cuda:0")
    output = wp.empty(1, dtype=wp.vec4f, device=device)
    cases = (
        (np.float32, [1.25, -2.5, 3.75, 0.0]),
        (np.int32, [-17, 0, 257, 16777217]),
        (np.uint32, [0, 257, 16777217, 4294967295]),
    )
    for dtype, values in cases:
        data = np.array([[values]], dtype=dtype)
        texture = wp.Texture2D(data, filter_mode=wp.TextureFilterMode.CLOSEST, device=device)
        wp.launch(sample_texture, dim=1, inputs=[texture], outputs=[output], device=device)
        np.testing.assert_array_equal(output.numpy(), data.reshape(1, 4).astype(np.float32))


if __name__ == "__main__":
    main()
