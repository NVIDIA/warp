# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np

import warp as wp
import warp.examples
import warp.optim
from warp.tests.unittest_utils import *


def create_layer(rng, dim_in, dim_hid, dtype=float):
    w = rng.uniform(-1.0 / np.sqrt(dim_in), 1.0 / np.sqrt(dim_in), (dim_hid, dim_in))
    b = rng.uniform(-1.0 / np.sqrt(dim_in), 1.0 / np.sqrt(dim_in), (dim_hid, 1))

    weights = wp.array(w, dtype=dtype, requires_grad=True)
    bias = wp.array(b, dtype=dtype, requires_grad=True)

    return (weights, bias)


def create_array(rng, dim_in, dim_hid, dtype=float):
    s = rng.uniform(-1.0 / np.sqrt(dim_in), 1.0 / np.sqrt(dim_in), (dim_hid, dim_in))
    a = wp.array(s, dtype=dtype, requires_grad=True)

    return a


def test_multi_layer_nn(test, device):
    import torch as tc

    if device.is_cuda and not wp.context.runtime.core.is_mathdx_enabled():
        test.skipTest("Skipping test on CUDA device without MathDx (tolerance)")

    NUM_FREQ = wp.constant(8)

    DIM_IN = wp.constant(4 * NUM_FREQ)  # sin,cos for both x,y at each frequency
    DIM_HID = 32
    DIM_OUT = 3

    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    BATCH_SIZE = min(512, int((IMG_WIDTH * IMG_HEIGHT) / 8))

    if device.is_cpu:
        NUM_THREADS = 1
    else:
        NUM_THREADS = 32

    dtype = wp.float16
    npdtype = wp.types.warp_type_to_np_dtype[dtype]

    @wp.func
    def relu(x: dtype):
        return wp.max(x, dtype(0.0))

    @wp.func
    def sigmoid(x: dtype):
        return dtype(1.0 / (1.0 + wp.exp(-float(x))))

    @wp.kernel
    def zero(loss: wp.array(dtype=float)):
        loss[0] = 0.0

    @wp.kernel(module="unique")
    def compute(
        batches: wp.array(dtype=int),
        input: wp.array2d(dtype=dtype),
        weights_0: wp.array2d(dtype=dtype),
        bias_0: wp.array2d(dtype=dtype),
        weights_1: wp.array2d(dtype=dtype),
        bias_1: wp.array2d(dtype=dtype),
        weights_2: wp.array2d(dtype=dtype),
        bias_2: wp.array2d(dtype=dtype),
        weights_3: wp.array2d(dtype=dtype),
        bias_3: wp.array2d(dtype=dtype),
        reference: wp.array2d(dtype=float),
        loss: wp.array1d(dtype=float),
        out: wp.array2d(dtype=float),
    ):
        linear = batches[wp.tid()]
        row = linear / IMG_WIDTH
        col = linear % IMG_WIDTH

        # normalize input coordinates to [-1, 1]
        x = (float(row) / float(IMG_WIDTH) - 0.5) * 2.0
        y = (float(col) / float(IMG_HEIGHT) - 0.5) * 2.0

        local = wp.vector(dtype=dtype, length=DIM_IN)

        # construct positional encoding
        for s in range(NUM_FREQ):
            scale = wp.pow(2.0, float(s)) * wp.pi

            # x-coord
            local[s * 4 + 0] = dtype(wp.sin(x * scale))
            local[s * 4 + 1] = dtype(wp.cos(x * scale))

            # y-coord
            local[s * 4 + 2] = dtype(wp.sin(y * scale))
            local[s * 4 + 3] = dtype(wp.cos(y * scale))

            # write input back to array so that torch can use it
            input[s * 4 + 0, linear] = local[s * 4 + 0]
            input[s * 4 + 1, linear] = local[s * 4 + 1]
            input[s * 4 + 2, linear] = local[s * 4 + 2]
            input[s * 4 + 3, linear] = local[s * 4 + 3]

        # tile feature vectors across the block, returns [dim(f), NUM_THREADS]
        f = wp.tile(local)

        # input layer
        w0 = wp.tile_load(weights_0, shape=(DIM_HID, DIM_IN))
        b0 = wp.tile_load(bias_0, shape=(DIM_HID, 1))
        z = wp.tile_map(relu, wp.tile_matmul(w0, f) + wp.tile_broadcast(b0, shape=(DIM_HID, NUM_THREADS)))

        # hidden layer
        w1 = wp.tile_load(weights_1, shape=(DIM_HID, DIM_HID))
        b1 = wp.tile_load(bias_1, shape=(DIM_HID, 1))
        z = wp.tile_map(relu, wp.tile_matmul(w1, z) + wp.tile_broadcast(b1, shape=(DIM_HID, NUM_THREADS)))

        w2 = wp.tile_load(weights_2, shape=(DIM_HID, DIM_HID))
        b2 = wp.tile_load(bias_2, shape=(DIM_HID, 1))
        z = wp.tile_map(relu, wp.tile_matmul(w2, z) + wp.tile_broadcast(b2, shape=(DIM_HID, NUM_THREADS)))

        # output layer
        w3 = wp.tile_load(weights_3, shape=(DIM_OUT, DIM_HID))
        b3 = wp.tile_load(bias_3, shape=(DIM_OUT, 1))
        o = wp.tile_map(relu, wp.tile_matmul(w3, z) + wp.tile_broadcast(b3, shape=(DIM_OUT, NUM_THREADS)))

        # untile back to SIMT
        output = wp.untile(o)

        # compute error
        error = wp.vec3(
            float(output[0]) - reference[0, linear],
            float(output[1]) - reference[1, linear],
            float(output[2]) - reference[2, linear],
        )

        # write MSE loss
        wp.atomic_add(loss, 0, wp.length_sq(error) / float(3 * BATCH_SIZE))

        # image output
        for i in range(DIM_OUT):
            out[i, linear] = float(output[i])

    with wp.ScopedDevice(device):
        torch_device = wp.device_to_torch(device)

        rng = np.random.default_rng(45)

        weights_0, bias_0 = create_layer(rng, DIM_IN, DIM_HID, dtype=dtype)
        weights_1, bias_1 = create_layer(rng, DIM_HID, DIM_HID, dtype=dtype)
        weights_2, bias_2 = create_layer(rng, DIM_HID, DIM_HID, dtype=dtype)
        weights_3, bias_3 = create_layer(rng, DIM_HID, DIM_OUT, dtype=dtype)

        input = create_array(rng, IMG_WIDTH * IMG_HEIGHT, DIM_IN, dtype=dtype)
        output = create_array(rng, IMG_WIDTH * IMG_HEIGHT, DIM_OUT)

        reference_np = (
            np.load(os.path.join(os.path.dirname(__file__), "..", "assets", "pixel.npy"), allow_pickle=True) / 255.0
        )
        reference = wp.array(reference_np, dtype=float)

        assert reference.shape[1] == IMG_WIDTH * IMG_HEIGHT

        loss = wp.zeros(1, dtype=float, requires_grad=True)

        params = [weights_0, bias_0, weights_1, bias_1, weights_2, bias_2, weights_3, bias_3]

        optimizer_grads = [p.grad.flatten() for p in params]
        optimizer_inputs = [p.flatten() for p in params]
        optimizer = warp.optim.Adam(optimizer_inputs, lr=0.01)

        num_batches = int((IMG_WIDTH * IMG_HEIGHT) / BATCH_SIZE)
        max_epochs = 30

        # create randomized batch indices
        batches = np.arange(0, IMG_WIDTH * IMG_HEIGHT, dtype=np.int32)
        rng.shuffle(batches)
        batches = wp.array(batches)

        with wp.ScopedTimer("Training", active=False):
            for epoch in range(max_epochs):
                for b in range(0, IMG_WIDTH * IMG_HEIGHT, BATCH_SIZE):
                    loss.zero_()

                    with wp.Tape() as tape:
                        wp.launch(
                            compute,
                            dim=[BATCH_SIZE],
                            inputs=[
                                batches[b : b + BATCH_SIZE],
                                input,
                                weights_0,
                                bias_0,
                                weights_1,
                                bias_1,
                                weights_2,
                                bias_2,
                                weights_3,
                                bias_3,
                                reference,
                                loss,
                                output,
                            ],
                            block_dim=NUM_THREADS,
                        )

                    tape.backward(loss)

                    # check outputs + grads on the first few epoch only
                    # since this is a relatively slow operation
                    verify = True
                    if verify and epoch < 3:
                        indices = batches[b : b + BATCH_SIZE].numpy()

                        z_np = np.maximum(weights_0.numpy() @ input.numpy()[:, indices] + bias_0.numpy(), 0.0)
                        z_np = np.maximum(weights_1.numpy() @ z_np + bias_1.numpy(), 0.0)
                        z_np = np.maximum(weights_2.numpy() @ z_np + bias_2.numpy(), 0.0)
                        z_np = np.maximum(weights_3.numpy() @ z_np + bias_3.numpy(), 0.0)

                        # test numpy forward
                        assert_np_equal(output.numpy()[:, indices].astype(npdtype), z_np, tol=1.0e-2)

                        # torch
                        input_tc = tc.tensor(input.numpy()[:, indices], requires_grad=True, device=torch_device)

                        weights_0_tc = tc.tensor(weights_0.numpy(), requires_grad=True, device=torch_device)
                        bias_0_tc = tc.tensor(bias_0.numpy(), requires_grad=True, device=torch_device)

                        weights_1_tc = tc.tensor(weights_1.numpy(), requires_grad=True, device=torch_device)
                        bias_1_tc = tc.tensor(bias_1.numpy(), requires_grad=True, device=torch_device)

                        weights_2_tc = tc.tensor(weights_2.numpy(), requires_grad=True, device=torch_device)
                        bias_2_tc = tc.tensor(bias_2.numpy(), requires_grad=True, device=torch_device)

                        weights_3_tc = tc.tensor(weights_3.numpy(), requires_grad=True, device=torch_device)
                        bias_3_tc = tc.tensor(bias_3.numpy(), requires_grad=True, device=torch_device)

                        z_tc = tc.clamp(weights_0_tc @ input_tc + bias_0_tc, min=0.0)
                        z_tc = tc.clamp(weights_1_tc @ z_tc + bias_1_tc, min=0.0)
                        z_tc = tc.clamp(weights_2_tc @ z_tc + bias_2_tc, min=0.0)
                        z_tc = tc.clamp(weights_3_tc @ z_tc + bias_3_tc, min=0.0)

                        ref_tc = tc.tensor(reference.numpy()[:, indices], requires_grad=True, device=torch_device)

                        l_tc = tc.mean((z_tc - ref_tc) ** 2)
                        l_tc.backward()

                        # test torch
                        assert_np_equal(
                            z_tc.cpu().detach().numpy(), output.numpy()[:, indices].astype(npdtype), tol=1.0e-2
                        )
                        assert_np_equal(weights_0.grad.numpy(), weights_0_tc.grad.cpu().detach().numpy(), tol=1.0e-2)
                        assert_np_equal(bias_0.grad.numpy(), bias_0_tc.grad.cpu().detach().numpy(), tol=1.0e-2)
                        assert_np_equal(weights_1.grad.numpy(), weights_1_tc.grad.cpu().detach().numpy(), tol=1.0e-2)
                        assert_np_equal(bias_1.grad.numpy(), bias_1_tc.grad.cpu().detach().numpy(), tol=1.0e-2)
                        assert_np_equal(weights_2.grad.numpy(), weights_2_tc.grad.cpu().detach().numpy(), tol=1.0e-2)
                        assert_np_equal(bias_2.grad.numpy(), bias_2_tc.grad.cpu().detach().numpy(), tol=1.0e-2)
                        assert_np_equal(weights_3.grad.numpy(), weights_3_tc.grad.cpu().detach().numpy(), tol=1.0e-2)
                        assert_np_equal(bias_3.grad.numpy(), bias_3_tc.grad.cpu().detach().numpy(), tol=1.0e-2)

                    optimizer.step(optimizer_grads)
                    tape.zero()

        # initial loss is ~0.061
        test.assertLess(loss.numpy()[0], 0.002)


def test_single_layer_nn(test, device):
    import torch as tc

    DIM_IN = 8
    DIM_HID = 32
    DIM_OUT = 16

    NUM_BLOCKS = 56

    if device.is_cpu:
        NUM_THREADS = 1
    else:
        NUM_THREADS = 32

    @wp.func
    def relu(x: float):
        return wp.max(x, 0.0)

    @wp.kernel(module="unique")
    def compute(
        input: wp.array2d(dtype=float),
        weights: wp.array2d(dtype=float),
        bias: wp.array2d(dtype=float),
        out: wp.array2d(dtype=float),
    ):
        i = wp.tid()

        f = wp.tile_load(input, shape=(DIM_IN, NUM_THREADS), offset=(0, i * NUM_THREADS))

        w = wp.tile_load(weights, shape=(DIM_OUT, DIM_IN))
        b = wp.tile_load(bias, shape=(DIM_OUT, 1))

        o = wp.tile_map(relu, wp.tile_matmul(w, f) + wp.tile_broadcast(b, shape=(DIM_OUT, NUM_THREADS)))

        wp.tile_store(out, o, offset=(0, i * NUM_THREADS))

    with wp.ScopedDevice(device):
        rng = np.random.default_rng(45)

        # single layer weights, bias
        weights, bias = create_layer(rng, DIM_IN, DIM_OUT, dtype=float)

        input = create_array(rng, NUM_THREADS * NUM_BLOCKS, DIM_IN)
        output = create_array(rng, NUM_THREADS * NUM_BLOCKS, DIM_OUT)

        with wp.Tape() as tape:
            wp.launch_tiled(compute, dim=[NUM_BLOCKS], inputs=[input, weights, bias, output], block_dim=NUM_THREADS)

        output.grad = wp.ones_like(output)
        tape.backward()

        # numpy
        output_np = np.maximum(weights.numpy() @ input.numpy() + bias.numpy(), 0.0)

        # test numpy forward
        assert_np_equal(output.numpy(), output_np, tol=1.0e-2)

        # torch
        weights_tc = tc.from_numpy(weights.numpy()).requires_grad_(True)  # use .numpy() to avoid any memory aliasing
        input_tc = tc.from_numpy(input.numpy()).requires_grad_(True)
        bias_tc = tc.from_numpy(bias.numpy()).requires_grad_(True)

        output_tc = tc.clamp(weights_tc @ input_tc + bias_tc, min=0.0)
        output_tc.backward(tc.ones_like(output_tc))

        # test torch
        assert_np_equal(output_tc.detach().numpy(), output.numpy(), tol=1.0e-2)
        assert_np_equal(input.grad.numpy(), input_tc.grad.detach().numpy(), tol=1.0e-2)


class TestTileMLP(unittest.TestCase):
    pass


test_devices = get_test_devices()

try:
    import torch

    # check which Warp devices work with Torch
    torch_compatible_devices = []
    torch_compatible_cuda_devices = []

    for d in test_devices:
        try:
            t = torch.arange(10, device=wp.device_to_torch(d))
            t += 1
            torch_compatible_devices.append(d)
            if d.is_cuda:
                torch_compatible_cuda_devices.append(d)
        except Exception as e:
            print(f"Skipping Torch tests on device '{d}' due to exception: {e}")

    add_function_test(
        TestTileMLP,
        "test_single_layer_nn",
        test_single_layer_nn,
        check_output=False,
        devices=torch_compatible_devices,
    )
    add_function_test(
        TestTileMLP,
        "test_multi_layer_nn",
        test_multi_layer_nn,
        check_output=False,
        devices=torch_compatible_cuda_devices,
    )

except Exception as e:
    print(f"Skipping Torch tests due to exception: {e}")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    wp.clear_lto_cache()
    unittest.main(verbosity=2, failfast=True)
