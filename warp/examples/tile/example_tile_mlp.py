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

###########################################################################
# Example Image Multilayer Perceptron (MLP)
#
# Shows how to train a coordinate-based MLP on an image to predict the RGB
# color at a given input position. By default, a positional encoding is
# applied to the input coordinates to improve the ability of the MLP to
# represent higher-frequency content. This can be disabled by passing the
# '--no_encoding' option.
#
# References:
#   Ben Mildenhall et al. 2021. NeRF: representing scenes
#   as neural radiance fields for view synthesis. Commun. ACM 65, 1
#   (January 2022), 99–106. https://doi.org/10.1145/3503250
#
###########################################################################

# ruff: noqa: RUF003

import math
import os

import numpy as np
from PIL import Image

import warp as wp
import warp.examples
import warp.optim

rng = np.random.default_rng(45)


def create_layer(dim_in, dim_hid, dtype=float):
    w = rng.uniform(-1.0 / np.sqrt(dim_in), 1.0 / np.sqrt(dim_in), (dim_hid, dim_in))
    b = rng.uniform(-1.0 / np.sqrt(dim_in), 1.0 / np.sqrt(dim_in), (dim_hid, 1))

    weights = wp.array(w, dtype=dtype, requires_grad=True)
    bias = wp.array(b, dtype=dtype, requires_grad=True)

    return (weights, bias)


def create_array(dim_in, dim_hid, dtype=float):
    s = rng.uniform(-1.0 / np.sqrt(dim_in), 1.0 / np.sqrt(dim_in), (dim_hid, dim_in))
    a = wp.array(s, dtype=dtype, requires_grad=True)

    return a


# number of frequencies for the positional encoding
NUM_FREQ = wp.constant(8)

DIM_IN = wp.constant(4 * NUM_FREQ)  # sin,cos for both x,y at each frequenecy
DIM_HID = 32
DIM_OUT = 3

# threads per-block
NUM_THREADS = 32

IMG_WIDTH = 512
IMG_HEIGHT = 512

BATCH_SIZE = min(1024, int((IMG_WIDTH * IMG_HEIGHT) / 8))

# dtype for our weights and bias matrices
dtype = wp.float16


@wp.func
def relu(x: dtype):
    return wp.max(x, dtype(0.0))


@wp.kernel
def compute(
    indices: wp.array(dtype=int),
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
    # batch indices
    linear = indices[wp.tid()]

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
    if loss:
        wp.atomic_add(loss, 0, wp.length_sq(error) / float(3 * BATCH_SIZE))

    #  write image output
    if out:
        for i in range(DIM_OUT):
            out[i, linear] = float(output[i])


class Example:
    def __init__(self, train_iters):
        self.weights_0, self.bias_0 = create_layer(DIM_IN, DIM_HID, dtype=dtype)
        self.weights_1, self.bias_1 = create_layer(DIM_HID, DIM_HID, dtype=dtype)
        self.weights_2, self.bias_2 = create_layer(DIM_HID, DIM_HID, dtype=dtype)
        self.weights_3, self.bias_3 = create_layer(DIM_HID, DIM_OUT, dtype=dtype)

        # reference
        reference_path = os.path.join(wp.examples.get_asset_directory(), "pixel.jpg")
        with Image.open(reference_path) as im:
            reference_image = np.asarray(im.resize((IMG_WIDTH, IMG_HEIGHT)).convert("RGB")) / 255.0
        self.reference = wp.array(reference_image.reshape(IMG_WIDTH * IMG_HEIGHT, 3).T, dtype=float)

        # create randomized batch indices
        indices = np.arange(0, IMG_WIDTH * IMG_HEIGHT, dtype=np.int32)
        rng.shuffle(indices)
        self.indices = wp.array(indices)

        self.num_batches = int((IMG_WIDTH * IMG_HEIGHT) / BATCH_SIZE)
        self.max_iters = train_iters
        self.max_epochs = max(1, int(self.max_iters / self.num_batches))

    def train_warp(self):
        params = [
            self.weights_0,
            self.bias_0,
            self.weights_1,
            self.bias_1,
            self.weights_2,
            self.bias_2,
            self.weights_3,
            self.bias_3,
        ]

        optimizer_grads = [p.grad.flatten() for p in params]
        optimizer_inputs = [p.flatten() for p in params]
        optimizer = warp.optim.Adam(optimizer_inputs, lr=0.01)

        loss = wp.zeros(1, dtype=float, requires_grad=True)
        output = create_array(IMG_WIDTH * IMG_HEIGHT, DIM_OUT)

        # capture graph for whole epoch
        wp.capture_begin()

        for b in range(0, IMG_WIDTH * IMG_HEIGHT, BATCH_SIZE):
            loss.zero_()

            with wp.Tape() as tape:
                wp.launch(
                    compute,
                    dim=[BATCH_SIZE],
                    inputs=[
                        self.indices[b : b + BATCH_SIZE],
                        self.weights_0,
                        self.bias_0,
                        self.weights_1,
                        self.bias_1,
                        self.weights_2,
                        self.bias_2,
                        self.weights_3,
                        self.bias_3,
                        self.reference,
                        loss,
                        None,
                    ],
                    block_dim=NUM_THREADS,
                )

            tape.backward(loss)
            optimizer.step(optimizer_grads)
            tape.zero()

        graph = wp.capture_end()

        with wp.ScopedTimer("Training"):
            for i in range(self.max_epochs):
                with wp.ScopedTimer("Epoch"):
                    wp.capture_launch(graph)
                    print(f"Epoch: {i} Loss: {loss.numpy()}")

        # evaluate full image
        wp.launch(
            compute,
            dim=[IMG_WIDTH * IMG_HEIGHT],
            inputs=[
                self.indices,
                self.weights_0,
                self.bias_0,
                self.weights_1,
                self.bias_1,
                self.weights_2,
                self.bias_2,
                self.weights_3,
                self.bias_3,
                self.reference,
                loss,
                output,
            ],
            block_dim=NUM_THREADS,
        )

        self.save_image("example_tile_mlp.jpg", output.numpy())

    def train_torch(self):
        import torch as tc

        weights_0 = tc.nn.Parameter(wp.to_torch(self.weights_0))
        weights_1 = tc.nn.Parameter(wp.to_torch(self.weights_1))
        weights_2 = tc.nn.Parameter(wp.to_torch(self.weights_2))
        weights_3 = tc.nn.Parameter(wp.to_torch(self.weights_3))

        bias_0 = tc.nn.Parameter(wp.to_torch(self.bias_0))
        bias_1 = tc.nn.Parameter(wp.to_torch(self.bias_1))
        bias_2 = tc.nn.Parameter(wp.to_torch(self.bias_2))
        bias_3 = tc.nn.Parameter(wp.to_torch(self.bias_3))

        indices = wp.to_torch(self.indices)
        reference = wp.to_torch(self.reference)

        optimizer = tc.optim.Adam(
            [weights_0, bias_0, weights_1, bias_1, weights_2, bias_2, weights_3, bias_3],
            capturable=True,
            lr=0.0001,
            betas=(0.9, 0.95),
            eps=1.0e-6,
        )

        # generate frequency space encoding of pixels
        # based on their linear index in the image
        def encode(linear):
            row = (linear // IMG_WIDTH).float()
            col = (linear % IMG_WIDTH).float()

            x = (row / float(IMG_WIDTH) - 0.5) * 2.0
            y = (col / float(IMG_HEIGHT) - 0.5) * 2.0

            encoding = tc.zeros((NUM_FREQ * 4, len(linear)), dtype=tc.float16, device="cuda")

            for s in range(NUM_FREQ):
                scale = math.pow(2.0, float(s)) * math.pi

                # Directly write the computed values into the encoding tensor
                encoding[s * 4 + 0, :] = tc.sin(scale * x)
                encoding[s * 4 + 1, :] = tc.cos(scale * x)
                encoding[s * 4 + 2, :] = tc.sin(scale * y)
                encoding[s * 4 + 3, :] = tc.cos(scale * y)

            return encoding

        stream = tc.cuda.Stream()
        graph = tc.cuda.CUDAGraph()

        # warm-up
        with tc.cuda.stream(stream):
            f = tc.rand((NUM_FREQ * 4, BATCH_SIZE), dtype=tc.float16, device="cuda")
            z = tc.relu(weights_0 @ f + bias_0)
            z = tc.relu(weights_1 @ z + bias_1)
            z = tc.relu(weights_2 @ z + bias_2)
            z = tc.relu(weights_3 @ z + bias_3)
            ref = tc.rand((3, BATCH_SIZE), dtype=tc.float16, device="cuda")
            loss = tc.mean((z - ref) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with tc.cuda.graph(graph):
            for b in range(0, IMG_WIDTH * IMG_HEIGHT, BATCH_SIZE):
                linear = indices[b : b + BATCH_SIZE]

                f = encode(linear)

                z = tc.relu(weights_0 @ f + bias_0)
                z = tc.relu(weights_1 @ z + bias_1)
                z = tc.relu(weights_2 @ z + bias_2)
                z = tc.relu(weights_3 @ z + bias_3)

                ref = reference[:, linear]
                loss = tc.mean((z - ref) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with wp.ScopedTimer("Training (Torch)"):
            for _i in range(self.max_epochs):
                with wp.ScopedTimer("Epoch"):
                    graph.replay()

                    print(loss)

        f = encode(tc.arange(0, IMG_WIDTH * IMG_HEIGHT))
        z = tc.relu(weights_0 @ f + bias_0)
        z = tc.relu(weights_1 @ z + bias_1)
        z = tc.relu(weights_2 @ z + bias_2)
        z = tc.relu(weights_3 @ z + bias_3)

        self.save_image("example_tile_mlp_torch.jpg", z.detach().cpu().numpy())

    def save_image(self, name, output):
        predicted_image = output.T.reshape(IMG_WIDTH, IMG_HEIGHT, 3)
        predicted_image = (predicted_image * 255).astype(np.uint8)

        predicted_image_pil = Image.fromarray(predicted_image)
        predicted_image_pil.save(name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_iters", type=int, default=20000, help="Total number of training iterations.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice("cuda:0"):
        example = Example(args.train_iters)
        example.train_warp()
        # example.train_torch()
