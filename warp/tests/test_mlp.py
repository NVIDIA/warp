# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.func
def mlp_activation(z: float):
    return wp.tanh(z)


@wp.kernel
def mlp_kernel(
    weights: wp.array2d(dtype=float),
    bias: wp.array(dtype=float),
    x: wp.array2d(dtype=float),
    y: wp.array2d(dtype=float),
):
    wp.mlp(weights, bias, mlp_activation, wp.tid(), x, y)


@wp.kernel
def loss_kernel(x: wp.array2d(dtype=float), loss: wp.array(dtype=float)):
    i, j = wp.tid()

    wp.atomic_add(loss, 0, x[i, j] * x[i, j])


def test_mlp(test, device):
    rng = np.random.default_rng(123)

    m = 10
    n = 200

    batches = 20000

    weights = wp.array(rng.random(size=(m, n)) * 0.5 - 0.5, dtype=float, device=device)
    bias = wp.array(rng.random(size=m) * 0.5 - 0.5, dtype=float, device=device)

    x = wp.array(rng.random(size=(n, batches)), dtype=float, device=device)
    y = wp.zeros(shape=(m, batches), device=device)

    with wp.ScopedTimer("warp", active=False):
        wp.launch(mlp_kernel, dim=batches, inputs=[weights, bias, x, y], device=device)
        wp.synchronize()

    # A*x + b
    with wp.ScopedTimer("numpy", active=False):
        expect = np.tanh(weights.numpy().reshape(m, n) @ x.numpy().reshape(-1, batches) + bias.numpy().reshape(m, 1))

    result = y.numpy().reshape(-1, batches)

    assert_np_equal(result, expect, tol=1.0e-6)


def create_mlp(m, n):
    import torch

    torch.manual_seed(0)

    class FeedForward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()

            self.input_size = input_size
            self.hidden_size = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.act = torch.nn.Tanh()

        def forward(self, x):
            out = self.fc1(x)
            out = self.act(out)
            return out

    return FeedForward(m, n)


def create_golden():
    import torch

    rng = np.random.default_rng(123)

    input_size = 32
    hidden_size = 16
    batch_size = 64

    network = create_mlp(input_size, hidden_size)

    x = torch.Tensor(rng.random(size=(batch_size, input_size)))
    x.requires_grad = True

    y = network.forward(x)
    y.retain_grad()

    loss = torch.inner(y.flatten(), y.flatten())
    loss.backward(retain_graph=True)

    results = {}
    results["weights"] = network.fc1.weight.cpu().detach().numpy()
    results["weights_grad"] = network.fc1.weight.grad.cpu().detach().numpy()
    results["bias"] = network.fc1.bias.cpu().detach().numpy()
    results["bias_grad"] = network.fc1.bias.grad.cpu().detach().numpy()
    results["x"] = x.cpu().detach().numpy()
    results["x_grad"] = x.grad.cpu().detach().numpy()
    results["y"] = y.cpu().detach().numpy()
    results["y_grad"] = y.grad.cpu().detach().numpy()
    results["loss"] = loss.cpu().detach().numpy()

    np.save(os.path.join(os.path.dirname(__file__), "assets/mlp_golden.npy"), results, allow_pickle=True)


def load_golden():
    return np.load(os.path.join(os.path.dirname(__file__), "assets/mlp_golden.npy"), allow_pickle=True).item()


def test_mlp_grad(test, device):
    # uncomment to re-build golden files
    # create_golden()

    results = load_golden()

    torch_weights = results["weights"]
    torch_weights_grad = results["weights_grad"]
    torch_bias = results["bias"]
    torch_bias_grad = results["bias_grad"]
    torch_x = results["x"].T
    torch_x_grad = results["x_grad"].T
    torch_y = results["y"].T
    torch_y_grad = results["y_grad"].T
    torch_loss = results["loss"].T

    weights = wp.array(torch_weights, dtype=float, device=device, requires_grad=True)
    bias = wp.array(torch_bias, dtype=float, device=device, requires_grad=True)

    x = wp.array(torch_x, dtype=float, device=device, requires_grad=True)
    y = wp.array(torch_y, dtype=float, device=device, requires_grad=True)
    y.zero_()

    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    m = torch_weights.shape[0]
    n = torch_weights.shape[1]
    b = torch_x.shape[1]

    tape = wp.Tape()
    with tape:
        wp.launch(mlp_kernel, dim=b, inputs=[weights, bias, x, y], device=device)
        wp.launch(loss_kernel, dim=y.shape, inputs=[y, loss], device=device)

    tape.backward(loss=loss)

    # check forward result
    assert_np_equal(y.numpy().reshape(-1, b), torch_y, tol=1.0e-1)
    assert_np_equal(loss.numpy(), torch_loss, tol=1.0e-1)

    # check backward result
    assert_np_equal(tape.gradients[weights].numpy().reshape(m, n), torch_weights_grad, tol=1.0e-1)
    assert_np_equal(tape.gradients[bias].numpy(), torch_bias_grad, tol=1.0e-1)
    assert_np_equal(tape.gradients[x].numpy().reshape(n, b), torch_x_grad, tol=1.0e-1)
    assert_np_equal(tape.gradients[y].numpy().reshape(m, b), torch_y_grad, tol=1.0e-1)


def profile_mlp_torch():
    import torch

    rng = np.random.default_rng(123)

    m = 128
    n = 64

    steps = 20

    for i in range(steps):
        b = 2**i

        network = create_mlp(m, n)

        x = torch.Tensor(rng.random(size=(b, m)))

        with wp.ScopedTimer("torch_forward" + str(b)):
            y = network.forward(x)
            torch.cuda.synchronize()

    for i in range(steps):
        b = 2**i

        network = create_mlp(m, n)

        x = torch.Tensor(rng.random(size=(b, m)))
        y = network.forward(x)

        loss = torch.norm(y)

        # run once to alloc all gradients
        loss.backward(retain_graph=True)

        with wp.ScopedTimer("torch-backward" + str(b)):
            loss.backward()
            torch.cuda.synchronize()


def profile_mlp_warp(device):
    rng = np.random.default_rng(123)

    m = 128
    n = 64

    steps = 20

    for i in range(steps):
        b = 2**i

        weights = wp.array(rng.random(size=(m, n)) * 0.5 - 0.5, dtype=float, device=device)
        bias = wp.array(rng.random(size=m) * 0.5 - 0.5, dtype=float, device=device)

        x = wp.array(rng.random(size=(n, b)), dtype=float, device=device)
        y = wp.zeros(shape=(m, b), device=device)

        with wp.ScopedTimer("warp-forward" + str(b)):
            wp.launch(mlp_kernel, dim=b, inputs=[weights, bias, x, y], device=device)
            wp.synchronize()

    for i in range(steps):
        b = 2**i

        weights = wp.array(rng.random(size=(m, n)) * 0.5 - 0.5, dtype=float, device=device, requires_grad=True)
        bias = wp.array(rng.random(size=m) * 0.5 - 0.5, dtype=float, device=device, requires_grad=True)

        x = wp.array(rng.random(size=(n, b)), dtype=float, device=device, requires_grad=True)
        y = wp.zeros(shape=(m, b), device=device, requires_grad=True)

        loss = wp.zeros(1, dtype=float, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(mlp_kernel, dim=b, inputs=[weights, bias, x, y], device=device)
            wp.launch(loss_kernel, dim=y.size, inputs=[y.flatten(), loss], device=device)

        # run backward once to ensure all adjoints are allocated
        tape.backward(loss)
        wp.synchronize()

        with wp.ScopedTimer("warp-backward" + str(b)):
            tape.backward(loss)
            wp.synchronize()


# profile_mlp_warp("cuda")
# profile_mlp_torch()


devices = get_test_devices()


class TestMLP(unittest.TestCase):
    pass


add_function_test(TestMLP, "test_mlp", test_mlp, devices=devices, check_output=False)
add_function_test(TestMLP, "test_mlp_grad", test_mlp_grad, devices=devices, check_output=False)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
