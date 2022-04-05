# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp
from warp.tests.test_base import *

wp.init()
wp.config.mode = "debug"

@wp.func
def mlp_activation(z: float):
    return wp.tanh(z)

@wp.kernel
def loss_kernel(x: wp.array(dtype=float),
                loss: wp.array(dtype=float)):

    l = x[wp.tid()]
    
    wp.atomic_add(loss, 0, l*l)


@wp.kernel
def mlp_kernel(dim_m: int,
               dim_n: int,
               dim_b: int,
               weights: wp.array(dtype=float),
               bias: wp.array(dtype=float),
               x: wp.array(dtype=float),
               y: wp.array(dtype=float)):


    wp.mlp(weights, bias, mlp_activation, dim_m, dim_n, dim_b, wp.tid(), x, y)
    

def test_mlp(test, device):

    m = 10
    n = 200

    batches = 20000
    
    weights = wp.array(np.random.rand(m, n)*0.5 - 0.5, dtype=float, device=device)
    bias = wp.array(np.random.rand(m)*0.5 - 0.5, dtype=float, device=device)

    x = wp.array(np.random.rand(n*batches), dtype=float, device=device)
    y = wp.zeros(m*batches, device=device)

    # profile = True
    # if profile:

    #     for i in range(18):

    #         batch_size = 2**i        
    #         p = wp.array(np.random.rand(n*batch_size), dtype=float, device=device)
    #         q = wp.zeros(m*batch_size, device=device)

    #         with wp.ScopedTimer("warp_" + str(batch_size), active=profile):
    #             wp.launch(mlp_kernel, dim=batch_size, inputs=[m, n, batch_size, weights, bias, p, q], device=device)
    #             wp.synchronize()


    with wp.ScopedTimer("warp", active=False):
        wp.launch(mlp_kernel, dim=batches, inputs=[m, n, batches, weights, bias, x, y], device=device)
        wp.synchronize()

    # A*x + b
    with wp.ScopedTimer("numpy", active=False):
        expect = np.tanh(weights.numpy().reshape(m,n)@x.numpy().reshape(-1, batches) + bias.numpy().reshape(m, 1))

    # x^T*A^T + b^T
    # with wp.ScopedTimer("numpy", active=profile):
    #     expect = np.tanh(x.numpy().reshape(batches, -1)@weights.numpy().reshape(m,n).T + bias.numpy().reshape(1, m))


    result = y.numpy().reshape(-1, batches)

    assert_np_equal(result, expect, tol=1.e-6)


def create_golden():

    import torch

    class FeedForward(torch.nn.Module):

        def __init__(self, input_size, hidden_size):
            super(FeedForward, self).__init__()

            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.act = torch.nn.Tanh()
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.act(out)
            return out


    input_size = 32
    hidden_size = 16
    batch_size = 64

    network = FeedForward(input_size, hidden_size)

    # profile = True
    # if profile:
    #     for i in range(18):       
    #         with torch.no_grad():
                
    #             batch_size = 2**i
    #             x = torch.Tensor(np.random.rand(batch_size, input_size))

    #             with wp.ScopedTimer("torch_" + str(batch_size)):
    #                 y = network.forward(x)
    #                 torch.cuda.synchronize()

    #     return


    x = torch.Tensor(np.random.rand(batch_size, input_size))
    x.requires_grad = True

    y = network.forward(x)

    loss = torch.norm(y)**2.0
    loss.backward(retain_graph=True)

    for i in range(16):
        with wp.ScopedTimer("torch-backward"):
            loss.backward(retain_graph=True)
            torch.cuda.synchronize()
    


    results = {}
    results["weights"] = network.fc1.weight.cpu().detach().numpy()
    results["weights_grad"] = network.fc1.weight.grad.cpu().detach().numpy()
    results["bias"] = network.fc1.bias.cpu().detach().numpy()
    results["bias_grad"] = network.fc1.bias.grad.cpu().detach().numpy()
    results["x"] = x.cpu().detach().numpy()
    results["x_grad"] = x.grad.cpu().detach().numpy()
    results["y"] = y.cpu().detach().numpy()
    results["loss"] = loss.cpu().detach().numpy()

    np.save(os.path.join(os.path.dirname(__file__), "assets/mlp_golden.npy"), results, allow_pickle=True)
    
def load_golden():
    
    return np.load(os.path.join(os.path.dirname(__file__), "assets/mlp_golden.npy"), allow_pickle=True).item()

# uncomment to re-build golden files
create_golden()


def test_mlp_grad(test, device):

    results = load_golden()

    torch_weights = results["weights"]
    torch_weights_grad = results["weights_grad"]
    torch_bias = results["bias"]
    torch_bias_grad = results["bias_grad"]
    torch_x = results["x"].T
    torch_x_grad = results["x_grad"].T
    torch_y = results["y"].T
    torch_loss = results["loss"].T


    weights = wp.array(torch_weights, dtype=float, device=device, requires_grad=True)
    bias = wp.array(torch_bias, dtype=float, device=device, requires_grad=True)

    x = wp.array(torch_x, dtype=float, device=device, requires_grad=True)
    y = wp.array(torch_y, dtype=float, device=device, requires_grad=True)
    y.zero_()

    loss = wp.zeros(1, dtype=float, device=device)

    m = torch_weights.shape[0]
    n = torch_weights.shape[1]
    b = torch_x.shape[1]

    tape = wp.Tape()
    with tape:
        wp.launch(mlp_kernel, dim=b, inputs=[m, n, b, weights, bias, x, y], device=device)
        wp.launch(loss_kernel, dim=len(y), inputs=[y, loss], device=device)

    tape.backward(loss=loss)

    # # check forward result
    # assert_np_equal(y.numpy().reshape(-1, b), torch_y, tol=1.e-3)
    # assert_np_equal(loss.numpy(), torch_loss, tol=1.e-3)

    # # check backward result
    # assert_np_equal(tape.gradients[weights].numpy().reshape(m, n), torch_weights_grad, tol=1.e-3)

    for i in range(16):
        with wp.ScopedTimer("warp-backward"):
            tape.backward(loss=loss)
            wp.synchronize()
    



def register(parent):

    devices = wp.get_devices()

    class TestMLP(parent):
        pass

    add_function_test(TestMLP, "test_mlp", test_mlp, devices=devices)
    add_function_test(TestMLP, "test_mlp_grad", test_mlp_grad, devices=devices)
    
    #add_function_test(TestMLP, "test_mlp_grad", test_for_loop_nested_if_grad, devices=devices)

    return TestMLP

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
