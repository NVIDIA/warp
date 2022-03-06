# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from warp.context import synchronize
import warp as wp

import numpy as np


@wp.kernel
def gd_step(arr_x: wp.array(dtype=float), 
            arr_dfdx: wp.array(dtype=float),
            alpha: float):

    tid = wp.tid()

    x = arr_x[tid]
    dfdx = arr_dfdx[tid]

    x = x - dfdx*alpha

    arr_x[tid] = x

@wp.kernel
def nesterov1(beta: float,
              x: wp.array(dtype=float),
              x_prev: wp.array(dtype=float),
              y: wp.array(dtype=float)):
    
    tid = wp.tid()

    y[tid] = x[tid] + beta*(x[tid] - x_prev[tid])

@wp.kernel
def nesterov2(alpha: float,
              beta: wp.array(dtype=float), 
              eta: wp.array(dtype=float), 
              x: wp.array(dtype=float), 
              x_prev: wp.array(dtype=float), 
              y: wp.array(dtype=float),
              dfdx: wp.array(dtype=float)):

    # if (eta > 0.0):
    #     # adaptive restart
    #     x_prev = x
    #     b = 0
    # else:
    #     # nesterov update
    #     x_prev = x
    #     x = y - alpha*dfdx

    tid = wp.tid()

    x_prev[tid] = x[tid]
    x[tid] = y[tid] - alpha*dfdx[tid]

def inner(a, b, out):

    if (a.device != b.device):
        raise RuntimeError("Inner product devices do not match")

    if a.device == "cpu":
        wp.runtime.array_inner_host(a, b, out, a.length)
    elif a.device == "cuda":
        wp.runtime.array_inner_device(a, b, out, a.length)

class Optimizer:

    def __init__(self, n, mode, device):
        
        self.n = n
        self.mode = mode
        self.device = device
               
        # allocate space for residual buffers
        self.dfdx = wp.zeros(n, dtype=float, device=device)

        if (mode == "nesterov"):
            self.x_prev = wp.zeros(n, dtype=float, device=device)
            self.y = wp.zeros(n, dtype=float, device=device)
            self.eta = wp.zeros(1, dtype=float, device=device)
            self.eta_prev = wp.zeros(1, dtype=float, device=device)
            self.beta = wp.zeros(1, dtype=int, device=device)



    def solve(self, x, grad_func, max_iters=20, alpha=0.01, report=False):
        

        if (report):

            stats = {}

            # reset stats
            stats["evals"] = 0
            stats["residual"] = []
            

        if (self.mode == "gd"):

            for i in range(max_iters):

                # compute residual
                grad_func(x, self.dfdx)

                # gradient step
                wp.launch(kernel=gd_step, dim=self.n, inputs=[x, self.dfdx, alpha], device=self.device)

                if (report):

                    stats["evals"] += 1
                    
                    r = np.linalg.norm(self.dfdx.to("cpu").numpy())
                    stats["residual"].append(r)

        elif (self.mode == "nesterov"):
            
            wp.copy(self.x_prev, x)

            # momentum index (reset after restart)
            b = 0
            for iter in range(max_iters):

                beta = (b-1.0)/(b+2.0)
                b += 1

                # y = x + beta*(x - x_prev)
                wp.launch(kernel=nesterov1, dim=self.n, inputs=[beta, x, self.x_prev, self.y], device=self.device)
                
                # grad
                grad_func(self.y, self.dfdx)

                #inner()
                #np.dot(dfdx, x - x_prev)

                # x = y - alpha*dfdx
                wp.launch(kernel=nesterov2, dim=self.n, inputs=[alpha, None, None, x, self.x_prev, self.y, self.dfdx], device=self.device)

                if (report):
    
                    stats["evals"] += 1
                    
                    r = np.linalg.norm(self.dfdx.to("cpu").numpy())
                    stats["residual"].append(r)
                
        else:
            raise RuntimeError("Unknown optimizer")

        if (report):
            print(stats)

    