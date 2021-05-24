from oglang.context import synchronize
import oglang as og

import numpy as np


@og.kernel
def gd_step(arr_x: og.array(dtype=float), 
            arr_dfdx: og.array(dtype=float),
            alpha: float):

    tid = og.tid()

    x = og.load(arr_x, tid)
    dfdx = og.load(arr_dfdx, tid)

    x = x - dfdx*alpha

    og.store(arr_x, tid, x)

@og.kernel
def nesterov1(beta: float,
              x: og.array(dtype=float),
              x_prev: og.array(dtype=float),
              y: og.array(dtype=float)):
    
    tid = og.tid()

    y[tid] = x[tid] + beta*(x[tid] - x_prev[tid])

@og.kernel
def nesterov2(alpha: float,
              beta: og.array(float), 
              eta: og.array(float), 
              x: og.array(dtype=float), 
              x_prev: og.array(dtype=float), 
              y: og.array(dtype=float),
              dfdx: og.array(dtype=float)):

    # if (eta > 0.0):
    #     # adaptive restart
    #     x_prev = x
    #     b = 0
    # else:
    #     # nesterov update
    #     x_prev = x
    #     x = y - alpha*dfdx

    tid = og.tid()

    x_prev[tid] = x[tid]
    x[tid] = y[tid] - alpha*dfdx[tid]

def inner(a, b, out):

    if (a.device != b.device):
        raise RuntimeError("Inner product devices do not match")

    if a.device == "cpu":
        og.runtime.array_inner_host(a, b, out, a.length)
    elif a.device == "cuda":
        og.runtime.array_inner_device(a, b, out, a.length)

class Optimizer:

    def __init__(self, n, mode, device):
        
        self.n = n
        self.mode = mode
        self.device = device
               
        # allocate space for residual buffers
        self.dfdx = og.zeros(n, dtype=float, device=device)

        if (mode == "nesterov"):
            self.x_prev = og.zeros(n, dtype=float, device=device)
            self.y = og.zeros(n, dtype=float, device=device)
            self.eta = og.zeros(1, dtype=float, device=device)
            self.eta_prev = og.zeros(1, dtype=float, device=device)
            self.beta = og.zeros(1, dtype=int, device=device)



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
                og.launch(kernel=gd_step, dim=self.n, inputs=[x, self.dfdx, alpha], device=self.device)

                if (report):

                    stats["evals"] += 1
                    
                    r = np.linalg.norm(self.dfdx.to("cpu").numpy())
                    stats["residual"].append(r)

        elif (self.mode == "nesterov"):
            
            og.copy(self.x_prev, x)

            # momentum index (reset after restart)
            b = 0
            for iter in range(max_iters):

                beta = (b-1.0)/(b+2.0)
                b += 1

                # y = x + beta*(x - x_prev)
                og.launch(kernel=nesterov1, dim=self.n, inputs=[beta, x, self.x_prev, self.y], device=self.device)
                
                # grad
                grad_func(self.y, self.dfdx)

                #inner()
                #np.dot(dfdx, x - x_prev)

                # x = y - alpha*dfdx
                og.launch(kernel=nesterov2, dim=self.n, inputs=[alpha, None, None, x, self.x_prev, self.y, self.dfdx], device=self.device)

                if (report):
    
                    stats["evals"] += 1
                    
                    r = np.linalg.norm(self.dfdx.to("cpu").numpy())
                    stats["residual"].append(r)
                
        else:
            raise RuntimeError("Unknown optimizer")

        if (report):
            print(stats)

    