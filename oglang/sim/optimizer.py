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



class Optimizer:

    def __init__(self, n, mode, device):
        
        self.n = n
        self.mode = mode
        self.device = device
               
        # allocate space for residual buffers
        self.dfdx = og.zeros(n, dtype=float, device=device)


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
                og.launch(
                    kernel=gd_step,
                    dim=self.n,
                    inputs=[x, self.dfdx, alpha],
                    device=self.device)

                if (report):

                    stats["evals"] += 1
                    
                    r = np.linalg.norm(self.dfdx.to("cpu").numpy())
                    stats["residual"].append(r)
                

        if (self.mode == "nesterov"):

            x_prev = np.copy(x)
            restart = self.args["adaptive_restart"]
            # momentum index (reset after restart)
            b = 0
            for iter in range(max_iters):
                beta = (b-1.0)/(b+2.0)
                b += 1
                y = x + beta*(x - x_prev)
                dfdx = func(y)
                if (restart and np.dot(dfdx, x - x_prev) > 0.0):
                    # adaptive restart
                    theta_prev = 1.0
                    x_prev = x
                    b = 0
                else:
                    # nesterov update
                    x_prev = x
                    x = y - alpha*dfdx
                err = np.linalg.norm(dfdx)
                # compute residual |dfdx|
                if (notify):
                    notify(iter, x, err)
                if (err < tol):
                    break

        if (report):
            print(stats)

        else:
            raise RuntimeError("Unknown optimizer")
    