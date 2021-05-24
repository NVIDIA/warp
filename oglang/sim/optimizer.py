from oglang.context import synchronize
import oglang as og


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


    def solve(self, x, grad_func, max_iters=20, alpha=0.01):
        
        if (self.mode == "gd"):

            for i in range(max_iters):

                # compute residual
                grad_func(x, self.dfdx)

                og.launch(
                    kernel=gd_step,
                    dim=self.n,
                    inputs=[x, self.dfdx, alpha],
                    device=self.device)

        else:
            raise RuntimeError("Unknown optimizer")
    