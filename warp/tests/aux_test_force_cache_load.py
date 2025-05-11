import warp as wp


@wp.kernel
def add_kernel(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32), res: wp.array(dtype=wp.int32)):
    i = wp.tid()
    res[i] = a[i] + b[i]


def run(a, b, res, device):
    wp.launch(add_kernel, dim=a.shape, inputs=[a, b], outputs=[res], device=device)
