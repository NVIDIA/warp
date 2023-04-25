import warp as wp

wp.init()


@wp.func
def multiply(x: float):
    return x * x


@wp.kernel
def kern(expect: float):
    wp.expect_eq(multiply(4.0), expect)


def run(expect, device):
    wp.launch(kern, dim=1, inputs=[expect], device=device)
