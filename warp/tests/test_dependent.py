import warp as wp
import warp.tests.test_dependency as dep

wp.init()

@wp.kernel
def kern(expect: float):
    wp.expect_eq(dep.magic(), expect)

def run(expect, device):
    wp.launch(kern, dim=1, inputs=[expect], device=device)
