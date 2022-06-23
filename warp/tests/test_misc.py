import warp as wp
import numpy as np

wp.init()
wp.config.verify_cuda = True

device = "cuda"

h = np.array([1.0, 2.0, 3.0, -3.14159], dtype=np.float16)

w = wp.array(h, dtype=wp.float16, device=device)

print(h)
print(w)

@wp.kernel
def test_half(a: wp.array(dtype=wp.float16)):

    #h = wp.float16(1.0)
    #b = 1 + 1
    h = a[wp.tid()]

    #print(float(h))



wp.launch(test_half, dim=len(w), inputs=[w], device=device)

