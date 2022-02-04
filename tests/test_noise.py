import warp as wp
import numpy as np
from PIL import Image

device = "cuda"

wp.init()

@wp.kernel
def test_pnoise(
    kernel_seed: int,
    W: int,
    px: int,
    py: int,
    noise_values: wp.array(dtype=float)):

    tid = wp.tid()

    state = wp.rand_init(kernel_seed, kernel_seed)
    x = float(tid % W) + 0.5
    y = float(tid / W) + 0.5
    p = wp.vec2(x, y)

    n = wp.pnoise(state, p, px, py)
    g = n * 255.0
    wp.store(noise_values, tid, g)


# period
px = 128
py = 512

# image dim
W = 1024
H = 1024
N = W * H

pixels_host = wp.zeros(N, dtype=float, device="cpu")
pixels = wp.zeros(N, dtype=float, device=device)
seed = 42

wp.launch(
    kernel=test_pnoise,
    dim=N,
    inputs=[seed, W, px, py, pixels],
    outputs=[],
    device=device
)

wp.copy(pixels_host, pixels)
wp.synchronize()

img = pixels_host.numpy()
img = np.reshape(img, (W, H))
img = img.astype(np.uint8)
img = Image.fromarray(img)
img.show()