import warp as wp
import numpy as np
import time

wp.init()


@wp.kernel
def saxpy(x: wp.array(dtype=float), y: wp.array(dtype=float), a: float):
    i = wp.tid()
    y[i] = a * x[i] + y[i]


n = 1_000_000
x = wp.array(np.ones(n, dtype=np.float32), device="cuda")
y = wp.array(np.zeros(n, dtype=np.float32), device="cuda")

# Warmup
wp.launch(saxpy, dim=n, inputs=[x, y, 2.0], device="cuda")
wp.synchronize()

t0 = time.perf_counter()
for _ in range(100):
    wp.launch(saxpy, dim=n, inputs=[x, y, 2.0], device="cuda")
wp.synchronize()
elapsed = (time.perf_counter() - t0) / 100
print(f"saxpy kernel: {elapsed*1000:.3f} ms/launch over 1M elements")
