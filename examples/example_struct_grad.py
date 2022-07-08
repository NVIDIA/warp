import os
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render

wp.init()

device = "cuda"


@wp.struct
class TestStruct:
    x: wp.vec3
    a: wp.array(dtype=wp.vec3)
    b: wp.array(dtype=wp.vec3)


@wp.kernel
def test_kernel(s: TestStruct):
    tid = wp.tid()
    s.b[tid] = s.a[tid] + s.x


@wp.kernel
# def loss_kernel(sb: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
def loss_kernel(s: TestStruct, loss: wp.array(dtype=float)):
    tid = wp.tid()
    v = s.b[tid]
    wp.atomic_add(loss, 0, float(tid + 1) * (v[0] + 2.0 * v[1] + 3.0 * v[2]))


ts = TestStruct()
ts.x = wp.vec3(1.0, 2.0, 3.0)
ts.a = wp.array(
    np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    dtype=wp.vec3,
    device=device,
    requires_grad=True,
)
ts.b = wp.zeros(2, dtype=wp.vec3, device=device, requires_grad=True)
loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)


tape = wp.Tape()
with tape:
    wp.launch(test_kernel, dim=2, inputs=[ts], device=device)
    wp.launch(loss_kernel, dim=2, inputs=[ts, loss], device=device)

tape.backward(loss)
wp.synchronize()
print(loss)
print(tape.gradients[ts].a)
