import warp as wp
import numpy as np

device = "cuda"
compare_to_numpy = False

wp.init()

@wp.kernel
def test_kernel(
    x: wp.array(dtype=float),
    x_round: wp.array(dtype=float),
    x_rint: wp.array(dtype=float),
    x_trunc: wp.array(dtype=float),
    x_cast: wp.array(dtype=float),
    x_floor: wp.array(dtype=float),
    x_ceil: wp.array(dtype=float)):

    tid = wp.tid()

    x_round[tid] = wp.round(x[tid])
    x_rint[tid] = wp.rint(x[tid])
    x_trunc[tid] = wp.trunc(x[tid])
    x_cast[tid] = float(int(x[tid]))
    x_floor[tid] = wp.floor(x[tid])
    x_ceil[tid] = wp.ceil(x[tid])


nx = np.array([ 4.9,  4.5,  4.1,  3.9,  3.5,  3.1,  2.9,  2.5,  2.1,  1.9,  1.5,  1.1,  0.9,  0.5,  0.1,
               -0.1, -0.5, -0.9, -1.1, -1.5, -1.9, -2.1, -2.5, -2.9, -3.1, -3.5, -3.9, -4.1, -4.5, -4.9],
               dtype=np.float32)

x = wp.array(nx, device=device)
N = x.length

x_round = wp.empty(N, dtype=float, device=device)
x_rint = wp.empty(N, dtype=float, device=device)
x_trunc = wp.empty(N, dtype=float, device=device)
x_cast = wp.empty(N, dtype=float, device=device)
x_floor = wp.empty(N, dtype=float, device=device)
x_ceil = wp.empty(N, dtype=float, device=device)

wp.launch(
    kernel=test_kernel,
    dim=N,
    inputs=[x, x_round, x_rint, x_trunc, x_cast, x_floor, x_ceil],
    device=device
)

wp.synchronize()

nx_round = x_round.numpy().reshape(N)
nx_rint = x_rint.numpy().reshape(N)
nx_trunc = x_trunc.numpy().reshape(N)
nx_cast = x_cast.numpy().reshape(N)
nx_floor = x_floor.numpy().reshape(N)
nx_ceil = x_ceil.numpy().reshape(N)

tab = np.stack([nx, nx_round, nx_rint, nx_trunc, nx_cast, nx_floor, nx_ceil], axis=1)

np.set_printoptions(formatter={'float': lambda x: "{:6.1f}".format(x).replace(".0", ".")})

print("----------------------------------------------")
print("   %5s %5s %5s %5s %5s %5s %5s" % ("x ", "round", "rint", "trunc", "cast", "floor", "ceil"))
print(tab)
print("----------------------------------------------")

if compare_to_numpy:
    nx_round = np.round(nx)
    nx_rint = np.rint(nx)
    nx_trunc = np.trunc(nx)
    nx_fix = np.fix(nx)
    nx_floor = np.floor(nx)
    nx_ceil = np.ceil(nx)

    tab = np.stack([nx, nx_round, nx_rint, nx_trunc, nx_fix, nx_floor, nx_ceil], axis=1)
    print("   %5s %5s %5s %5s %5s %5s %5s" % ("x ", "round", "rint", "trunc", "fix", "floor", "ceil"))
    print(tab)
    print("----------------------------------------------")
