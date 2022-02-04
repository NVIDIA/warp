import warp as wp

device = "cuda"

wp.init()

@wp.kernel
def test_kernel(
    kernel_seed: int,
    int_a: wp.array(dtype=int),
    int_ab: wp.array(dtype=int),
    float_01: wp.array(dtype=float),
    float_ab: wp.array(dtype=float)):

    tid = wp.tid()

    state = wp.rand_init(kernel_seed, tid)

    wp.store(int_a, tid, wp.randi(state))
    wp.store(int_ab, tid, wp.randi(state, 0, 100))
    wp.store(float_01, tid, wp.randf(state))
    wp.store(float_ab, tid, wp.randf(state, 0.0, 100.0))

N = 10

int_a_device = wp.zeros(N, dtype=int, device=device)
int_ab_device = wp.zeros(N, dtype=int, device=device)

float_01_device = wp.zeros(N, dtype=float, device=device)
float_ab_device = wp.zeros(N, dtype=float, device=device)

seed = 42

wp.launch(
    kernel=test_kernel,
    dim=N,
    inputs=[seed, int_a_device, int_ab_device, float_01_device, float_ab_device],
    outputs=[],
    device=device
)

wp.synchronize()

print("Test Rand")
print("randi(state) array")
print(int_a_device)
print("randi(state, 0, 100) array")
print(int_ab_device)
print("randf(state) array")
print(float_01_device)
print("randf(state, 0, 100) array")
print(float_ab_device)