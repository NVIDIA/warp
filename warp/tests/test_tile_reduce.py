import numpy as np
import warp as wp

wp.init()
wp.set_module_options({"enable_backward": True})
wp.set_device("cuda:0")
wp.set_module_options({"fast_math": True})
#wp.config.mode = "debug"
#wp.config.verify_cuda = True

wp.build.clear_kernel_cache()

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 64


@wp.kernel
def tile_sum_kernel(input: wp.array3d(dtype=float),
                    output: wp.array(dtype=float)):

    # output tile index
    i, _ = wp.tid()

    a = wp.tile_load(input[i], 0, 0, m=TILE_M, n=TILE_N)
    s = wp.tile_sum(a)*0.5

    wp.tile_store(output, i, 0, s)

def test_tile_sum():

    batch_count = 56

    M = TILE_M
    N = TILE_N

    rng = np.random.default_rng(42)
    input = rng.random((batch_count, M, N), dtype=np.float32)

    input_wp = wp.array(input, requires_grad=True)
    output_wp = wp.zeros(batch_count, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(tile_sum_kernel, dim=[batch_count, TILE_DIM], inputs=[input_wp, output_wp], block_dim=TILE_DIM)


    for i in range(batch_count):
        sum_np = np.sum(input[i])*0.5
        sum_wp = output_wp.numpy()[i]

        assert(np.allclose(sum_np, sum_wp, rtol=1.e-4))

    print("Sum forward passed")

    output_wp.grad.fill_(1.0)

    tape.backward()

    assert(np.allclose(input_wp.grad.numpy(), np.ones_like(input)*0.5, rtol=1.e-4))

    print("Sum backward passed")



@wp.kernel
def tile_reduce_1d_kernel(output: wp.array(dtype=int)):

    # output tile index
    i = wp.tid()
    
    t = wp.tile(i)      # convert to block wide tile    
    s = wp.tile_sum(t)  # sum over block

    # update global sum
    wp.tile_atomic_add(output, i, 0, s)

def test_tile_reduce_1d():

    N = int(TILE_DIM*3/2)

    output = wp.zeros(shape=1, dtype=int, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(tile_reduce_1d_kernel, dim=[N], inputs=[output], block_dim=TILE_DIM)

    assert(np.sum(np.arange(N)), output.numpy())

    print("Sum 1D forward passed")

    # output_wp.grad.fill_(1.0)

    # tape.backward()

    # assert(np.allclose(input_wp.grad.numpy(), np.ones_like(input)*0.5, rtol=1.e-4))

    # print("Sum backward passed")


test_tile_sum()
test_tile_reduce_1d()



















