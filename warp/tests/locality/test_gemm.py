import warp as wp

TILE_M = wp.constant(32)
TILE_N = wp.constant(64)
TILE_K = wp.constant(64)

_streams = {}
ndevices = len([d for d in wp.get_devices() if d.is_cuda])
print(f"Found {ndevices} CUDA device(s)")


# One stream per device
def get_stream(device_name: str):
    if device_name not in _streams:
        _streams[device_name] = wp.Stream(device=device_name)
    return _streams[device_name]


@wp.kernel
def gemm_tiled(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    i, j = wp.tid()

    # allocate output tile
    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=float)
    count = int(K / TILE_K)

    # iterate over inner dimension
    for k in range(count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))

        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

        # perform gemm + accumulate
        wp.tile_matmul(a, b, sum)

    # store result
    wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))


# test with 1024^2 inputs
M, N, K = 4 * 1024, 4 * 1024, 4 * 1024

# Use blocked policy to automatically compute everything
result = wp.blocked(dim=(M // TILE_M, N // TILE_N), places=8)

# Localized version: multiple devices, non localized memory (using managed memory)

# Create streams for each place, distributed across devices
streams = [get_stream(f"cuda:{i % ndevices}") for i in range(len(result.offsets))]

A = wp.empty_tiled((M, K), tile_dim=(TILE_M, TILE_K), partition_desc=result, streams=streams, dtype=wp.float32)
B = wp.empty_tiled((K, N), tile_dim=(TILE_K, TILE_N), partition_desc=result, streams=streams, dtype=wp.float32)
C = wp.empty_tiled((M, N), tile_dim=(TILE_M, TILE_N), partition_desc=result, streams=streams, dtype=wp.float32)

# launch kernel with 128 threads per-block
wp.launch_tiled_localized(
    gemm_tiled,
    dim=(int(M // TILE_M), int(N // TILE_N)),
    inputs=[A, B],
    outputs=[C],
    block_dim=128,
    mapping=result,
    streams=streams,
)
