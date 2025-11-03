import cupy as cp
from cupy.cuda import memory

import warp as wp

BLOCKSIZE = wp.constant(64)
nx = 8 * 20 * BLOCKSIZE
ny = 10 * BLOCKSIZE


@wp.kernel
def copy(A: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    x, y = wp.tid()

    # Load tile from A and transpose it
    a = wp.tile_load(A, shape=(BLOCKSIZE, BLOCKSIZE), offset=(BLOCKSIZE * x, BLOCKSIZE * y))
    wp.tile_store(C, a, offset=(BLOCKSIZE * x, BLOCKSIZE * y))


wp.init()

_streams = {}


# One stream per device
def get_stream(device_name: str):
    if device_name not in _streams:
        _streams[device_name] = wp.Stream(device=device_name)
    return _streams[device_name]


dtype = cp.float32  # or cp.float64, but must match wp dtype
nbytes = nx * ny * cp.dtype(dtype).itemsize

memptr = memory.malloc_managed(nbytes)

cupy_arr = cp.ndarray((nx, ny), dtype=dtype, memptr=memptr)
A = wp.from_dlpack(cupy_arr.toDlpack())


@wp.kernel
def range_fill_kernel(out: wp.array2d(dtype=float)):
    i, j = wp.tid()
    # Row-major ordering: row_index * num_cols + col_index
    out[i, j] = wp.float(i * nx + j)


wp.launch(range_fill_kernel, dim=(nx, ny), outputs=[A])
# print(A.numpy())

refC = wp.zeros((nx, ny), dtype=float)
wp.launch_tiled(copy, dim=(nx // BLOCKSIZE, ny // BLOCKSIZE), inputs=[A], outputs=[refC], device="cuda:0", block_dim=32)

# Use blocked policy to automatically compute everything
ndevices = wp.get_cuda_device_count()
nplaces = min(8, ndevices)  # Use up to 8 GPUs, or whatever is available
print(
    f"Running with {nplaces} place{'s' if nplaces != 1 else ''} on {ndevices} CUDA device{'s' if ndevices != 1 else ''}"
)

result = wp.blocked(dim=(nx // BLOCKSIZE, ny // BLOCKSIZE), places=nplaces)

for iter in range(10):
    memptrC = memory.malloc_managed(nbytes)
    cupy_arrC = cp.ndarray((nx, ny), dtype=dtype, memptr=memptrC)
    C = wp.from_dlpack(cupy_arrC.toDlpack())

    e0 = wp.Event(device="cuda:0")
    get_stream("cuda:0").record_event(e0)
    for devid in range(len(result.offsets)):
        device_name = f"cuda:{devid % ndevices}"
        if devid != 0:
            # print(f"Sync 0 with devid {device_name}")
            get_stream(device_name).wait_event(e0)

    for devid, offset in enumerate(result.offsets):
        device_name = f"cuda:{devid % ndevices}"
        stream = get_stream(device_name)
        # print(f"launch work on {device_name} in stream {stream} stream.device {stream.device}")
        # When using stream argument, device can be ignored
        wp.launch_tiled(
            copy,
            dim=(nx // BLOCKSIZE, ny // BLOCKSIZE),
            inputs=[A],
            outputs=[C],
            partition=result.partition,
            offset=offset,
            block_dim=32,
            stream=stream,
        )

    for devid in range(len(result.offsets)):
        device_name = f"cuda:{devid % ndevices}"
        if devid != 0:
            # print(f"Sync {device_name} with 0 (0 waits)")
            ei = wp.Event(device=device_name)
            get_stream(device_name).record_event(ei)
            get_stream("cuda:0").wait_event(ei)

    wp.synchronize_stream(get_stream("cuda:0"))

    # Check that C matches refC
    C_numpy = C.numpy()
    refC_numpy = refC.numpy()
    if cp.allclose(C_numpy, refC_numpy):
        print(f"Iteration {iter}: PASS - C matches refC")
    else:
        print(f"Iteration {iter}: FAIL - C does not match refC")
        diff = cp.abs(C_numpy - refC_numpy)
        print(f"  Max difference: {cp.max(diff)}")
        print(f"  Mean difference: {cp.mean(diff)}")
