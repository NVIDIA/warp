import time

import cupy as cp
import numpy as np

import warp as wp

BLOCKSIZE = wp.constant(64)
nx = 128 * BLOCKSIZE
ny = 128 * BLOCKSIZE
nplaces = 8


@wp.kernel
def copy(A: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    x, y = wp.tid()

    a = wp.tile_load(A, shape=(BLOCKSIZE, BLOCKSIZE), offset=(BLOCKSIZE * x, BLOCKSIZE * y))
    wp.tile_store(C, a, offset=(BLOCKSIZE * x, BLOCKSIZE * y))


wp.init()

# Query number of CUDA devices
ndevices = wp.get_cuda_device_count()
nplaces = min(nplaces, ndevices)  # Adapt to available GPUs
print(
    f"Running with {nplaces} place{'s' if nplaces != 1 else ''} on {ndevices} CUDA device{'s' if ndevices != 1 else ''}"
)
_streams = {}


# One stream per device
def get_stream(device_name: str):
    if device_name not in _streams:
        _streams[device_name] = wp.Stream(device=device_name)
    return _streams[device_name]


dtype = cp.float32  # or cp.float64, but must match wp dtype
nbytes = nx * ny * cp.dtype(dtype).itemsize


@wp.kernel
def range_fill_kernel(out: wp.array2d(dtype=float)):
    i, j = wp.tid()
    # Row-major ordering: row_index * num_cols + col_index
    out[i, j] = wp.float(i * nx + j)


# Legacy version: single stream, single device

refA = wp.zeros((nx, ny), dtype=float)
wp.launch(range_fill_kernel, dim=(nx, ny), outputs=[refA], device="cuda:0", block_dim=32)

refC = wp.zeros((nx, ny), dtype=float)
wp.launch_tiled(
    copy, dim=(nx // BLOCKSIZE, ny // BLOCKSIZE), inputs=[refA], outputs=[refC], device="cuda:0", block_dim=32
)

wp.synchronize_device(device="cuda:0")
time.sleep(0.05)

# Use blocked policy to automatically compute everything
result = wp.blocked(dim=(nx // BLOCKSIZE, ny // BLOCKSIZE), places=nplaces)

# Localized version: multiple devices, non localized memory (using managed memory)

# Create streams for each place, distributed across devices
streams = [get_stream(f"cuda:{i % ndevices}") for i in range(len(result.offsets))]

managedA = wp.zeros_managed((nx, ny), dtype=float)
managedC = wp.zeros_managed((nx, ny), dtype=float)

# Old way (kept for reference):
# memptr = memory.malloc_managed(nbytes)
# cupy_arr = cp.ndarray((nx, ny), dtype=dtype, memptr=memptr)
# managedA = wp.from_dlpack(cupy_arr.toDlpack())
#
# memptrC = memory.malloc_managed(nbytes)
# cupy_arrC = cp.ndarray((nx, ny), dtype=dtype, memptr=memptrC)
# managedC = wp.from_dlpack(cupy_arrC.toDlpack())

wp.launch(range_fill_kernel, dim=(nx, ny), outputs=[managedA], device="cuda:0", block_dim=32)

for iter in range(5):
    wp.launch_tiled_localized(
        copy,
        dim=(nx // BLOCKSIZE, ny // BLOCKSIZE),
        inputs=[managedA],
        outputs=[managedC],
        primary_stream=get_stream("cuda:0"),
        block_dim=32,
        mapping=result,
        streams=streams,
    )

# Localized version: multiple devices, localized memory (using VMM)
wp.synchronize_stream(get_stream("cuda:0"))
time.sleep(0.05)

localizedC = wp.empty_localized(
    shape=(nx, ny), tile_dim=(BLOCKSIZE, BLOCKSIZE), partition_desc=result, streams=streams, dtype=dtype
)

localizedA = wp.empty_localized(
    shape=(nx, ny), tile_dim=(BLOCKSIZE, BLOCKSIZE), partition_desc=result, streams=streams, dtype=dtype
)

wp.launch(range_fill_kernel, dim=(nx, ny), outputs=[localizedA])
for iter in range(5):
    wp.launch_tiled_localized(
        copy,
        dim=(nx // BLOCKSIZE, ny // BLOCKSIZE),
        inputs=[localizedA],
        outputs=[localizedC],
        primary_stream=get_stream("cuda:0"),
        block_dim=32,
        mapping=result,
        streams=streams,
    )
wp.synchronize_stream(get_stream("cuda:0"))
time.sleep(0.05)

# Check that C matches refC
C_numpy = localizedC.numpy()
refC_numpy = refC.numpy()
if np.allclose(C_numpy, refC_numpy):
    print(f"Iteration {iter}: PASS - C matches refC")
else:
    print(f"Iteration {iter}: FAIL - C does not match refC")
    diff = cp.abs(C_numpy - refC_numpy)
    print(f"  Max difference: {cp.max(diff)}")
    print(f"  Mean difference: {cp.mean(diff)}")
