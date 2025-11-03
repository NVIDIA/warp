import cupy as cp
from cupy.cuda import memory

import warp as wp

BLOCKSIZE = wp.constant(2)


@wp.kernel
def copy(A: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    x, y = wp.tid()

    # Load tile from A and transpose it
    a = wp.tile_load(A, shape=(BLOCKSIZE, BLOCKSIZE), offset=(BLOCKSIZE * x, BLOCKSIZE * y))
    wp.tile_store(C, a, offset=(BLOCKSIZE * x, BLOCKSIZE * y))


wp.init()

nx = 1200
ny = 800
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
print(A.numpy())

refC = wp.zeros((nx, ny), dtype=float)
wp.launch_tiled(copy, dim=(nx // BLOCKSIZE, ny // BLOCKSIZE), inputs=[A], outputs=[refC], device="cuda:0", block_dim=32)
print(refC.numpy())

# Use blocked policy to automatically compute everything
ndevices = wp.get_cuda_device_count()
nplaces = min(3, ndevices)  # Use up to 3 GPUs, or whatever is available
print(
    f"Running with {nplaces} place{'s' if nplaces != 1 else ''} on {ndevices} CUDA device{'s' if ndevices != 1 else ''}"
)

result = wp.blocked(dim=(nx // BLOCKSIZE, ny // BLOCKSIZE), places=nplaces)

for iter in range(10):
    memptrC = memory.malloc_managed(nbytes)
    cupy_arrC = cp.ndarray((nx, ny), dtype=dtype, memptr=memptrC)
    C = wp.from_dlpack(cupy_arrC.toDlpack())

    for devid, offset in enumerate(result.offsets):
        device_name = f"cuda:{devid % ndevices}"
        wp.launch_tiled(
            copy,
            dim=(nx // BLOCKSIZE, ny // BLOCKSIZE),
            inputs=[A],
            outputs=[C],
            device=device_name,
            partition=result.partition,
            offset=offset,
            block_dim=32,
        )

    wp.synchronize()
    print(C.numpy())
