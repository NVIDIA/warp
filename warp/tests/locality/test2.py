import warp as wp

BLOCKSIZE = wp.constant(2)


@wp.kernel
def copy(A: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    x, y = wp.tid()

    # Load tile from A and transpose it
    a = wp.tile_load(A, shape=(BLOCKSIZE, BLOCKSIZE), offset=(BLOCKSIZE * x, BLOCKSIZE * y))
    wp.tile_store(C, a, offset=(BLOCKSIZE * x, BLOCKSIZE * y))


nx = 12
ny = 8
A = wp.empty((nx, ny), dtype=float)


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
result = wp.blocked(dim=(nx // BLOCKSIZE, ny // BLOCKSIZE), places=3)

for offset in result.offsets:
    C = wp.zeros((nx, ny), dtype=float)
    wp.launch_tiled(
        copy,
        dim=(nx // BLOCKSIZE, ny // BLOCKSIZE),
        inputs=[A],
        outputs=[C],
        device="cuda:0",
        partition=result.partition,
        offset=offset,
        block_dim=32,
    )
    print(C.numpy())
