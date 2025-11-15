import warp as wp

BLOCKSIZE = wp.constant(2)


@wp.kernel
def copy(A: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    x, y = wp.tid()

    # Load tile from A and transpose it
    a = wp.tile_load(A, shape=(BLOCKSIZE, BLOCKSIZE), offset=(BLOCKSIZE * x, BLOCKSIZE * y))
    wp.tile_store(C, a, offset=(BLOCKSIZE * x, BLOCKSIZE * y))


nx = 12
ny = 12
A = wp.empty((nx, ny), dtype=float)


@wp.kernel
def range_fill_kernel(out: wp.array2d(dtype=float)):
    i, j = wp.tid()
    # Row-major ordering: row_index * num_cols + col_index
    out[i, j] = wp.float(i * ny + j)


wp.launch(range_fill_kernel, dim=(nx, ny), outputs=[A])
print(A.numpy())

refC = wp.zeros((nx, ny), dtype=float)
wp.launch_tiled(copy, dim=(nx // BLOCKSIZE, ny // BLOCKSIZE), inputs=[A], outputs=[refC], device="cuda:0", block_dim=32)
print(refC.numpy())

# Use cyclic policy for round-robin distribution
# dim is in terms of tiles: (nx//BLOCKSIZE, ny//BLOCKSIZE) = (6, 4)
# places (2, 2) means 2x2 grid
result = wp.cyclic(dim=(nx // BLOCKSIZE, ny // BLOCKSIZE), places=(2, 2))
# result = wp.cyclic(dim=(nx//BLOCKSIZE, ny//BLOCKSIZE), places=1)

print("\nCyclic policy result:")
print(f"  Partition: {result.partition}")
print("  Expected:  (6, 4):(2, 2)")
print(f"  Offsets: {result.offsets}")
print(f"  Block shape: {result.block_shape}")

for i, offset in enumerate(result.offsets):
    C = wp.zeros((nx, ny), dtype=float)
    print(f"\n=== Place {i}, Offset {offset} ===")
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

C = wp.zeros((nx, ny), dtype=float)
wp.launch_tiled(
    copy,
    dim=(nx // BLOCKSIZE, ny // BLOCKSIZE),
    inputs=[A],
    outputs=[C],
    device="cuda:0",
    partition=wp.Layout(shape=(3, 3), stride=(2, 12)),
    offset=7,
    block_dim=32,
)
print(C.numpy())
